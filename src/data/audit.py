"""Duplicate and near-duplicate detection — the core of the audit.

The metadata already tells us which images share a `lesion_id`. This module looks for
the redundancy the metadata does *not* declare: pairs of images that are visually the
same (or near-identical) but are labelled as different lesions. Published audits of
HAM10000 have found exactly this, including pairs that straddle train/test boundaries.

Two complementary detectors, because they fail differently:

  * Perceptual hashing (pHash) -- fast, exact-ish. Catches re-encodings, crops, minor
    colour shifts. Cheap enough to run over all pairs via BK-tree-free bucketing.
  * CNN embedding similarity -- slower, semantic. Catches the same lesion photographed
    at a different angle/zoom, which pHash misses entirely.

Both emit candidate pairs. Candidates are NOT ground truth: `make_contact_sheet`
renders them for manual verification so detector precision can be reported honestly
rather than assumed.

Usage
-----
    python -m src.data.audit --phash
    python -m src.data.audit --embeddings
    python -m src.data.audit --phash --embeddings --contact-sheet
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, set_seed
from src.data.download import index_images


# --------------------------------------------------------------------------------------
# Perceptual hashing
# --------------------------------------------------------------------------------------

def compute_phashes(image_paths: dict[str, Path], hash_size: int = 16) -> pd.DataFrame:
    """Compute a perceptual hash per image.

    Returns a DataFrame with columns [image_id, hash_hex, bits] where `bits` is the
    hash as a flat boolean array, kept for fast Hamming distance computation.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pHash needs ImageHash and Pillow: pip install ImageHash Pillow"
        ) from exc

    records = []
    for image_id, path in tqdm(sorted(image_paths.items()), desc="pHash", unit="img"):
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img, hash_size=hash_size)
        except Exception as exc:  # corrupt/unreadable file -- report, don't crash
            print(f"⚠ Could not hash {image_id}: {exc}")
            continue
        records.append({
            "image_id": image_id,
            "hash_hex": str(h),
            "bits": h.hash.flatten(),
        })
    return pd.DataFrame(records)


def phash_candidate_pairs(hashes: pd.DataFrame, max_hamming: int = 12) -> pd.DataFrame:
    """Find image pairs whose pHashes are within `max_hamming` bits.

    Exact duplicates are found by grouping identical hashes (cheap). Near-duplicates
    need pairwise comparison; this is done as a vectorised XOR-popcount over a packed
    bit matrix, which handles ~10k images comfortably.
    """
    if hashes.empty:
        return pd.DataFrame(columns=["image_id_a", "image_id_b", "hamming", "method"])

    ids = hashes["image_id"].to_numpy()
    bits = np.vstack(hashes["bits"].to_numpy()).astype(np.uint8)  # (n, n_bits)
    n = len(ids)

    pairs: list[dict] = []
    # Chunked to bound memory: compare block i against all j > i.
    chunk = 512
    for start in tqdm(range(0, n, chunk), desc="pHash pairs", unit="blk"):
        stop = min(start + chunk, n)
        block = bits[start:stop]                       # (b, n_bits)
        # Hamming distance = count of differing bits.
        dist = (block[:, None, :] != bits[None, :, :]).sum(axis=2)  # (b, n)
        for local_i in range(stop - start):
            global_i = start + local_i
            # Only consider j > i to avoid duplicate pairs and self-comparison.
            js = np.where(dist[local_i] <= max_hamming)[0]
            for j in js:
                if j <= global_i:
                    continue
                pairs.append({
                    "image_id_a": ids[global_i],
                    "image_id_b": ids[j],
                    "hamming": int(dist[local_i, j]),
                    "method": "phash",
                })

    return pd.DataFrame(pairs)


# --------------------------------------------------------------------------------------
# Embedding similarity
# --------------------------------------------------------------------------------------

def compute_embeddings(
    image_paths: dict[str, Path],
    backbone: str = "resnet50",
    batch_size: int = 64,
    image_size: int = 224,
) -> tuple[list[str], np.ndarray]:
    """Extract penultimate-layer CNN features for every image.

    Uses ImageNet-pretrained weights purely as a feature extractor -- no training.
    Returns (image_ids, embeddings) with embeddings L2-normalised so that a dot
    product is cosine similarity.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as T
        from torchvision import models
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Embeddings need torch/torchvision: pip install torch torchvision"
        ) from exc

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Extracting embeddings on: {device}")

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _ImageDataset(Dataset):
        def __init__(self, items: list[tuple[str, Path]]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            image_id, path = self.items[idx]
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB"))
            return image_id, tensor

    items = sorted(image_paths.items())
    loader = DataLoader(_ImageDataset(items), batch_size=batch_size, shuffle=False)

    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone for embeddings: {backbone}")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()          # keep the 2048-d penultimate features
    net.eval().to(device)

    ids: list[str] = []
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids, batch in tqdm(loader, desc="embeddings", unit="batch"):
            feats = net(batch.to(device))
            feats = torch.nn.functional.normalize(feats, dim=1)   # -> cosine via dot
            ids.extend(batch_ids)
            chunks.append(feats.cpu().numpy())

    return ids, np.vstack(chunks)


def embedding_candidate_pairs(
    ids: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.98,
    n_neighbours: int = 5,
) -> pd.DataFrame:
    """Find pairs whose embeddings exceed `threshold` cosine similarity.

    Uses a nearest-neighbour search rather than the full N^2 matrix so this stays
    tractable as the dataset grows.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(n_neighbours + 1, len(ids))   # +1 because the nearest neighbour is self
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    seen: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    for i in range(len(ids)):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            similarity = 1.0 - float(dist)
            if similarity < threshold:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "image_id_a": ids[key[0]],
                "image_id_b": ids[key[1]],
                "cosine": round(similarity, 5),
                "method": "embedding",
            })
    return pd.DataFrame(pairs)


# --------------------------------------------------------------------------------------
# Reconciliation against the metadata
# --------------------------------------------------------------------------------------

def annotate_pairs(pairs: pd.DataFrame, meta: pd.DataFrame,
                   group_col: str = "lesion_id", id_col: str = "image_id",
                   label_col: str = "dx") -> pd.DataFrame:
    """Tag each candidate pair with lesion/label agreement.

    The interesting rows are `same_lesion == False`: visually near-identical images
    that the metadata claims are *different* lesions. Those are the undeclared
    duplicates, and the ones that can silently leak across a lesion-grouped split.
    """
    if pairs.empty:
        return pairs

    lookup = meta.set_index(id_col)
    for side in ("a", "b"):
        pairs[f"lesion_{side}"] = pairs[f"image_id_{side}"].map(lookup[group_col])
        pairs[f"dx_{side}"] = pairs[f"image_id_{side}"].map(lookup[label_col])

    pairs["same_lesion"] = pairs["lesion_a"] == pairs["lesion_b"]
    pairs["same_dx"] = pairs["dx_a"] == pairs["dx_b"]
    return pairs


def build_duplicate_groups(pairs: pd.DataFrame) -> dict[str, str]:
    """Collapse undeclared duplicate pairs into connected components.

    Returns image_id -> canonical group id. Union-find over pairs, so a chain
    A~B, B~C produces a single group {A,B,C}. These groups are merged into the
    lesion groups when building the honest split, so near-duplicates cannot be
    separated across train and test.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)   # deterministic root

    for a, b in zip(pairs["image_id_a"], pairs["image_id_b"]):
        union(a, b)

    return {img: find(img) for img in parent}


def make_contact_sheet(pairs: pd.DataFrame, image_paths: dict[str, Path],
                       out_path: Path, max_pairs: int = 20) -> Path | None:
    """Render candidate pairs side by side for manual verification.

    Automated detection produces false positives. Looking at them is not optional.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("⚠ matplotlib/Pillow missing — skipping contact sheet")
        return None

    if pairs.empty:
        print("No pairs to render.")
        return None

    sample = pairs.head(max_pairs)
    n = len(sample)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
    if n == 1:
        axes = np.array([axes])

    for row, (_, pair) in enumerate(sample.iterrows()):
        for col, side in enumerate(("a", "b")):
            ax = axes[row, col]
            image_id = pair[f"image_id_{side}"]
            path = image_paths.get(image_id)
            if path and path.exists():
                with Image.open(path) as img:
                    ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            lesion = pair.get(f"lesion_{side}", "?")
            ax.set_title(f"{image_id}\n{lesion}", fontsize=6)
        metric = (f"hamming={pair['hamming']}" if "hamming" in pair
                  else f"cos={pair.get('cosine', float('nan')):.4f}")
        same = pair.get("same_lesion", None)
        flag = "SAME lesion" if same else "DIFFERENT lesions (undeclared)"
        axes[row, 0].set_ylabel(f"{metric}\n{flag}", fontsize=6)

    fig.suptitle("Candidate duplicate pairs — verify manually", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit HAM10000 for duplicate images.")
    parser.add_argument("--phash", action="store_true", help="run perceptual-hash detection")
    parser.add_argument("--embeddings", action="store_true", help="run embedding-similarity detection")
    parser.add_argument("--contact-sheet", action="store_true", help="render pairs for manual review")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if not (args.phash or args.embeddings):
        parser.error("choose at least one of --phash / --embeddings")

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    raw_dir = cfg.path("paths.raw_dir")
    meta = pd.read_csv(cfg.path("paths.metadata"))
    image_paths = index_images(raw_dir)
    if not image_paths:
        print(f"✗ No images found under {raw_dir}. Run `python -m src.data.download --check`.")
        return 1
    print(f"Indexed {len(image_paths):,} images\n")

    reports_dir = cfg.path("paths.reports_dir")
    cache_dir = cfg.path("paths.cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    if args.phash:
        hash_size = int(cfg.get("audit.phash.hash_size", 16))
        max_hamming = int(cfg.get("audit.phash.max_hamming", 12))
        hashes = compute_phashes(image_paths, hash_size=hash_size)
        pairs = phash_candidate_pairs(hashes, max_hamming=max_hamming)
        print(f"pHash candidate pairs (hamming <= {max_hamming}): {len(pairs):,}")
        frames.append(pairs)

    if args.embeddings:
        ids, embeddings = compute_embeddings(
            image_paths,
            backbone=cfg.get("audit.embedding.backbone", "resnet50"),
            batch_size=int(cfg.get("audit.embedding.batch_size", 64)),
            image_size=int(cfg.get("audit.embedding.image_size", 224)),
        )
        np.save(cache_dir / "embeddings.npy", embeddings)
        pd.Series(ids).to_csv(cache_dir / "embedding_ids.csv", index=False, header=["image_id"])
        pairs = embedding_candidate_pairs(
            ids, embeddings,
            threshold=float(cfg.get("audit.embedding.cosine_threshold", 0.98)),
            n_neighbours=int(cfg.get("audit.embedding.n_neighbours", 5)),
        )
        print(f"Embedding candidate pairs (cosine >= "
              f"{cfg.get('audit.embedding.cosine_threshold')}): {len(pairs):,}")
        frames.append(pairs)

    all_pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_pairs = annotate_pairs(
        all_pairs, meta,
        group_col=cfg.get("dataset.group_col", "lesion_id"),
        id_col=cfg.get("dataset.id_col", "image_id"),
        label_col=cfg.get("dataset.label_col", "dx"),
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "duplicates.csv"
    all_pairs.to_csv(out_csv, index=False)

    print()
    print("=" * 62)
    print("AUDIT SUMMARY")
    print("=" * 62)
    print(f"  Total candidate pairs          : {len(all_pairs):,}")
    if not all_pairs.empty:
        declared = int(all_pairs["same_lesion"].sum())
        undeclared = int((~all_pairs["same_lesion"]).sum())
        print(f"  Already declared (same lesion) : {declared:,}")
        print(f"  UNDECLARED (different lesions) : {undeclared:,}  <- the hidden leakage risk")
        if undeclared:
            mismatched = int((~all_pairs["same_lesion"] & ~all_pairs["same_dx"]).sum())
            print(f"    ...of which labelled differently: {mismatched:,}")
        groups = build_duplicate_groups(all_pairs[~all_pairs["same_lesion"]])
        print(f"  Images in undeclared dup groups: {len(groups):,}")
    print(f"\n  Written: {out_csv}")

    if args.contact_sheet and not all_pairs.empty:
        undeclared_pairs = all_pairs[~all_pairs["same_lesion"]]
        target = undeclared_pairs if not undeclared_pairs.empty else all_pairs
        sheet = make_contact_sheet(
            target, image_paths, cfg.path("paths.figures_dir") / "duplicate_pairs.png"
        )
        if sheet:
            print(f"  Contact sheet: {sheet}")
            print("  -> Review these by eye and record detector precision in the README.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
