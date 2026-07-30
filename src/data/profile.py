"""Metadata profiling — the evidence that motivates the whole project.

Produces the numbers and figures that answer: how much redundancy is actually in
HAM10000, and what does that imply for a naive train/test split?

Usage
-----
    python -m src.data.profile
    python -m src.data.profile --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. Run `python -m src.data.download --check` first."
        )
    return pd.read_csv(path)


def class_distribution(meta: pd.DataFrame, label_col: str) -> pd.DataFrame:
    counts = meta[label_col].value_counts()
    return pd.DataFrame({
        "count": counts,
        "share": (counts / len(meta)).round(4),
    })


def lesion_redundancy(meta: pd.DataFrame, group_col: str, id_col: str) -> dict[str, float]:
    """Quantify how many images are 'extra' views of a lesion already in the dataset."""
    per_lesion = meta.groupby(group_col)[id_col].count()
    n_images = len(meta)
    n_lesions = int(per_lesion.size)
    redundant = n_images - n_lesions

    return {
        "n_images": n_images,
        "n_lesions": n_lesions,
        "lesions_with_multiple_images": int((per_lesion > 1).sum()),
        "max_images_per_lesion": int(per_lesion.max()),
        "redundant_images": redundant,
        "redundant_share": round(redundant / n_images, 4) if n_images else 0.0,
    }


def expected_naive_contamination(
    meta: pd.DataFrame, group_col: str, id_col: str, test_size: float
) -> float:
    """Estimate the share of a naive random test set that is contaminated.

    For each lesion with k images, an image assigned to test is 'contaminated' if at
    least one of the other k-1 images of the same lesion falls in train/val. Under
    uniform random assignment that probability is 1 - test_size**(k-1).

    This is an analytic estimate, deliberately simple; the empirical figure is
    measured directly once the splits are built (see src/data/splits.py).
    """
    per_lesion = meta.groupby(group_col)[id_col].count()
    # Weight each lesion by how many of its images land in test (proportional to k).
    weighted = 0.0
    total = 0.0
    for k, n_lesions in per_lesion.value_counts().items():
        images = k * n_lesions
        prob_contaminated = 1.0 - (test_size ** (k - 1)) if k > 1 else 0.0
        weighted += images * prob_contaminated
        total += images
    return round(weighted / total, 4) if total else 0.0


def make_figures(meta: pd.DataFrame, cfg, out_dir: Path) -> list[Path]:
    """Save the two figures that tell the story. Returns paths written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed — skipping figures")
        return []

    label_col = cfg.get("dataset.label_col", "dx")
    group_col = cfg.get("dataset.group_col", "lesion_id")
    id_col = cfg.get("dataset.id_col", "image_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Class imbalance — motivates balanced accuracy over plain accuracy.
    counts = meta[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#4C6EF5")
    ax.set_title("HAM10000 class distribution")
    ax.set_xlabel("diagnosis")
    ax.set_ylabel("images")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2. Images per lesion — the redundancy that breaks naive splitting.
    per_lesion = meta.groupby(group_col)[id_col].count().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(per_lesion.index.astype(str), per_lesion.values, color="#E8590C")
    ax.set_title("Images per lesion — why image-level splits leak")
    ax.set_xlabel("images of the same lesion")
    ax.set_ylabel("number of lesions")
    ax.set_yscale("log")
    for i, v in enumerate(per_lesion.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "images_per_lesion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the HAM10000 metadata.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    label_col = cfg.get("dataset.label_col", "dx")
    group_col = cfg.get("dataset.group_col", "lesion_id")
    id_col = cfg.get("dataset.id_col", "image_id")
    test_size = float(cfg.get("splits.test_size", 0.15))

    meta = load_metadata(cfg.path("paths.metadata"))

    print("=" * 62)
    print("CLASS DISTRIBUTION")
    print("=" * 62)
    dist = class_distribution(meta, label_col)
    print(dist.to_string())
    majority = dist["share"].max()
    print(f"\nMajority class is {majority:.1%} of the data.")
    print("-> Plain accuracy is close to meaningless here; use balanced accuracy / macro-F1.")

    print()
    print("=" * 62)
    print("LESION REDUNDANCY  (the reason this project exists)")
    print("=" * 62)
    stats = lesion_redundancy(meta, group_col, id_col)
    for key, value in stats.items():
        label = key.replace("_", " ").capitalize()
        print(f"  {label:<32}: {value:,}" if isinstance(value, int) else
              f"  {label:<32}: {value}")

    contamination = expected_naive_contamination(meta, group_col, id_col, test_size)
    print(f"\n  Estimated share of a naive random test set whose lesion also")
    print(f"  appears in training: ~{contamination:.1%}")
    print("  (analytic estimate; measured empirically once splits are built)")

    if not args.no_figures:
        written = make_figures(meta, cfg, cfg.path("paths.figures_dir"))
        if written:
            print("\nFigures written:")
            for p in written:
                print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
