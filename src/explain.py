"""Interpretability: Grad-CAM attention maps and probability calibration.

Two questions a metrics table cannot answer:

1. **Where is the model looking?** Dermatoscopic images are full of artefacts — rulers,
   ink marks, hair, dark vignetting from the lens. Models are well documented to latch
   onto these instead of the lesion, which produces good test numbers and a model that
   fails the moment the imaging setup changes. Grad-CAM makes that visible.

2. **Should we believe its confidence?** A model that is 95% sure and wrong is more
   dangerous in a medical setting than one that is uncertain. Reliability diagrams and
   expected calibration error quantify that.

The calibration maths is pure NumPy so it can be unit-tested without torch.

Usage
-----
    python -m src.explain --split grouped
    python -m src.explain --split grouped --n-samples 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.models.build import build_model, resolve_device
from src.models.dataset import IMAGENET_MEAN, IMAGENET_STD, load_split, make_loader

SPLIT_KEYS = {"naive": "splits.naive_name", "grouped": "splits.grouped_name"}


# --------------------------------------------------------------------------------------
# Calibration (torch-free)
# --------------------------------------------------------------------------------------

def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error: |confidence - accuracy| averaged over bins.

    A perfectly calibrated model that says "80% sure" is right 80% of the time, giving
    ECE 0. Deep networks are typically overconfident, so ECE > 0 with accuracy below
    confidence in the high bins.
    """
    if len(confidences) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        # Include the right edge in the final bin so confidence == 1.0 is counted.
        mask = (confidences > low) & (confidences <= high) if high < 1.0 \
            else (confidences > low) & (confidences <= 1.0)
        if mask.sum() == 0:
            continue
        bin_weight = mask.mean()
        ece += bin_weight * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_bins(confidences: np.ndarray, correct: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Per-bin accuracy vs mean confidence, for the reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high) if high < 1.0 \
            else (confidences > low) & (confidences <= 1.0)
        rows.append({
            "bin_low": round(low, 3),
            "bin_high": round(high, 3),
            "n": int(mask.sum()),
            "mean_confidence": float(confidences[mask].mean()) if mask.sum() else np.nan,
            "accuracy": float(correct[mask].mean()) if mask.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def plot_reliability(bins: pd.DataFrame, ece: float, out_path: Path,
                     title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    populated = bins.dropna(subset=["accuracy"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.bar(populated["mean_confidence"], populated["accuracy"], width=0.08,
           edgecolor="black", color="#4C6EF5", alpha=0.85, label="observed")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------------------

def find_target_layer(model):
    """Pick the last convolutional layer — where spatial detail survives.

    Searching for the final Conv2d works across backbones rather than hard-coding a
    timm-specific attribute path that breaks the moment the backbone changes.
    """
    import torch.nn as nn

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found — cannot place Grad-CAM hooks.")
    return last_conv


def denormalise(tensor: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation so the image can be shown under the heatmap."""
    image = tensor.transpose(1, 2, 0)
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(image, 0, 1)


def generate_cams(model, images, target_indices, device):
    """Return Grad-CAM maps for a batch, one per requested target class."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(int(i)) for i in target_indices]
    with GradCAM(model=model, target_layers=[find_target_layer(model)]) as cam:
        return cam(input_tensor=images.to(device), targets=targets)


def plot_cam_grid(records: list[dict], out_path: Path, title: str) -> Path | None:
    """Image / heatmap pairs, annotated with true and predicted class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not records:
        return None

    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.6 * n))
    if n == 1:
        axes = np.array([axes])

    for row, record in enumerate(records):
        axes[row, 0].imshow(record["image"])
        axes[row, 0].set_title(f"true: {record['true']}", fontsize=8)
        axes[row, 1].imshow(record["image"])
        axes[row, 1].imshow(record["cam"], cmap="jet", alpha=0.45)
        colour = "green" if record["true"] == record["pred"] else "red"
        axes[row, 1].set_title(f"pred: {record['pred']} ({record['confidence']:.2f})",
                               fontsize=8, color=colour)
        for col in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.suptitle(title, fontsize=11, y=0.997)
    # Leave headroom for the suptitle, otherwise it collides with the first row's titles.
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM and calibration analysis.")
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_KEYS))
    parser.add_argument("--n-samples", type=int, default=8,
                        help="how many correct and how many misclassified examples to render")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    import torch

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    classes = cfg.classes

    checkpoint_path = Path("checkpoints") / f"{args.split}_best.pt"
    if not checkpoint_path.exists():
        print(f"✗ No checkpoint at {checkpoint_path}. Train first:")
        print(f"    python -m src.models.train --split {args.split}")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = resolve_device()
    model = build_model(
        backbone=checkpoint.get("backbone", cfg.get("model.backbone")),
        n_classes=len(classes), pretrained=False,
        dropout=float(cfg.get("model.dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    print(f"Loaded {checkpoint_path} (val bal_acc "
          f"{checkpoint.get('val_balanced_accuracy', float('nan')):.4f})")

    test_df = load_split(cfg.path("paths.splits_dir"), cfg.get(SPLIT_KEYS[args.split]), "test")
    loader = make_loader(
        test_df, cfg.path("paths.raw_dir") / "images", classes,
        int(cfg.get("model.image_size", 224)), int(cfg.get("train.batch_size", 32)),
        train=False, num_workers=args.num_workers, seed=cfg.seed,
    )

    # --- collect predictions + keep a few images for Grad-CAM ---------------------
    all_conf, all_correct, all_true, all_pred = [], [], [], []
    # Keyed by (outcome, true_class) so the rendered examples span the label space.
    # Taking the first N examples instead yields whatever the (unshuffled) loader emits
    # first -- in practice a single class, which makes the figure look representative
    # while showing nothing of the sort.
    pools: dict[tuple[str, int], list[tuple]] = {}
    per_class_cap = max(1, args.n_samples)

    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            confidence, prediction = probs.max(dim=1)

            hit = (prediction == targets)
            all_conf.append(confidence.numpy())
            all_correct.append(hit.numpy())
            all_true.append(targets.numpy())
            all_pred.append(prediction.numpy())

            for i in range(len(targets)):
                key = ("correct" if hit[i] else "misclassified", int(targets[i]))
                pool = pools.setdefault(key, [])
                if len(pool) < per_class_cap:
                    pool.append((images[i], int(targets[i]), int(prediction[i]),
                                 float(confidence[i])))

    def stratified_sample(outcome: str, n_wanted: int) -> list[tuple]:
        """Round-robin across true classes so no single class dominates the figure."""
        by_class = {cls: pools.get((outcome, cls), []) for cls in range(len(classes))}
        chosen: list[tuple] = []
        depth = 0
        while len(chosen) < n_wanted and any(len(v) > depth for v in by_class.values()):
            for cls in range(len(classes)):
                if len(chosen) >= n_wanted:
                    break
                if len(by_class[cls]) > depth:
                    chosen.append(by_class[cls][depth])
            depth += 1
        return chosen

    correct_examples = stratified_sample("correct", args.n_samples)
    wrong_examples = stratified_sample("misclassified", args.n_samples)

    confidences = np.concatenate(all_conf)
    correct = np.concatenate(all_correct).astype(float)

    # --- calibration ---------------------------------------------------------------
    ece = expected_calibration_error(confidences, correct)
    bins = reliability_bins(confidences, correct)

    print("\n" + "=" * 60)
    print(f"CALIBRATION — {args.split} split")
    print("=" * 60)
    print(f"  Mean confidence : {confidences.mean():.4f}")
    print(f"  Accuracy        : {correct.mean():.4f}")
    print(f"  ECE             : {ece:.4f}")
    gap = confidences.mean() - correct.mean()
    if gap > 0.05:
        print(f"  → Overconfident by {gap:.3f}. Typical of deep nets; matters clinically.")
    elif gap < -0.05:
        print(f"  → Underconfident by {abs(gap):.3f}.")
    else:
        print("  → Reasonably calibrated.")
    print("\n" + bins.to_string(index=False))

    figures_dir = cfg.path("paths.figures_dir")
    reports_dir = cfg.path("paths.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    bins.to_csv(reports_dir / f"calibration_{args.split}.csv", index=False)

    reliability = plot_reliability(
        bins, ece, figures_dir / f"reliability_{args.split}.png",
        f"Reliability — {args.split} split",
    )
    if reliability:
        print(f"\n  Reliability diagram: {reliability}")

    # --- Grad-CAM ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GRAD-CAM")
    print("=" * 60)
    try:
        for label, bucket in (("correct", correct_examples), ("misclassified", wrong_examples)):
            if not bucket:
                print(f"  (no {label} examples to render)")
                continue

            batch = torch.stack([item[0] for item in bucket])
            predictions = [item[2] for item in bucket]
            cams = generate_cams(model, batch, predictions, device)

            records = [{
                "image": denormalise(batch[i].numpy()),
                "cam": cams[i],
                "true": classes[bucket[i][1]],
                "pred": classes[bucket[i][2]],
                "confidence": bucket[i][3],
            } for i in range(len(bucket))]

            covered = sorted({classes[item[1]] for item in bucket})
            path = plot_cam_grid(
                records, figures_dir / f"gradcam_{label}_{args.split}.png",
                f"Grad-CAM — {label} ({args.split} split)",
            )
            if path:
                print(f"  {label}: {path}")
                print(f"    classes shown: {', '.join(covered)}")

        print("\n  Look for attention on rulers, ink marks, hair or vignetting rather than")
        print("  the lesion — that is shortcut learning, and it will not transfer.")
    except ImportError:
        print("  pytorch-grad-cam not installed — skipping. pip install grad-cam")
    except Exception as exc:                     # hooks are fragile across versions
        print(f"  Grad-CAM failed: {type(exc).__name__}: {exc}")
        print("  Calibration results above are unaffected.")

    with open(reports_dir / f"calibration_{args.split}.json", "w") as fh:
        json.dump({
            "split": args.split,
            "ece": ece,
            "mean_confidence": float(confidences.mean()),
            "accuracy": float(correct.mean()),
            "n_test": int(len(confidences)),
        }, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
