"""Test-set evaluation with bootstrap confidence intervals.

Reports the metrics that mean something on imbalanced medical data:

  * balanced accuracy and macro-F1 as headline numbers
  * per-class precision / recall / F1, with melanoma recall called out
  * a normalised confusion matrix
  * bootstrap CIs, because a single point estimate on 1,700 images invites
    over-reading small differences

`--compare` renders the naive-vs-grouped table that is the point of the project.

Usage
-----
    python -m src.evaluate --split grouped
    python -m src.evaluate --compare
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
from src.models.dataset import load_split, make_loader

SPLIT_KEYS = {"naive": "splits.naive_name", "grouped": "splits.grouped_name"}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    return pd.DataFrame({
        "class": classes,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
        "support": support,
    })


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                 n_iterations: int = 1000, confidence: float = 0.95,
                 seed: int = 42) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over the test set.

    Resamples test *images* with replacement. Note this treats images as independent,
    which is exactly true for the grouped split and NOT true for the naive split —
    another way its numbers are optimistic.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}

    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue                      # degenerate resample
        metrics = compute_metrics(y_true[idx], y_pred[idx], classes)
        for key in samples:
            samples[key].append(metrics[key])

    alpha = (1.0 - confidence) / 2.0
    return {
        key: (float(np.percentile(values, 100 * alpha)),
              float(np.percentile(values, 100 * (1 - alpha))))
        for key, values in samples.items() if values
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                          out_path: Path, title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return None

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    normalised = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if normalised[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share of true class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------

def predict_test_set(split: str, cfg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the best checkpoint for `split` and predict its test partition."""
    import torch

    classes = cfg.classes
    checkpoint_path = Path("checkpoints") / f"{split}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run `python -m src.models.train "
            f"--split {split}` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = resolve_device()
    model = build_model(
        backbone=checkpoint.get("backbone", cfg.get("model.backbone")),
        n_classes=len(classes),
        pretrained=False,
        dropout=float(cfg.get("model.dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)

    test_df = load_split(cfg.path("paths.splits_dir"), cfg.get(SPLIT_KEYS[split]), "test")
    loader = make_loader(
        test_df, cfg.path("paths.raw_dir") / "images", classes,
        int(cfg.get("model.image_size", 224)), int(cfg.get("train.batch_size", 32)),
        train=False, num_workers=int(cfg.get("train.num_workers", 4)), seed=cfg.seed,
    )

    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            all_true.append(targets.numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred), classes


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def evaluate_split(split: str, cfg) -> dict:
    y_true, y_pred, classes = predict_test_set(split, cfg)

    metrics = compute_metrics(y_true, y_pred, classes)
    cis = bootstrap_ci(
        y_true, y_pred, classes,
        n_iterations=int(cfg.get("eval.bootstrap_iterations", 1000)),
        confidence=float(cfg.get("eval.confidence_level", 0.95)),
        seed=cfg.seed,
    )
    per_class = per_class_report(y_true, y_pred, classes)

    print("=" * 66)
    print(f"TEST RESULTS — {split} split  (n = {len(y_true):,})")
    print("=" * 66)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        line = f"  {key:<20}: {metrics[key]:.4f}"
        if key in cis:
            line += f"   95% CI [{cis[key][0]:.4f}, {cis[key][1]:.4f}]"
        print(line)

    print("\n  Per class:")
    print("    " + per_class.to_string(index=False).replace("\n", "\n    "))

    if "mel" in classes:
        mel_recall = float(per_class.loc[per_class["class"] == "mel", "recall"].iloc[0])
        print(f"\n  Melanoma recall: {mel_recall:.4f}  "
              f"({'a missed melanoma is the costly error' if mel_recall < 0.8 else 'reasonable'})")

    reports_dir = cfg.path("paths.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(reports_dir / f"per_class_{split}.csv", index=False)

    figure = save_confusion_matrix(
        y_true, y_pred, classes,
        cfg.path("paths.figures_dir") / f"confusion_matrix_{split}.png",
        f"Confusion matrix — {split} split (row-normalised)",
    )
    if figure:
        print(f"\n  Confusion matrix: {figure}")

    payload = {"split": split, "n_test": int(len(y_true)),
               "metrics": metrics, "confidence_intervals": {k: list(v) for k, v in cis.items()}}
    with open(reports_dir / f"test_metrics_{split}.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def compare(cfg) -> int:
    """Render the headline table: same model, both splits."""
    reports_dir = cfg.path("paths.reports_dir")
    loaded = {}
    for split in SPLIT_KEYS:
        path = reports_dir / f"test_metrics_{split}.json"
        if not path.exists():
            print(f"✗ Missing {path.name} — run `python -m src.evaluate --split {split}` first")
            return 1
        with open(path) as fh:
            loaded[split] = json.load(fh)

    naive, grouped = loaded["naive"], loaded["grouped"]
    print("=" * 66)
    print("THE LEAKAGE EFFECT")
    print("=" * 66)
    print(f"{'metric':<22}{'naive':>12}{'grouped':>12}{'Δ':>12}")
    print("-" * 66)
    rows = []
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        a, b = naive["metrics"][key], grouped["metrics"][key]
        delta = a - b
        print(f"{key:<22}{a:>12.4f}{b:>12.4f}{delta:>+12.4f}")
        rows.append({"metric": key, "naive": round(a, 4),
                     "grouped": round(b, 4), "delta": round(delta, 4)})

    pd.DataFrame(rows).to_csv(reports_dir / "leakage_comparison.csv", index=False)

    gap = naive["metrics"]["balanced_accuracy"] - grouped["metrics"]["balanced_accuracy"]
    print("-" * 66)
    print(f"\nNaive splitting inflates balanced accuracy by {gap * 100:+.1f} percentage points.")
    print("Same model, same hyperparameters, same seed — only the split differs.")

    naive_ci = naive["confidence_intervals"].get("balanced_accuracy")
    grouped_ci = grouped["confidence_intervals"].get("balanced_accuracy")
    if naive_ci and grouped_ci:
        overlap = naive_ci[0] <= grouped_ci[1] and grouped_ci[0] <= naive_ci[1]
        print(f"\n95% CIs: naive [{naive_ci[0]:.4f}, {naive_ci[1]:.4f}]  "
              f"grouped [{grouped_ci[0]:.4f}, {grouped_ci[1]:.4f}]")
        print("  → intervals OVERLAP; the gap is not clearly significant at this sample size."
              if overlap else
              "  → intervals are DISJOINT; the gap is unlikely to be sampling noise.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on its test set.")
    parser.add_argument("--split", choices=sorted(SPLIT_KEYS))
    parser.add_argument("--compare", action="store_true", help="render the naive-vs-grouped table")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if not args.split and not args.compare:
        parser.error("pass --split {naive,grouped} or --compare")

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.split:
        evaluate_split(args.split, cfg)
    if args.compare:
        return compare(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
