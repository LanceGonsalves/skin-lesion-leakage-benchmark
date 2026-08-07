"""Dose-response: how much accuracy does test-set contamination actually buy?

The headline experiment compares two splits and reports one number: +10.3 points of
balanced accuracy. That answers "does leakage matter?" but not "how much leakage
produces how much illusion?" -- and it carries a caveat that will not go away, namely
that the two splits have different test sets, so the model differs *and* the evaluation
data differs at the same time.

This module removes that confound. **One model, trained honestly on the grouped split,
is held fixed.** Only the test set changes: images are swapped out for images of lesions
the model saw during training, at controlled rates from 0% to `max_rate`. Every point on
the resulting curve is the same weights scored against a differently-contaminated test
set, so the only thing that varies is evaluation integrity.

The substitution is stratified by class and the test-set size is held constant, because
otherwise a shift in class mix would move balanced accuracy on its own and the curve
would measure the wrong thing.

    python -m src.contamination --split grouped --repeats 5

Outputs reports/contamination_curve.csv, a fitted slope in points per 10 percentage
points of contamination, and reports/figures/contamination_curve.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.evaluate import SPLIT_KEYS, compute_metrics
from src.models.build import build_model, resolve_device
from src.models.dataset import load_split, make_loader

DEFAULT_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


# --------------------------------------------------------------------------------------
# Building contaminated test sets
# --------------------------------------------------------------------------------------

def donor_pool(splits_dir: Path, split_file: str, group_col: str = "group") -> pd.DataFrame:
    """Images from lesions the model trained on — the material leakage is made of.

    A naive split leaks precisely because images like these end up in the test set: the
    lesion is familiar even though the individual photograph is not.
    """
    train = load_split(splits_dir, split_file, "train")
    return train.copy()


def contaminate(
    test_df: pd.DataFrame,
    donors: pd.DataFrame,
    rate: float,
    label_col: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """Replace `rate` of the test set with donor images, stratified by class.

    Substitution rather than addition, so the test set keeps its size and its class
    balance. Both matter: balanced accuracy is sensitive to class mix, and a curve that
    silently changed the denominator would be measuring two things at once.

    Returns the contaminated frame and the contamination rate actually achieved (which
    can fall short of the target if some class has too few donors).
    """
    if rate <= 0:
        return test_df.copy(), 0.0

    keep_parts, swap_parts = [], []

    for cls, block in test_df.groupby(label_col, sort=True):
        n_swap = int(round(len(block) * rate))
        pool = donors[donors[label_col] == cls]

        # Never substitute more than the donor pool can supply without replacement:
        # duplicated donors would inflate the effect with the same image scored twice.
        n_swap = min(n_swap, len(pool))
        if n_swap == 0:
            keep_parts.append(block)
            continue

        drop_idx = rng.choice(block.index.to_numpy(), size=n_swap, replace=False)
        keep_parts.append(block.drop(index=drop_idx))

        take_idx = rng.choice(pool.index.to_numpy(), size=n_swap, replace=False)
        swap_parts.append(pool.loc[take_idx])

    kept = pd.concat(keep_parts) if keep_parts else test_df.iloc[0:0]
    swapped = pd.concat(swap_parts) if swap_parts else test_df.iloc[0:0]
    out = pd.concat([kept, swapped], ignore_index=True)

    achieved = len(swapped) / len(out) if len(out) else 0.0
    return out, achieved


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def load_checkpoint_model(checkpoint_path: Path, cfg, n_classes: int):
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train the grouped split first:\n"
            f"  python -m src.models.train --split grouped"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        backbone=checkpoint.get("backbone", cfg.get("model.backbone")),
        n_classes=n_classes,
        pretrained=False,
        dropout=float(cfg.get("model.dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device()
    return model.eval().to(device), device


def score(model, device, frame: pd.DataFrame, cfg, classes: list[str]) -> dict:
    import torch

    loader = make_loader(
        frame, cfg.path("paths.raw_dir") / "images", classes,
        int(cfg.get("model.image_size", 224)), int(cfg.get("train.batch_size", 32)),
        train=False, num_workers=int(cfg.get("train.num_workers", 4)), seed=cfg.seed,
    )
    trues, preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            trues.append(targets.numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return compute_metrics(np.concatenate(trues), np.concatenate(preds), classes)


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------

def fit_slope(rates: np.ndarray, values: np.ndarray) -> dict:
    """Least-squares line through the curve, reported per 10 percentage points.

    Per-10pp because that is the unit people can hold in their head: "a tenth of your
    test set contaminated buys you N points you did not earn."
    """
    if len(rates) < 2:
        return {"slope_per_10pp": float("nan"), "r_squared": float("nan")}

    slope, intercept = np.polyfit(rates, values, 1)
    predicted = slope * rates + intercept
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope_per_10pp": float(slope * 0.10),
        "intercept": float(intercept),
        "r_squared": float(r2),
    }


def sweep(cfg, split: str, rates: list[float], repeats: int) -> pd.DataFrame:
    """Score the fixed model against test sets at each contamination rate.

    `repeats` re-draws which images get swapped. The curve then carries its own error
    bars, so a reader can see whether a wobble is signal or just which images happened
    to be chosen.
    """
    classes = cfg.classes
    label_col = cfg.get("dataset.label_col", "dx")
    splits_dir = cfg.path("paths.splits_dir")
    split_file = cfg.get(SPLIT_KEYS[split])

    test_df = load_split(splits_dir, split_file, "test")
    donors = donor_pool(splits_dir, split_file)

    model, device = load_checkpoint_model(
        Path("checkpoints") / f"{split}_best.pt", cfg, len(classes)
    )

    print(f"  model      : checkpoints/{split}_best.pt (fixed across every point)")
    print(f"  test set   : {len(test_df):,} images")
    print(f"  donor pool : {len(donors):,} images from lesions seen in training")
    print()

    rows = []
    for rate in rates:
        # rate 0 is deterministic — no substitution happens, so repeating it would
        # produce identical numbers and imply a precision that isn't there.
        n_draws = 1 if rate <= 0 else repeats

        for draw in range(n_draws):
            rng = np.random.default_rng(cfg.seed + draw)
            frame, achieved = contaminate(test_df, donors, rate, label_col, rng)
            metrics = score(model, device, frame, cfg, classes)
            rows.append({
                "target_rate": rate,
                "achieved_rate": achieved,
                "draw": draw,
                "n_test": len(frame),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            })
            print(f"  contamination {rate:5.0%} (draw {draw + 1}/{n_draws}) "
                  f"-> balanced acc {metrics['balanced_accuracy']:.4f}")

    return pd.DataFrame(rows)


def plot_curve(summary: pd.DataFrame, fit: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = summary["target_rate"] * 100
    y = summary["mean"]
    err = summary["std"].fillna(0.0)

    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=1.8, color="#C0392B")
    if np.isfinite(fit.get("slope_per_10pp", float("nan"))):
        line = fit["intercept"] + (fit["slope_per_10pp"] / 0.10) * summary["target_rate"]
        ax.plot(x, line, linestyle="--", linewidth=1, color="#555",
                label=f"fit: +{fit['slope_per_10pp'] * 100:.2f} pts per 10pp "
                      f"(R²={fit['r_squared']:.3f})")
        ax.legend(frameon=False, fontsize=9)

    ax.set_xlabel("Test-set contamination (%)")
    ax.set_ylabel("Reported balanced accuracy")
    ax.set_title("The same model, scored against progressively dishonest test sets")
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure reported accuracy as a function of test-set contamination."
    )
    parser.add_argument("--split", default="grouped", choices=sorted(SPLIT_KEYS),
                        help="Which trained model to hold fixed (default: grouped, the "
                             "honestly-trained one).")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES,
                        help="Contamination rates to sweep.")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Re-draws per rate, for error bars.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    print("=" * 78)
    print("CONTAMINATION DOSE-RESPONSE")
    print("=" * 78)
    print("One fixed model. Only the test set changes.")
    print()

    raw = sweep(cfg, args.split, sorted(args.rates), args.repeats)

    summary = (raw.groupby("target_rate")["balanced_accuracy"]
                  .agg(["mean", "std", "count"])
                  .reset_index())
    fit = fit_slope(summary["target_rate"].to_numpy(), summary["mean"].to_numpy())

    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    raw.to_csv(reports / "contamination_raw.csv", index=False)
    summary.to_csv(reports / "contamination_curve.csv", index=False)

    baseline = float(summary.loc[summary["target_rate"] == 0, "mean"].iloc[0])
    top = float(summary["mean"].iloc[-1])

    result = {
        "split": args.split,
        "baseline_balanced_accuracy": baseline,
        "max_rate": float(summary["target_rate"].iloc[-1]),
        "balanced_accuracy_at_max_rate": top,
        "total_inflation": top - baseline,
        **fit,
    }
    (reports / "contamination_fit.json").write_text(json.dumps(result, indent=2))

    fig_path = plot_curve(summary, fit, cfg.path("paths.figures_dir") / "contamination_curve.png")

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  Honest baseline (0% contamination) : {baseline:.4f}")
    print(f"  At {summary['target_rate'].iloc[-1]:.0%} contamination            : {top:.4f}")
    print(f"  Total inflation                    : {top - baseline:+.4f}")
    print()
    print(f"  Slope: {fit['slope_per_10pp'] * 100:+.2f} balanced-accuracy points "
          f"per 10pp of contamination  (R²={fit['r_squared']:.3f})")
    print()
    print(f"  Curve : {reports / 'contamination_curve.csv'}")
    print(f"  Figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
