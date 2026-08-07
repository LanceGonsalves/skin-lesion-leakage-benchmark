"""Replicate the leakage experiment across seeds and architectures.

The original result — naive evaluation reports +10.3 balanced-accuracy points over a
grouped one — came from a single training run per split. That is enough to notice an
effect and not enough to defend one. Bootstrap confidence intervals do not help here:
they quantify uncertainty in the *test sample*, while the open question is uncertainty
in the *training process*. Re-running with a different seed changes initialisation,
augmentation draws and batch order, and a gap that survives that is a different kind
of claim from a gap that doesn't.

Design
------
For each (backbone, seed) pair both splits are trained and evaluated, giving one paired
observation of the gap. Pairing matters: seed-to-seed variation is shared between the
two arms, so the *difference* is far less noisy than either arm alone, and a paired test
is the right one.

    python -m src.experiments.replicate --seeds 42 43 44 45 46
    python -m src.experiments.replicate --seeds 42 43 44 --backbones efficientnet_b0 resnet18

Every run is a subprocess so a crash in one configuration cannot take the sweep with it,
and results are appended to disk as they complete — an overnight run that dies at 3am
still leaves everything it had finished.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluate import SPLIT_KEYS

RESULTS_CSV = "replication_runs.csv"


# --------------------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------------------

def train_one(split: str, seed: int, backbone: str, num_workers: int | None,
              extra: list[str]) -> bool:
    """Train a single (split, seed, backbone). Returns True on success."""
    cmd = [sys.executable, "-m", "src.models.train",
           "--split", split, "--seed", str(seed), "--backbone", backbone]
    if num_workers is not None:
        cmd += ["--num-workers", str(num_workers)]
    cmd += extra

    print(f"    $ {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - started

    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}) after {elapsed / 60:.1f} min")
        return False
    print(f"    done in {elapsed / 60:.1f} min")
    return True


def score_one(split: str, seed: int, backbone: str, cfg) -> dict | None:
    """Evaluate the checkpoint a run just produced."""
    import torch  # noqa: F401  (imported for the side effect of a clear error if absent)

    from src.evaluate import compute_metrics
    from src.models.build import build_model, checkpoint_name, resolve_device
    from src.models.dataset import load_split, make_loader

    classes = cfg.classes
    path = Path("checkpoints") / checkpoint_name(
        split, backbone, seed, cfg.get("model.backbone", "efficientnet_b0"), cfg.seed
    )
    if not path.exists():
        print(f"    no checkpoint at {path} — skipping")
        return None

    import torch
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(backbone=ckpt.get("backbone", backbone), n_classes=len(classes),
                        pretrained=False, dropout=float(cfg.get("model.dropout", 0.2)))
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device()
    model.eval().to(device)

    test_df = load_split(cfg.path("paths.splits_dir"), cfg.get(SPLIT_KEYS[split]), "test")
    loader = make_loader(
        test_df, cfg.path("paths.raw_dir") / "images", classes,
        int(cfg.get("model.image_size", 224)), int(cfg.get("train.batch_size", 32)),
        train=False, num_workers=int(cfg.get("train.num_workers", 4)), seed=seed,
    )

    trues, preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            trues.append(targets.numpy())
            preds.append(model(images.to(device)).argmax(dim=1).cpu().numpy())

    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    metrics = compute_metrics(y_true, y_pred, classes)

    # Melanoma recall is carried separately because it is the number that would matter
    # clinically; a mean over classes can hide it moving in the wrong direction.
    mel_recall = float("nan")
    if "mel" in classes:
        mel = classes.index("mel")
        mask = y_true == mel
        if mask.sum():
            mel_recall = float((y_pred[mask] == mel).mean())

    return {
        "split": split, "seed": seed, "backbone": backbone,
        "n_test": int(len(y_true)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mel_recall": mel_recall,
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def paired_gaps(runs: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """One row per (backbone, seed): naive minus grouped on the same seed."""
    wide = runs.pivot_table(index=["backbone", "seed"], columns="split",
                            values=metric, aggfunc="first")
    wide = wide.dropna(subset=[s for s in ("naive", "grouped") if s in wide.columns])
    if not {"naive", "grouped"}.issubset(wide.columns):
        return pd.DataFrame(columns=["backbone", "seed", "naive", "grouped", "gap"])
    wide["gap"] = wide["naive"] - wide["grouped"]
    return wide.reset_index()


def paired_test(gaps: np.ndarray) -> dict:
    """Paired t-test plus Wilcoxon, on the per-seed differences.

    Both are reported because with a handful of seeds neither is decisive on its own:
    the t-test assumes normality that n=5 cannot evidence, and Wilcoxon has so little
    power at that size that a null result means very little. Agreement between them is
    the useful signal. Cohen's d is included because with n this small the effect size
    is more informative than the p-value.
    """
    from scipy import stats

    n = len(gaps)
    out = {
        "n_pairs": int(n),
        "mean_gap": float(np.mean(gaps)) if n else float("nan"),
        "std_gap": float(np.std(gaps, ddof=1)) if n > 1 else float("nan"),
    }
    if n > 1:
        se = out["std_gap"] / np.sqrt(n)
        out["ci95_low"] = float(out["mean_gap"] - 1.96 * se)
        out["ci95_high"] = float(out["mean_gap"] + 1.96 * se)
        out["cohens_d"] = float(out["mean_gap"] / out["std_gap"]) if out["std_gap"] else float("nan")

        t_stat, t_p = stats.ttest_rel(gaps, np.zeros_like(gaps))
        out["t_statistic"] = float(t_stat)
        out["t_p_value"] = float(t_p)
        try:
            w_stat, w_p = stats.wilcoxon(gaps)
            out["wilcoxon_statistic"] = float(w_stat)
            out["wilcoxon_p_value"] = float(w_p)
        except ValueError as exc:      # raised when every difference is identical
            out["wilcoxon_note"] = str(exc)
    return out


def plot_gaps(gaps_df: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (backbone, block) in enumerate(gaps_df.groupby("backbone")):
        xs = np.full(len(block), i, dtype=float) + np.linspace(-0.06, 0.06, len(block))
        ax.scatter(xs, block["gap"] * 100, s=60, zorder=3, label=backbone)
        ax.hlines(block["gap"].mean() * 100, i - 0.18, i + 0.18,
                  linewidth=2.5, color="#C0392B", zorder=4)

    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xticks(range(gaps_df["backbone"].nunique()))
    ax.set_xticklabels(sorted(gaps_df["backbone"].unique()))
    ax.set_ylabel("Inflation from naive evaluation\n(balanced-accuracy points)")
    ax.set_title("Every point is one seed; the bar is the mean")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    if gaps_df["backbone"].nunique() > 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replicate the leakage experiment across seeds and architectures."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--backbones", nargs="+", default=["efficientnet_b0"])
    parser.add_argument("--splits", nargs="+", default=["naive", "grouped"],
                        choices=sorted(SPLIT_KEYS))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny subset, 1 epoch — verifies the sweep wiring, not the science")
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip training; re-aggregate whatever is already on disk")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    results_path = reports / RESULTS_CSV

    n_runs = len(args.seeds) * len(args.backbones) * len(args.splits)
    print("=" * 78)
    print("REPLICATION SWEEP")
    print("=" * 78)
    print(f"  seeds     : {args.seeds}")
    print(f"  backbones : {args.backbones}")
    print(f"  splits    : {args.splits}")
    print(f"  total runs: {n_runs}")
    print()

    rows: list[dict] = []
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        print(f"  resuming — {len(rows)} run(s) already on disk\n")

    done = {(r["split"], int(r["seed"]), r["backbone"]) for r in rows}

    if not args.analyse_only:
        extra = ["--smoke-test"] if args.smoke_test else []
        run_i = 0
        for backbone in args.backbones:
            for seed in args.seeds:
                for split in args.splits:
                    run_i += 1
                    key = (split, seed, backbone)
                    if key in done:
                        print(f"  [{run_i}/{n_runs}] {backbone} seed={seed} {split} "
                              f"— already done, skipping")
                        continue

                    print(f"  [{run_i}/{n_runs}] {backbone} seed={seed} split={split}")
                    if not train_one(split, seed, backbone, args.num_workers, extra):
                        continue

                    result = score_one(split, seed, backbone, cfg)
                    if result is None:
                        continue
                    rows.append(result)
                    # Written after every run, not at the end: an overnight sweep that
                    # dies at 3am should not lose the runs that already succeeded.
                    pd.DataFrame(rows).to_csv(results_path, index=False)
                    print(f"    balanced acc {result['balanced_accuracy']:.4f} "
                          f"| mel recall {result['mel_recall']:.4f}")
                    print()

    if not rows:
        print("No results to analyse.")
        return 1

    runs = pd.DataFrame(rows)
    runs.to_csv(results_path, index=False)

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    summary = (runs.groupby(["backbone", "split"])[["balanced_accuracy", "mel_recall"]]
                   .agg(["mean", "std", "count"]).round(4))
    print(summary.to_string())
    print()

    gaps_df = paired_gaps(runs)
    if gaps_df.empty:
        print("Need both splits for at least one seed to compute a paired gap.")
        return 0

    gaps_df.to_csv(reports / "replication_gaps.csv", index=False)

    stats_all = paired_test(gaps_df["gap"].to_numpy())
    print("Paired gap (naive − grouped), balanced accuracy:")
    print(f"  seeds       : {stats_all['n_pairs']}")
    print(f"  mean        : {stats_all['mean_gap'] * 100:+.2f} points")
    if stats_all["n_pairs"] > 1:
        print(f"  std         : {stats_all['std_gap'] * 100:.2f} points")
        print(f"  95% CI      : [{stats_all['ci95_low'] * 100:+.2f}, "
              f"{stats_all['ci95_high'] * 100:+.2f}]")
        print(f"  Cohen's d   : {stats_all['cohens_d']:.2f}")
        print(f"  paired t    : t={stats_all['t_statistic']:.3f}, p={stats_all['t_p_value']:.5f}")
        if "wilcoxon_p_value" in stats_all:
            print(f"  Wilcoxon    : p={stats_all['wilcoxon_p_value']:.5f}")

    per_backbone = {}
    if runs["backbone"].nunique() > 1:
        print("\nPer architecture:")
        for backbone, block in gaps_df.groupby("backbone"):
            st = paired_test(block["gap"].to_numpy())
            per_backbone[backbone] = st
            print(f"  {backbone:20s} {st['mean_gap'] * 100:+.2f} points "
                  f"(n={st['n_pairs']})")

    payload = {"overall": stats_all, "per_backbone": per_backbone,
               "metric": "balanced_accuracy"}
    (reports / "replication_stats.json").write_text(json.dumps(payload, indent=2))

    fig = plot_gaps(gaps_df, cfg.path("paths.figures_dir") / "replication_gaps.png")
    print(f"\n  Runs  : {results_path}")
    print(f"  Stats : {reports / 'replication_stats.json'}")
    print(f"  Figure: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
