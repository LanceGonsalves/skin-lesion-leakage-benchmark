"""Build the two splits whose comparison IS the experiment.

Split A (naive)   -- stratified random split at the *image* level.
                     Images of the same lesion can land on both sides. This is what
                     most tutorials do. It is kept deliberately, as the control.

Split B (grouped) -- stratified split at the *lesion* level, with any undeclared
                     duplicate groups discovered by the audit merged in first.
                     No lesion (and no near-duplicate cluster) spans two partitions.

Everything else -- ratios, seed, stratification -- is identical between them, because
the split is the only independent variable in this study.

Usage
-----
    python -m src.data.splits
    python -m src.data.splits --no-duplicates    # ignore audit findings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, set_seed


# --------------------------------------------------------------------------------------
# Group construction
# --------------------------------------------------------------------------------------

def load_duplicate_groups(duplicates_csv: Path) -> dict[str, str]:
    """Read audit output and return image_id -> duplicate-cluster id.

    Only *undeclared* duplicates matter here (pairs the metadata says are different
    lesions). Pairs already sharing a lesion_id are handled by lesion grouping.
    """
    if not duplicates_csv.exists():
        return {}

    pairs = pd.read_csv(duplicates_csv)
    if pairs.empty or "same_lesion" not in pairs.columns:
        return {}

    undeclared = pairs[~pairs["same_lesion"].astype(bool)]
    if undeclared.empty:
        return {}

    from src.data.audit import build_duplicate_groups
    return build_duplicate_groups(undeclared)


def effective_groups(meta: pd.DataFrame, duplicate_groups: dict[str, str],
                     group_col: str, id_col: str) -> pd.Series:
    """Merge lesion ids with discovered duplicate clusters into one grouping key.

    If images X and Y are near-identical but sit under different lesion ids, their
    lesions must be treated as a single group -- otherwise the honest split isn't.
    Implemented as union-find over (lesion_id, duplicate_cluster) memberships.
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
            parent[max(rx, ry)] = min(rx, ry)

    for lesion in meta[group_col].unique():
        find(str(lesion))

    if duplicate_groups:
        lookup = meta.set_index(id_col)[group_col]
        # Images sharing a duplicate cluster pull their lesions into one group.
        cluster_to_lesions: dict[str, set[str]] = {}
        for image_id, cluster in duplicate_groups.items():
            lesion = lookup.get(image_id)
            if lesion is not None:
                cluster_to_lesions.setdefault(cluster, set()).add(str(lesion))
        for lesions in cluster_to_lesions.values():
            lesions = sorted(lesions)
            for other in lesions[1:]:
                union(lesions[0], other)

    return meta[group_col].astype(str).map(find)


# --------------------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------------------

def _safe_stratify(labels: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray | None:
    """Return labels for stratification, or None if stratification is impossible.

    train_test_split raises if any class has fewer than 2 members. Rather than crash
    on a rare class, fall back to an unstratified split and say so.
    """
    values = labels if subset is None else labels[subset]
    _, counts = np.unique(values, return_counts=True)
    if counts.min() < 2:
        return None
    return values


def naive_split(meta: pd.DataFrame, label_col: str, test_size: float,
                val_size: float, seed: int) -> pd.Series:
    """Stratified random split at the image level. Leaky by construction."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(meta))
    labels = meta[label_col].to_numpy()

    stratify = _safe_stratify(labels)
    if stratify is None:
        print("⚠ A class has <2 members — falling back to an unstratified naive split.")

    train_idx, holdout_idx = train_test_split(
        idx, test_size=test_size + val_size, random_state=seed, stratify=stratify
    )
    relative_val = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=1 - relative_val, random_state=seed,
        stratify=_safe_stratify(labels, holdout_idx),
    )

    split = pd.Series("train", index=meta.index, dtype=object)
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split


def grouped_split(meta: pd.DataFrame, groups: pd.Series, label_col: str,
                  test_size: float, val_size: float, seed: int) -> pd.Series:
    """Stratified split that keeps every group wholly within one partition.

    Uses StratifiedGroupKFold, which balances the label distribution as far as the
    grouping constraint allows. Groups are atomic: a lesion never spans partitions.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels = meta[label_col].to_numpy()
    group_values = groups.to_numpy()

    # Carve out the holdout (val+test) first, then halve it.
    n_splits_outer = max(2, int(round(1 / (test_size + val_size))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
    train_idx, holdout_idx = next(sgkf.split(np.zeros(len(meta)), labels, group_values))

    holdout_labels = labels[holdout_idx]
    holdout_groups = group_values[holdout_idx]
    relative_test = test_size / (test_size + val_size)
    n_splits_inner = max(2, int(round(1 / relative_test)))
    sgkf_inner = StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=seed
    )
    val_local, test_local = next(
        sgkf_inner.split(np.zeros(len(holdout_idx)), holdout_labels, holdout_groups)
    )

    split = pd.Series("train", index=meta.index, dtype=object)
    split.iloc[holdout_idx[val_local]] = "val"
    split.iloc[holdout_idx[test_local]] = "test"
    return split


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

def measure_contamination(frame: pd.DataFrame, group_key: str = "group") -> dict[str, float]:
    """Empirically measure leakage: what share of test rows share a group with train?

    This is the number the analytic estimate in profile.py approximates, now measured
    directly. For the grouped split it must be exactly zero -- that's the point.
    """
    train_groups = set(frame.loc[frame["split"] == "train", group_key])
    val_groups = set(frame.loc[frame["split"] == "val", group_key])
    test = frame[frame["split"] == "test"]
    if test.empty:
        return {"test_rows": 0, "contaminated_rows": 0, "contaminated_share": 0.0}

    contaminated = test[group_key].isin(train_groups | val_groups)
    return {
        "test_rows": int(len(test)),
        "contaminated_rows": int(contaminated.sum()),
        "contaminated_share": round(float(contaminated.mean()), 4),
    }


def summarise(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-split row counts and class shares, to confirm stratification held."""
    counts = frame.groupby("split").size().rename("rows")
    shares = (
        frame.groupby("split")[label_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .round(3)
    )
    return pd.concat([counts, shares], axis=1)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build naive and grouped splits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-duplicates", action="store_true",
                        help="ignore audit duplicate findings when grouping")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    label_col = cfg.get("dataset.label_col", "dx")
    group_col = cfg.get("dataset.group_col", "lesion_id")
    id_col = cfg.get("dataset.id_col", "image_id")
    test_size = float(cfg.get("splits.test_size", 0.15))
    val_size = float(cfg.get("splits.val_size", 0.15))

    meta = pd.read_csv(cfg.path("paths.metadata")).reset_index(drop=True)

    use_dups = (not args.no_duplicates) and bool(cfg.get("splits.merge_discovered_duplicates", True))
    duplicate_groups = (
        load_duplicate_groups(cfg.path("paths.reports_dir") / "duplicates.csv")
        if use_dups else {}
    )
    if duplicate_groups:
        print(f"Merging {len(duplicate_groups):,} images from undeclared duplicate clusters "
              f"into their lesion groups")
    elif use_dups:
        print("No duplicate audit findings found — grouping on lesion_id alone.")
        print("(Run `python -m src.data.audit --phash` first to include discovered duplicates.)")

    groups = effective_groups(meta, duplicate_groups, group_col, id_col)
    print(f"Effective groups: {groups.nunique():,} (from {meta[group_col].nunique():,} lesions)\n")

    splits_dir = cfg.path("paths.splits_dir")
    splits_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, split_series in (
        ("naive", naive_split(meta, label_col, test_size, val_size, cfg.seed)),
        ("grouped", grouped_split(meta, groups, label_col, test_size, val_size, cfg.seed)),
    ):
        frame = meta[[id_col, group_col, label_col]].copy()
        frame["group"] = groups
        frame["split"] = split_series.to_numpy()

        filename = cfg.get(f"splits.{name}_name", f"split_{name}.csv")
        frame.to_csv(splits_dir / filename, index=False)

        contamination = measure_contamination(frame)
        results[name] = contamination

        print("=" * 62)
        print(f"SPLIT {'A' if name == 'naive' else 'B'} — {name}")
        print("=" * 62)
        print(summarise(frame, label_col).to_string())
        print(f"\n  Test rows                    : {contamination['test_rows']:,}")
        print(f"  Sharing a group with train/val: {contamination['contaminated_rows']:,} "
              f"({contamination['contaminated_share']:.1%})")
        print(f"  Written: {splits_dir / filename}\n")

    print("=" * 62)
    print("LEAKAGE MEASURED")
    print("=" * 62)
    print(f"  Naive split test contamination  : {results['naive']['contaminated_share']:.1%}")
    print(f"  Grouped split test contamination: {results['grouped']['contaminated_share']:.1%}")
    if results["grouped"]["contaminated_rows"] != 0:
        print("\n  ✗ Grouped split is contaminated — this is a bug, investigate before training.")
        return 1
    print("\n  ✓ Grouped split is clean. Train the same model on both to measure the effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
