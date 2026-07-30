"""Tests for split construction.

The headline test is `test_grouped_split_has_no_group_leakage`: it asserts in code the
property this entire project is about. If it ever fails, every downstream result is
invalid, so it runs in CI on every commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    effective_groups,
    grouped_split,
    measure_contamination,
    naive_split,
)

SEED = 42
LABEL_COL = "dx"
GROUP_COL = "lesion_id"
ID_COL = "image_id"


@pytest.fixture
def meta() -> pd.DataFrame:
    """Synthetic metadata mimicking HAM10000: imbalanced classes, multi-image lesions."""
    rng = np.random.RandomState(SEED)
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    weights = [0.03, 0.05, 0.11, 0.02, 0.11, 0.66, 0.02]

    rows = []
    image_no = 0
    for lesion_no in range(400):
        n_images = rng.choice([1, 2, 3, 4], p=[0.65, 0.20, 0.10, 0.05])
        dx = rng.choice(classes, p=weights)
        for _ in range(n_images):
            rows.append({
                ID_COL: f"ISIC_{image_no:07d}",
                GROUP_COL: f"HAM_{lesion_no:07d}",
                LABEL_COL: dx,
            })
            image_no += 1
    return pd.DataFrame(rows)


def _build(meta: pd.DataFrame, split: pd.Series, groups: pd.Series) -> pd.DataFrame:
    frame = meta.copy()
    frame["group"] = groups
    frame["split"] = split.to_numpy()
    return frame


# --------------------------------------------------------------------------------------
# The test that matters
# --------------------------------------------------------------------------------------

def test_grouped_split_has_no_group_leakage(meta):
    """No group may appear in more than one partition. This is the project's thesis."""
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    split = grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    frame = _build(meta, split, groups)

    per_group_partitions = frame.groupby("group")["split"].nunique()
    offenders = per_group_partitions[per_group_partitions > 1]

    assert offenders.empty, (
        f"{len(offenders)} group(s) span multiple partitions, e.g. "
        f"{offenders.head().to_dict()}"
    )


def test_grouped_split_measures_zero_contamination(meta):
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    split = grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    frame = _build(meta, split, groups)

    assert measure_contamination(frame)["contaminated_rows"] == 0


def test_naive_split_does_leak(meta):
    """The control must actually be contaminated, or the comparison is meaningless."""
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    split = naive_split(meta, LABEL_COL, 0.15, 0.15, SEED)
    frame = _build(meta, split, groups)

    contamination = measure_contamination(frame)
    assert contamination["contaminated_rows"] > 0, (
        "Naive split showed no leakage — unexpected given multi-image lesions."
    )


# --------------------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("splitter", ["naive", "grouped"])
def test_every_row_assigned_exactly_once(meta, splitter):
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    split = (
        naive_split(meta, LABEL_COL, 0.15, 0.15, SEED) if splitter == "naive"
        else grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    )
    assert len(split) == len(meta)
    assert set(split.unique()) <= {"train", "val", "test"}
    assert split.isna().sum() == 0


@pytest.mark.parametrize("splitter", ["naive", "grouped"])
def test_all_partitions_non_empty(meta, splitter):
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    split = (
        naive_split(meta, LABEL_COL, 0.15, 0.15, SEED) if splitter == "naive"
        else grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    )
    counts = split.value_counts()
    for partition in ("train", "val", "test"):
        assert counts.get(partition, 0) > 0, f"'{partition}' partition is empty"


def test_splits_are_deterministic(meta):
    """Same seed must give the same split, or results aren't reproducible."""
    groups = effective_groups(meta, {}, GROUP_COL, ID_COL)
    first = grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    second = grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    assert first.equals(second)


# --------------------------------------------------------------------------------------
# Duplicate merging
# --------------------------------------------------------------------------------------

def test_duplicate_clusters_merge_lesion_groups(meta):
    """Two lesions linked by a near-duplicate image must collapse into one group."""
    lesion_a, lesion_b = meta[GROUP_COL].unique()[:2]
    image_a = meta.loc[meta[GROUP_COL] == lesion_a, ID_COL].iloc[0]
    image_b = meta.loc[meta[GROUP_COL] == lesion_b, ID_COL].iloc[0]

    baseline = effective_groups(meta, {}, GROUP_COL, ID_COL)
    assert baseline[meta[ID_COL] == image_a].iloc[0] != baseline[meta[ID_COL] == image_b].iloc[0]

    # Audit says these two images are the same lesion in reality.
    duplicate_groups = {image_a: "cluster_0", image_b: "cluster_0"}
    merged = effective_groups(meta, duplicate_groups, GROUP_COL, ID_COL)

    assert merged[meta[ID_COL] == image_a].iloc[0] == merged[meta[ID_COL] == image_b].iloc[0], (
        "Images flagged as duplicates did not end up in the same effective group."
    )


def test_merged_duplicates_stay_together_in_grouped_split(meta):
    """The merge must actually hold through splitting, not just in group assignment."""
    lesion_a, lesion_b = meta[GROUP_COL].unique()[:2]
    image_a = meta.loc[meta[GROUP_COL] == lesion_a, ID_COL].iloc[0]
    image_b = meta.loc[meta[GROUP_COL] == lesion_b, ID_COL].iloc[0]

    groups = effective_groups(
        meta, {image_a: "cluster_0", image_b: "cluster_0"}, GROUP_COL, ID_COL
    )
    split = grouped_split(meta, groups, LABEL_COL, 0.15, 0.15, SEED)
    frame = _build(meta, split, groups)

    partition_a = frame.loc[frame[ID_COL] == image_a, "split"].iloc[0]
    partition_b = frame.loc[frame[ID_COL] == image_b, "split"].iloc[0]
    assert partition_a == partition_b
