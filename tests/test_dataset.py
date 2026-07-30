"""Tests for dataset construction and split loading.

`test_dataset_is_picklable` guards a bug that only appears at runtime on macOS:
DataLoader workers are started with `spawn`, which pickles the dataset. A dataset
class defined inside a factory function is a local object and cannot be pickled, so
training crashed the instant num_workers > 0 — after the model had already been built.
Cheap to test, expensive to rediscover.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.models.dataset import (
    SkinLesionDataset,
    assert_no_group_leakage,
    load_split,
    make_dataset,
)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"],
        "lesion_id": ["HAM_0000001", "HAM_0000001", "HAM_0000002"],
        "dx": ["nv", "mel", "nv"],
        "group": ["HAM_0000001", "HAM_0000001", "HAM_0000002"],
        "split": ["train", "train", "test"],
    })


# --------------------------------------------------------------------------------------
# Pickling — the regression guard
# --------------------------------------------------------------------------------------

def test_dataset_is_picklable(frame):
    """DataLoader workers pickle the dataset on macOS/Windows spawn."""
    dataset = make_dataset(frame, Path("data/raw/images"), CLASSES, 224, train=True)

    restored = pickle.loads(pickle.dumps(dataset))

    assert isinstance(restored, SkinLesionDataset)
    assert restored.image_ids == dataset.image_ids
    assert restored.targets == dataset.targets
    assert len(restored) == len(dataset)


def test_transform_is_not_pickled(frame):
    """The transform is rebuilt per worker, so it must not bloat the pickled payload."""
    dataset = make_dataset(frame, Path("data/raw/images"), CLASSES, 224, train=True)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._transform is None


def test_dataset_class_is_module_level():
    """A locally-defined class would be unpicklable; keep it importable."""
    assert SkinLesionDataset.__qualname__ == "SkinLesionDataset", (
        "SkinLesionDataset must stay at module level (no nesting inside a factory)"
    )


# --------------------------------------------------------------------------------------
# Label mapping
# --------------------------------------------------------------------------------------

def test_targets_map_to_class_indices(frame):
    dataset = make_dataset(frame, Path("data/raw/images"), CLASSES, 224, train=False)
    assert dataset.targets == [CLASSES.index("nv"), CLASSES.index("mel"), CLASSES.index("nv")]


def test_dataset_length_matches_frame(frame):
    dataset = make_dataset(frame, Path("data/raw/images"), CLASSES, 224, train=False)
    assert len(dataset) == len(frame)


# --------------------------------------------------------------------------------------
# Split loading
# --------------------------------------------------------------------------------------

def test_load_split_filters_partition(tmp_path, frame):
    frame.to_csv(tmp_path / "split_test.csv", index=False)

    train = load_split(tmp_path, "split_test.csv", "train")
    assert len(train) == 2
    assert set(train["split"]) == {"train"}


def test_load_split_rejects_missing_columns(tmp_path):
    pd.DataFrame({"image_id": ["a"], "dx": ["nv"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_split(tmp_path, "bad.csv")


def test_load_split_errors_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "does_not_exist.csv")


def test_leakage_guard_accepts_clean_split(frame):
    assert_no_group_leakage(frame)      # each group sits in exactly one partition


def test_leakage_guard_rejects_contaminated_split(frame):
    contaminated = frame.copy()
    contaminated.loc[1, "split"] = "test"   # group HAM_0000001 now spans train and test
    with pytest.raises(ValueError, match="contaminated"):
        assert_no_group_leakage(contaminated)
