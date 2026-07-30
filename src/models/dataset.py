"""Dataset, transforms and class-imbalance handling.

The split CSV is the single source of truth for which images belong to which
partition. Nothing here re-derives a split -- that would risk the two experiments
diverging in a way that isn't the variable under test.

The imbalance helpers (`compute_class_weights`, `sampler_weights`) are pure
NumPy/pandas so they can be unit-tested without a GPU or even torch installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ImageNet statistics -- the pretrained backbones expect inputs normalised this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------
# Split loading
# --------------------------------------------------------------------------------------

def load_split(splits_dir: Path, filename: str, partition: str | None = None) -> pd.DataFrame:
    """Load a split CSV, optionally filtered to one partition.

    Expected columns: image_id, lesion_id, dx, group, split.
    """
    path = splits_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}. Run `python -m src.data.splits` first."
        )
    frame = pd.read_csv(path)

    required = {"image_id", "dx", "group", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if partition is not None:
        frame = frame[frame["split"] == partition].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No rows for partition '{partition}' in {path.name}")
    return frame


def assert_no_group_leakage(frame: pd.DataFrame) -> None:
    """Fail loudly if a group spans partitions.

    Cheap insurance: if a split file is ever regenerated incorrectly, this catches it
    before a training run produces numbers that look fine and mean nothing.
    """
    spans = frame.groupby("group")["split"].nunique()
    offenders = spans[spans > 1]
    if not offenders.empty:
        raise ValueError(
            f"{len(offenders)} group(s) span multiple partitions "
            f"(e.g. {list(offenders.index[:3])}). This split is contaminated."
        )


# --------------------------------------------------------------------------------------
# Class imbalance
# --------------------------------------------------------------------------------------

def compute_class_weights(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    Normalising keeps the loss on a comparable scale to the unweighted case, so the
    learning rate doesn't need retuning when the strategy changes.
    Classes absent from this partition get weight 1 rather than infinity.
    """
    counts = labels.value_counts()
    n_samples = len(labels)
    n_classes = len(classes)

    weights = np.ones(n_classes, dtype=np.float32)
    for i, cls in enumerate(classes):
        count = int(counts.get(cls, 0))
        if count > 0:
            weights[i] = n_samples / (n_classes * count)

    return weights / weights.mean()


def sampler_weights(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Per-sample weights for a WeightedRandomSampler (balanced batches)."""
    class_weights = compute_class_weights(labels, classes)
    index = {cls: i for i, cls in enumerate(classes)}
    return np.array([class_weights[index[label]] for label in labels], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Torch-dependent pieces
# --------------------------------------------------------------------------------------

def build_transforms(image_size: int, train: bool):
    """Augmentation for training, deterministic resize for eval.

    Dermatoscopic images have no canonical orientation, so flips and full rotations are
    legitimate. Colour jitter is kept mild: hue is diagnostically meaningful in skin
    lesions, so aggressive jitter would destroy signal rather than regularise.

    Augmentation is applied *inside* the dataset, after the split has been fixed, so an
    augmented view can never cross the train/test boundary.
    """
    import torchvision.transforms as T

    if not train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_dataset(frame: pd.DataFrame, image_dir: Path, classes: list[str],
                 image_size: int, train: bool):
    """Build a torch Dataset over the rows of a split partition."""
    from PIL import Image
    from torch.utils.data import Dataset

    transform = build_transforms(image_size, train=train)
    class_index = {cls: i for i, cls in enumerate(classes)}

    image_ids = frame["image_id"].tolist()
    targets = [class_index[label] for label in frame["dx"]]

    class _SkinLesionDataset(Dataset):
        def __init__(self):
            self.image_ids = image_ids
            self.targets = targets

        def __len__(self) -> int:
            return len(self.image_ids)

        def __getitem__(self, idx: int):
            path = image_dir / f"{self.image_ids[idx]}.jpg"
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB"))
            return tensor, self.targets[idx]

    return _SkinLesionDataset()


def make_loader(frame: pd.DataFrame, image_dir: Path, classes: list[str],
                image_size: int, batch_size: int, train: bool,
                imbalance_strategy: str = "class_weights",
                num_workers: int = 4, seed: int = 42):
    """Build a DataLoader for one partition.

    Only `weighted_sampler` changes the sampling distribution; `class_weights` is applied
    in the loss instead (see train.py), and `none` does neither.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = make_dataset(frame, image_dir, classes, image_size, train=train)

    sampler = None
    shuffle = train
    if train and imbalance_strategy == "weighted_sampler":
        weights = sampler_weights(frame["dx"], classes)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False   # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,      # MPS/CPU friendly; harmless on CUDA at this scale
        drop_last=False,
    )
