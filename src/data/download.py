"""Verify (and help locate) the HAM10000 dataset.

The images are CC BY-NC licensed and are NOT redistributed with this repo, so this
module does not fetch them automatically. Instead it:

  * tells you exactly where to get them,
  * checks that what you downloaded is complete and internally consistent,
  * flattens the two-folder image layout that both Kaggle and Dataverse ship.

Usage
-----
    python -m src.data.download --check
    python -m src.data.download --check --flatten
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

DOWNLOAD_INSTRUCTIONS = """
HAM10000 is not bundled with this repository (CC BY-NC licence). Download it from
either source below, then place the files under `data/raw/`.

  Option 1 — Kaggle (easiest; free GPU notebooks alongside)
    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    Or via CLI:  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

  Option 2 — Harvard Dataverse (canonical source, DOI 10.7910/DVN/DBW86T)
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Expected layout in data/raw/ (either form is accepted):

    data/raw/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/     <- or a single flat `images/` folder
    │   └── ISIC_0024306.jpg ...
    └── HAM10000_images_part_2/
        └── ISIC_0024307.jpg ...

Run `python -m src.data.download --check --flatten` to merge the two part folders
into `data/raw/images/`.
"""

IMAGE_DIR_CANDIDATES = (
    "images",
    "HAM10000_images_part_1",
    "HAM10000_images_part_2",
    "ham10000_images_part_1",
    "ham10000_images_part_2",
)

REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}


def find_image_dirs(raw_dir: Path) -> list[Path]:
    """Return every directory under raw_dir that plausibly holds the JPEGs."""
    dirs = [raw_dir / name for name in IMAGE_DIR_CANDIDATES]
    found = [d for d in dirs if d.is_dir()]
    if found:
        return found
    # Fall back: any subdirectory containing at least one .jpg
    return [d for d in raw_dir.iterdir() if d.is_dir() and any(d.glob("*.jpg"))]


def index_images(raw_dir: Path) -> dict[str, Path]:
    """Map image_id -> path on disk, across however many folders the images live in."""
    index: dict[str, Path] = {}
    for directory in find_image_dirs(raw_dir):
        for path in directory.glob("*.jpg"):
            index[path.stem] = path
    return index


def flatten_images(raw_dir: Path) -> int:
    """Merge the part_1 / part_2 folders into a single `images/` directory.

    Returns the number of files moved. Idempotent: files already in place are skipped.
    """
    target = raw_dir / "images"
    target.mkdir(exist_ok=True)
    moved = 0
    for directory in find_image_dirs(raw_dir):
        if directory.resolve() == target.resolve():
            continue
        for path in directory.glob("*.jpg"):
            destination = target / path.name
            if not destination.exists():
                shutil.move(str(path), str(destination))
                moved += 1
    return moved


def check(raw_dir: Path, metadata_path: Path, expected: int) -> bool:
    """Validate the download. Returns True if everything looks right.

    Checks, in order: metadata present -> required columns -> image count ->
    metadata/disk correspondence in both directions.
    """
    ok = True

    if not metadata_path.exists():
        print(f"✗ Metadata not found at {metadata_path}")
        print(DOWNLOAD_INSTRUCTIONS)
        return False

    meta = pd.read_csv(metadata_path)
    print(f"✓ Metadata found: {len(meta):,} rows")

    missing_cols = REQUIRED_COLUMNS - set(meta.columns)
    if missing_cols:
        print(f"✗ Metadata is missing expected columns: {sorted(missing_cols)}")
        ok = False
    else:
        print(f"✓ All required columns present (incl. 'lesion_id' — the one this project turns on)")

    if len(meta) != expected:
        print(f"⚠ Expected {expected:,} metadata rows, found {len(meta):,}")

    images = index_images(raw_dir)
    if not images:
        print(f"✗ No .jpg files found under {raw_dir}")
        print(DOWNLOAD_INSTRUCTIONS)
        return False
    print(f"✓ Images found on disk: {len(images):,}")

    if "image_id" in meta.columns:
        meta_ids = set(meta["image_id"])
        disk_ids = set(images)

        missing_on_disk = meta_ids - disk_ids
        if missing_on_disk:
            print(f"✗ {len(missing_on_disk):,} images referenced in metadata are missing on disk")
            print(f"   e.g. {sorted(missing_on_disk)[:5]}")
            ok = False
        else:
            print("✓ Every image referenced in the metadata is present on disk")

        orphans = disk_ids - meta_ids
        if orphans:
            print(f"⚠ {len(orphans):,} images on disk have no metadata row (ignored downstream)")

    # A first glimpse of the thing this whole project is about.
    if {"lesion_id", "image_id"}.issubset(meta.columns):
        per_lesion = meta.groupby("lesion_id")["image_id"].count()
        multi = int((per_lesion > 1).sum())
        extra = int(len(meta) - per_lesion.size)
        print()
        print(f"  Unique lesions        : {per_lesion.size:,}")
        print(f"  Lesions with >1 image : {multi:,}")
        print(f"  Redundant images      : {extra:,} "
              f"({extra / max(len(meta), 1):.1%} of the dataset)")
        print("  ^ these are why an image-level random split leaks.")

    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the HAM10000 download.")
    parser.add_argument("--check", action="store_true", help="validate the dataset on disk")
    parser.add_argument("--flatten", action="store_true",
                        help="merge part_1/part_2 image folders into images/")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = cfg.path("paths.raw_dir")
    metadata_path = cfg.path("paths.metadata")
    expected = int(cfg.get("dataset.n_images_expected", 10015))

    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {raw_dir}")
        print(DOWNLOAD_INSTRUCTIONS)
        return 1

    if args.flatten:
        moved = flatten_images(raw_dir)
        print(f"Flattened image folders: {moved:,} file(s) moved into {raw_dir / 'images'}")

    if args.check or not args.flatten:
        return 0 if check(raw_dir, metadata_path, expected) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
