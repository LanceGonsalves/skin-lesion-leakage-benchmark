"""Normalise a Harvard Dataverse download into the layout this repo expects.

Dataverse ships HAM10000 differently to Kaggle:

    dataverse_files/
    ├── HAM10000_images_part_1.zip
    ├── HAM10000_images_part_2.zip
    ├── HAM10000_metadata                    <- no extension, often TAB-separated
    ├── HAM10000_segmentations_lesion_tschandl.zip
    ├── ISIC2018_Task3_Test_Images.zip       <- separate ISIC test set, not needed here
    ├── ISIC2018_Task3_Test_GroundTruth.csv
    └── ISIC2018_Task3_Test_NatureMedicine...csv

This script unzips the two image parts into `data/raw/images/` and writes a proper
`data/raw/HAM10000_metadata.csv`, sniffing the delimiter rather than assuming it.
The segmentation and ISIC-2018 archives are left alone -- they aren't used by the
classification benchmark, though the segmentations are there if you extend to that task.

Usage
-----
    python -m src.data.setup_dataverse ~/Downloads/dataverse_files
    python -m src.data.setup_dataverse ~/Downloads/dataverse_files --move
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.config import load_config

IMAGE_ARCHIVES = ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip")
METADATA_CANDIDATES = (
    "HAM10000_metadata",
    "HAM10000_metadata.csv",
    "HAM10000_metadata.tab",
    "HAM10000_metadata.txt",
)
EXPECTED_COLUMNS = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}


def find_metadata(source: Path) -> Path | None:
    """Locate the metadata file, whatever extension Dataverse gave it."""
    for name in METADATA_CANDIDATES:
        candidate = source / name
        if candidate.is_file():
            return candidate
    # Last resort: anything starting with the expected stem.
    matches = sorted(p for p in source.glob("HAM10000_metadata*") if p.is_file())
    return matches[0] if matches else None


def sniff_and_read(path: Path) -> pd.DataFrame:
    """Read the metadata without assuming comma or tab separation.

    Dataverse commonly converts uploaded CSVs to tab-separated `.tab` files and
    strips the extension on download, so guessing wrong yields a single-column
    DataFrame rather than an error -- hence the explicit column check.
    """
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        # Fall back to whichever candidate appears most in the header line.
        header = sample.splitlines()[0] if sample else ""
        delimiter = max(",\t;|", key=header.count)

    frame = pd.read_csv(path, sep=delimiter)

    if not EXPECTED_COLUMNS.issubset(frame.columns):
        # Delimiter guess was wrong -- try the alternatives explicitly.
        for alternative in (",", "\t", ";", "|"):
            if alternative == delimiter:
                continue
            retry = pd.read_csv(path, sep=alternative)
            if EXPECTED_COLUMNS.issubset(retry.columns):
                print(f"  (delimiter sniffed as {delimiter!r}, actually {alternative!r})")
                return retry
        missing = EXPECTED_COLUMNS - set(frame.columns)
        raise ValueError(
            f"Could not parse {path.name}: missing columns {sorted(missing)}. "
            f"Parsed columns were {list(frame.columns)}"
        )

    print(f"  Parsed with delimiter {delimiter!r}")
    return frame


def extract_images(source: Path, target: Path, move: bool = False) -> int:
    """Unzip both image parts into a single flat directory. Idempotent."""
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for archive_name in IMAGE_ARCHIVES:
        archive = source / archive_name
        if not archive.is_file():
            print(f"  ⚠ Missing archive: {archive_name}")
            continue

        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist()
                       if m.lower().endswith(".jpg") and not m.startswith("__MACOSX")]
            new_members = [m for m in members if not (target / Path(m).name).exists()]

            print(f"  {archive_name}: {len(members):,} images "
                  f"({len(new_members):,} to extract)")

            for member in new_members:
                # Flatten: ignore any directory structure inside the zip.
                with zf.open(member) as src_file:
                    destination = target / Path(member).name
                    with open(destination, "wb") as dst_file:
                        shutil.copyfileobj(src_file, dst_file)
                extracted += 1

        if move:
            archive.unlink()
            print(f"  Removed {archive_name}")

    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalise a Harvard Dataverse HAM10000 download."
    )
    parser.add_argument("source", help="path to the downloaded dataverse_files folder")
    parser.add_argument("--move", action="store_true",
                        help="delete the source zips after extracting (saves ~3 GB)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"✗ Not a directory: {source}")
        return 1

    cfg = load_config(args.config)
    raw_dir = cfg.path("paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}")
    print(f"Target: {raw_dir}\n")

    # --- metadata -------------------------------------------------------------
    print("Metadata")
    metadata_path = find_metadata(source)
    if metadata_path is None:
        print(f"  ✗ No HAM10000_metadata* file found in {source}")
        return 1

    print(f"  Found: {metadata_path.name}")
    frame = sniff_and_read(metadata_path)
    destination = cfg.path("paths.metadata")
    frame.to_csv(destination, index=False)
    print(f"  ✓ Wrote {len(frame):,} rows -> {destination}")

    # --- images ---------------------------------------------------------------
    print("\nImages")
    extracted = extract_images(source, raw_dir / "images", move=args.move)
    print(f"  ✓ Extracted {extracted:,} new image(s)")

    total_images = len(list((raw_dir / "images").glob("*.jpg")))
    print(f"  Total on disk: {total_images:,}")

    # --- verdict --------------------------------------------------------------
    print()
    if total_images == len(frame):
        print(f"✓ Setup complete — {total_images:,} images match {len(frame):,} metadata rows.")
    else:
        print(f"⚠ {total_images:,} images on disk vs {len(frame):,} metadata rows — "
              f"run the check below for detail.")

    print("\nNext:")
    print("  python -m src.data.download --check")
    print("  python -m src.data.profile")
    print("  python -m src.data.audit --phash --contact-sheet")
    print("  python -m src.data.splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
