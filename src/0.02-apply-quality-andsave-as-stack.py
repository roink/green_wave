#!/usr/bin/env python
"""Build a filtered NDVI stack from raw MODIS HDF files."""

from __future__ import annotations

import glob
import re
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "NDVI"
OUTPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
NEW_CHUNK_SIZE = (1, 256, 256)


def discover_files() -> list[Path]:
    """Locate NDVI HDF files to process."""

    files = sorted(Path(path) for path in glob.glob(str(DATA_PATH / "MOD13C1.A*.hdf")))
    if not files:
        raise FileNotFoundError(f"No HDF files matching MOD13C1 pattern found in {DATA_PATH}")

    print(f"Discovered {len(files)} files in {DATA_PATH}")
    return files


def extract_date(filename: Path) -> tuple[int | None, int | None]:
    """Extract (year, day-of-year) information from the file name."""

    match = re.search(r"A(\d{4})(\d{3})", filename.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


@contextmanager
def open_sd(path: Path):
    """Provide a context manager for ``pyhdf`` SD objects."""

    hdf = SD(str(path), SDC.READ)
    try:
        yield hdf
    finally:
        hdf.end()


def determine_shape(example_file: Path) -> tuple[int, int]:
    """Inspect the first file to determine the spatial dimensions."""

    with open_sd(example_file) as hdf:
        shape = hdf.select("CMG 0.05 Deg 16 days NDVI")[:].shape
    print(f"Detected NDVI grid shape {shape} from {example_file.name}")
    return shape


def build_stack(hdf_files: list[Path], ndvi_shape: tuple[int, int]) -> None:
    """Create an HDF5 stack that stores filtered NDVI data."""

    num_timesteps = len(hdf_files)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunk_layout = (
        NEW_CHUNK_SIZE[0],
        min(NEW_CHUNK_SIZE[1], ndvi_shape[0]),
        min(NEW_CHUNK_SIZE[2], ndvi_shape[1]),
    )

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        dset_ndvi = h5f.create_dataset(
            "ndvi_stack",
            shape=(num_timesteps, *ndvi_shape),
            dtype=np.float32,
            chunks=chunk_layout,
            compression="lzf",
        )

        dset_metadata = h5f.create_dataset(
            "metadata",
            shape=(num_timesteps, 2),
            dtype=np.int32,
        )

        for i, file_path in enumerate(
            tqdm(hdf_files, desc="Processing HDF files", unit="file")
        ):
            year, doy = extract_date(file_path)
            if year is None:
                print(f"Skipping {file_path.name}: no date metadata found in file name")
                continue

            dset_metadata[i] = (year, doy)

            with open_sd(file_path) as hdf:
                try:
                    ndvi_data = hdf.select("CMG 0.05 Deg 16 days NDVI")[:]
                    ndvi_reliability = hdf.select(
                        "CMG 0.05 Deg 16 days pixel reliability"
                    )[:]
                except Exception as exc:  # pragma: no cover - diagnostic output
                    print(f"Could not read {file_path.name}: {exc}")
                    continue

            ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)
            mask = (ndvi_reliability < 0) | (ndvi_reliability > 2)
            ndvi_data[mask] = np.nan

            dset_ndvi[i, :, :] = ndvi_data.astype(np.float32)

    print(f"Optimized NDVI stack saved at {OUTPUT_PATH}")


def main() -> None:
    hdf_files = discover_files()
    ndvi_shape = determine_shape(hdf_files[0])
    build_stack(hdf_files, ndvi_shape)


if __name__ == "__main__":
    main()
