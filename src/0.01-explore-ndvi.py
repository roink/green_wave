#!/usr/bin/env python
"""Inspect a sample NDVI file and export quick-look visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "NDVI" / "MOD13C1.A2009241.061.2021141172023.hdf"


def figure_directory() -> Path:
    """Return the directory where figures for this script are stored."""

    directory = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def open_dataset(file_path: Path) -> SD:
    """Open the provided HDF file and return the dataset handle."""

    print(f"Opening HDF file: {file_path}")
    return SD(str(file_path), SDC.READ)


def describe_file(hdf: SD) -> None:
    """Print dataset names and metadata contained in the HDF file."""

    datasets = hdf.datasets()
    print("Datasets in file:")
    for key, value in datasets.items():
        print(f"  {key}: {value}")

    attrs = hdf.attributes()
    print("\nMetadata:")
    for attr, value in attrs.items():
        print(f"  {attr}: {value}")


def plot_ndvi(hdf: SD, output_dir: Path) -> None:
    """Plot the NDVI data layer and save it as a PNG image."""

    ndvi_data = hdf.select("CMG 0.05 Deg 16 days NDVI")[:]
    ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(ndvi_data, cmap="RdYlGn", vmin=-2000, vmax=10000)
    fig.colorbar(img, label="NDVI Value", ax=ax)
    ax.set_title("MODIS NDVI, 2009 day 241")

    output_path = output_dir / "global-ndvi-sample.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NDVI preview to {output_path}")


def plot_quality(hdf: SD, output_dir: Path) -> None:
    """Plot the pixel reliability layer and save it as a PNG image."""

    ndvi_quality = hdf.select("CMG 0.05 Deg 16 days pixel reliability")[:]

    unique_values, counts = np.unique(ndvi_quality, return_counts=True)
    print("Unique values in the quality dataset and their frequencies:")
    for value, count in zip(unique_values, counts):
        print(f"  {value}: {count}")

    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(ndvi_quality, cmap="RdYlGn")
    fig.colorbar(img, label="NDVI Quality", ax=ax)
    ax.set_title("NDVI Quality, 2009 day 241")

    output_path = output_dir / "global-ndvi-quality.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NDVI quality preview to {output_path}")


def main() -> None:
    output_dir = figure_directory()
    hdf = open_dataset(FILE_PATH)
    describe_file(hdf)
    plot_ndvi(hdf, output_dir)
    plot_quality(hdf, output_dir)


if __name__ == "__main__":
    main()

