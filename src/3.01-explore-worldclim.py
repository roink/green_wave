#!/usr/bin/env python
"""Inspect WorldClim bioclimatic rasters and generate summary plots."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figure" / "worldclim"

MAX_PREVIEW_DIMENSION = 800


@dataclass
class BandStatistics:
    """Statistics for an individual raster band."""

    name: str
    dtype: str
    nodata_value: float | None
    min_value: float | None
    mean_value: float | None
    max_value: float | None


@dataclass
class RasterSummary:
    """Store metadata and lightweight previews for a raster dataset."""

    path: Path
    width: int
    height: int
    band_count: int
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    approx_area_deg2: float
    crs: str | None
    band_statistics: tuple[BandStatistics, ...]
    preview: np.ndarray


def discover_worldclim_files(data_dir: Path) -> list[Path]:
    """Return all TIFF files in the WorldClim directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"WorldClim directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files discovered in {data_dir}")

    print(f"Discovered {len(files)} raster files in {data_dir}")
    return files


def derive_band_names(descriptions: Iterable[str | None], count: int) -> tuple[str, ...]:
    """Generate human-friendly band names."""

    descriptions_list = list(descriptions)
    names: list[str] = []
    for idx in range(count):
        description = descriptions_list[idx] if idx < len(descriptions_list) else None
        if description is None or description.strip() == "":
            names.append(f"Band {idx + 1}")
        else:
            names.append(description.strip())
    return tuple(names)


def compute_preview(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample the data array for quick plotting."""

    max_dimension = max(width, height)
    if max_dimension <= MAX_PREVIEW_DIMENSION:
        return arr.copy()

    scale_factor = int(math.ceil(max_dimension / MAX_PREVIEW_DIMENSION))
    downsampled = arr[::scale_factor, ::scale_factor]
    return downsampled.copy()


def summarise_raster(path: Path) -> RasterSummary:
    """Open a raster file and collect descriptive metadata."""

    with rasterio.open(path) as dataset:
        band_names = derive_band_names(dataset.descriptions, dataset.count)
        nodata_values = tuple(dataset.nodatavals)
        dtypes = tuple(dataset.dtypes)
        band_stats: list[BandStatistics] = []
        preview: np.ndarray | None = None

        for band_index in range(dataset.count):
            data = dataset.read(band_index + 1, masked=True).astype(np.float32)
            array = data.filled(np.nan)

            valid_mask = np.isfinite(array)
            if valid_mask.any():
                min_value = float(np.nanmin(array))
                mean_value = float(np.nanmean(array))
                max_value = float(np.nanmax(array))
            else:
                min_value = mean_value = max_value = None

            nodata_value = (
                nodata_values[band_index] if band_index < len(nodata_values) else None
            )
            dtype = dtypes[band_index] if band_index < len(dtypes) else "unknown"

            band_stats.append(
                BandStatistics(
                    name=band_names[band_index],
                    dtype=dtype,
                    nodata_value=nodata_value,
                    min_value=min_value,
                    mean_value=mean_value,
                    max_value=max_value,
                )
            )

            if preview is None:
                preview = compute_preview(array, dataset.width, dataset.height)

        if preview is None:
            preview = np.empty((dataset.height, dataset.width), dtype=np.float32)
            preview.fill(np.nan)

        bounds = dataset.bounds
        approx_area = float(abs((bounds.right - bounds.left) * (bounds.top - bounds.bottom)))

        resolution = dataset.res
        crs = dataset.crs.to_string() if dataset.crs else None

        summary = RasterSummary(
            path=path,
            width=dataset.width,
            height=dataset.height,
            band_count=dataset.count,
            resolution=resolution,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            approx_area_deg2=approx_area,
            crs=crs,
            band_statistics=tuple(band_stats),
            preview=preview,
        )

    print(f"\nFile: {path.name}")
    print(f"  Driver: GTiff")
    print(f"  Dimensions (width x height): {summary.width} x {summary.height} pixels")
    print(
        "  Resolution: "
        f"{summary.resolution[0]:.6f}° lon x {summary.resolution[1]:.6f}° lat"
    )
    left, bottom, right, top = summary.bounds
    print(
        "  Geographic coverage: "
        f"left={left:.2f}°, right={right:.2f}°, bottom={bottom:.2f}°, top={top:.2f}°"
    )
    print(f"  Approximate area: {summary.approx_area_deg2:.2f} square degrees")
    print(f"  Coordinate reference system: {summary.crs or 'None'}")
    print(f"  Band count: {summary.band_count}")
    for band_stat in summary.band_statistics:
        nodata_display = "None" if band_stat.nodata_value is None else str(band_stat.nodata_value)
        if band_stat.min_value is None:
            stats_display = "no valid data"
        else:
            stats_display = (
                f"min={band_stat.min_value:.2f}, "
                f"mean={band_stat.mean_value:.2f}, "
                f"max={band_stat.max_value:.2f}"
            )
        print(
            "    - "
            f"{band_stat.name}: dtype={band_stat.dtype}, nodata={nodata_display}, {stats_display}"
        )

    return summary


def build_metadata_table(summaries: list[RasterSummary]) -> None:
    """Print a compact table of raster metadata and statistics."""

    if not summaries:
        return

    header = (
        f"{'File':30} {'Size (px)':>15} {'Resolution (°)':>20} "
        f"{'Bands':>7} {'Min':>10} {'Mean':>10} {'Max':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for summary in summaries:
        res_str = f"{summary.resolution[0]:.4f}×{summary.resolution[1]:.4f}"
        size_str = f"{summary.width}×{summary.height}"
        first_band = summary.band_statistics[0] if summary.band_statistics else None
        if first_band is not None and first_band.min_value is not None:
            min_str = f"{first_band.min_value:.2f}"
            mean_str = f"{first_band.mean_value:.2f}"
            max_str = f"{first_band.max_value:.2f}"
        else:
            min_str = mean_str = max_str = "n/a"
        print(
            f"{summary.path.stem:30} {size_str:>15} {res_str:>20} "
            f"{summary.band_count:>7} {min_str:>10} {mean_str:>10} {max_str:>10}"
        )


def plot_raster_previews(
    summaries: list[RasterSummary], figure_dir: Path
) -> Path | None:
    """Generate a multi-panel figure of the raster previews."""

    if not summaries:
        print("No rasters available for plotting previews.")
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)

    columns = 4
    rows = math.ceil(len(summaries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for axis in axes.flat[len(summaries):]:
        axis.axis("off")

    for summary, axis in zip(summaries, axes.flat):
        image = axis.imshow(summary.preview, cmap="viridis")
        axis.set_title(summary.path.stem, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        first_band = summary.band_statistics[0] if summary.band_statistics else None
        if first_band is not None and first_band.min_value is not None:
            axis.text(
                0.02,
                0.02,
                f"min {first_band.min_value:.1f}\n"
                f"mean {first_band.mean_value:.1f}\n"
                f"max {first_band.max_value:.1f}",
                transform=axis.transAxes,
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )

    fig.suptitle("WorldClim bioclimatic variables (preview)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = figure_dir / "worldclim_bioclim_previews.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved preview figure to {output_path}")
    return output_path


def plot_summary_statistics(
    summaries: list[RasterSummary], figure_dir: Path
) -> Path | None:
    """Create a bar chart summarising the value ranges for each raster."""

    valid_summaries = [
        s for s in summaries if s.band_statistics and s.band_statistics[0].min_value is not None
    ]
    if not valid_summaries:
        print("No valid raster statistics available for summary plot.")
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = [s.path.stem for s in valid_summaries]
    means = np.array([s.band_statistics[0].mean_value for s in valid_summaries], dtype=float)
    min_values = np.array([s.band_statistics[0].min_value for s in valid_summaries], dtype=float)
    max_values = np.array([s.band_statistics[0].max_value for s in valid_summaries], dtype=float)

    lower_error = means - min_values
    upper_error = max_values - means

    x_positions = np.arange(len(valid_summaries))

    fig, ax = plt.subplots(figsize=(max(12, len(valid_summaries) * 0.6), 6))
    ax.bar(x_positions, means, color="#1b9e77", alpha=0.8)
    ax.errorbar(
        x_positions,
        means,
        yerr=[lower_error, upper_error],
        fmt="none",
        ecolor="#d95f02",
        capsize=4,
        linewidth=1,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("WorldClim raster value ranges (mean ± min/max)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    output_path = figure_dir / "worldclim_bioclim_statistics.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved summary statistics figure to {output_path}")
    return output_path


def main() -> None:
    """Entry point for the WorldClim inspection script."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarise WorldClim GeoTIFF rasters by printing metadata and creating "
            "preview plots."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing WorldClim .tif files.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Destination directory for generated figures.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()

    try:
        files = discover_worldclim_files(data_dir)
    except FileNotFoundError as exc:
        print(exc)
        print("No rasters processed.")
        return

    summaries = [summarise_raster(path) for path in files]
    build_metadata_table(summaries)
    plot_raster_previews(summaries, figure_dir)
    plot_summary_statistics(summaries, figure_dir)
    print("Inspection complete.")


if __name__ == "__main__":
    main()
