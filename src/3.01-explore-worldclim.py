#!/usr/bin/env python
"""Inspect WorldClim bioclimatic rasters and generate summary plots."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show_hist

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

logger = __import__("logging").getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_STEM = Path(__file__).stem
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figure" / SCRIPT_STEM
SUMMARY_JSON_PATH = PROJECT_ROOT / "logs" / f"{SCRIPT_STEM}.summary.json"

MAX_PREVIEW_DIMENSION = 800


@dataclass
class RasterSummary:
    """Store metadata and lightweight previews for a raster dataset."""

    path: Path
    variable_code: str
    variable_number: int | None
    variable_name: str
    units: str
    scale_applied: float
    scale_note: str
    width: int
    height: int
    band_count: int
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    approx_area_deg2: float
    crs: str | None
    dtype: str
    nodata_value: float | None
    nodata_count: int
    valid_pixel_count: int
    min_value_raw: float | None
    mean_value_raw: float | None
    max_value_raw: float | None
    percentile_2_raw: float | None
    percentile_98_raw: float | None
    min_value_scaled: float | None
    mean_value_scaled: float | None
    max_value_scaled: float | None
    percentile_2_scaled: float | None
    percentile_98_scaled: float | None
    preview: np.ma.MaskedArray
    histogram_sample: np.ndarray


BIO_VARIABLES = {
    1: ("Annual Mean Temperature", "°C", "temperature"),
    2: ("Mean Diurnal Range", "°C", "temperature"),
    3: ("Isothermality (BIO2/BIO7) ×100", "%", "special"),
    4: ("Temperature Seasonality (σ ×100)", "°C", "temperature_seasonality"),
    5: ("Max Temperature of Warmest Month", "°C", "temperature"),
    6: ("Min Temperature of Coldest Month", "°C", "temperature"),
    7: ("Temperature Annual Range", "°C", "temperature"),
    8: ("Mean Temperature of Wettest Quarter", "°C", "temperature"),
    9: ("Mean Temperature of Driest Quarter", "°C", "temperature"),
    10: ("Mean Temperature of Warmest Quarter", "°C", "temperature"),
    11: ("Mean Temperature of Coldest Quarter", "°C", "temperature"),
    12: ("Annual Precipitation", "mm", "precipitation"),
    13: ("Precipitation of Wettest Month", "mm", "precipitation"),
    14: ("Precipitation of Driest Month", "mm", "precipitation"),
    15: ("Precipitation Seasonality (CV)", "%", "special"),
    16: ("Precipitation of Wettest Quarter", "mm", "precipitation"),
    17: ("Precipitation of Driest Quarter", "mm", "precipitation"),
    18: ("Precipitation of Warmest Quarter", "mm", "precipitation"),
    19: ("Precipitation of Coldest Quarter", "mm", "precipitation"),
}

BANDLESS_DESCRIPTION = "Band 1"


def discover_worldclim_files(data_dir: Path) -> list[Path]:
    """Return all TIFF files in the WorldClim directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"WorldClim directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files discovered in {data_dir}")

    logger.info("Discovered %d raster files in %s", len(files), data_dir)
    if len(files) != 19:
        logger.warning(
            "Expected 19 bioclim rasters but found %d. Check dataset completeness.",
            len(files),
        )
    return files


def parse_variable_metadata(path: Path) -> tuple[str, int | None, str, str, str]:
    """Extract the BIO code and human-readable metadata from the filename."""

    match = re.search(r"bio[_-]?(\d{1,2})", path.stem, re.IGNORECASE)
    variable_number = int(match.group(1)) if match else None

    if variable_number and variable_number in BIO_VARIABLES:
        name, units, category = BIO_VARIABLES[variable_number]
        variable_code = f"BIO{variable_number:02d}"
    else:
        name = BANDLESS_DESCRIPTION
        units = "unknown"
        category = "unknown"
        variable_code = path.stem

    return variable_code, variable_number, name, units, category


def compute_preview(dataset: rasterio.DatasetReader) -> np.ma.MaskedArray:
    """Downsample the raster band using on-read resampling."""

    max_dimension = max(dataset.width, dataset.height)
    if max_dimension <= MAX_PREVIEW_DIMENSION:
        out_height = dataset.height
        out_width = dataset.width
    else:
        scale_factor = max_dimension / MAX_PREVIEW_DIMENSION
        out_height = max(1, int(round(dataset.height / scale_factor)))
        out_width = max(1, int(round(dataset.width / scale_factor)))

    preview = dataset.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear,
        masked=True,
    )
    return preview


def detect_temperature_scaling(
    category: str, min_value: float | None, max_value: float | None, mean_value: float | None
) -> tuple[float, str]:
    """Determine if temperature-like rasters require a 0.1 scaling factor."""

    if category != "temperature":
        return 1.0, "No scaling required for non-temperature variable."

    values = [v for v in (min_value, max_value, mean_value) if v is not None]
    if any(abs(v) >= 120 for v in values):
        return 0.1, "Values appear to be stored in tenths; dividing by 10 for reporting."

    return 1.0, "Values fall within expected °C ranges; no scaling applied."


def base_display_scale(category: str, variable_number: int | None) -> float:
    """Provide baseline scale factors for special-case variables."""

    if category == "temperature_seasonality":
        return 0.01
    return 1.0


def summarise_raster(path: Path) -> RasterSummary:
    """Open a raster file and collect descriptive metadata."""

    with rasterio.open(path) as dataset:
        variable_code, variable_number, variable_name, units, category = parse_variable_metadata(
            path
        )

        data = dataset.read(1, masked=True)
        preview = compute_preview(dataset)

        if data.mask is np.ma.nomask:
            nodata_count = 0
        else:
            nodata_count = int(np.sum(data.mask))

        valid_values = data.compressed().astype(float)

        if valid_values.size:
            min_value = float(np.min(valid_values))
            mean_value = float(np.mean(valid_values))
            max_value = float(np.max(valid_values))
            percentile_2 = float(np.percentile(valid_values, 2))
            percentile_98 = float(np.percentile(valid_values, 98))
        else:
            min_value = mean_value = max_value = None
            percentile_2 = percentile_98 = None

        temp_scale, temp_note = detect_temperature_scaling(
            category, min_value, max_value, mean_value
        )
        base_scale = base_display_scale(category, variable_number)
        scale_applied = base_scale * temp_scale

        if valid_values.size:
            scaled_values = valid_values * scale_applied
            max_hist_size = 100_000
            if scaled_values.size > max_hist_size:
                step = max(1, math.ceil(scaled_values.size / max_hist_size))
                histogram_sample = scaled_values[::step]
            else:
                histogram_sample = scaled_values
        else:
            histogram_sample = np.array([], dtype=float)

        def scale_optional(value: float | None) -> float | None:
            return None if value is None else float(value * scale_applied)

        bounds = dataset.bounds
        approx_area = float(abs((bounds.right - bounds.left) * (bounds.top - bounds.bottom)))

        resolution = dataset.res
        crs = dataset.crs.to_string() if dataset.crs else None
        dtype = dataset.dtypes[0] if dataset.count else "unknown"
        nodata_value = dataset.nodatavals[0] if dataset.nodatavals else None

        note_parts: list[str] = [temp_note]
        if category == "temperature_seasonality":
            note_parts.append("Divided by 100 to express standard deviation in °C.")
        if not note_parts:
            note_parts.append("No scaling applied.")

        summary = RasterSummary(
            path=path,
            variable_code=variable_code,
            variable_number=variable_number,
            variable_name=variable_name,
            units=units,
            scale_applied=scale_applied,
            scale_note=" ".join(note_parts),
            width=dataset.width,
            height=dataset.height,
            band_count=dataset.count,
            resolution=resolution,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            approx_area_deg2=approx_area,
            crs=crs,
            dtype=dtype,
            nodata_value=nodata_value,
            nodata_count=nodata_count,
            valid_pixel_count=int(valid_values.size),
            min_value_raw=min_value,
            mean_value_raw=mean_value,
            max_value_raw=max_value,
            percentile_2_raw=percentile_2,
            percentile_98_raw=percentile_98,
            min_value_scaled=scale_optional(min_value),
            mean_value_scaled=scale_optional(mean_value),
            max_value_scaled=scale_optional(max_value),
            percentile_2_scaled=scale_optional(percentile_2),
            percentile_98_scaled=scale_optional(percentile_98),
            preview=preview,
            histogram_sample=histogram_sample,
        )

    left, bottom, right, top = summary.bounds
    logger.info("\n%s (%s) — %s", summary.path.name, summary.variable_code, summary.variable_name)
    logger.info("  Driver: GTiff")
    logger.info(
        "  Dimensions: %d × %d px | Resolution: %.6f° × %.6f°",
        summary.width,
        summary.height,
        summary.resolution[0],
        summary.resolution[1],
    )
    logger.info(
        "  Geographic coverage: left=%.2f°, right=%.2f°, bottom=%.2f°, top=%.2f°",
        left,
        right,
        bottom,
        top,
    )
    logger.info("  Approximate area: %.2f square degrees", summary.approx_area_deg2)
    logger.info("  CRS: %s", summary.crs or "None")
    logger.info("  Band count: %d", summary.band_count)
    logger.info("  Data type: %s", summary.dtype)
    logger.info("  Nodata value: %s", "None" if summary.nodata_value is None else summary.nodata_value)
    logger.info("  Valid pixels: %d | Nodata pixels: %d", summary.valid_pixel_count, summary.nodata_count)

    if summary.min_value_scaled is None:
        logger.warning("  No valid data found in raster.")
    else:
        logger.info(
            "  Value summary (%s): min=%.2f, mean=%.2f, max=%.2f",
            summary.units,
            summary.min_value_scaled,
            summary.mean_value_scaled,
            summary.max_value_scaled,
        )
        if summary.percentile_2_scaled is not None and summary.percentile_98_scaled is not None:
            logger.info(
                "  2–98%% percentile clip: %.2f – %.2f %s",
                summary.percentile_2_scaled,
                summary.percentile_98_scaled,
                summary.units,
            )

    logger.info("  Scaling note: %s", summary.scale_note)

    return summary


def build_metadata_table(summaries: list[RasterSummary]) -> None:
    """Log a compact table of raster metadata and statistics."""

    if not summaries:
        return

    header = (
        f"{'Variable':12} {'Name':35} {'Size (px)':>15} {'Resolution (°)':>20} "
        f"{'Min':>10} {'Mean':>10} {'Max':>10} {'Nodata':>10}"
    )
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for summary in summaries:
        res_str = f"{summary.resolution[0]:.4f}×{summary.resolution[1]:.4f}"
        size_str = f"{summary.width}×{summary.height}"
        min_str = f"{summary.min_value_scaled:.2f}" if summary.min_value_scaled is not None else "n/a"
        mean_str = f"{summary.mean_value_scaled:.2f}" if summary.mean_value_scaled is not None else "n/a"
        max_str = f"{summary.max_value_scaled:.2f}" if summary.max_value_scaled is not None else "n/a"
        logger.info(
            "%12s %-35s %15s %20s %10s %10s %10s %10d",
            summary.variable_code,
            summary.variable_name[:35],
            size_str,
            res_str,
            min_str,
            mean_str,
            max_str,
            summary.nodata_count,
        )


def sanitise_descriptor(text: str) -> str:
    """Return a filesystem-friendly descriptor string."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "preview"


def save_preview_figure(summary: RasterSummary, figure_dir: Path) -> Path | None:
    """Save a georeferenced preview image for a raster."""

    if summary.min_value_scaled is None:
        logger.warning("Skipping preview for %s; raster lacks valid data.", summary.path.name)
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)

    scaled_preview = summary.preview.astype(float) * summary.scale_applied
    vmin = summary.percentile_2_scaled if summary.percentile_2_scaled is not None else summary.min_value_scaled
    vmax = summary.percentile_98_scaled if summary.percentile_98_scaled is not None else summary.max_value_scaled
    if vmin is None or vmax is None or np.isclose(vmin, vmax):
        vmin = summary.min_value_scaled
        vmax = summary.max_value_scaled

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(
        np.ma.masked_invalid(scaled_preview),
        extent=summary.bounds,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"{summary.variable_code} — {summary.variable_name} ({summary.units})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(summary.units)

    ax.text(
        0.01,
        0.01,
        (
            f"min {summary.min_value_scaled:.2f}\n"
            f"mean {summary.mean_value_scaled:.2f}\n"
            f"max {summary.max_value_scaled:.2f}\n"
            f"nodata {summary.nodata_count}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    descriptor = sanitise_descriptor(summary.variable_code.lower())
    output_path = figure_dir / f"{SCRIPT_STEM}__{descriptor}-preview.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info("Saved preview figure to %s", output_path)
    return output_path


def save_histogram(summary: RasterSummary, figure_dir: Path) -> Path | None:
    """Save a histogram showing the value distribution for a raster."""

    if summary.min_value_scaled is None or summary.histogram_sample.size == 0:
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    show_hist(
        summary.histogram_sample,
        bins=64,
        lw=1.5,
        histtype="stepfilled",
        alpha=0.7,
        ax=ax,
        title=f"{summary.variable_code} — {summary.variable_name}",
    )
    ax.set_xlabel(summary.units)
    ax.set_ylabel("Pixel count")
    ax.grid(True, linestyle="--", alpha=0.5)

    descriptor = sanitise_descriptor(summary.variable_code.lower())
    output_path = figure_dir / f"{SCRIPT_STEM}__{descriptor}-histogram.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info("Saved histogram to %s", output_path)
    return output_path


def save_summary_ranges_plot(summaries: list[RasterSummary], figure_dir: Path) -> Path | None:
    """Save a bar plot summarising min/mean/max ranges across rasters."""

    valid_summaries = [s for s in summaries if s.min_value_scaled is not None]
    if not valid_summaries:
        logger.warning("No valid raster statistics available for summary plot.")
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = [s.variable_code for s in valid_summaries]
    means = np.array([s.mean_value_scaled for s in valid_summaries], dtype=float)
    min_values = np.array([s.min_value_scaled for s in valid_summaries], dtype=float)
    max_values = np.array([s.max_value_scaled for s in valid_summaries], dtype=float)

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
    ax.set_ylabel("Value (see individual units)")
    ax.set_title("WorldClim raster value ranges (mean ± min/max)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    output_path = figure_dir / f"{SCRIPT_STEM}__summary-ranges.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info("Saved summary statistics figure to %s", output_path)
    return output_path


def build_summary_json(
    summaries: list[RasterSummary], json_path: Path, data_dir: Path, figure_dir: Path
) -> None:
    """Persist a structured summary of raster statistics."""

    records = []
    for summary in summaries:
        record = {
            "file_name": summary.path.name,
            "variable_code": summary.variable_code,
            "variable_number": summary.variable_number,
            "variable_name": summary.variable_name,
            "units": summary.units,
            "scale_applied": summary.scale_applied,
            "scale_note": summary.scale_note,
            "width": summary.width,
            "height": summary.height,
            "resolution": summary.resolution,
            "bounds": summary.bounds,
            "crs": summary.crs,
            "nodata_value": summary.nodata_value,
            "nodata_count": summary.nodata_count,
            "valid_pixel_count": summary.valid_pixel_count,
            "min": summary.min_value_scaled,
            "mean": summary.mean_value_scaled,
            "max": summary.max_value_scaled,
            "percentile_2": summary.percentile_2_scaled,
            "percentile_98": summary.percentile_98_scaled,
        }
        records.append(record)

    payload = {
        "script": SCRIPT_STEM,
        "data_directory": str(data_dir),
        "figure_directory": str(figure_dir),
        "raster_count": len(summaries),
        "rasters": records,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved summary JSON to %s", json_path)


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
        logger.error("%s", exc)
        logger.error("No rasters processed.")
        return

    summaries = [summarise_raster(path) for path in files]
    summaries.sort(
        key=lambda summary: (
            summary.variable_number is None,
            summary.variable_number if summary.variable_number is not None else float("inf"),
        )
    )
    build_metadata_table(summaries)
    for summary in summaries:
        save_preview_figure(summary, figure_dir)
        save_histogram(summary, figure_dir)

    save_summary_ranges_plot(summaries, figure_dir)
    build_summary_json(summaries, SUMMARY_JSON_PATH, data_dir, figure_dir)

    logger.info("Inspection complete.")


if __name__ == "__main__":
    main()
