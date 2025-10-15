#!/usr/bin/env python
"""Summarise WorldClim bioclimatic rasters and emit quick-look figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from logging_setup import initialize_script_logging


initialize_script_logging(__file__)
logger = __import__("logging").getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_STEM = Path(__file__).stem
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figure" / SCRIPT_STEM
SUMMARY_JSON_PATH = PROJECT_ROOT / "logs" / f"{SCRIPT_STEM}.summary.json"

MAX_PREVIEW_DIMENSION = 800
HISTOGRAM_SAMPLE_CAP = 100_000


BIO_LOOKUP = {
    1: ("BIO01", "Annual Mean Temperature", "°C", "temperature"),
    2: ("BIO02", "Mean Diurnal Range", "°C", "temperature"),
    3: ("BIO03", "Isothermality (BIO2/BIO7) ×100", "%", "ratio_scaled"),
    4: ("BIO04", "Temperature Seasonality (σ ×100)", "°C", "temperature_sd"),
    5: ("BIO05", "Max Temperature of Warmest Month", "°C", "temperature"),
    6: ("BIO06", "Min Temperature of Coldest Month", "°C", "temperature"),
    7: ("BIO07", "Temperature Annual Range", "°C", "temperature"),
    8: ("BIO08", "Mean Temperature of Wettest Quarter", "°C", "temperature"),
    9: ("BIO09", "Mean Temperature of Driest Quarter", "°C", "temperature"),
    10: ("BIO10", "Mean Temperature of Warmest Quarter", "°C", "temperature"),
    11: ("BIO11", "Mean Temperature of Coldest Quarter", "°C", "temperature"),
    12: ("BIO12", "Annual Precipitation", "mm", "precipitation"),
    13: ("BIO13", "Precipitation of Wettest Month", "mm", "precipitation"),
    14: ("BIO14", "Precipitation of Driest Month", "mm", "precipitation"),
    15: ("BIO15", "Precipitation Seasonality (CV)", "%", "ratio"),
    16: ("BIO16", "Precipitation of Wettest Quarter", "mm", "precipitation"),
    17: ("BIO17", "Precipitation of Driest Quarter", "mm", "precipitation"),
    18: ("BIO18", "Precipitation of Warmest Quarter", "mm", "precipitation"),
    19: ("BIO19", "Precipitation of Coldest Quarter", "mm", "precipitation"),
}


def discover_worldclim_files(data_dir: Path) -> list[Path]:
    """Return sorted GeoTIFF paths or raise when no rasters are present."""

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


def decode_metadata(path: Path) -> tuple[str, str, str, str]:
    """Return (code, label, units, category) derived from the filename."""

    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    try:
        number = int(digits)
    except ValueError:
        return stem, stem, "unknown", "unknown"

    if number in BIO_LOOKUP:
        return BIO_LOOKUP[number]
    return f"BIO{number:02d}", stem, "unknown", "unknown"


def compute_preview(dataset: rasterio.DatasetReader) -> np.ma.MaskedArray:
    """Read a down-sampled preview using rasterio resampling."""

    max_dim = max(dataset.width, dataset.height)
    if max_dim <= MAX_PREVIEW_DIMENSION:
        out_height, out_width = dataset.height, dataset.width
    else:
        scale = max_dim / MAX_PREVIEW_DIMENSION
        out_height = max(1, int(round(dataset.height / scale)))
        out_width = max(1, int(round(dataset.width / scale)))

    preview = dataset.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear,
        masked=True,
    )
    return preview


def choose_scale(category: str, values: Iterable[float]) -> tuple[float, str]:
    """Return a scale factor plus a note about the applied heuristic."""

    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 1.0, "No valid data; scale defaults to 1.0."

    scale = 1.0
    note_parts: list[str] = []

    if category == "temperature" and any(abs(v) >= 120 for v in vals):
        scale *= 0.1
        note_parts.append("Temperature range suggests ×0.1 rescaling.")
    else:
        note_parts.append("Temperature values fall within expected °C range.")

    if category == "temperature_sd":
        scale *= 0.01
        note_parts.append("Dividing BIO4 by 100 to express σ in °C.")
    if category == "ratio_scaled":
        scale *= 0.01
        note_parts.append("Dividing BIO3 by 100 to return 0–1 ratios.")

    return scale, " ".join(note_parts)


def summarise_file(path: Path, figure_dir: Path) -> dict[str, object]:
    """Compute statistics, log them, and emit preview + histogram figures."""

    with rasterio.open(path) as dataset:
        code, label, units, category = decode_metadata(path)
        data = dataset.read(1, masked=True)
        preview = compute_preview(dataset)

        nodata_count = int(np.sum(data.mask)) if data.mask is not np.ma.nomask else 0
        valid = data.compressed().astype("float64")

        if valid.size:
            min_val = float(valid.min())
            mean_val = float(valid.mean())
            max_val = float(valid.max())
            pct2 = float(np.percentile(valid, 2))
            pct98 = float(np.percentile(valid, 98))
        else:
            min_val = mean_val = max_val = pct2 = pct98 = None

        scale, scale_note = choose_scale(category, (min_val, mean_val, max_val))

        left, bottom, right, top = dataset.bounds
        logger.info("\n%s — %s", code, label)
        logger.info("  File: %s", path.name)
        logger.info("  Driver: GTiff | CRS: %s", dataset.crs or "None")
        logger.info(
            "  Dimensions: %d × %d px | Resolution: %.6f° × %.6f°",
            dataset.width,
            dataset.height,
            dataset.res[0],
            dataset.res[1],
        )
        logger.info(
            "  Bounds: left=%.2f°, right=%.2f°, bottom=%.2f°, top=%.2f°",
            left,
            right,
            bottom,
            top,
        )
        logger.info("  Data type: %s | Band count: %d", dataset.dtypes[0], dataset.count)
        logger.info("  Nodata: %s | Nodata pixels: %d", dataset.nodatavals[0], nodata_count)

        if min_val is None:
            logger.warning("  Raster contains no valid data; skipping figures.")
        else:
            logger.info(
                "  Values (%s): min=%.2f mean=%.2f max=%.2f", units, min_val * scale, mean_val * scale, max_val * scale
            )
            logger.info(
                "  2–98%% percentile clip: %.2f – %.2f %s",
                pct2 * scale,
                pct98 * scale,
                units,
            )
        logger.info("  Scaling note: %s", scale_note)

        figure_dir.mkdir(parents=True, exist_ok=True)

        if min_val is not None:
            scaled_preview = preview.astype("float64") * scale
            vmin = pct2 * scale if pct2 is not None else min_val * scale
            vmax = pct98 * scale if pct98 is not None else max_val * scale
            if vmin is None or vmax is None or math.isclose(vmin, vmax):
                vmin = min_val * scale
                vmax = max_val * scale

            fig, ax = plt.subplots(figsize=(8, 4.5))
            im = ax.imshow(
                np.ma.masked_invalid(scaled_preview),
                extent=(left, right, bottom, top),
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(f"{code} — {label} ({units})")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(units)
            ax.text(
                0.01,
                0.01,
                f"min {min_val * scale:.2f}\nmean {mean_val * scale:.2f}\nmax {max_val * scale:.2f}\nnodata {nodata_count}",
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )
            preview_path = figure_dir / f"{SCRIPT_STEM}__{code.lower()}-preview.png"
            fig.tight_layout()
            fig.savefig(preview_path, dpi=200)
            plt.close(fig)
            logger.info("  Saved preview to %s", preview_path)

            sample = valid
            if sample.size > HISTOGRAM_SAMPLE_CAP:
                step = max(1, sample.size // HISTOGRAM_SAMPLE_CAP)
                sample = sample[::step]
            sample = sample.astype("float64") * scale

            counts, edges = np.histogram(sample, bins=64)
            centers = (edges[:-1] + edges[1:]) / 2
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.step(centers, counts, where="mid", color="tab:blue")
            ax.set_xlabel(units)
            ax.set_ylabel("Pixel count")
            ax.set_title(f"{code} — {label}")
            ax.grid(True, linestyle="--", alpha=0.5)
            hist_path = figure_dir / f"{SCRIPT_STEM}__{code.lower()}-histogram.png"
            fig.tight_layout()
            fig.savefig(hist_path, dpi=200)
            plt.close(fig)
            logger.info("  Saved histogram to %s", hist_path)

        return {
            "file_name": path.name,
            "variable_code": code,
            "variable_name": label,
            "units": units,
            "category": category,
            "width": dataset.width,
            "height": dataset.height,
            "resolution": dataset.res,
            "bounds": (left, bottom, right, top),
            "crs": str(dataset.crs) if dataset.crs else None,
            "dtype": dataset.dtypes[0],
            "nodata_value": dataset.nodatavals[0],
            "nodata_count": nodata_count,
            "valid_pixel_count": int(valid.size),
            "min": None if min_val is None else min_val * scale,
            "mean": None if mean_val is None else mean_val * scale,
            "max": None if max_val is None else max_val * scale,
            "percentile_2": None if pct2 is None else pct2 * scale,
            "percentile_98": None if pct98 is None else pct98 * scale,
            "scale_applied": scale,
            "scale_note": scale_note,
        }


def log_metadata_table(summaries: list[dict[str, object]]) -> None:
    """Emit a compact table of key statistics."""

    if not summaries:
        return

    header = (
        f"{'Code':6} {'Variable':35} {'Size (px)':>15} {'Resolution (°)':>20} "
        f"{'Min':>10} {'Mean':>10} {'Max':>10} {'Nodata':>10}"
    )
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))
    for summary in summaries:
        res = summary["resolution"]
        size_str = f"{summary['width']}×{summary['height']}"
        res_str = f"{res[0]:.4f}×{res[1]:.4f}"
        min_str = "n/a" if summary["min"] is None else f"{summary['min']:.2f}"
        mean_str = "n/a" if summary["mean"] is None else f"{summary['mean']:.2f}"
        max_str = "n/a" if summary["max"] is None else f"{summary['max']:.2f}"
        logger.info(
            "%6s %-35s %15s %20s %10s %10s %10s %10d",
            summary["variable_code"],
            str(summary["variable_name"])[:35],
            size_str,
            res_str,
            min_str,
            mean_str,
            max_str,
            summary["nodata_count"],
        )


def write_summary_json(summaries: list[dict[str, object]], json_path: Path, figure_dir: Path, data_dir: Path) -> None:
    """Persist aggregated metadata for later reference."""

    payload = {
        "script": SCRIPT_STEM,
        "data_directory": str(data_dir),
        "figure_directory": str(figure_dir),
        "raster_count": len(summaries),
        "rasters": summaries,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved summary JSON to %s", json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise WorldClim GeoTIFF rasters, log metadata, and create preview figures."
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

    summaries = []
    for path in files:
        summary = summarise_file(path, figure_dir)
        summaries.append(summary)

    log_metadata_table(summaries)
    write_summary_json(summaries, SUMMARY_JSON_PATH, figure_dir, data_dir)
    logger.info("Inspection complete.")


if __name__ == "__main__":
    main()

