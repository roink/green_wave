#!/usr/bin/env python
"""Summarise WorldClim bioclimatic rasters and emit quick-look figures."""

from __future__ import annotations

import math
import re
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_STEM = Path(__file__).stem
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figure" / SCRIPT_STEM
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

    print(f"Discovered {len(files)} raster files in {data_dir}")
    if len(files) != 19:
        print(
            "WARNING: Expected 19 bioclim rasters but found "
            f"{len(files)}. Check dataset completeness."
        )
    return files


def decode_metadata(path: Path) -> tuple[str, str, str, str]:
    """Return (code, label, units, category) derived from the filename."""

    stem = path.stem
    match = re.search(r"(\d{1,2})$", stem)
    if match:
        number = int(match.group(1))
    else:
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
        normalized_units = units.strip().lower() if isinstance(units, str) else "unknown"
        unit_suffix = ""
        if units and normalized_units != "unknown":
            unit_suffix = f" ({units})"
        value_label = f"{label}{unit_suffix}" if label else units or "Value"
        colorbar_label = units if units and normalized_units != "unknown" else label

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
        print(f"\nFile: {path.name}")
        print(f"  Variable: {code} — {label}")
        print(f"  Driver: {dataset.driver}")
        print(f"  Dimensions (width x height): {dataset.width} x {dataset.height} pixels")
        print(
            "  Resolution: "
            f"{dataset.res[0]:.6f}° lon x {dataset.res[1]:.6f}° lat"
        )
        print(
            "  Geographic coverage: "
            f"left={left:.2f}°, right={right:.2f}°, bottom={bottom:.2f}°, top={top:.2f}°"
        )
        width_deg = abs(right - left)
        height_deg = abs(top - bottom)
        print(f"  Approximate area: {width_deg * height_deg:.2f} square degrees")
        print(
            "  Coordinate reference system: "
            f"{dataset.crs.to_string() if dataset.crs else 'None'}"
        )
        dtype = dataset.dtypes[0]
        nodata_val = dataset.nodatavals[0]
        print(f"  Band count: {dataset.count}")
        scaled_min = None if min_val is None else min_val * scale
        scaled_mean = None if mean_val is None else mean_val * scale
        scaled_max = None if max_val is None else max_val * scale
        scaled_pct2 = None if pct2 is None else pct2 * scale
        scaled_pct98 = None if pct98 is None else pct98 * scale
        min_str = "n/a" if scaled_min is None else f"{scaled_min:.2f}"
        mean_str = "n/a" if scaled_mean is None else f"{scaled_mean:.2f}"
        max_str = "n/a" if scaled_max is None else f"{scaled_max:.2f}"
        print(
            f"    - Band 1: dtype={dtype}, nodata={nodata_val}, "
            f"min={min_str}, mean={mean_str}, max={max_str}"
        )
        print(f"      Valid pixels: {int(valid.size)} | Nodata pixels: {nodata_count}")
        print(f"      Scaling note: {scale_note}")
        if min_val is not None:
            pct_low = "n/a" if scaled_pct2 is None else f"{scaled_pct2:.2f}"
            pct_high = "n/a" if scaled_pct98 is None else f"{scaled_pct98:.2f}"
            print(f"      2–98% percentile clip: {pct_low} – {pct_high} {units}")
            print(
                f"      Values ({units}): min={scaled_min:.2f}"
                f" mean={scaled_mean:.2f} max={scaled_max:.2f}"
            )
        else:
            print("      Raster contains no valid data; skipping figures.")

        figure_dir.mkdir(parents=True, exist_ok=True)

        if min_val is not None:
            scaled_preview = preview.astype("float64") * scale
            vmin = scaled_pct2 if scaled_pct2 is not None else scaled_min
            vmax = scaled_pct98 if scaled_pct98 is not None else scaled_max
            if vmin is None or vmax is None or math.isclose(vmin, vmax):
                vmin = scaled_min
                vmax = scaled_max

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
            title_unit_suffix = f" ({units})" if units and normalized_units != "unknown" else ""
            ax.set_title(f"{code} — {label}{title_unit_suffix}")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            if colorbar_label:
                cbar.set_label(colorbar_label)
            ax.text(
                0.01,
                0.01,
                f"min {min_val * scale:.2f}\nmean {mean_val * scale:.2f}\nmax {max_val * scale:.2f}\nnodata {nodata_count}",
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )
            preview_path = figure_dir / f"{code}-preview.png"
            fig.tight_layout()
            fig.savefig(preview_path, dpi=200)
            plt.close(fig)
            print(f"      Saved preview to {preview_path}")

            sample = valid
            if sample.size > HISTOGRAM_SAMPLE_CAP:
                step = max(1, sample.size // HISTOGRAM_SAMPLE_CAP)
                sample = sample[::step]
            sample = sample.astype("float64") * scale

            counts, edges = np.histogram(sample, bins=64)
            centers = (edges[:-1] + edges[1:]) / 2
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.step(centers, counts, where="mid", color="tab:blue")
            ax.set_xlabel(value_label if value_label else colorbar_label or code)
            ax.set_ylabel("Pixel count")
            ax.set_title(f"{code} — {label}")
            ax.grid(True, linestyle="--", alpha=0.5)
            hist_path = figure_dir / f"{code}-histogram.png"
            fig.tight_layout()
            fig.savefig(hist_path, dpi=200)
            plt.close(fig)
            print(f"      Saved histogram to {hist_path}")

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
            "band_count": dataset.count,
            "nodata_value": dataset.nodatavals[0],
            "nodata_count": nodata_count,
            "valid_pixel_count": int(valid.size),
            "min": scaled_min,
            "mean": scaled_mean,
            "max": scaled_max,
            "percentile_2": scaled_pct2,
            "percentile_98": scaled_pct98,
            "scale_applied": scale,
            "scale_note": scale_note,
        }


def log_metadata_table(summaries: list[dict[str, object]]) -> None:
    """Emit a compact table of key statistics."""

    if not summaries:
        return

    header = (
        f"{'File':35} {'Size (px)':>12} {'Resolution (°)':>16} {'Bands':>7}"
        f" {'Min':>10} {'Mean':>10} {'Max':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for summary in summaries:
        res = summary["resolution"]
        size_str = f"{summary['width']}×{summary['height']}"
        res_str = f"{res[0]:.4f}×{res[1]:.4f}"
        min_str = "n/a" if summary["min"] is None else f"{summary['min']:.2f}"
        mean_str = "n/a" if summary["mean"] is None else f"{summary['mean']:.2f}"
        max_str = "n/a" if summary["max"] is None else f"{summary['max']:.2f}"
        file_stem = Path(summary["file_name"]).stem
        print(
            f"{file_stem:<35} {size_str:>12} {res_str:>16} {summary['band_count']:>7}"
            f" {min_str:>10} {mean_str:>10} {max_str:>10}"
        )
def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    figure_dir = DEFAULT_FIGURE_DIR

    try:
        files = discover_worldclim_files(data_dir)
    except FileNotFoundError as exc:
        print(exc)
        print("No rasters processed.")
        return

    summaries = []
    for path in files:
        summary = summarise_file(path, figure_dir)
        summaries.append(summary)

    log_metadata_table(summaries)
    print("Inspection complete.")


if __name__ == "__main__":
    main()

