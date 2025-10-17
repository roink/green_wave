#!/usr/bin/env python3
"""Inspect the harmonic semiannual trend fit outputs and generate quicklooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

from logging_setup import initialize_script_logging
from ndvi_analysis_utils import ensure_script_figure_dir, _save_figure

initialize_script_logging(__file__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARMONIC_PATH = (
    PROJECT_ROOT / "data" / "intermediate" / "ndvi_harmonic_fit_semiannual_trend.h5"
)

EUROPE_LAT_BOUNDS = (30.0, 72.0)
EUROPE_LON_BOUNDS = (-12.0, 40.0)

class GridGeometry:
    """Describe the spatial layout of the global NDVI grid."""

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)

    @property
    def lat_step(self) -> float:
        return 180.0 / self.n_rows

    @property
    def lon_step(self) -> float:
        return 360.0 / self.n_cols

    @property
    def global_extent(self) -> tuple[float, float, float, float]:
        return (-180.0, 180.0, -90.0, 90.0)

    def region_slices(
        self,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
    ) -> tuple[slice, slice, tuple[float, float, float, float]]:
        """Return array slices and extent for a lat/lon bounding box."""

        lat_min, lat_max = sorted(lat_bounds)
        lon_min, lon_max = sorted(lon_bounds)

        lat_min = float(np.clip(lat_min, -90.0, 90.0))
        lat_max = float(np.clip(lat_max, -90.0, 90.0))
        lon_min = float(np.clip(lon_min, -180.0, 180.0))
        lon_max = float(np.clip(lon_max, -180.0, 180.0))

        row_start = int(np.floor((90.0 - lat_max) / self.lat_step))
        row_end = int(np.ceil((90.0 - lat_min) / self.lat_step))
        col_start = int(np.floor((lon_min + 180.0) / self.lon_step))
        col_end = int(np.ceil((lon_max + 180.0) / self.lon_step))

        row_start = int(np.clip(row_start, 0, self.n_rows))
        row_end = int(np.clip(row_end, row_start + 1, self.n_rows))
        col_start = int(np.clip(col_start, 0, self.n_cols))
        col_end = int(np.clip(col_end, col_start + 1, self.n_cols))

        extent = (lon_min, lon_max, lat_min, lat_max)
        return slice(row_start, row_end), slice(col_start, col_end), extent


def _decode_attr_value(value) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return [v.decode("utf-8") for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _print_attributes(header: str, attrs: h5py.AttributeManager) -> None:
    print(header)
    if len(attrs) == 0:
        print("  (no attributes)")
        return
    for key, value in attrs.items():
        decoded = _decode_attr_value(value)
        print(f"  - {key}: {decoded}")


def _summarise_array(name: str, array: np.ndarray) -> None:
    total = array.size
    finite_mask = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite_mask))
    print(
        f"    {name}: shape={array.shape}, dtype={array.dtype}, "
        f"valid={finite_count:,}/{total:,} ({finite_count / total:.1%})"
    )
    if finite_count == 0:
        print("      No finite values present.")
        return

    valid_values = array[finite_mask]
    stats = {
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "median": float(np.median(valid_values)),
        "std": float(np.std(valid_values)),
    }
    print(
        "      "
        + ", ".join(f"{key}={value:.3f}" for key, value in stats.items())
    )


def _choose_hist_bins(values: np.ndarray) -> int | np.ndarray:
    if values.size == 0:
        return 10
    if np.issubdtype(values.dtype, np.integer):
        min_val = int(values.min())
        max_val = int(values.max())
        if max_val - min_val <= 100:
            return np.arange(min_val - 0.5, max_val + 1.5)
    return 100


def _plot_maps(
    data: np.ndarray,
    label: str,
    geometry: GridGeometry,
    script_stem: str,
    figure_dir: Path,
    *,
    cmap: str = "viridis",
) -> None:
    finite = np.isfinite(data)
    if not np.any(finite):
        print(f"      Skipping maps for {label}: no finite values.")
        return

    valid = data[finite]
    vmin, vmax = np.nanpercentile(valid, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    im = axes[0].imshow(
        data,
        origin="upper",
        extent=geometry.global_extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"{label} – Global")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    region_rows, region_cols, region_extent = geometry.region_slices(
        EUROPE_LAT_BOUNDS, EUROPE_LON_BOUNDS
    )
    region_data = data[region_rows, region_cols]
    if region_data.size == 0 or not np.any(np.isfinite(region_data)):
        axes[1].set_title(f"{label} – Europe (no data)")
        axes[1].set_axis_off()
    else:
        axes[1].imshow(
            region_data,
            origin="upper",
            extent=region_extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(f"{label} – Europe")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(label)

    _save_figure(fig, f"{label} maps", script_stem=script_stem, figure_dir=figure_dir)


def _plot_histogram(
    data: np.ndarray, label: str, script_stem: str, figure_dir: Path
) -> None:
    finite = np.isfinite(data)
    if not np.any(finite):
        print(f"      Skipping histogram for {label}: no finite values.")
        return
    values = data[finite]
    bins = _choose_hist_bins(values)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.85)
    ax.set_title(f"{label} distribution")
    ax.set_xlabel(label)
    ax.set_ylabel("Pixel count")
    _save_figure(fig, f"{label} histogram", script_stem=script_stem, figure_dir=figure_dir)


def _iter_spatial_layers(
    name: str, dataset: h5py.Dataset
) -> Iterator[tuple[str, np.ndarray]]:
    if dataset.ndim == 2:
        yield name, dataset[:, :]
        return

    if dataset.ndim == 3 and dataset.shape[2] <= 20:
        param_names: Sequence[str] | None = None
        if "parameter_names" in dataset.attrs:
            raw_names = dataset.attrs["parameter_names"]
            param_names = [
                v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in raw_names
            ]
        for idx in range(dataset.shape[2]):
            if param_names is not None and idx < len(param_names):
                layer_name = f"{name}/{param_names[idx]}"
            else:
                layer_name = f"{name}/component_{idx}"
            yield layer_name, dataset[:, :, idx]
    else:
        print(
            f"  Warning: dataset {name} has unsupported dimensions {dataset.shape}; skipping maps."
        )


def _summarise_metadata(dataset: h5py.Dataset) -> None:
    array = dataset[:, :]
    print(f"  Dataset '{dataset.name}' preview (first 5 rows):")
    preview = array[:5]
    with np.printoptions(edgeitems=5, linewidth=120):
        print(preview)


def _summarise_time_offsets(dataset: h5py.Dataset) -> None:
    offsets = dataset[:]
    print(
        "  Time offsets statistics (days): min={:.1f}, max={:.1f}, mean={:.1f}".format(
            float(np.min(offsets)), float(np.max(offsets)), float(np.mean(offsets))
        )
    )


def main() -> None:
    if not HARMONIC_PATH.exists():
        raise FileNotFoundError(
            "Harmonic fit file missing. "
            f"Expected {HARMONIC_PATH}. Run 0.11-fit-harmonic-models.py first."
        )

    script_stem, figure_dir = ensure_script_figure_dir(Path(__file__))

    print(f"Opening harmonic fit file: {HARMONIC_PATH}")
    with h5py.File(HARMONIC_PATH, "r") as h5f:
        _print_attributes("File attributes:", h5f.attrs)

        spatial_reference = None
        for key in ("r_squared", "parameters", "adjusted_r_squared"):
            if key in h5f:
                ds = h5f[key]
                spatial_reference = GridGeometry(ds.shape[0], ds.shape[1])
                break
        if spatial_reference is None:
            raise ValueError("Could not determine spatial grid dimensions from datasets.")

        print(
            "Spatial grid: rows={}, cols={}, lat_step={:.3f}°, lon_step={:.3f}°".format(
                spatial_reference.n_rows,
                spatial_reference.n_cols,
                spatial_reference.lat_step,
                spatial_reference.lon_step,
            )
        )
        region_rows, region_cols, region_extent = spatial_reference.region_slices(
            EUROPE_LAT_BOUNDS, EUROPE_LON_BOUNDS
        )
        print(
            "Europe slice: rows[{}:{}], cols[{}:{}], extent={}".format(
                region_rows.start,
                region_rows.stop,
                region_cols.start,
                region_cols.stop,
                region_extent,
            )
        )

        for name in sorted(h5f.keys()):
            dataset = h5f[name]
            print(f"\nDataset '{name}': shape={dataset.shape}, dtype={dataset.dtype}")
            _print_attributes("  Attributes:", dataset.attrs)

            if name == "metadata":
                _summarise_metadata(dataset)
                continue
            if name == "time_offsets_days":
                _summarise_time_offsets(dataset)
                continue

            for layer_name, layer_data in _iter_spatial_layers(name, dataset):
                _summarise_array(layer_name, layer_data)
                _plot_maps(layer_data, layer_name, spatial_reference, script_stem, figure_dir)
                _plot_histogram(layer_data, layer_name, script_stem, figure_dir)

    print("Investigation complete.")


if __name__ == "__main__":
    main()
