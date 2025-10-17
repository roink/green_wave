#!/usr/bin/env python3
"""Combine harmonic semiannual trend fits with resampled bioclim features."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from bioclim_alignment_utils import (
    NdviGridSpec,
    RAW_BIOCLIM_DIR,
    grid_from_h5,
    list_bioclim_files,
    resample_bioclim_layers,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
HARMONIC_PATH = INTERMEDIATE_DIR / "ndvi_harmonic_fit_semiannual_trend.h5"
OUTPUT_PATH = (
    INTERMEDIATE_DIR / "ndvi_harmonic_semiannual_trend_bioclim_combined.npz"
)

FALLBACK_GRID_SPEC = NdviGridSpec(
    row_start=0, row_end=3600, col_start=0, col_end=7200
)

HARMONIC_EXPORTS: list[tuple[str, str]] = [
    ("parameters", "harmonic_parameters"),
    ("r_squared", "harmonic_r_squared"),
    ("adjusted_r_squared", "harmonic_adjusted_r_squared"),
    ("aic", "harmonic_aic"),
    ("amplitude_annual", "harmonic_amplitude_annual"),
    ("amplitude_semiannual", "harmonic_amplitude_semiannual"),
    ("phase_annual_days", "harmonic_phase_annual_days"),
    ("phase_semiannual_days", "harmonic_phase_semiannual_days"),
    ("num_observations", "harmonic_num_observations"),
]


def _decode_parameter_names(raw) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, np.bytes_)):
        return [raw.decode("utf-8")]
    if isinstance(raw, np.ndarray):
        return [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in raw.tolist()
        ]
    if isinstance(raw, Iterable):
        return [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in raw
        ]
    return [str(raw)]


def _summarise_array(label: str, array: np.ndarray) -> None:
    total = array.size
    finite_mask = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite_mask))
    print(
        f"  {label}: shape={array.shape}, dtype={array.dtype}, "
        f"valid={finite_count:,}/{total:,}"
    )
    if finite_count == 0:
        return
    valid_values = array[finite_mask]
    stats = {
        "min": float(np.min(valid_values)),
        "median": float(np.median(valid_values)),
        "max": float(np.max(valid_values)),
    }
    print(
        "    " + ", ".join(f"{key}={value:.3f}" for key, value in stats.items())
    )


def _load_harmonic_subset(
    h5f: h5py.File, grid: NdviGridSpec
) -> tuple[dict[str, np.ndarray], list[str] | None]:
    row_slice, col_slice = grid.slices()
    outputs: dict[str, np.ndarray] = {}
    parameter_names: list[str] | None = None

    for dataset_name, output_key in HARMONIC_EXPORTS:
        if dataset_name not in h5f:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {Path(h5f.filename).name}."
            )
        dataset = h5f[dataset_name]
        if dataset.shape[0] < grid.row_end or dataset.shape[1] < grid.col_end:
            raise ValueError(
                "Dataset '{name}' is smaller than the requested NDVI subset: "
                "{current} vs {expected}.".format(
                    name=dataset_name,
                    current=dataset.shape[:2],
                    expected=grid.shape,
                )
            )
        if dataset.ndim == 2:
            subset = dataset[row_slice, col_slice]
        elif dataset.ndim == 3:
            subset = dataset[row_slice, col_slice, :]
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' has unsupported dimensions {dataset.shape}."
            )
        array = np.asarray(subset)
        if dataset_name == "num_observations":
            array = array.astype(np.uint16, copy=False)
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        outputs[output_key] = array
        _summarise_array(output_key, array)

        if dataset_name == "parameters":
            parameter_names = _decode_parameter_names(
                dataset.attrs.get("parameter_names")
            )

    print(
        f"Loaded harmonic datasets: {', '.join(key for _, key in HARMONIC_EXPORTS)}"
    )
    return outputs, parameter_names


def main() -> None:
    if not HARMONIC_PATH.exists():
        raise FileNotFoundError(
            "Harmonic semiannual trend file missing. "
            f"Expected {HARMONIC_PATH}. Run 0.11-fit-harmonic-models.py first."
        )

    bioclim_files = list_bioclim_files(RAW_BIOCLIM_DIR)
    if not bioclim_files:
        raise FileNotFoundError(
            "No WorldClim bioclim files found. Expected GeoTIFFs in "
            f"{RAW_BIOCLIM_DIR}. Run 3.01-explore-worldclim.py or download the rasters first."
        )
    print(f"Found {len(bioclim_files)} bioclim layers to resample.")

    with h5py.File(HARMONIC_PATH, "r") as h5f:
        grid_spec = grid_from_h5(h5f, FALLBACK_GRID_SPEC)
        harmonic_data, parameter_names = _load_harmonic_subset(h5f, grid_spec)

    bioclim_stack, bioclim_names = resample_bioclim_layers(
        bioclim_files, grid_spec
    )
    latitudes, longitudes = grid_spec.coordinate_vectors()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        **harmonic_data,
        "bioclim": bioclim_stack,
        "bioclim_names": np.array(bioclim_names, dtype=object),
        "latitudes": latitudes,
        "longitudes": longitudes,
    }
    if parameter_names is not None:
        payload["harmonic_parameter_names"] = np.array(parameter_names, dtype=object)
    payload["harmonic_layer_names"] = np.array(
        [
            "r_squared",
            "adjusted_r_squared",
            "aic",
            "amplitude_annual",
            "phase_annual_days",
            "amplitude_semiannual",
            "phase_semiannual_days",
            "num_observations",
        ],
        dtype=object,
    )

    np.savez_compressed(OUTPUT_PATH, **payload)
    print(f"Saved combined harmonic dataset to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
