#!/usr/bin/env python3
"""Investigate correlations between harmonic semiannual trend outputs and WorldClim bioclim variables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioclim_correlation_utils import (
    FeatureLayerSpec,
    compute_correlation_table,
    load_npz_arrays,
    load_bioclim_layers,
    load_feature_layers,
    _normalise_string_list,
    print_top_correlations,
    save_correlation_table,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_harmonic_semiannual_trend_bioclim_combined.npz"
OUTPUT_TABLE_PATH = INTERMEDIATE_DIR / "bioclim_harmonic_correlations.csv"

QUALITY_LAYER_KEYS: list[tuple[str, str]] = [
    ("harmonic_r_squared", "r_squared"),
    ("harmonic_adjusted_r_squared", "adjusted_r_squared"),
    ("harmonic_aic", "aic"),
    ("harmonic_amplitude_annual", "amplitude_annual"),
    ("harmonic_phase_annual_days", "phase_annual_days"),
    ("harmonic_amplitude_semiannual", "amplitude_semiannual"),
    ("harmonic_phase_semiannual_days", "phase_semiannual_days"),
    ("harmonic_num_observations", "num_observations"),
]


def _infer_phase_period(feature_name: str) -> float | None:
    name = feature_name.lower()
    if "phase" not in name:
        return None
    if "semi" in name:
        return 182.5
    if "annual" in name:
        return 365.0
    return 365.0


def _load_combined_dataset() -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    arrays = load_npz_arrays(
        COMBINED_PATH,
        required_keys=[
            "bioclim",
            "bioclim_names",
            "harmonic_parameters",
            "harmonic_parameter_names",
        ],
        optional_keys=["harmonic_layer_names", *[key for key, _ in QUALITY_LAYER_KEYS]],
        missing_file_hint="Run 0.13-merge-bioclim-with-harmonic-semiannual-trend.py first.",
    )

    bioclim_stack, bioclim_names = load_bioclim_layers(arrays)
    feature_layers = load_feature_layers(
        arrays,
        FeatureLayerSpec(
            array_key="harmonic_parameters",
            names_key="harmonic_parameter_names",
        ),
    )

    layer_names_raw = arrays.get("harmonic_layer_names")
    layer_names = (
        _normalise_string_list(layer_names_raw)
        if layer_names_raw is not None
        else None
    )
    for idx, (dataset_key, fallback_name) in enumerate(QUALITY_LAYER_KEYS):
        if dataset_key not in arrays:
            print(f"Warning: '{dataset_key}' missing from combined dataset.")
            continue
        layer_array = np.asarray(arrays[dataset_key], dtype=np.float32)
        name = (
            layer_names[idx]
            if layer_names is not None and idx < len(layer_names)
            else fallback_name
        )
        feature_layers[name] = layer_array.ravel()

    harmonic_shape = (
        arrays["harmonic_parameters"].shape
        if "harmonic_parameters" in arrays
        else "unknown"
    )
    print(
        "Loaded harmonic combined dataset: "
        f"bioclim stack {bioclim_stack.shape}, harmonic parameter cube {harmonic_shape}."
    )
    print(f"Prepared {len(feature_layers)} feature layers for correlation analysis.")
    return bioclim_stack, bioclim_names, feature_layers


def main() -> None:
    bioclim_stack, bioclim_names, feature_layers = _load_combined_dataset()

    circular_periods = {
        name: period
        for name in feature_layers
        if (period := _infer_phase_period(name)) is not None
    }

    df = compute_correlation_table(
        bioclim_stack,
        bioclim_names,
        feature_layers,
        circular_feature_periods=circular_periods,
    )

    save_correlation_table(df, OUTPUT_TABLE_PATH)
    print_top_correlations(df)

    print("Harmonic correlation analysis complete.")


if __name__ == "__main__":
    main()
