#!/usr/bin/env python3
"""Investigate correlations between harmonic semiannual trend outputs and WorldClim bioclim variables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioclim_correlation_utils import (
    FeatureLayerSpec,
    compute_correlation_table,
    load_bioclim_layers,
    load_feature_layers,
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


def _decode_layer_names(raw: np.ndarray | None) -> list[str] | None:
    if raw is None:
        return None
    decoded = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in raw
    ]
    return decoded


def _load_combined_dataset() -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            "Combined harmonic dataset missing. "
            f"Expected to find {COMBINED_PATH}. Run 0.13-merge-bioclim-with-harmonic-semiannual-trend.py first."
        )

    with np.load(COMBINED_PATH, allow_pickle=True) as data:
        bioclim_stack, bioclim_names = load_bioclim_layers(data)
        feature_layers = load_feature_layers(
            data,
            FeatureLayerSpec(
                array_key="harmonic_parameters",
                names_key="harmonic_parameter_names",
            ),
        )

        layer_names = _decode_layer_names(data.get("harmonic_layer_names"))
        for idx, (dataset_key, fallback_name) in enumerate(QUALITY_LAYER_KEYS):
            if dataset_key not in data:
                print(f"Warning: '{dataset_key}' missing from combined dataset.")
                continue
            layer_array = np.asarray(data[dataset_key], dtype=np.float32)
            name = (
                layer_names[idx]
                if layer_names is not None and idx < len(layer_names)
                else fallback_name
            )
            feature_layers[name] = layer_array.ravel()

        harmonic_shape = (
            data["harmonic_parameters"].shape
            if "harmonic_parameters" in data
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

    df = compute_correlation_table(
        bioclim_stack,
        bioclim_names,
        feature_layers,
    )

    save_correlation_table(df, OUTPUT_TABLE_PATH)
    print_top_correlations(df)

    print("Harmonic correlation analysis complete.")


if __name__ == "__main__":
    main()
