"""Shared helpers for analysing correlations between bioclim layers and NDVI-derived features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "FeatureLayerSpec",
    "compute_correlation_table",
    "load_npz_arrays",
    "load_bioclim_layers",
    "load_feature_layers",
    "print_top_correlations",
    "save_correlation_table",
]


@dataclass(frozen=True)
class FeatureLayerSpec:
    """Describe how to extract feature vectors from an array stored in an ``.npz`` bundle."""

    array_key: str
    """Key of the array within the ``.npz`` bundle."""

    names_key: str | None = None
    """Optional key that stores human-readable names for the feature axis."""

    axis: int | None = None
    """Axis index of the feature dimension. Defaults to the last axis for >=3D arrays."""

    name_prefix: str | None = None
    """Prefix to use when generating feature names in the absence of ``names_key``."""


def _normalise_string_list(raw: Iterable[object], *, expected_length: int | None = None) -> list[str]:
    strings = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in raw
    ]
    if expected_length is not None and len(strings) != expected_length:
        raise ValueError(
            "Name list length mismatch: expected {expected}, got {actual}.".format(
                expected=expected_length, actual=len(strings)
            )
        )
    return strings


def load_npz_arrays(
    path: Path,
    *,
    required_keys: Iterable[str] | None = None,
    optional_keys: Iterable[str] | None = None,
    missing_file_hint: str | None = None,
) -> dict[str, np.ndarray]:
    """Load an ``.npz`` archive and report which arrays were extracted."""

    path = Path(path)
    if not path.exists():
        hint = f" {missing_file_hint}" if missing_file_hint else ""
        raise FileNotFoundError(
            f"Combined dataset missing. Expected to find {path}.{hint}"
        )

    required = set(required_keys or [])
    optional = set(optional_keys or [])

    with np.load(path, allow_pickle=True) as data:
        available = set(data.files)
        missing = sorted(required - available)
        if missing:
            hint = f" {missing_file_hint}" if missing_file_hint else ""
            raise KeyError(
                "Dataset {path} is missing required arrays: {missing}.{hint}".format(
                    path=path, missing=", ".join(missing)
                )
            )
        arrays = {key: np.array(data[key]) for key in data.files}

    loaded_keys = sorted(arrays)
    print(
        "Loaded arrays from {path}: {keys}.".format(
            path=path,
            keys=", ".join(loaded_keys) if loaded_keys else "<none>",
        )
    )
    unexpected_missing = optional - available
    if unexpected_missing:
        print(
            "Optional arrays missing from {path}: {keys}.".format(
                path=path, keys=", ".join(sorted(unexpected_missing))
            )
        )
    return arrays


def load_bioclim_layers(
    npz_data: Mapping[str, np.ndarray],
    *,
    bioclim_key: str = "bioclim",
    bioclim_names_key: str = "bioclim_names",
) -> tuple[np.ndarray, list[str]]:
    """Extract the bioclim stack and its layer names from an ``.npz`` mapping."""

    try:
        bioclim_stack = np.asarray(npz_data[bioclim_key], dtype=np.float32)
        raw_names = npz_data[bioclim_names_key]
    except KeyError as error:
        raise KeyError(
            "Required bioclim arrays missing from dataset. Expected keys "
            f"'{bioclim_key}' and '{bioclim_names_key}'."
        ) from error

    names = _normalise_string_list(raw_names, expected_length=bioclim_stack.shape[0])
    return bioclim_stack, names


def load_feature_layers(
    npz_data: Mapping[str, np.ndarray],
    spec: FeatureLayerSpec,
) -> dict[str, np.ndarray]:
    """Extract named feature layers from the array described by ``spec``."""

    if spec.array_key not in npz_data:
        raise KeyError(f"Feature array '{spec.array_key}' not found in dataset.")

    array = np.asarray(npz_data[spec.array_key], dtype=np.float32)
    if array.ndim <= 2:
        names: list[str]
        if spec.names_key:
            if spec.names_key not in npz_data:
                raise KeyError(
                    f"Names key '{spec.names_key}' referenced by '{spec.array_key}' is missing."
                )
            names = _normalise_string_list(npz_data[spec.names_key])
            if len(names) == 1:
                name = names[0]
            elif len(names) > 1:
                raise ValueError(
                    "Received multiple names for a 2D feature array. "
                    f"Array '{spec.array_key}' has shape {array.shape}."
                )
            else:
                name = spec.name_prefix or spec.array_key
        else:
            name = spec.name_prefix or spec.array_key
        return {name: array.ravel()}

    axis = spec.axis if spec.axis is not None else array.ndim - 1
    axis = axis % array.ndim
    feature_count = array.shape[axis]

    if spec.names_key:
        if spec.names_key not in npz_data:
            raise KeyError(
                f"Names key '{spec.names_key}' referenced by '{spec.array_key}' is missing."
            )
        names = _normalise_string_list(npz_data[spec.names_key], expected_length=feature_count)
    else:
        prefix = spec.name_prefix or spec.array_key
        names = [f"{prefix}_{idx}" for idx in range(feature_count)]

    features: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        values = np.take(array, indices=idx, axis=axis)
        features[name] = np.asarray(values, dtype=np.float32).ravel()
    return features


def _pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    min_overlap: int = 5,
) -> float:
    mask = mask if mask is not None else (np.isfinite(x) & np.isfinite(y))
    if np.count_nonzero(mask) < min_overlap:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    x_std = np.std(x_valid)
    y_std = np.std(y_valid)
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return float("nan")
    covariance = np.mean((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid)))
    return float(covariance / (x_std * y_std))


def _circular_linear_correlation(
    linear_values: np.ndarray,
    circular_radians: np.ndarray,
    *,
    min_overlap: int,
) -> float:
    if linear_values.size < min_overlap:
        return float("nan")

    sin_component = np.sin(circular_radians)
    cos_component = np.cos(circular_radians)

    rxs = _pearson_correlation(linear_values, sin_component, min_overlap=min_overlap)
    rxc = _pearson_correlation(linear_values, cos_component, min_overlap=min_overlap)
    rsc = _pearson_correlation(sin_component, cos_component, min_overlap=min_overlap)

    if np.isnan(rxs) or np.isnan(rxc) or np.isnan(rsc):
        return float("nan")

    denominator = 1.0 - rsc**2
    if np.isclose(denominator, 0.0):
        return float("nan")

    numerator = rxc**2 + rxs**2 - 2.0 * rxc * rxs * rsc
    if numerator < 0:
        numerator = 0.0
    correlation = float(np.sqrt(numerator / denominator))

    # Recover the sign using the orientation of the sine and cosine correlations.
    delta = np.arctan2(rxs - rxc * rsc, rxc - rxs * rsc)
    return float(np.sign(delta) * correlation)


def compute_correlation_table(
    bioclim_stack: np.ndarray,
    bioclim_names: Iterable[str],
    feature_layers: Mapping[str, np.ndarray],
    *,
    feature_column: str = "feature",
    min_overlap: int = 5,
    circular_feature_periods: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return a tidy table of correlations for the supplied feature layers."""

    circular_periods = dict(circular_feature_periods or {})

    results: list[dict[str, object]] = []
    for feature_name, feature_values in feature_layers.items():
        feature_flat = np.asarray(feature_values, dtype=np.float32).ravel()
        for bioclim_idx, bioclim_name in enumerate(bioclim_names):
            bioclim_values = np.asarray(bioclim_stack[bioclim_idx], dtype=np.float32).ravel()
            mask = np.isfinite(feature_flat) & np.isfinite(bioclim_values)
            overlap = int(np.count_nonzero(mask))
            correlation = _pearson_correlation(
                feature_flat, bioclim_values, mask=mask, min_overlap=min_overlap
            )
            base_record = {
                feature_column: feature_name,
                "bioclim_variable": str(bioclim_name),
                "method": "pearson",
                "correlation": correlation,
                "overlap_pixels": overlap,
            }
            results.append(base_record)

            if feature_name in circular_periods:
                period = circular_periods[feature_name]
                if not np.isfinite(period) or period <= 0:
                    continue
                if overlap < min_overlap:
                    circular_corr = float("nan")
                    sin_corr = float("nan")
                    cos_corr = float("nan")
                else:
                    wrapped = np.mod(feature_flat[mask], period)
                    radians = wrapped / period * (2.0 * np.pi)
                    linear_values = bioclim_values[mask]
                    circular_corr = _circular_linear_correlation(
                        linear_values, radians, min_overlap=min_overlap
                    )
                    sin_corr = _pearson_correlation(
                        linear_values,
                        np.sin(radians),
                        min_overlap=min_overlap,
                    )
                    cos_corr = _pearson_correlation(
                        linear_values,
                        np.cos(radians),
                        min_overlap=min_overlap,
                    )

                for method_name, value in (
                    ("circular_linear", circular_corr),
                    ("sine_embedding", sin_corr),
                    ("cosine_embedding", cos_corr),
                ):
                    results.append(
                        {
                            feature_column: feature_name,
                            "bioclim_variable": str(bioclim_name),
                            "method": method_name,
                            "correlation": value,
                            "overlap_pixels": overlap,
                        }
                    )

    df = pd.DataFrame(results)
    df["abs_correlation"] = df["correlation"].abs()
    df.sort_values(
        [feature_column, "method", "abs_correlation"],
        ascending=[True, True, False],
        inplace=True,
    )
    return df


def save_correlation_table(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the correlation table to ``output_path`` in CSV format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved correlation table to {output_path}.")


def print_top_correlations(df: pd.DataFrame, *, top_n: int = 5) -> None:
    """Pretty-print the strongest correlations for each feature layer."""

    feature_column = "feature" if "feature" in df.columns else df.columns[0]
    for feature in df[feature_column].unique():
        print(f"\nTop correlations for {feature}:")
        feature_subset = df[df[feature_column] == feature]
        for method in feature_subset["method"].unique():
            method_subset = feature_subset[feature_subset["method"] == method].head(top_n)
            print(f"  Method: {method}")
            print(
                method_subset.drop(columns=["abs_correlation"]).to_string(
                    index=False
                )
            )
