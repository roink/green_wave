"""Utilities for loading bioclim bundles and preparing regression targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = [
    "BioclimTargetBundle",
    "PhaseTargetMetadata",
    "build_target_transform",
    "load_bioclim_target_bundle",
    "prepare_regression_samples",
]


@dataclass(frozen=True)
class BioclimTargetBundle:
    """Container for matched bioclim predictors and target feature cubes."""

    bioclim_stack: np.ndarray
    bioclim_names: list[str]
    target_cube: np.ndarray
    target_names: list[str]
    extras: dict[str, np.ndarray]


@dataclass(frozen=True)
class PhaseTargetMetadata:
    """Metadata required to decode circular phase outputs represented as sine/cosine."""

    sin_name: str
    cos_name: str
    period: float

    def as_dict(self) -> dict[str, float | str]:
        """Serialise metadata to a JSON-friendly mapping."""

        return {"sin_name": self.sin_name, "cos_name": self.cos_name, "period": float(self.period)}


class PerColumnTargetTransform:
    """Column-wise transformer supporting reversible log and standard scaling."""

    def __init__(self, per_column_ops: list[list[tuple[str, object | None]]]):
        self.per_column_ops = per_column_ops

    def transform(self, y: np.ndarray) -> np.ndarray:
        transformed = np.asarray(y, dtype=float).copy()
        for idx, ops in enumerate(self.per_column_ops):
            column = transformed[:, idx]
            for name, obj in ops:
                if name == "log1p":
                    column = np.log1p(np.clip(column, a_min=0.0, a_max=None))
                elif name == "standard" and obj is not None:
                    column = obj.transform(column.reshape(-1, 1)).ravel()
            transformed[:, idx] = column
        return transformed

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        restored = np.asarray(y, dtype=float).copy()
        for idx, ops in enumerate(self.per_column_ops):
            column = restored[:, idx]
            for name, obj in reversed(ops):
                if name == "standard" and obj is not None:
                    column = obj.inverse_transform(column.reshape(-1, 1)).ravel()
                elif name == "log1p":
                    column = np.expm1(column)
            restored[:, idx] = column
        return restored


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


def load_bioclim_target_bundle(
    path: Path | str,
    *,
    target_array_key: str,
    target_names_key: str,
    extra_keys: Iterable[str] | None = None,
    missing_file_hint: str | None = None,
) -> BioclimTargetBundle:
    """Load a combined bioclim/target bundle from ``path`` and report its contents."""

    path = Path(path)
    if not path.exists():
        hint = f" {missing_file_hint}" if missing_file_hint else ""
        raise FileNotFoundError(
            f"Combined dataset missing. Expected to find {path}.{hint}"
        )

    with np.load(path, allow_pickle=True) as data:
        required = {"bioclim", "bioclim_names", target_array_key, target_names_key}
        available = set(data.files)
        missing = sorted(required - available)
        if missing:
            hint = f" {missing_file_hint}" if missing_file_hint else ""
            raise KeyError(
                "Dataset {path} is missing required arrays: {missing}.{hint}".format(
                    path=path, missing=", ".join(missing)
                )
            )

        bioclim_stack = np.asarray(data["bioclim"], dtype=np.float32)
        bioclim_names = _normalise_string_list(
            data["bioclim_names"], expected_length=bioclim_stack.shape[0]
        )

        target_cube = np.asarray(data[target_array_key], dtype=np.float32)
        target_names = _normalise_string_list(data[target_names_key])

        extras: dict[str, np.ndarray] = {}
        for key in extra_keys or []:
            if key not in data:
                print(f"Warning: optional array '{key}' missing from {path}.")
                continue
            extras[key] = np.asarray(data[key])

    print(
        "Loaded combined dataset {path}: bioclim stack {bshape}, target cube {tshape}.".format(
            path=path,
            bshape=bioclim_stack.shape,
            tshape=target_cube.shape,
        )
    )
    if extras:
        print(
            "  Extracted optional arrays: {keys}.".format(keys=", ".join(sorted(extras)))
        )
    return BioclimTargetBundle(bioclim_stack, bioclim_names, target_cube, target_names, extras)


def prepare_regression_samples(
    bioclim_stack: np.ndarray,
    bioclim_names: Sequence[str],
    target_cube: np.ndarray,
    target_names: Sequence[str],
    selected_targets: Sequence[str],
    *,
    quality_layers: Sequence[tuple[np.ndarray, float | None]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten grids into ML matrices and apply quality-based sample filtering."""

    feature_index = {name: idx for idx, name in enumerate(target_names)}
    missing_targets = [name for name in selected_targets if name not in feature_index]
    if missing_targets:
        raise KeyError(
            "Target cube does not contain all requested features: "
            + ", ".join(missing_targets)
        )

    target_indices = [feature_index[name] for name in selected_targets]

    rows, cols = target_cube.shape[:2]
    samples = rows * cols

    bioclim_features = len(bioclim_names)
    X = bioclim_stack.reshape(bioclim_features, samples).T
    y = target_cube[:, :, target_indices].reshape(samples, len(target_indices))

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)

    if quality_layers:
        for layer, threshold in quality_layers:
            flat = np.asarray(layer).reshape(samples)
            layer_mask = np.isfinite(flat)
            if threshold is not None:
                layer_mask &= flat >= threshold
            mask &= layer_mask

    X_valid = X[mask]
    y_valid = y[mask]
    print(
        "Prepared training matrices with {n:,} samples and {f} predictors.".format(
            n=X_valid.shape[0], f=X_valid.shape[1]
        )
    )
    return X_valid, y_valid, mask


def build_target_transform(
    y: np.ndarray,
    target_names: Sequence[str],
    *,
    log1p_then_standardize: Iterable[str] | None = None,
    standardize_only: Iterable[str] | None = None,
    identity: Iterable[str] | None = None,
) -> PerColumnTargetTransform:
    """Construct a reversible transform for each target column."""

    log1p_set = set(log1p_then_standardize or [])
    identity_set = set(identity or [])
    if standardize_only is None:
        standard_set = {
            name for name in target_names if name not in log1p_set and name not in identity_set
        }
    else:
        standard_set = set(standardize_only)

    per_column_ops: list[list[tuple[str, object | None]]] = []
    for idx, name in enumerate(target_names):
        column_ops: list[tuple[str, object | None]] = []
        if name in identity_set:
            per_column_ops.append(column_ops)
            continue

        column = y[:, idx]
        if name in log1p_set:
            column_ops.append(("log1p", None))
            column = np.log1p(np.clip(column, a_min=0.0, a_max=None))

        if name in standard_set:
            scaler = StandardScaler().fit(column.reshape(-1, 1))
            column_ops.append(("standard", scaler))
        per_column_ops.append(column_ops)
    return PerColumnTargetTransform(per_column_ops)
