#!/usr/bin/env python3
"""Train a machine-learning model that predicts NDVI fit parameters from bioclim features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_bioclim_combined.npz"
MODEL_PATH = INTERMEDIATE_DIR / "bioclim_to_ndvi_random_forest.joblib"
METRICS_PATH = INTERMEDIATE_DIR / "bioclim_to_ndvi_random_forest_metrics.json"

TARGET_FEATURES: Sequence[str] = (
    "xmid_spring",
    "scale_spring",
    "xmid_autumn",
    "scale_autumn",
    "bias",
    "scale",
)
QUALITY_FEATURE = "r_squared"
R2_THRESHOLD = 0.6


LOG1P_THEN_STANDARDIZE = {"scale_spring", "scale_autumn", "scale"}
STANDARDIZE_ONLY = {"xmid_spring", "xmid_autumn", "bias"}


class _YTransform:
    """Column-wise target transformer with inverse-transform support."""

    def __init__(self, per_col_ops: list[list[tuple[str, object | None]]]):
        self.per_col_ops = per_col_ops

    def transform(self, y: np.ndarray) -> np.ndarray:
        transformed = np.empty_like(y, dtype=float)
        for idx, ops in enumerate(self.per_col_ops):
            col = y[:, idx]
            for name, obj in ops:
                if name == "log1p":
                    col = np.log1p(np.clip(col, a_min=0.0, a_max=None))
                elif name == "standard" and obj is not None:
                    col = obj.transform(col.reshape(-1, 1)).ravel()
            transformed[:, idx] = col
        return transformed

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        restored = np.empty_like(y, dtype=float)
        for idx, ops in enumerate(self.per_col_ops):
            col = y[:, idx]
            for name, obj in reversed(ops):
                if name == "standard" and obj is not None:
                    col = obj.inverse_transform(col.reshape(-1, 1)).ravel()
                elif name == "log1p":
                    col = np.expm1(col)
            restored[:, idx] = col
        return restored


def _fit_y_transform(y: np.ndarray, target_names: Sequence[str]) -> _YTransform:
    per_col_ops: list[list[tuple[str, object | None]]] = []
    for idx, name in enumerate(target_names):
        ops: list[tuple[str, object | None]] = []
        column = y[:, idx]
        if name in LOG1P_THEN_STANDARDIZE:
            ops.append(("log1p", None))
            log_column = np.log1p(np.clip(column, a_min=0.0, a_max=None)).reshape(-1, 1)
            scaler = StandardScaler().fit(log_column)
            ops.append(("standard", scaler))
        else:
            scaler = StandardScaler().fit(column.reshape(-1, 1))
            ops.append(("standard", scaler))
        per_col_ops.append(ops)
    return _YTransform(per_col_ops)


def _load_combined_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            "Combined dataset missing. "
            f"Expected to find {COMBINED_PATH}. Run 4.02-merge-bioclim-with-ndvi-fit-params.py first."
        )
    with np.load(COMBINED_PATH, allow_pickle=True) as data:
        try:
            bioclim_stack = data["bioclim"]
            ndvi_cube = data["ndvi_fit_params"]
            bioclim_names = data["bioclim_names"].tolist()
            ndvi_feature_names = data["ndvi_feature_names"].tolist()
        except KeyError as error:
            raise KeyError(
                "Combined dataset is missing required arrays. Expected keys: "
                "'bioclim', 'ndvi_fit_params', 'bioclim_names', 'ndvi_feature_names'."
            ) from error
    print(
        "Loaded combined dataset: "
        f"bioclim stack {bioclim_stack.shape}, ndvi cube {ndvi_cube.shape}."
    )
    return bioclim_stack, ndvi_cube, bioclim_names, ndvi_feature_names


def _prepare_training_data(
    bioclim_stack: np.ndarray,
    ndvi_cube: np.ndarray,
    bioclim_names: Sequence[str],
    ndvi_feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_index = {name: idx for idx, name in enumerate(ndvi_feature_names)}
    missing_targets = [name for name in TARGET_FEATURES if name not in feature_index]
    if missing_targets:
        raise KeyError(
            "Combined dataset does not contain all required NDVI features: "
            + ", ".join(missing_targets)
        )

    target_indices = [feature_index[name] for name in TARGET_FEATURES]
    quality_index = feature_index.get(QUALITY_FEATURE)

    rows, cols = ndvi_cube.shape[:2]
    samples = rows * cols
    X = bioclim_stack.reshape(len(bioclim_names), samples).T
    y = ndvi_cube[:, :, target_indices].reshape(samples, len(target_indices))

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    if quality_index is not None:
        quality = ndvi_cube[:, :, quality_index].reshape(samples)
        mask &= ~np.isnan(quality) & (quality >= R2_THRESHOLD)

    X_valid = X[mask]
    y_valid = y[mask]
    print(
        f"Prepared training data with {X_valid.shape[0]:,} samples and {X_valid.shape[1]} features."
    )
    return X_valid, y_valid


def main() -> None:
    bioclim_stack, ndvi_cube, bioclim_names, ndvi_feature_names = _load_combined_dataset()
    X, y = _prepare_training_data(bioclim_stack, ndvi_cube, bioclim_names, ndvi_feature_names)

    if X.shape[0] < 1000:
        print(
            "Warning: fewer than 1000 samples available after filtering. Results may be noisy."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_transform = _fit_y_transform(y_train, TARGET_FEATURES)
    y_train_transformed = y_transform.transform(y_train)

    print(
        f"Training RandomForestRegressor on {X_train.shape[0]:,} samples; "
        f"evaluating on {X_test.shape[0]:,} samples."
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_transformed)
    predictions_transformed = model.predict(X_test)
    predictions = y_transform.inverse_transform(predictions_transformed)

    per_target_r2 = r2_score(y_test, predictions, multioutput="raw_values")
    per_target_mae = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    overall_r2 = r2_score(y_test, predictions, multioutput="variance_weighted")
    overall_mae = mean_absolute_error(y_test, predictions)

    print("Model evaluation:")
    for name, r2_value, mae_value in zip(TARGET_FEATURES, per_target_r2, per_target_mae):
        print(f"  - {name}: R²={r2_value:.3f}, MAE={mae_value:.3f}")
    print(f"Overall variance-weighted R²: {overall_r2:.3f}")
    print(f"Overall mean absolute error: {overall_mae:.3f}")

    feature_importances = model.feature_importances_
    sorted_importances = sorted(
        zip(bioclim_names, feature_importances), key=lambda item: item[1], reverse=True
    )
    print("Top 10 feature importances:")
    for name, importance in sorted_importances[:10]:
        print(f"  - {name}: {importance:.4f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "y_transform": y_transform}, MODEL_PATH)
    metrics = {
        "target_features": list(TARGET_FEATURES),
        "bioclim_features": list(bioclim_names),
        "target_transform": {
            "log1p_then_standardize": sorted(LOG1P_THEN_STANDARDIZE),
            "standardize_only": sorted(STANDARDIZE_ONLY),
        },
        "overall_r2": float(overall_r2),
        "overall_mae": float(overall_mae),
        "per_target_r2": {name: float(value) for name, value in zip(TARGET_FEATURES, per_target_r2)},
        "per_target_mae": {name: float(value) for name, value in zip(TARGET_FEATURES, per_target_mae)},
        "feature_importances": {name: float(value) for name, value in zip(bioclim_names, feature_importances)},
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "r2_threshold": R2_THRESHOLD,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Saved trained model to {MODEL_PATH} and metrics to {METRICS_PATH}.")
    print("Training routine complete.")


if __name__ == "__main__":
    main()
