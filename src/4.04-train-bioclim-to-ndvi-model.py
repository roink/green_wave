#!/usr/bin/env python3
"""Train a machine-learning model that predicts NDVI fit parameters from bioclim features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bioclim_model_utils import (
    build_target_transform,
    load_bioclim_target_bundle,
    prepare_regression_samples,
)

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


def main() -> None:
    bundle = load_bioclim_target_bundle(
        COMBINED_PATH,
        target_array_key="ndvi_fit_params",
        target_names_key="ndvi_feature_names",
        missing_file_hint="Run 4.02-merge-bioclim-with-ndvi-fit-params.py first.",
    )

    bioclim_names = bundle.bioclim_names
    quality_layers = []
    if QUALITY_FEATURE in bundle.target_names:
        quality_index = bundle.target_names.index(QUALITY_FEATURE)
        quality_layers.append((bundle.target_cube[:, :, quality_index], R2_THRESHOLD))
    else:
        print(
            f"Warning: quality feature '{QUALITY_FEATURE}' missing from combined dataset."
        )

    X, y, _ = prepare_regression_samples(
        bundle.bioclim_stack,
        bundle.bioclim_names,
        bundle.target_cube,
        bundle.target_names,
        TARGET_FEATURES,
        quality_layers=quality_layers or None,
    )

    if X.shape[0] < 1000:
        print(
            "Warning: fewer than 1000 samples available after filtering. Results may be noisy."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    feature_scaler = StandardScaler().fit(X_train)
    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)

    y_transform = build_target_transform(
        y_train,
        TARGET_FEATURES,
        log1p_then_standardize=LOG1P_THEN_STANDARDIZE,
        standardize_only=STANDARDIZE_ONLY,
    )
    y_train_transformed = y_transform.transform(y_train)

    print(
        f"Training RandomForestRegressor on {X_train.shape[0]:,} samples; "
        f"evaluating on {X_test.shape[0]:,} samples."
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_samples=0.8,
        max_features=0.5,
    )
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
    dump(
        {
            "model": model,
            "feature_scaler": feature_scaler,
            "y_transform": y_transform,
        },
        MODEL_PATH,
    )
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
