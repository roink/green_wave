#!/usr/bin/env python3
"""Train a random forest ensemble with bioclim and insolation predictors for harmonic fit parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from bioclim_correlation_utils import (
    FeatureLayerSpec,
    load_bioclim_layers,
    load_feature_layers,
    load_npz_arrays,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_harmonic_semiannual_trend_bioclim_combined.npz"
MODEL_PATH = INTERMEDIATE_DIR / "harmonic_bioclim_insolation_random_forest.joblib"
METRICS_PATH = INTERMEDIATE_DIR / "harmonic_bioclim_insolation_random_forest_metrics.json"
INSOLATION_PATH = PROJECT_ROOT / "data" / "raw" / "insolation" / "orbit91"

SELECTED_BIOCLIM_NUMBERS: tuple[int, ...] = (1, *range(4, 20))
INSOLATION_DAYS: tuple[int, ...] = (20, 80, 140, 200, 260, 320)
R2_THRESHOLD = 0.6
MIN_OBSERVATIONS = 24
TRAIN_FRACTION = 0.8
MAX_TRAINING_SAMPLES = 500_000
N_ESTIMATORS = 20
TREE_SAMPLE_FRACTION = 0.8
TREE_FEATURE_FRACTION = 0.8
RANDOM_STATE = 42
So = 1365.0

_PRESENT_DAY_ORBITAL_PARAMS: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TrainingData:
    """Container for processed training arrays and metadata."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: list[str]
    bioclim_feature_indices: list[int]
    target_names: list[str]
    insolation_days: list[int]


def _extract_bioclim_number(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def _select_bioclim_indices(names: Sequence[str], allowed_numbers: Iterable[int]) -> tuple[list[int], list[str]]:
    allowed_set = set(allowed_numbers)
    indices: list[int] = []
    selected_names: list[str] = []
    for idx, name in enumerate(names):
        number = _extract_bioclim_number(name)
        if number is not None and number in allowed_set:
            indices.append(idx)
            selected_names.append(str(name))
    if not indices:
        raise ValueError(
            "No bioclim variables matched the requested numbers."
        )
    return indices, selected_names


def _load_present_day_orbital_parameters() -> tuple[float, float, float]:
    global _PRESENT_DAY_ORBITAL_PARAMS
    if _PRESENT_DAY_ORBITAL_PARAMS is not None:
        return _PRESENT_DAY_ORBITAL_PARAMS

    if not INSOLATION_PATH.exists():
        raise FileNotFoundError(
            "Orbital parameter file not found at {path}. Run download_insolation_data.py first.".format(
                path=INSOLATION_PATH
            )
        )

    data = np.loadtxt(INSOLATION_PATH, skiprows=2, usecols=(0, 1, 2, 3))
    if data.ndim == 1:
        data = data[np.newaxis, :]

    kyears = -data[:, 0]
    idx = int(np.argmin(np.abs(kyears)))
    ecc = float(data[idx, 1])
    long_perh = float(data[idx, 2] + 180.0)
    obliquity = float(data[idx, 3])

    _PRESENT_DAY_ORBITAL_PARAMS = (ecc, obliquity, long_perh)

    print(
        "Loaded present-day orbital parameters from {path}: ecc={ecc:.5f}, obliquity={obl:.3f}°, longitude_perihelion={lon:.3f}°.".format(
            path=INSOLATION_PATH,
            ecc=ecc,
            obl=obliquity,
            lon=long_perh,
        )
    )
    return _PRESENT_DAY_ORBITAL_PARAMS


def _daily_insolation(
    lat_deg: np.ndarray | float,
    day_of_year: float,
    ecc: float,
    obliquity_deg: float,
    longitude_perihelion_deg: float,
) -> np.ndarray:
    epsilon = np.deg2rad(obliquity_deg)
    omega = np.deg2rad(longitude_perihelion_deg)
    phi = np.deg2rad(lat_deg)
    day = float(day_of_year)

    delta_lambda_m = (day - 80.0) * 2 * np.pi / 365.2422
    beta = np.sqrt(1 - ecc**2)
    lambda_m0 = -2.0 * (
        (0.5 * ecc + 0.125 * ecc**3) * (1 + beta) * np.sin(-omega)
        - 0.25 * ecc**2 * (0.5 + beta) * np.sin(-2 * omega)
        + 0.125 * ecc**3 * (1.0 / 3.0 + beta) * np.sin(-3 * omega)
    )
    lambda_m = lambda_m0 + delta_lambda_m
    lam = (
        lambda_m
        + (2 * ecc - 0.25 * ecc**3) * np.sin(lambda_m - omega)
        + 1.25 * ecc**2 * np.sin(2 * (lambda_m - omega))
        + (13.0 / 12.0) * ecc**3 * np.sin(3 * (lambda_m - omega))
    )

    delta = np.arcsin(np.sin(epsilon) * np.sin(lam))
    cos_h0_arg = -np.tan(phi) * np.tan(delta)
    cos_h0_arg = np.clip(cos_h0_arg, -1.0, 1.0)
    h0 = np.arccos(cos_h0_arg)

    mask_polar_day = (np.abs(phi) >= (np.pi / 2 - np.abs(delta))) & (phi * delta > 0)
    mask_polar_night = (np.abs(phi) >= (np.pi / 2 - np.abs(delta))) & (phi * delta <= 0)
    h0 = np.where(mask_polar_day, np.pi, h0)
    h0 = np.where(mask_polar_night, 0.0, h0)

    f_sw = (
        So
        / np.pi
        * (1 + ecc * np.cos(lam - omega)) ** 2
        / (1 - ecc**2) ** 2
        * (h0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(h0))
    )
    return f_sw


def _compute_insolation_layers(latitudes: np.ndarray, days: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    ecc, obliquity, long_perh = _load_present_day_orbital_parameters()
    layers: list[np.ndarray] = []
    names: list[str] = []
    for day in days:
        insolation = _daily_insolation(latitudes, day, ecc, obliquity, long_perh)
        layers.append(np.asarray(insolation, dtype=np.float32))
        names.append(f"insolation_day_{int(day):03d}")
    stack = np.stack(layers, axis=0)
    print(
        "Computed {count} insolation feature layers for days of year: {days}.".format(
            count=len(days),
            days=", ".join(str(day) for day in days),
        )
    )
    return stack, names


def _load_training_arrays() -> TrainingData:
    arrays = load_npz_arrays(
        COMBINED_PATH,
        required_keys=[
            "bioclim",
            "bioclim_names",
            "harmonic_parameters",
            "harmonic_parameter_names",
            "latitudes",
        ],
        optional_keys=[
            "harmonic_r_squared",
            "harmonic_num_observations",
        ],
        missing_file_hint="Run 0.13-merge-bioclim-with-harmonic-semiannual-trend.py first.",
    )

    bioclim_stack, bioclim_names = load_bioclim_layers(arrays)
    bioclim_indices, selected_names = _select_bioclim_indices(
        bioclim_names,
        SELECTED_BIOCLIM_NUMBERS,
    )
    selected_stack = bioclim_stack[bioclim_indices]

    latitudes = np.asarray(arrays["latitudes"], dtype=np.float32)
    if latitudes.shape != selected_stack.shape[1:]:
        raise ValueError(
            "Latitude grid shape {lat_shape} does not match feature grid shape {feat_shape}.".format(
                lat_shape=latitudes.shape,
                feat_shape=selected_stack.shape[1:],
            )
        )

    insolation_stack, insolation_names = _compute_insolation_layers(
        latitudes,
        INSOLATION_DAYS,
    )

    feature_layers = load_feature_layers(
        arrays,
        FeatureLayerSpec(
            array_key="harmonic_parameters",
            names_key="harmonic_parameter_names",
        ),
    )
    target_names = list(feature_layers)
    target_matrix = np.stack(
        [feature_layers[name] for name in target_names],
        axis=1,
    ).astype(np.float32)

    rows, cols = selected_stack.shape[1:]
    feature_names = [*selected_names, *insolation_names]
    combined_feature_stack = np.concatenate(
        [selected_stack, insolation_stack],
        axis=0,
    )

    print(
        "Selected {bioclim_count} bioclim features and {insolation_count} insolation features for modelling.".format(
            bioclim_count=len(selected_names),
            insolation_count=len(insolation_names),
        )
    )

    feature_mask = np.isfinite(combined_feature_stack).all(axis=0)
    target_mask_flat = np.isfinite(target_matrix).all(axis=1)
    target_mask = target_mask_flat.reshape(rows, cols)

    combined_mask = feature_mask & target_mask

    if "harmonic_r_squared" in arrays:
        r_squared = np.asarray(arrays["harmonic_r_squared"], dtype=np.float32)
        combined_mask &= np.isfinite(r_squared) & (r_squared >= R2_THRESHOLD)
    if "harmonic_num_observations" in arrays:
        num_obs = np.asarray(arrays["harmonic_num_observations"], dtype=np.float32)
        combined_mask &= num_obs >= MIN_OBSERVATIONS

    valid_indices = np.flatnonzero(combined_mask.ravel())
    if valid_indices.size == 0:
        raise ValueError("No valid samples remain after applying quality filters.")

    flat_features = combined_feature_stack.reshape(len(feature_names), -1).T
    features = flat_features[valid_indices]
    targets = target_matrix[valid_indices]

    print(
        "Prepared "
        f"{features.shape[0]:,d} samples with {features.shape[1]} features and "
        f"{targets.shape[1]} targets."
    )

    if features.shape[0] > MAX_TRAINING_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = rng.choice(features.shape[0], size=MAX_TRAINING_SAMPLES, replace=False)
        features = features[selected]
        targets = targets[selected]
        print(f"Subsampled to {features.shape[0]:,d} samples for manageable training.")

    return TrainingData(
        features=features.astype(np.float32),
        targets=targets.astype(np.float32),
        feature_names=feature_names,
        bioclim_feature_indices=bioclim_indices,
        target_names=target_names,
        insolation_days=list(INSOLATION_DAYS),
    )


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_samples=TREE_SAMPLE_FRACTION,
                    max_features=TREE_FEATURE_FRACTION,
                    bootstrap=True,
                    oob_score=True,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _predict_per_tree(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    feature_transform = pipeline[:-1]
    X_transformed = feature_transform.transform(X)
    model: RandomForestRegressor = pipeline.named_steps["model"]
    predictions = np.stack([
        estimator.predict(X_transformed) for estimator in model.estimators_
    ])
    return predictions


def main() -> None:
    training_data = _load_training_arrays()
    X_train, X_test, y_train, y_test = train_test_split(
        training_data.features,
        training_data.targets,
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_STATE,
    )

    pipeline = _build_pipeline()
    print(
        "Training RandomForestRegressor with "
        f"{N_ESTIMATORS} trees on {X_train.shape[0]:,d} samples."
    )
    pipeline.fit(X_train, y_train)

    model: RandomForestRegressor = pipeline.named_steps["model"]
    print(f"Random forest OOB R² score: {model.oob_score_:.3f}")

    if hasattr(model, "oob_prediction_") and model.oob_prediction_ is not None:
        oob_predictions = model.oob_prediction_
        oob_per_target_r2 = r2_score(y_train, oob_predictions, multioutput="raw_values")
        oob_per_target_mae = mean_absolute_error(
            y_train, oob_predictions, multioutput="raw_values"
        )
        oob_overall_r2 = r2_score(
            y_train, oob_predictions, multioutput="variance_weighted"
        )
        oob_overall_mae = mean_absolute_error(y_train, oob_predictions)
        print("OOB evaluation:")
        for name, r2_value, mae_value in zip(
            training_data.target_names, oob_per_target_r2, oob_per_target_mae
        ):
            print(f"  - {name}: R²={r2_value:.3f}, MAE={mae_value:.3f}")
        print(f"  Overall variance-weighted R²: {oob_overall_r2:.3f}")
        print(f"  Overall mean absolute error: {oob_overall_mae:.3f}")
    else:
        oob_predictions = None
        oob_per_target_r2 = None
        oob_per_target_mae = None
        oob_overall_r2 = None
        oob_overall_mae = None

    predictions = pipeline.predict(X_test)
    per_tree_predictions = _predict_per_tree(pipeline, X_test)

    per_target_r2 = r2_score(y_test, predictions, multioutput="raw_values")
    per_target_mae = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    overall_r2 = r2_score(y_test, predictions, multioutput="variance_weighted")
    overall_mae = mean_absolute_error(y_test, predictions)

    print("Model evaluation (hold-out set):")
    for name, r2_value, mae_value in zip(
        training_data.target_names, per_target_r2, per_target_mae
    ):
        print(f"  - {name}: R²={r2_value:.3f}, MAE={mae_value:.3f}")
    print(f"Overall variance-weighted R²: {overall_r2:.3f}")
    print(f"Overall mean absolute error: {overall_mae:.3f}")

    print(f"Generated per-tree predictions with shape {per_tree_predictions.shape}.")

    impurity_importances = dict(
        zip(
            training_data.feature_names,
            map(float, model.feature_importances_),
        )
    )
    print("Top impurity-based feature importances:")
    for name, importance in sorted(
        impurity_importances.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"  - {name}: {importance:.4f}")

    print("Computing permutation importances on the hold-out set...")
    perm_result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_importances = {
        name: {
            "mean": float(mean),
            "std": float(std),
        }
        for name, mean, std in zip(
            training_data.feature_names,
            perm_result.importances_mean,
            perm_result.importances_std,
        )
    }
    print("Top permutation-based feature importances:")
    for name, stats in sorted(
        permutation_importances.items(),
        key=lambda item: item[1]["mean"],
        reverse=True,
    )[:10]:
        print(
            f"  - {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "pipeline": pipeline,
            "feature_names": training_data.feature_names,
            "bioclim_feature_indices": training_data.bioclim_feature_indices,
            "insolation_days": training_data.insolation_days,
            "target_names": training_data.target_names,
            "bioclim_numbers": list(SELECTED_BIOCLIM_NUMBERS),
            "r2_threshold": R2_THRESHOLD,
            "min_observations": MIN_OBSERVATIONS,
            "n_estimators": N_ESTIMATORS,
            "max_samples": TREE_SAMPLE_FRACTION,
            "max_features": TREE_FEATURE_FRACTION,
            "oob_score": float(model.oob_score_),
        },
        MODEL_PATH,
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_estimators": N_ESTIMATORS,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_fraction": TRAIN_FRACTION,
        "tree_sample_fraction": TREE_SAMPLE_FRACTION,
        "tree_feature_fraction": TREE_FEATURE_FRACTION,
        "overall_r2": float(overall_r2),
        "overall_mae": float(overall_mae),
        "per_target_r2": {
            name: float(value)
            for name, value in zip(training_data.target_names, per_target_r2)
        },
        "per_target_mae": {
            name: float(value)
            for name, value in zip(training_data.target_names, per_target_mae)
        },
        "feature_names": training_data.feature_names,
        "bioclim_feature_indices": training_data.bioclim_feature_indices,
        "insolation_days": training_data.insolation_days,
        "feature_importances": impurity_importances,
        "permutation_importance": permutation_importances,
        "per_tree_prediction_shape": list(per_tree_predictions.shape),
        "oob_score": float(model.oob_score_),
        "oob_overall_r2": None if oob_overall_r2 is None else float(oob_overall_r2),
        "oob_overall_mae": None if oob_overall_mae is None else float(oob_overall_mae),
        "oob_per_target_r2": None
        if oob_per_target_r2 is None
        else {
            name: float(value)
            for name, value in zip(training_data.target_names, oob_per_target_r2)
        },
        "oob_per_target_mae": None
        if oob_per_target_mae is None
        else {
            name: float(value)
            for name, value in zip(training_data.target_names, oob_per_target_mae)
        },
    }
    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}.")
    print("Training routine complete.")


if __name__ == "__main__":
    main()
