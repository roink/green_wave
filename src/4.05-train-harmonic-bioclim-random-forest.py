#!/usr/bin/env python3
"""Train a random forest ensemble to predict detrended harmonic fit parameters from bioclim variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from joblib import dump, parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from bioclim_correlation_utils import (
    FeatureLayerSpec,
    load_bioclim_layers,
    load_feature_layers,
    load_npz_arrays,
)
from tile_sampling_utils import (
    compute_tile_ids,
    construct_coordinate_grid,
    summarize_tile_statistics,
    tile_bootstrap,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
JOBLIB_TEMP_DIR = PROJECT_ROOT / "tmp"
COMBINED_PATH = INTERMEDIATE_DIR / "detrended_ndvi_bioclim_combined.npz"
MODEL_PATH = INTERMEDIATE_DIR / "detrended_harmonic_bioclim_random_forest.joblib"
METRICS_PATH = INTERMEDIATE_DIR / "detrended_harmonic_bioclim_random_forest_metrics.json"

SELECTED_BIOCLIM_NUMBERS: tuple[int, ...] = (1, *range(4, 20))
R2_THRESHOLD = 0.6
MIN_OBSERVATIONS = 24
TRAIN_FRACTION = 0.8
MAX_TRAINING_SAMPLES = 1_500_000
N_ESTIMATORS = 40
TREE_MAX_SAMPLES: int | float | None = None
TREE_MAX_FEATURES: str | int | float | None = "sqrt"
RANDOM_STATE = 42
TILE_SIZE_DEGREES = 1.00
TREE_TILE_FRACTION = 0.5


@dataclass(frozen=True)
class TrainingData:
    """Container for processed training arrays and metadata."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: list[str]
    feature_indices: list[int]
    target_names: list[str]
    tile_ids: np.ndarray
    tile_metadata: dict[str, float]


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


def _load_training_arrays() -> TrainingData:
    arrays = load_npz_arrays(
        COMBINED_PATH,
        required_keys=[
            "bioclim",
            "bioclim_names",
            "harmonic_parameters",
            "harmonic_parameter_names",
            "latitudes",
            "longitudes",
        ],
        optional_keys=[
            "harmonic_r_squared",
            "harmonic_num_observations",
        ],
        missing_file_hint="Run 0.132-merge-bioclim-with-detrended-harmonic.py first.",
    )

    bioclim_stack, bioclim_names = load_bioclim_layers(arrays)
    bioclim_indices, selected_names = _select_bioclim_indices(
        bioclim_names,
        SELECTED_BIOCLIM_NUMBERS,
    )
    selected_stack = bioclim_stack[bioclim_indices]

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
    print(f"Selected {len(selected_names)} bioclim features for modelling.")

    feature_mask = np.isfinite(selected_stack).all(axis=0)
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

    latitude_grid = construct_coordinate_grid(arrays, "latitudes", (rows, cols))
    longitude_grid = construct_coordinate_grid(arrays, "longitudes", (rows, cols))

    flat_features = selected_stack.reshape(len(selected_names), -1).T
    features = flat_features[valid_indices]
    targets = target_matrix[valid_indices]

    tile_ids, tile_metadata = compute_tile_ids(
        latitude_grid,
        longitude_grid,
        valid_indices,
        tile_size_deg=TILE_SIZE_DEGREES,
    )

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
        tile_ids = tile_ids[selected]
        print(f"Subsampled to {features.shape[0]:,d} samples for manageable training.")

    assert features.shape[0] == targets.shape[0] == tile_ids.shape[0]

    return TrainingData(
        features=features.astype(np.float32),
        targets=targets.astype(np.float32),
        feature_names=selected_names,
        feature_indices=bioclim_indices,
        target_names=target_names,
        tile_ids=tile_ids,
        tile_metadata=tile_metadata,
    )


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_samples=TREE_MAX_SAMPLES,
                    max_features=TREE_MAX_FEATURES,
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

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(
            training_data.features,
            training_data.targets,
            groups=training_data.tile_ids,
        )
    )
    X_train = training_data.features[train_idx]
    X_test = training_data.features[test_idx]
    y_train = training_data.targets[train_idx]
    y_test = training_data.targets[test_idx]
    tile_ids_train = training_data.tile_ids[train_idx]
    tile_ids_test = training_data.tile_ids[test_idx]
    assert tile_ids_test.shape[0] == X_test.shape[0]
    tile_stats = summarize_tile_statistics(tile_ids_train)

    pipeline = _build_pipeline()
    training_samples = X_train.shape[0]
    unique_train_tiles = np.unique(tile_ids_train)
    print(
        "Tile-aware bootstrap will operate on "
        f"{unique_train_tiles.size} spatial tiles (tile size {TILE_SIZE_DEGREES:.2f}°) "
        f"with a sampling fraction of {TREE_TILE_FRACTION:.2f}."
    )
    print(
        "Training tile occupancy: min={min_count}, median={median:.1f}, max={max_count} cells".format(
            min_count=tile_stats["train_tile_cell_count_min"],
            median=tile_stats["train_tile_cell_count_median"],
            max_count=tile_stats["train_tile_cell_count_max"],
        )
    )
    tree_sample_desc = (
        "all samples"
        if TREE_MAX_SAMPLES is None
        else f"{TREE_MAX_SAMPLES:,d} samples"
        if isinstance(TREE_MAX_SAMPLES, int)
        else f"{TREE_MAX_SAMPLES:.2f} fraction"
    )
    print(
        "Training RandomForestRegressor with "
        f"{N_ESTIMATORS} trees on {training_samples:,d} samples "
        f"(per-tree draw: {tree_sample_desc}, max_features={TREE_MAX_FEATURES!r})."
    )
    with tile_bootstrap(tile_ids_train, TREE_TILE_FRACTION):
        with parallel_backend("threading"):
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
    JOBLIB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TEMP_DIR))
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
            "feature_indices": training_data.feature_indices,
            "target_names": training_data.target_names,
            "bioclim_numbers": list(SELECTED_BIOCLIM_NUMBERS),
            "r2_threshold": R2_THRESHOLD,
            "min_observations": MIN_OBSERVATIONS,
            "n_estimators": N_ESTIMATORS,
            "max_samples": TREE_MAX_SAMPLES,
            "max_features": TREE_MAX_FEATURES,
            "oob_score": float(model.oob_score_),
            "tile_size_degrees": TILE_SIZE_DEGREES,
            "tile_sampling_fraction": TREE_TILE_FRACTION,
            "tile_metadata": training_data.tile_metadata,
            "tile_statistics": tile_stats,
            "combined_dataset": COMBINED_PATH.name,
            "joblib_temp_folder": str(JOBLIB_TEMP_DIR),
        },
        MODEL_PATH,
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_estimators": N_ESTIMATORS,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_fraction": TRAIN_FRACTION,
        "tree_max_samples": TREE_MAX_SAMPLES,
        "tree_max_features": TREE_MAX_FEATURES,
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
        "feature_indices": training_data.feature_indices,
        "feature_importances": impurity_importances,
        "permutation_importance": permutation_importances,
        "per_tree_prediction_shape": list(per_tree_predictions.shape),
        "oob_score": float(model.oob_score_),
        "tile_size_degrees": TILE_SIZE_DEGREES,
        "tile_sampling_fraction": TREE_TILE_FRACTION,
        "tile_metadata": training_data.tile_metadata,
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
        "combined_dataset": COMBINED_PATH.name,
        "joblib_temp_folder": str(JOBLIB_TEMP_DIR),
    }
    metrics.update(tile_stats)
    metrics["tile_statistics"] = tile_stats
    metrics["training_tile_count"] = tile_stats["training_tile_count"]
    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}.")
    print("Training routine complete.")


if __name__ == "__main__":
    main()
