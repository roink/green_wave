#!/usr/bin/env python3
"""Train a spatially validated random forest that maps bioclim variables to harmonic parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from bioclim_model_utils import (
    PhaseTargetMetadata,
    build_target_transform,
    load_bioclim_target_bundle,
    prepare_regression_samples,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_harmonic_semiannual_trend_bioclim_combined.npz"
RANDOM_FOREST_MODEL_PATH = INTERMEDIATE_DIR / "bioclim_to_harmonic_random_forest.joblib"
METRICS_PATH = INTERMEDIATE_DIR / "bioclim_to_harmonic_model_metrics.json"
CV_CURVES_PATH = INTERMEDIATE_DIR / "bioclim_to_harmonic_random_forest_cv_curves.json"

QUALITY_KEY = "harmonic_r_squared"
AMPLITUDE_FEATURES = ["harmonic_amplitude_annual", "harmonic_amplitude_semiannual"]
PHASE_PERIODS = {
    "harmonic_phase_annual_days": 365.0,
    "harmonic_phase_semiannual_days": 182.5,
}
R2_THRESHOLD = 0.6
N_SPLITS = 5
FOREST_GROWTH_STAGES = [50, 100, 200, 400, 800]
SPATIAL_TILE_DEGREES = 5.0


@dataclass(frozen=True)
class PhaseTargetInfo:
    """Describe how a circular phase is represented in the target matrix."""

    phase_name: str
    sin_name: str
    cos_name: str
    sin_index: int
    cos_index: int
    period: float
    raw_values: np.ndarray


def _json_ready(value):
    """Convert nested structures to JSON-friendly types."""

    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def _compute_phase_metrics(
    period: float,
    sin_pred: np.ndarray,
    cos_pred: np.ndarray,
    true_values_days: np.ndarray,
) -> tuple[float, float]:
    """Return circular R² and MAE in days for a phase prediction."""

    pred_angles = np.arctan2(sin_pred, cos_pred)
    pred_angles = np.mod(pred_angles, 2.0 * np.pi)

    true_angles = np.mod(true_values_days, period) / period * (2.0 * np.pi)
    angle_diff = np.angle(np.exp(1j * (pred_angles - true_angles)))

    mae_days = float(np.mean(np.abs(angle_diff)) * period / (2.0 * np.pi))

    mean_angle = np.angle(np.mean(np.exp(1j * true_angles)))
    centered = np.angle(np.exp(1j * (true_angles - mean_angle)))
    sst = np.sum(centered ** 2)
    sse = np.sum(angle_diff ** 2)
    if np.isclose(sst, 0.0):
        circular_r2 = 1.0 if np.isclose(sse, 0.0) else float("nan")
    else:
        circular_r2 = 1.0 - sse / sst
    return circular_r2, mae_days


def _build_random_forest() -> RandomForestRegressor:
    """Return a configured random forest regressor with warm-start support."""

    return RandomForestRegressor(
        n_estimators=FOREST_GROWTH_STAGES[0],
        max_depth=None,
        min_samples_leaf=2,
        max_samples=0.8,
        max_features=0.5,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=42,
    )


def _compute_spatial_groups(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    base_mask: np.ndarray,
    refinement_mask: np.ndarray | None,
    tile_degrees: float,
) -> np.ndarray:
    """Compute 5°×5° spatial block IDs for each retained pixel."""

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError(
            "Latitude and longitude vectors are expected to be 1D arrays matching the grid axes."
        )

    lat_grid = np.repeat(latitudes[:, None], len(longitudes), axis=1)
    lon_grid = np.repeat(longitudes[None, :], len(latitudes), axis=0)
    flat_lat = lat_grid.reshape(-1)[base_mask]
    flat_lon = lon_grid.reshape(-1)[base_mask]
    if refinement_mask is not None:
        flat_lat = flat_lat[refinement_mask]
        flat_lon = flat_lon[refinement_mask]

    lat_bins = np.floor((flat_lat + 90.0) / tile_degrees).astype(int)
    lon_bins = np.floor((flat_lon + 180.0) / tile_degrees).astype(int)
    group_ids = lat_bins * 1000 + lon_bins
    return group_ids


def _describe_groups(group_ids: np.ndarray) -> None:
    """Print summary statistics for the spatial group assignments."""

    unique_groups, counts = np.unique(group_ids, return_counts=True)
    print(
        "Constructed {n_groups} spatial tiles (size {tile}°); "
        "median {median:.0f} samples per tile (min {min_count}, max {max_count}).".format(
            n_groups=len(unique_groups),
            tile=SPATIAL_TILE_DEGREES,
            median=float(np.median(counts)),
            min_count=int(counts.min()),
            max_count=int(counts.max()),
        )
    )


def _prepare_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    dict[str, PhaseTargetInfo],
    list[str],
    list[str],
    np.ndarray,
]:
    """Load the harmonic bundle and construct ML-ready matrices."""

    print(f"Loading combined harmonic and bioclim dataset from {COMBINED_PATH}.")
    bundle = load_bioclim_target_bundle(
        COMBINED_PATH,
        target_array_key="harmonic_parameters",
        target_names_key="harmonic_parameter_names",
        extra_keys=[
            QUALITY_KEY,
            *AMPLITUDE_FEATURES,
            *PHASE_PERIODS,
            "latitudes",
            "longitudes",
        ],
        missing_file_hint="Run 0.13-merge-bioclim-with-harmonic-semiannual-trend.py first.",
    )

    print(
        "Loaded {bioclim} bioclim features and {targets} harmonic targets.".format(
            bioclim=len(bundle.bioclim_names), targets=len(bundle.target_names)
        )
    )

    quality_layers = []
    if QUALITY_KEY in bundle.extras:
        quality_layers.append((bundle.extras[QUALITY_KEY], R2_THRESHOLD))
        print(
            f"Applying quality filter '{QUALITY_KEY}' with threshold >= {R2_THRESHOLD}."
        )
    else:
        print(
            f"Warning: quality layer '{QUALITY_KEY}' missing from combined dataset; proceeding without an R² filter."
        )

    X, y_params, base_mask = prepare_regression_samples(
        bundle.bioclim_stack,
        bundle.bioclim_names,
        bundle.target_cube,
        bundle.target_names,
        bundle.target_names,
        quality_layers=quality_layers or None,
    )

    print(
        "Prepared {samples:,} preliminary samples with {features} features.".format(
            samples=X.shape[0], features=X.shape[1]
        )
    )

    extras_masked: dict[str, np.ndarray] = {}
    for name in set(AMPLITUDE_FEATURES) | set(PHASE_PERIODS):
        if name not in bundle.extras:
            continue
        extras_masked[name] = np.asarray(bundle.extras[name]).reshape(-1)[base_mask]

    valid_mask = np.ones(X.shape[0], dtype=bool)
    for name in AMPLITUDE_FEATURES:
        if name in extras_masked:
            valid_mask &= np.isfinite(extras_masked[name])
    for name in PHASE_PERIODS:
        if name in extras_masked:
            valid_mask &= np.isfinite(extras_masked[name])

    if not np.all(valid_mask):
        dropped = int(np.count_nonzero(~valid_mask))
        print(f"Dropping {dropped:,} samples due to missing amplitude or phase targets.")
        X = X[valid_mask]
        y_params = y_params[valid_mask]
        for key in list(extras_masked):
            extras_masked[key] = extras_masked[key][valid_mask]

    latitudes_raw = bundle.extras.get("latitudes")
    longitudes_raw = bundle.extras.get("longitudes")
    if latitudes_raw is None or longitudes_raw is None:
        raise KeyError(
            "Latitude/longitude vectors are required for spatial cross-validation but were not found in the bundle."
        )
    group_ids = _compute_spatial_groups(
        np.asarray(latitudes_raw, dtype=np.float32),
        np.asarray(longitudes_raw, dtype=np.float32),
        base_mask,
        valid_mask,
        SPATIAL_TILE_DEGREES,
    )
    _describe_groups(group_ids)

    target_names = list(bundle.target_names)
    direct_target_names = list(target_names)
    log1p_targets: set[str] = set()

    target_parts = [y_params]
    target_names_extended = list(target_names)
    phase_infos: dict[str, PhaseTargetInfo] = {}

    for amplitude_name in AMPLITUDE_FEATURES:
        if amplitude_name not in extras_masked:
            print(
                f"Warning: amplitude layer '{amplitude_name}' not available; skipping this target."
            )
            continue
        values = extras_masked[amplitude_name].astype(np.float32)
        target_parts.append(values.reshape(-1, 1))
        target_names_extended.append(amplitude_name)
        direct_target_names.append(amplitude_name)
        log1p_targets.add(amplitude_name)

    for phase_name, period in PHASE_PERIODS.items():
        if phase_name not in extras_masked:
            print(
                f"Warning: phase layer '{phase_name}' not available; skipping this target."
            )
            continue
        values = extras_masked[phase_name].astype(np.float32)
        angles = np.mod(values, period) / period * (2.0 * np.pi)
        sin_values = np.sin(angles).reshape(-1, 1)
        cos_values = np.cos(angles).reshape(-1, 1)
        sin_name = f"{phase_name}_sin"
        cos_name = f"{phase_name}_cos"
        start_index = len(target_names_extended)
        target_parts.append(np.hstack([sin_values, cos_values]))
        target_names_extended.extend([sin_name, cos_name])
        phase_infos[phase_name] = PhaseTargetInfo(
            phase_name=phase_name,
            sin_name=sin_name,
            cos_name=cos_name,
            sin_index=start_index,
            cos_index=start_index + 1,
            period=period,
            raw_values=values,
        )

    y_full = np.hstack(target_parts)
    log1p_targets = sorted(log1p_targets)

    print(
        "Prepared harmonic regression dataset with {samples:,} samples, {features} features, "
        "and {targets} targets.".format(
            samples=X.shape[0], features=X.shape[1], targets=y_full.shape[1]
        )
    )
    print(f"Target columns: {', '.join(target_names_extended)}")
    if log1p_targets:
        print(
            "Log1p+standardize targets: {targets}.".format(
                targets=", ".join(log1p_targets)
            )
        )

    return (
        X,
        y_full,
        target_names_extended,
        direct_target_names,
        phase_infos,
        log1p_targets,
        bundle.bioclim_names,
        group_ids,
    )


def _aggregate(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _cross_validate_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    target_names: list[str],
    direct_target_names: list[str],
    phase_infos: dict[str, PhaseTargetInfo],
    log1p_targets: list[str],
) -> tuple[dict, list[dict]]:
    """Evaluate the configured random forest with GroupKFold CV."""

    print(
        f"Evaluating random forest with {N_SPLITS}-fold spatial GroupKFold cross-validation."
    )
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups=group_ids))
    stage_records: dict[int, list[dict[str, float]]] = {
        stage: [] for stage in FOREST_GROWTH_STAGES
    }

    direct_indices = [target_names.index(name) for name in direct_target_names]
    direct_metrics = {name: {"r2": [], "mae": []} for name in direct_target_names}
    phase_metrics = {
        name: {"circular_r2": [], "circular_mae": []} for name in phase_infos
    }

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc="GroupKFold CV", unit="fold", leave=False), start=1
    ):
        print(
            f"  Fold {fold}/{N_SPLITS}: train={len(train_idx):,} samples, test={len(test_idx):,} samples."
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        feature_scaler = StandardScaler().fit(X_train)
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        y_transform = build_target_transform(
            y_train,
            target_names,
            log1p_then_standardize=log1p_targets,
        )
        y_train_transformed = y_transform.transform(y_train)

        estimator = _build_random_forest()
        increment_iterator = tqdm(
            FOREST_GROWTH_STAGES,
            desc=f"    Fold {fold} forest growth",
            unit="stage",
            leave=False,
        )
        for stage in increment_iterator:
            increment_iterator.set_postfix_str(f"trees={stage}")
            estimator.set_params(n_estimators=stage)
            estimator.fit(X_train_scaled, y_train_transformed)
            predictions = estimator.predict(X_test_scaled)
            predictions = np.asarray(predictions, dtype=float)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            predictions = y_transform.inverse_transform(predictions)

            overall_r2 = r2_score(
                y_test, predictions, multioutput="variance_weighted"
            )
            overall_mae = mean_absolute_error(y_test, predictions)
            oob_score = getattr(estimator, "oob_score_", float("nan"))
            stage_records[stage].append(
                {
                    "fold": float(fold),
                    "overall_r2": float(overall_r2),
                    "overall_mae": float(overall_mae),
                    "oob_score": float(oob_score) if np.isfinite(oob_score) else float("nan"),
                }
            )
            print(
                "    Fold {fold}: {stage} trees -> CV R²={r2:.3f}, MAE={mae:.3f}, OOB={oob:.3f}.".format(
                    fold=fold,
                    stage=stage,
                    r2=float(overall_r2),
                    mae=float(overall_mae),
                    oob=float(oob_score) if np.isfinite(oob_score) else float("nan"),
                )
            )

            if stage != FOREST_GROWTH_STAGES[-1]:
                continue

            for idx, name in zip(direct_indices, direct_target_names):
                truth = y_test[:, idx]
                pred = predictions[:, idx]
                try:
                    r2_value = r2_score(truth, pred)
                except ValueError:
                    r2_value = float("nan")
                mae_value = mean_absolute_error(truth, pred)
                direct_metrics[name]["r2"].append(float(r2_value))
                direct_metrics[name]["mae"].append(float(mae_value))

            for phase_name, info in phase_infos.items():
                sin_pred = predictions[:, info.sin_index]
                cos_pred = predictions[:, info.cos_index]
                true_values = info.raw_values[test_idx]
                circular_r2, circular_mae = _compute_phase_metrics(
                    info.period, sin_pred, cos_pred, true_values
                )
                phase_metrics[phase_name]["circular_r2"].append(float(circular_r2))
                phase_metrics[phase_name]["circular_mae"].append(float(circular_mae))

        fold_final_scores = [
            record["overall_r2"]
            for record in stage_records[FOREST_GROWTH_STAGES[-1]]
            if record["fold"] == float(fold)
        ]
        if fold_final_scores:
            print(
                "    Fold {fold} summary: final-stage mean R²={score:.3f}.".format(
                    fold=fold,
                    score=float(np.nanmean(fold_final_scores)),
                )
            )

    def _mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    per_target_summary = {
        name: {
            "r2_mean": _mean(metrics["r2"]),
            "mae_mean": _mean(metrics["mae"]),
        }
        for name, metrics in direct_metrics.items()
    }

    per_phase_summary = {
        name: {
            "circular_r2_mean": _mean(metrics["circular_r2"]),
            "circular_mae_mean": _mean(metrics["circular_mae"]),
        }
        for name, metrics in phase_metrics.items()
    }

    non_phase_r2 = [_mean(metrics["r2"]) for metrics in direct_metrics.values()]
    phase_r2 = [_mean(metrics["circular_r2"]) for metrics in phase_metrics.values()]
    all_r2 = [value for value in non_phase_r2 + phase_r2 if not np.isnan(value)]
    overall_r2_mean = float(np.mean(all_r2)) if all_r2 else float("nan")

    non_phase_mae = [
        _mean(metrics["mae"]) for metrics in direct_metrics.values() if metrics["mae"]
    ]
    phase_mae = [
        _mean(metrics["circular_mae"]) for metrics in phase_metrics.values() if metrics["circular_mae"]
    ]

    summary = {
        "model": "random_forest",
        "overall_r2_mean": overall_r2_mean,
        "non_phase_r2_mean": _mean(non_phase_r2),
        "non_phase_mae_mean": _mean(non_phase_mae),
        "phase_circular_r2_mean": _mean(phase_r2),
        "phase_circular_mae_mean": _mean(phase_mae),
        "per_target": per_target_summary,
        "per_phase": per_phase_summary,
    }

    print(
        f"  Cross-validation complete: overall mean R²={overall_r2_mean:.3f} at {FOREST_GROWTH_STAGES[-1]} trees."
    )

    stage_curves: list[dict[str, object]] = []
    for stage, records in stage_records.items():
        mean_r2, std_r2 = _aggregate(record["overall_r2"] for record in records)
        mean_mae, std_mae = _aggregate(record["overall_mae"] for record in records)
        mean_oob, std_oob = _aggregate(record["oob_score"] for record in records)
        stage_curves.append(
            {
                "n_estimators": stage,
                "mean_overall_r2": mean_r2,
                "std_overall_r2": std_r2,
                "mean_overall_mae": mean_mae,
                "std_overall_mae": std_mae,
                "mean_oob_score": mean_oob,
                "std_oob_score": std_oob,
                "records": records,
            }
        )
        print(
            "    Stage {stage}: mean CV R²={mean_r2:.3f}±{std_r2:.3f}, "
            "mean OOB={mean_oob:.3f}±{std_oob:.3f}.".format(
                stage=stage,
                mean_r2=mean_r2,
                std_r2=std_r2,
                mean_oob=mean_oob,
                std_oob=std_oob,
            )
        )

    return summary, stage_curves


def _train_full_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    log1p_targets: list[str],
    direct_target_names: list[str],
    phase_infos: dict[str, PhaseTargetInfo],
    bioclim_names: list[str],
) -> dict:
    """Fit the random forest on the full dataset and package inference metadata."""

    print(
        "Training final random forest model on {samples:,} samples.".format(
            samples=X.shape[0]
        )
    )
    feature_scaler = StandardScaler().fit(X)
    X_scaled = feature_scaler.transform(X)

    y_transform = build_target_transform(
        y,
        target_names,
        log1p_then_standardize=log1p_targets,
    )
    y_transformed = y_transform.transform(y)

    estimator = _build_random_forest()
    training_curve: list[dict[str, float]] = []
    for stage in tqdm(
        FOREST_GROWTH_STAGES,
        desc="Full-data forest growth",
        unit="stage",
        leave=False,
    ):
        estimator.set_params(n_estimators=stage)
        estimator.fit(X_scaled, y_transformed)
        oob_score = getattr(estimator, "oob_score_", float("nan"))
        training_curve.append(
            {
                "n_estimators": stage,
                "oob_score": float(oob_score) if np.isfinite(oob_score) else float("nan"),
            }
        )
        print(
            f"  Full-data fit: {stage} trees complete with OOB R²={float(oob_score):.3f}."
        )

    phase_metadata = {
        name: PhaseTargetMetadata(info.sin_name, info.cos_name, info.period)
        for name, info in phase_infos.items()
    }

    return {
        "model_label": "random_forest",
        "model": estimator,
        "feature_scaler": feature_scaler,
        "target_transform": y_transform,
        "target_names": target_names,
        "direct_target_names": direct_target_names,
        "phase_targets": {
            name: metadata.as_dict() for name, metadata in phase_metadata.items()
        },
        "log1p_targets": list(log1p_targets),
        "bioclim_features": bioclim_names,
        "training_curve": training_curve,
    }


def main() -> None:
    (
        X,
        y,
        target_names,
        direct_target_names,
        phase_infos,
        log1p_targets,
        bioclim_names,
        group_ids,
    ) = _prepare_dataset()

    if X.shape[0] < 1000:
        print(
            "Warning: fewer than 1000 samples available after filtering. Results may be noisy."
        )

    cv_summary, stage_curves = _cross_validate_random_forest(
        X,
        y,
        group_ids,
        target_names,
        direct_target_names,
        phase_infos,
        log1p_targets,
    )

    print("Cross-validation complete; proceeding to full-data training.")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    artifact = _train_full_random_forest(
        X,
        y,
        target_names,
        log1p_targets,
        direct_target_names,
        phase_infos,
        bioclim_names,
    )
    dump(artifact, RANDOM_FOREST_MODEL_PATH)
    print(f"Saved random forest model artefact to {RANDOM_FOREST_MODEL_PATH}.")

    metrics_payload = _json_ready(
        {
            "combined_dataset": str(COMBINED_PATH),
            "n_splits": N_SPLITS,
            "r2_threshold": R2_THRESHOLD,
            "targets": target_names,
            "direct_targets": direct_target_names,
            "phase_targets": list(phase_infos),
            "model": cv_summary,
            "cv_stage_curves": stage_curves,
            "training_curve": artifact["training_curve"],
            "spatial_tile_degrees": SPATIAL_TILE_DEGREES,
            "group_count": int(np.unique(group_ids).size),
        }
    )
    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)
    print(f"Wrote cross-validation metrics to {METRICS_PATH}.")

    cv_curves_payload = _json_ready(stage_curves)
    with CV_CURVES_PATH.open("w", encoding="utf-8") as fp:
        json.dump(cv_curves_payload, fp, indent=2)
    print(f"Saved CV curve data to {CV_CURVES_PATH}.")

    print("Harmonic random forest training routine complete.")


if __name__ == "__main__":
    main()
