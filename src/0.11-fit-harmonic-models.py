#!/usr/bin/env python3
"""Fit harmonic seasonal models to the NDVI time series stack."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from logging_setup import initialize_script_logging
from ndvi_analysis_utils import (
    _coordinate_tag,
    _save_figure,
    ensure_script_figure_dir,
)

initialize_script_logging(__file__)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate"
PERIOD_DAYS = 365.2422
OMEGA = 2 * np.pi / PERIOD_DAYS

# Representative lat/lon pairs for plotting
EXAMPLE_LOCATIONS: Sequence[tuple[float, float]] = (
    (52.0, 13.0),  # Temperate Northern Hemisphere
    (0.0, 36.0),  # Equatorial East Africa
    (-33.0, 18.0),  # Southern Hemisphere Mediterranean climate
)


@dataclass(frozen=True)
class FitSpec:
    """Description of a harmonic fit configuration."""

    name: str
    include_semiannual: bool
    include_trend: bool
    parameter_names: Sequence[str]

    @property
    def num_params(self) -> int:
        return len(self.parameter_names)

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / f"ndvi_harmonic_fit_{self.name}.h5"


FIT_SPECS: Sequence[FitSpec] = (
    FitSpec(
        name="annual",
        include_semiannual=False,
        include_trend=False,
        parameter_names=("beta0", "beta1_cos1", "beta2_sin1"),
    ),
    FitSpec(
        name="annual_trend",
        include_semiannual=False,
        include_trend=True,
        parameter_names=("beta0", "beta1_cos1", "beta2_sin1", "beta5_trend"),
    ),
    FitSpec(
        name="semiannual",
        include_semiannual=True,
        include_trend=False,
        parameter_names=(
            "beta0",
            "beta1_cos1",
            "beta2_sin1",
            "beta3_cos2",
            "beta4_sin2",
        ),
    ),
    FitSpec(
        name="semiannual_trend",
        include_semiannual=True,
        include_trend=True,
        parameter_names=(
            "beta0",
            "beta1_cos1",
            "beta2_sin1",
            "beta3_cos2",
            "beta4_sin2",
            "beta5_trend",
        ),
    ),
)


@dataclass
class ModelContext:
    """Bundle of resources required to evaluate and store a model fit."""

    spec: FitSpec
    columns: Sequence[np.ndarray]
    h5file: h5py.File
    parameters_ds: h5py.Dataset | None = None
    r_squared_ds: h5py.Dataset | None = None
    adj_r_squared_ds: h5py.Dataset | None = None
    aic_ds: h5py.Dataset | None = None
    n_obs_ds: h5py.Dataset | None = None


@dataclass
class FitResult:
    """Container for per-pixel fit diagnostics."""

    params: np.ndarray
    r_squared: float
    adj_r_squared: float
    aic: float
    n_obs: int

    @property
    def success(self) -> bool:
        return np.isfinite(self.r_squared)


def _build_time_vectors(metadata: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return observation timestamps and elapsed days for the metadata."""

    if metadata.ndim != 2 or metadata.shape[1] != 2:
        raise ValueError("Metadata must be shaped (N, 2) with year and day of year")

    dates = pd.to_datetime(
        [f"{int(year)}-{int(doy):03d}" for year, doy in metadata],
        format="%Y-%j",
    )
    deltas = dates - dates[0]
    elapsed_days = deltas.total_seconds() / 86400.0
    return dates, elapsed_days.astype(np.float64)


def _evaluate_columns(elapsed_days: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-compute harmonic basis columns for all timestamps."""

    harmonics = {
        "intercept": np.ones_like(elapsed_days, dtype=np.float64),
        "cos1": np.cos(OMEGA * elapsed_days),
        "sin1": np.sin(OMEGA * elapsed_days),
        "cos2": np.cos(2 * OMEGA * elapsed_days),
        "sin2": np.sin(2 * OMEGA * elapsed_days),
        "trend": elapsed_days,
    }
    return harmonics


def _design_matrix_columns(
    base_columns: dict[str, np.ndarray],
    spec: FitSpec,
) -> list[np.ndarray]:
    """Return the ordered column set for ``spec``."""

    columns = [
        base_columns["intercept"],
        base_columns["cos1"],
        base_columns["sin1"],
    ]
    if spec.include_semiannual:
        columns.extend((base_columns["cos2"], base_columns["sin2"]))
    if spec.include_trend:
        columns.append(base_columns["trend"])
    return columns


def _fit_harmonic(
    series: np.ndarray,
    columns: Sequence[np.ndarray],
) -> FitResult:
    """Fit the harmonic regression model to ``series``."""

    mask = np.isfinite(series)
    n_obs = int(np.count_nonzero(mask))
    num_params = len(columns)
    if n_obs <= num_params or n_obs == 0:
        nan_params = np.full(num_params, np.nan, dtype=np.float32)
        return FitResult(nan_params, float("nan"), float("nan"), float("nan"), n_obs)

    design = np.column_stack([col[mask] for col in columns])
    targets = series[mask]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        nan_params = np.full(num_params, np.nan, dtype=np.float32)
        return FitResult(nan_params, float("nan"), float("nan"), float("nan"), n_obs)

    predictions = design @ coeffs
    residuals = targets - predictions
    ss_res = float(np.sum(residuals**2))
    mean_target = float(np.mean(targets))
    ss_tot = float(np.sum((targets - mean_target) ** 2))

    if ss_tot <= 0 or not np.isfinite(ss_tot):
        r_squared = float("nan")
    else:
        r_squared = 1.0 - ss_res / ss_tot
        r_squared = float(np.clip(r_squared, -1.0, 1.0))

    if not np.isfinite(r_squared) or n_obs <= num_params + 1:
        adj_r_squared = float("nan")
    else:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - num_params - 1)
        adj_r_squared = float(np.clip(adj_r_squared, -1.0, 1.0))

    if ss_res <= 0 or not np.isfinite(ss_res):
        aic = float("nan")
    else:
        aic = float(n_obs * np.log(ss_res / n_obs) + 2 * num_params)

    return FitResult(coeffs.astype(np.float32), r_squared, adj_r_squared, aic, n_obs)


def _create_output_datasets(
    stack_shape: Sequence[int],
    contexts: Sequence[ModelContext],
) -> None:
    """Initialise the datasets inside each output file."""

    _, n_rows, n_cols = stack_shape
    chunk_rows = 1
    for context in contexts:
        ds_params = context.h5file.create_dataset(
            "parameters",
            shape=(n_rows, n_cols, context.spec.num_params),
            dtype="f4",
            chunks=(chunk_rows, n_cols, context.spec.num_params),
            fillvalue=np.nan,
        )
        ds_params.attrs["parameter_names"] = np.array(
            context.spec.parameter_names, dtype="S"
        )
        ds_r2 = context.h5file.create_dataset(
            "r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, n_cols),
            fillvalue=np.nan,
        )
        ds_adj_r2 = context.h5file.create_dataset(
            "adjusted_r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, n_cols),
            fillvalue=np.nan,
        )
        ds_aic = context.h5file.create_dataset(
            "aic",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, n_cols),
            fillvalue=np.nan,
        )
        ds_nobs = context.h5file.create_dataset(
            "num_observations",
            shape=(n_rows, n_cols),
            dtype="i2",
            chunks=(chunk_rows, n_cols),
            fillvalue=0,
        )

        context.h5file.attrs.update(
            {
                "model_name": context.spec.name,
                "include_semiannual": int(context.spec.include_semiannual),
                "include_trend": int(context.spec.include_trend),
                "period_days": PERIOD_DAYS,
                "omega": OMEGA,
            }
        )

        context.parameters_ds = ds_params
        context.r_squared_ds = ds_r2
        context.adj_r_squared_ds = ds_adj_r2
        context.aic_ds = ds_aic
        context.n_obs_ds = ds_nobs


def _latlon_to_indices(lat: float, lon: float, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Convert latitude/longitude to array indices, clamped to valid ranges."""

    row = int(np.clip((90.0 - lat) / 0.05, 0, n_rows - 1))
    col = int(np.clip((lon + 180.0) / 0.05, 0, n_cols - 1))
    return row, col


def _evaluate_model(
    params: Sequence[float],
    t_days: np.ndarray,
    include_semiannual: bool,
    include_trend: bool,
) -> np.ndarray:
    """Evaluate a fitted harmonic model for the supplied timestamps."""

    params = np.asarray(params, dtype=float)
    idx = 0
    estimate = np.full_like(t_days, params[idx], dtype=float)
    idx += 1

    estimate += params[idx] * np.cos(OMEGA * t_days)
    idx += 1
    estimate += params[idx] * np.sin(OMEGA * t_days)
    idx += 1

    if include_semiannual:
        estimate += params[idx] * np.cos(2 * OMEGA * t_days)
        idx += 1
        estimate += params[idx] * np.sin(2 * OMEGA * t_days)
        idx += 1

    if include_trend:
        estimate += params[idx] * t_days

    return estimate


def _plot_examples(
    example_results: Sequence[dict[str, object]],
    all_dates: Sequence[pd.Timestamp],
    elapsed_days: np.ndarray,
) -> None:
    """Plot sample time series alongside all fitted models."""

    if not example_results:
        print("No example results available for plotting.")
        return

    script_stem, figure_dir = ensure_script_figure_dir(__file__)
    base_date = all_dates[0]
    dense_days = np.linspace(elapsed_days[0], elapsed_days[-1], 2000)
    dense_dates = base_date + pd.to_timedelta(dense_days, unit="D")

    for result in example_results:
        if not result or result.get("series") is None:
            continue

        lat = float(result["lat"])
        lon = float(result["lon"])
        series = np.asarray(result["series"], dtype=float)
        fits = result["fits"]

        valid_mask = np.isfinite(series)
        if np.count_nonzero(valid_mask) == 0:
            print(f"Skipping ({lat}, {lon}) due to lack of valid observations.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(
            np.asarray(all_dates)[valid_mask],
            series[valid_mask],
            s=10,
            color="black",
            alpha=0.6,
            label="NDVI observations",
        )

        for spec_name, display_name in (
            ("annual", "Annual"),
            ("annual_trend", "Annual + trend"),
            ("semiannual", "Annual + semiannual"),
            ("semiannual_trend", "Annual + semiannual + trend"),
        ):
            fit_info = fits.get(spec_name)
            if not fit_info:
                continue

            params = np.asarray(fit_info["params"], dtype=float)
            include_semiannual = bool(fit_info["include_semiannual"])
            include_trend = bool(fit_info["include_trend"])
            if not np.isfinite(params).all():
                continue

            dense_estimates = _evaluate_model(
                params,
                dense_days,
                include_semiannual=include_semiannual,
                include_trend=include_trend,
            )
            ax.plot(dense_dates, dense_estimates, label=f"{display_name} fit")

        if lon >= 0:
            title = f"NDVI harmonic fits at ({lat:.1f}°N, {lon:.1f}°E)"
        else:
            title = f"NDVI harmonic fits at ({lat:.1f}°N, {abs(lon):.1f}°W)"
        ax.set_title(title)
        ax.set_ylabel("NDVI")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        filename = f"{_coordinate_tag(lat, lon)}_harmonic_fits"
        _save_figure(
            fig,
            filename,
            script_stem=script_stem,
            figure_dir=figure_dir,
        )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NDVI stack not found at {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NDVI stack from {DATA_PATH} …")
    with h5py.File(DATA_PATH, "r") as ndvi_file:
        ndvi_stack = ndvi_file["ndvi_stack"]
        metadata = ndvi_file["metadata"][:]

        dates, elapsed_days = _build_time_vectors(metadata)
        base_columns = _evaluate_columns(elapsed_days)

        with ExitStack() as stack:
            contexts: list[ModelContext] = []
            for spec in FIT_SPECS:
                h5file = stack.enter_context(h5py.File(spec.output_path, "w"))
                context = ModelContext(
                    spec=spec,
                    columns=_design_matrix_columns(base_columns, spec),
                    h5file=h5file,
                )
                contexts.append(context)

            _create_output_datasets(ndvi_stack.shape, contexts)

            n_time, n_rows, n_cols = ndvi_stack.shape
            print(f"Stack shape: time={n_time}, rows={n_rows}, cols={n_cols}")

            example_lookup: dict[int, dict[int, int]] = {}
            example_results: list[dict[str, object]] = [
                {"lat": lat, "lon": lon, "series": None, "fits": {}}
                for lat, lon in EXAMPLE_LOCATIONS
            ]

            for idx, (lat, lon) in enumerate(EXAMPLE_LOCATIONS):
                row, col = _latlon_to_indices(lat, lon, n_rows, n_cols)
                example_lookup.setdefault(row, {})[col] = idx
                print(
                    f"Example point {idx + 1}: lat={lat}, lon={lon} -> row={row}, col={col}"
                )

            semiannual_counts = {"comparisons": 0, "aic": 0, "adj": 0}
            semiannual_trend_counts = {"comparisons": 0, "aic": 0, "adj": 0}
            trend_counts = {"comparisons": 0, "aic": 0, "adj": 0}
            trend_semi_counts = {"comparisons": 0, "aic": 0, "adj": 0}

            for row_idx in tqdm(range(n_rows), desc="Fitting harmonics", unit="row"):
                row_series = ndvi_stack[:, row_idx, :]

                row_param_buffers = [
                    np.full((n_cols, ctx.spec.num_params), np.nan, dtype=np.float32)
                    for ctx in contexts
                ]
                row_r2_buffers = [np.full(n_cols, np.nan, dtype=np.float32) for _ in contexts]
                row_adj_buffers = [
                    np.full(n_cols, np.nan, dtype=np.float32) for _ in contexts
                ]
                row_aic_buffers = [np.full(n_cols, np.nan, dtype=np.float32) for _ in contexts]
                row_nobs_buffers = [np.zeros(n_cols, dtype=np.int16) for _ in contexts]

                for col_idx in range(n_cols):
                    series = row_series[:, col_idx]
                    results = [
                        _fit_harmonic(series, ctx.columns) for ctx in contexts
                    ]

                    for ctx_idx, result in enumerate(results):
                        row_param_buffers[ctx_idx][col_idx, :] = result.params
                        row_r2_buffers[ctx_idx][col_idx] = result.r_squared
                        row_adj_buffers[ctx_idx][col_idx] = result.adj_r_squared
                        row_aic_buffers[ctx_idx][col_idx] = result.aic
                        row_nobs_buffers[ctx_idx][col_idx] = result.n_obs

                    # Comparison metrics for semiannual terms (no trend)
                    base_res, _, semi_res, semi_trend_res = results

                    if base_res.success and semi_res.success:
                        semiannual_counts["comparisons"] += 1
                        if (
                            np.isfinite(semi_res.aic)
                            and np.isfinite(base_res.aic)
                            and semi_res.aic < base_res.aic
                        ):
                            semiannual_counts["aic"] += 1
                        if (
                            np.isfinite(semi_res.adj_r_squared)
                            and np.isfinite(base_res.adj_r_squared)
                            and semi_res.adj_r_squared > base_res.adj_r_squared
                        ):
                            semiannual_counts["adj"] += 1

                    # Semiannual with trend comparison
                    trend_base = results[1]
                    if trend_base.success and semi_trend_res.success:
                        semiannual_trend_counts["comparisons"] += 1
                        if (
                            np.isfinite(semi_trend_res.aic)
                            and np.isfinite(trend_base.aic)
                            and semi_trend_res.aic < trend_base.aic
                        ):
                            semiannual_trend_counts["aic"] += 1
                        if (
                            np.isfinite(semi_trend_res.adj_r_squared)
                            and np.isfinite(trend_base.adj_r_squared)
                            and semi_trend_res.adj_r_squared > trend_base.adj_r_squared
                        ):
                            semiannual_trend_counts["adj"] += 1

                    # Trend term comparisons
                    if base_res.success and trend_base.success:
                        trend_counts["comparisons"] += 1
                        if (
                            np.isfinite(trend_base.aic)
                            and np.isfinite(base_res.aic)
                            and trend_base.aic < base_res.aic
                        ):
                            trend_counts["aic"] += 1
                        if (
                            np.isfinite(trend_base.adj_r_squared)
                            and np.isfinite(base_res.adj_r_squared)
                            and trend_base.adj_r_squared > base_res.adj_r_squared
                        ):
                            trend_counts["adj"] += 1

                    if semi_res.success and semi_trend_res.success:
                        trend_semi_counts["comparisons"] += 1
                        if (
                            np.isfinite(semi_trend_res.aic)
                            and np.isfinite(semi_res.aic)
                            and semi_trend_res.aic < semi_res.aic
                        ):
                            trend_semi_counts["aic"] += 1
                        if (
                            np.isfinite(semi_trend_res.adj_r_squared)
                            and np.isfinite(semi_res.adj_r_squared)
                            and semi_trend_res.adj_r_squared > semi_res.adj_r_squared
                        ):
                            trend_semi_counts["adj"] += 1

                    # Store example series and fits if relevant
                    example_for_row = example_lookup.get(row_idx)
                    if example_for_row and col_idx in example_for_row:
                        example_idx = example_for_row[col_idx]
                        example_entry = example_results[example_idx]
                        example_entry["series"] = np.array(series, dtype=np.float32)
                        fits = example_entry["fits"]
                        for ctx, result in zip(contexts, results):
                            fits[ctx.spec.name] = {
                                "params": result.params,
                                "include_semiannual": ctx.spec.include_semiannual,
                                "include_trend": ctx.spec.include_trend,
                                "r_squared": result.r_squared,
                                "adj_r_squared": result.adj_r_squared,
                                "aic": result.aic,
                                "n_obs": result.n_obs,
                            }

                for ctx_idx, ctx in enumerate(contexts):
                    assert ctx.parameters_ds is not None
                    assert ctx.r_squared_ds is not None
                    assert ctx.adj_r_squared_ds is not None
                    assert ctx.aic_ds is not None
                    assert ctx.n_obs_ds is not None
                    ctx.parameters_ds[row_idx, :, :] = row_param_buffers[ctx_idx]
                    ctx.r_squared_ds[row_idx, :] = row_r2_buffers[ctx_idx]
                    ctx.adj_r_squared_ds[row_idx, :] = row_adj_buffers[ctx_idx]
                    ctx.aic_ds[row_idx, :] = row_aic_buffers[ctx_idx]
                    ctx.n_obs_ds[row_idx, :] = row_nobs_buffers[ctx_idx]

            def _summarise(label: str, counts: dict[str, int]) -> None:
                comps = counts["comparisons"]
                if comps == 0:
                    print(f"{label}: no comparable pixels")
                    return
                aic_pct = 100 * counts["aic"] / comps
                adj_pct = 100 * counts["adj"] / comps
                print(
                    f"{label}: AIC improved for {counts['aic']}/{comps} pixels ({aic_pct:.1f}%), "
                    f"adj. R^2 improved for {counts['adj']}/{comps} pixels ({adj_pct:.1f}%)"
                )

            _summarise("Semiannual (no trend)", semiannual_counts)
            _summarise("Semiannual (with trend)", semiannual_trend_counts)
            _summarise("Trend (annual model)", trend_counts)
            _summarise("Trend (semiannual model)", trend_semi_counts)

            print("Writing example plots …")
            _plot_examples(example_results, dates, elapsed_days)

        for context in contexts:
            print(f"Wrote {context.spec.name} results to {context.spec.output_path}")


if __name__ == "__main__":
    main()
