#!/usr/bin/env python3
"""Fit harmonic seasonal models to the NDVI time series stack."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from multiprocessing import Pool, cpu_count
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
BLOCK_ROWS = 16
# Parallelism knobs (can be overridden via environment variables)
N_WORKERS = int(os.environ.get("GW_WORKERS", str(cpu_count() or 1)))
NICENESS_DELTA = int(os.environ.get("GW_NICE_DELTA", "10"))
COL_CHUNK = int(os.environ.get("GW_COL_CHUNK", "64"))

MIN_OBSERVATIONS = 12

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
    amplitude_annual_ds: h5py.Dataset | None = None
    phase_annual_ds: h5py.Dataset | None = None
    amplitude_semiannual_ds: h5py.Dataset | None = None
    phase_semiannual_ds: h5py.Dataset | None = None


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


def _evaluate_columns(
    elapsed_days: np.ndarray, time_center: float
) -> dict[str, np.ndarray]:
    """Pre-compute harmonic basis columns for all timestamps.

    ``time_center`` is subtracted from the elapsed days when generating the trend
    column so that the intercept and trend coefficients are less collinear.
    """

    harmonics = {
        "intercept": np.ones_like(elapsed_days, dtype=np.float64),
        "cos1": np.cos(OMEGA * elapsed_days),
        "sin1": np.sin(OMEGA * elapsed_days),
        "cos2": np.cos(2 * OMEGA * elapsed_days),
        "sin2": np.sin(2 * OMEGA * elapsed_days),
        "trend": elapsed_days - time_center,
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
    if n_obs < MIN_OBSERVATIONS or n_obs <= num_params or n_obs == 0:
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
    metadata: np.ndarray,
    elapsed_days: np.ndarray,
    time_origin: pd.Timestamp,
    time_center: float,
) -> None:
    """Initialise the datasets inside each output file."""

    _, n_rows, n_cols = stack_shape
    chunk_rows = min(32, n_rows)
    for context in contexts:
        ds_params = context.h5file.create_dataset(
            "parameters",
            shape=(n_rows, n_cols, context.spec.num_params),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols), context.spec.num_params),
            compression="lzf",
            fillvalue=np.nan,
        )
        ds_params.attrs["parameter_names"] = np.array(
            context.spec.parameter_names, dtype="S"
        )
        ds_r2 = context.h5file.create_dataset(
            "r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        ds_adj_r2 = context.h5file.create_dataset(
            "adjusted_r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        ds_aic = context.h5file.create_dataset(
            "aic",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        ds_nobs = context.h5file.create_dataset(
            "num_observations",
            shape=(n_rows, n_cols),
            dtype="i2",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=0,
        )

        amp1_ds = context.h5file.create_dataset(
            "amplitude_annual",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        phase1_ds = context.h5file.create_dataset(
            "phase_annual_days",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        amp2_ds = context.h5file.create_dataset(
            "amplitude_semiannual",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        phase2_ds = context.h5file.create_dataset(
            "phase_semiannual_days",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(chunk_rows, min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )

        context.h5file.attrs.update(
            {
                "model_name": context.spec.name,
                "include_semiannual": int(context.spec.include_semiannual),
                "include_trend": int(context.spec.include_trend),
                "period_days": PERIOD_DAYS,
                "omega": OMEGA,
                "time_origin": time_origin.isoformat(),
                "time_units": "days since time_origin",
                "time_center_days": float(time_center),
            }
        )

        context.h5file.create_dataset(
            "metadata", data=metadata, compression="lzf"
        )
        context.h5file.create_dataset(
            "time_offsets_days", data=elapsed_days.astype("f4"), compression="lzf"
        )

        context.parameters_ds = ds_params
        context.r_squared_ds = ds_r2
        context.adj_r_squared_ds = ds_adj_r2
        context.aic_ds = ds_aic
        context.n_obs_ds = ds_nobs
        context.amplitude_annual_ds = amp1_ds
        context.phase_annual_ds = phase1_ds
        context.amplitude_semiannual_ds = amp2_ds
        context.phase_semiannual_ds = phase2_ds


def _latlon_to_indices(lat: float, lon: float, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Convert latitude/longitude to array indices for a regular lat/lon grid."""

    lat_step = 180.0 / n_rows
    lon_step = 360.0 / n_cols
    row = int(np.clip((90.0 - lat) / lat_step, 0, n_rows - 1))
    col = int(np.clip((lon + 180.0) / lon_step, 0, n_cols - 1))
    return row, col


def _evaluate_model(
    params: Sequence[float],
    t_days: np.ndarray,
    include_semiannual: bool,
    include_trend: bool,
    *,
    time_center: float,
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
        estimate += params[idx] * (t_days - time_center)

    return estimate


def _derived_from_params(
    params: np.ndarray, spec: FitSpec
) -> tuple[float, float, float, float]:
    """Return annual/semiannual amplitudes and phases (in days)."""

    def _index(name: str) -> int | None:
        try:
            return spec.parameter_names.index(name)
        except ValueError:
            return None

    amp1 = np.nan
    phase1 = np.nan
    amp2 = np.nan
    phase2 = np.nan

    cos1_idx = _index("beta1_cos1")
    sin1_idx = _index("beta2_sin1")
    if cos1_idx is not None and sin1_idx is not None:
        coeffs = np.asarray([params[cos1_idx], params[sin1_idx]], dtype=float)
        if np.all(np.isfinite(coeffs)):
            amp1 = float(np.hypot(*coeffs))
            phase1 = float((np.arctan2(coeffs[1], coeffs[0]) % (2 * np.pi)) / OMEGA)

    cos2_idx = _index("beta3_cos2")
    sin2_idx = _index("beta4_sin2")
    if cos2_idx is not None and sin2_idx is not None:
        coeffs = np.asarray([params[cos2_idx], params[sin2_idx]], dtype=float)
        if np.all(np.isfinite(coeffs)):
            amp2 = float(np.hypot(*coeffs))
            phase2 = float((np.arctan2(coeffs[1], coeffs[0]) % (2 * np.pi)) / (2 * OMEGA))

    return amp1, phase1, amp2, phase2


# ----------------------------- multiprocessing -----------------------------
# Workers open the NDVI stack independently (recommended by h5py for
# read-mostly workloads) while the parent process performs all HDF5 writes.
# Each worker is launched with a higher niceness so the job yields to
# foreground work on the host.
_WORKER_STATE: dict[str, object] = {}


def _init_worker(
    ndvi_path: str,
    elapsed_days: np.ndarray,
    time_center: float,
    spec_dicts: list[dict],
    niceness_delta: int,
) -> None:
    """Initialise per-process state for the worker pool."""

    try:
        os.nice(max(0, niceness_delta))
    except OSError:
        # Raising niceness may fail under some environments (e.g. Windows).
        pass

    h5file = h5py.File(ndvi_path, "r")
    _WORKER_STATE["file"] = h5file
    _WORKER_STATE["dataset"] = h5file["ndvi_stack"]

    base_columns = _evaluate_columns(elapsed_days.astype(np.float64), float(time_center))
    specs = [FitSpec(**spec_dict) for spec_dict in spec_dicts]
    _WORKER_STATE["columns_by_spec"] = [
        _design_matrix_columns(base_columns, spec) for spec in specs
    ]
    _WORKER_STATE["specs"] = specs
    _WORKER_STATE["time_center"] = float(time_center)


def _process_col_chunk(
    task: tuple[int, int, int, int, dict[int, dict[int, int]]]
) -> tuple[
    int,
    int,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[int, dict]],
]:
    """Fit harmonic models for a rectangular subset of pixels."""

    row_start, row_end, col_start, col_end, example_lookup = task
    dataset = _WORKER_STATE["dataset"]
    columns_by_spec: list[Sequence[np.ndarray]] = _WORKER_STATE["columns_by_spec"]
    specs: list[FitSpec] = _WORKER_STATE["specs"]

    data_block = np.asarray(dataset[:, row_start:row_end, col_start:col_end])
    _, block_rows, block_cols = data_block.shape

    params = [
        np.full((block_rows, block_cols, spec.num_params), np.nan, dtype=np.float32)
        for spec in specs
    ]
    r_squared = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]
    adj_r_squared = [
        np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs
    ]
    aic = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]
    n_obs = [np.zeros((block_rows, block_cols), dtype=np.int16) for _ in specs]
    amp1 = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]
    phase1 = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]
    amp2 = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]
    phase2 = [np.full((block_rows, block_cols), np.nan, dtype=np.float32) for _ in specs]

    examples: list[tuple[int, dict]] = []

    for row_idx in range(block_rows):
        row_series = data_block[:, row_idx, :]
        global_row = row_start + row_idx
        lookup_for_row = example_lookup.get(global_row, {})
        for col_idx in range(block_cols):
            series = row_series[:, col_idx]
            results = [
                _fit_harmonic(series, columns)
                for columns in columns_by_spec
            ]

            for spec_idx, (spec, result) in enumerate(zip(specs, results)):
                params[spec_idx][row_idx, col_idx, :] = result.params
                r_squared[spec_idx][row_idx, col_idx] = result.r_squared
                adj_r_squared[spec_idx][row_idx, col_idx] = result.adj_r_squared
                aic[spec_idx][row_idx, col_idx] = result.aic
                n_obs[spec_idx][row_idx, col_idx] = result.n_obs
                amp1_val, phase1_val, amp2_val, phase2_val = _derived_from_params(
                    result.params, spec
                )
                amp1[spec_idx][row_idx, col_idx] = np.float32(amp1_val)
                phase1[spec_idx][row_idx, col_idx] = np.float32(phase1_val)
                amp2[spec_idx][row_idx, col_idx] = np.float32(amp2_val)
                phase2[spec_idx][row_idx, col_idx] = np.float32(phase2_val)

            global_col = col_start + col_idx
            example_col_lookup = lookup_for_row.get(global_col)
            if example_col_lookup is not None:
                example_payload = {
                    "series": np.array(series, dtype=np.float32),
                    "fits": {
                        spec.name: {
                            "params": result.params,
                            "include_semiannual": spec.include_semiannual,
                            "include_trend": spec.include_trend,
                            "r_squared": result.r_squared,
                            "adj_r_squared": result.adj_r_squared,
                            "aic": result.aic,
                            "n_obs": result.n_obs,
                        }
                        for spec, result in zip(specs, results)
                    },
                }
                examples.append((example_col_lookup, example_payload))

    return (
        col_start,
        col_end,
        params,
        r_squared,
        adj_r_squared,
        aic,
        n_obs,
        amp1,
        phase1,
        amp2,
        phase2,
        examples,
    )


def _collect_example_results(
    dataset: h5py.Dataset,
    base_columns: dict[str, np.ndarray],
    time_center: float,
    example_locations: Sequence[tuple[float, float]],
    n_rows: int,
    n_cols: int,
) -> list[dict[str, object]]:
    """Return fitted results for the requested example pixels."""

    results: list[dict[str, object]] = []
    for lat, lon in example_locations:
        row_idx, col_idx = _latlon_to_indices(lat, lon, n_rows, n_cols)
        series = np.asarray(dataset[:, row_idx, col_idx], dtype=np.float32)

        fits: dict[str, dict[str, object]] = {}
        for spec in FIT_SPECS:
            columns = _design_matrix_columns(base_columns, spec)
            fit_result = _fit_harmonic(series, columns)
            fits[spec.name] = {
                "params": fit_result.params,
                "include_semiannual": spec.include_semiannual,
                "include_trend": spec.include_trend,
                "r_squared": fit_result.r_squared,
                "adj_r_squared": fit_result.adj_r_squared,
                "aic": fit_result.aic,
                "n_obs": fit_result.n_obs,
            }

        results.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "series": series,
                "fits": fits,
            }
        )

    return results


def _plot_examples(
    example_results: Sequence[dict[str, object]],
    all_dates: Sequence[pd.Timestamp],
    elapsed_days: np.ndarray,
    time_center: float,
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
                time_center=time_center,
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
        ndvi_dataset = ndvi_file["ndvi_stack"]
        metadata = ndvi_file["metadata"][:]
        stack_shape = ndvi_dataset.shape

        dates, elapsed_days = _build_time_vectors(metadata)
        time_center = float(np.nanmean(elapsed_days))
        base_columns = _evaluate_columns(elapsed_days, time_center)

        print("Generating example plots before bulk fitting …")
        example_results = _collect_example_results(
            ndvi_dataset,
            base_columns,
            time_center,
            EXAMPLE_LOCATIONS,
            stack_shape[1],
            stack_shape[2],
        )
        _plot_examples(example_results, dates, elapsed_days, time_center)

    with ExitStack() as stack:
        contexts: list[ModelContext] = []
        spec_dicts = [
            {
                "name": spec.name,
                "include_semiannual": spec.include_semiannual,
                "include_trend": spec.include_trend,
                "parameter_names": tuple(spec.parameter_names),
            }
            for spec in FIT_SPECS
        ]
        for spec in FIT_SPECS:
            h5file = stack.enter_context(h5py.File(spec.output_path, "w"))
            contexts.append(
                ModelContext(
                    spec=spec,
                    columns=(),
                    h5file=h5file,
                )
            )

        _create_output_datasets(
            stack_shape,
            contexts,
            metadata=metadata,
            elapsed_days=elapsed_days,
            time_origin=dates[0],
            time_center=time_center,
        )

        n_time, n_rows, n_cols = stack_shape
        print(f"Stack shape: time={n_time}, rows={n_rows}, cols={n_cols}")

        example_lookup: dict[int, dict[int, int]] = {}
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

        for row_start in tqdm(
            range(0, n_rows, BLOCK_ROWS), desc="Fitting harmonics", unit="block"
        ):
            row_end = min(row_start + BLOCK_ROWS, n_rows)
            block_rows = row_end - row_start

            param_buffers = [
                np.full((block_rows, n_cols, ctx.spec.num_params), np.nan, dtype=np.float32)
                for ctx in contexts
            ]
            r2_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            adj_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            aic_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            nobs_buffers = [
                np.zeros((block_rows, n_cols), dtype=np.int16) for _ in contexts
            ]
            amp1_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            phase1_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            amp2_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]
            phase2_buffers = [
                np.full((block_rows, n_cols), np.nan, dtype=np.float32) for _ in contexts
            ]

            with Pool(
                processes=N_WORKERS,
                initializer=_init_worker,
                initargs=(
                    str(DATA_PATH),
                    elapsed_days,
                    time_center,
                    spec_dicts,
                    NICENESS_DELTA,
                ),
            ) as pool:
                tasks = [
                    (row_start, row_end, col_start, min(col_start + COL_CHUNK, n_cols), example_lookup)
                    for col_start in range(0, n_cols, COL_CHUNK)
                ]

                for (
                    col_start,
                    col_end,
                    params,
                    r2_values,
                    adj_values,
                    aic_values,
                    n_obs_values,
                    amp1_values,
                    phase1_values,
                    amp2_values,
                    phase2_values,
                    examples,
                ) in pool.imap_unordered(_process_col_chunk, tasks):
                    col_slice = slice(col_start, col_end)
                    for ctx_idx in range(len(contexts)):
                        param_buffers[ctx_idx][:, col_slice, :] = params[ctx_idx]
                        r2_buffers[ctx_idx][:, col_slice] = r2_values[ctx_idx]
                        adj_buffers[ctx_idx][:, col_slice] = adj_values[ctx_idx]
                        aic_buffers[ctx_idx][:, col_slice] = aic_values[ctx_idx]
                        nobs_buffers[ctx_idx][:, col_slice] = n_obs_values[ctx_idx]
                        amp1_buffers[ctx_idx][:, col_slice] = amp1_values[ctx_idx]
                        phase1_buffers[ctx_idx][:, col_slice] = phase1_values[ctx_idx]
                        amp2_buffers[ctx_idx][:, col_slice] = amp2_values[ctx_idx]
                        phase2_buffers[ctx_idx][:, col_slice] = phase2_values[ctx_idx]

                    for example_idx, payload in examples:
                        if example_results[example_idx]["series"] is None:
                            example_results[example_idx]["series"] = payload["series"]
                            example_results[example_idx]["fits"] = payload["fits"]

            success_masks = [np.isfinite(r2) for r2 in r2_buffers]

            def _count_improvement(
                mask_a: np.ndarray,
                mask_b: np.ndarray,
                metric_a: np.ndarray,
                metric_b: np.ndarray,
                comparator,
            ) -> tuple[int, int]:
                valid = mask_a & mask_b & np.isfinite(metric_a) & np.isfinite(metric_b)
                return int(np.count_nonzero(comparator(metric_a, metric_b) & valid)), int(
                    np.count_nonzero(valid)
                )

            base_idx, trend_idx, semi_idx, semi_trend_idx = 0, 1, 2, 3

            improved, comparisons = _count_improvement(
                success_masks[base_idx],
                success_masks[semi_idx],
                aic_buffers[semi_idx],
                aic_buffers[base_idx],
                np.less,
            )
            semiannual_counts["aic"] += improved
            semiannual_counts["comparisons"] += comparisons
            improved, _ = _count_improvement(
                success_masks[base_idx],
                success_masks[semi_idx],
                adj_buffers[semi_idx],
                adj_buffers[base_idx],
                np.greater,
            )
            semiannual_counts["adj"] += improved

            improved, comparisons = _count_improvement(
                success_masks[base_idx],
                success_masks[trend_idx],
                aic_buffers[trend_idx],
                aic_buffers[base_idx],
                np.less,
            )
            trend_counts["aic"] += improved
            trend_counts["comparisons"] += comparisons
            improved, _ = _count_improvement(
                success_masks[base_idx],
                success_masks[trend_idx],
                adj_buffers[trend_idx],
                adj_buffers[base_idx],
                np.greater,
            )
            trend_counts["adj"] += improved

            improved, comparisons = _count_improvement(
                success_masks[trend_idx],
                success_masks[semi_trend_idx],
                aic_buffers[semi_trend_idx],
                aic_buffers[trend_idx],
                np.less,
            )
            semiannual_trend_counts["aic"] += improved
            semiannual_trend_counts["comparisons"] += comparisons
            improved, _ = _count_improvement(
                success_masks[trend_idx],
                success_masks[semi_trend_idx],
                adj_buffers[semi_trend_idx],
                adj_buffers[trend_idx],
                np.greater,
            )
            semiannual_trend_counts["adj"] += improved

            improved, comparisons = _count_improvement(
                success_masks[semi_idx],
                success_masks[semi_trend_idx],
                aic_buffers[semi_trend_idx],
                aic_buffers[semi_idx],
                np.less,
            )
            trend_semi_counts["aic"] += improved
            trend_semi_counts["comparisons"] += comparisons
            improved, _ = _count_improvement(
                success_masks[semi_idx],
                success_masks[semi_trend_idx],
                adj_buffers[semi_trend_idx],
                adj_buffers[semi_idx],
                np.greater,
            )
            trend_semi_counts["adj"] += improved

            for ctx_idx, ctx in enumerate(contexts):
                assert ctx.parameters_ds is not None
                assert ctx.r_squared_ds is not None
                assert ctx.adj_r_squared_ds is not None
                assert ctx.aic_ds is not None
                assert ctx.n_obs_ds is not None
                assert ctx.amplitude_annual_ds is not None
                assert ctx.phase_annual_ds is not None
                assert ctx.amplitude_semiannual_ds is not None
                assert ctx.phase_semiannual_ds is not None

                ctx.parameters_ds[row_start:row_end, :, :] = param_buffers[ctx_idx]
                ctx.r_squared_ds[row_start:row_end, :] = r2_buffers[ctx_idx]
                ctx.adj_r_squared_ds[row_start:row_end, :] = adj_buffers[ctx_idx]
                ctx.aic_ds[row_start:row_end, :] = aic_buffers[ctx_idx]
                ctx.n_obs_ds[row_start:row_end, :] = nobs_buffers[ctx_idx]
                ctx.amplitude_annual_ds[row_start:row_end, :] = amp1_buffers[ctx_idx]
                ctx.phase_annual_ds[row_start:row_end, :] = phase1_buffers[ctx_idx]
                ctx.amplitude_semiannual_ds[row_start:row_end, :] = amp2_buffers[ctx_idx]
                ctx.phase_semiannual_ds[row_start:row_end, :] = phase2_buffers[ctx_idx]

        def _summarise(label: str, counts: dict[str, int]) -> None:
            comparisons = counts["comparisons"]
            if comparisons == 0:
                print(f"{label}: no comparable pixels")
                return
            aic_pct = 100.0 * counts["aic"] / comparisons
            adj_pct = 100.0 * counts["adj"] / comparisons
            print(
                f"{label}: tested={comparisons:,}, "
                f"AIC improved in {counts['aic']:,} ({aic_pct:.1f}%), "
                f"adj. R^2 improved in {counts['adj']:,} ({adj_pct:.1f}%)"
            )

        _summarise("Semiannual (no trend)", semiannual_counts)
        _summarise("Semiannual (with trend)", semiannual_trend_counts)
        _summarise("Trend (annual model)", trend_counts)
        _summarise("Trend (semiannual model)", trend_semi_counts)

    for context in contexts:
        print(f"Wrote {context.spec.name} results to {context.spec.output_path}")


if __name__ == "__main__":
    main()
