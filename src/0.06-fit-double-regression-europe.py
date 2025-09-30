#!/usr/bin/env python3
"""Fit a double logistic model for every pixel in the European subset."""

from __future__ import annotations

import warnings
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import median_filter
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import linregress
from tqdm.auto import tqdm

from process_priority import lower_process_priority

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NDVI_STACK_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_filtered.npz"
OUTPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_fit_params.npz"

ROW_START, ROW_END = 320, 1198
COL_START, COL_END = 3335, 4553
NUM_PARAMS = 6
NUM_METRICS = 2  # R^2 and covariance quality
RESULT_LENGTH = NUM_PARAMS + NUM_METRICS

# Globals populated in ``main`` so that worker processes can access them.
METADATA: np.ndarray | None = None
NDVI_STACK_SHAPE: Tuple[int, ...] | None = None
NDVI_STACK_DTYPE: np.dtype | None = None


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def double_logistic(
    t: np.ndarray,
    xmid_spring: float,
    scale_spring: float,
    xmid_autumn: float,
    scale_autumn: float,
    bias: float,
    scale: float,
) -> np.ndarray:
    """Evaluate the double logistic model at ``t``."""

    spring = 1 / (1 + np.exp((xmid_spring - t) / scale_spring))
    autumn = 1 / (1 + np.exp((xmid_autumn - t) / scale_autumn))
    return bias + scale * (spring - autumn)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination for the valid observations."""

    mask = ~np.isnan(y_true)
    if np.sum(mask) < 5:
        return float("nan")

    slope, intercept, r_value, _, _ = linregress(y_true[mask], y_pred[mask])
    _ = (slope, intercept)  # explicitly ignore unused values
    return r_value**2


def _initial_guesses(
    bias_guess: float, scale_guess: float
) -> Iterable[Tuple[float, ...]]:
    """Yield a collection of initial guesses for curve fitting."""

    yield (120, 20, 270, 25, bias_guess, scale_guess)
    yield (240, 20, 60, 25, bias_guess + scale_guess, scale_guess)
    yield (240, 20, 60, 25, bias_guess + 0.5 * scale_guess, 1)
    yield (120, 20, 270, 25, bias_guess + 0.5 * scale_guess, 1)


def process_pixel(row: int, col: int, shm_name: str) -> np.ndarray:
    """Fit the double logistic model for a single pixel."""

    assert METADATA is not None, "Metadata must be loaded before processing"
    assert NDVI_STACK_SHAPE is not None, "Stack shape must be available"
    assert NDVI_STACK_DTYPE is not None, "Stack dtype must be available"

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    try:
        ndvi_stack = np.ndarray(
            NDVI_STACK_SHAPE, dtype=NDVI_STACK_DTYPE, buffer=existing_shm.buf
        )

        ndvi_timeseries = ndvi_stack[:, ROW_START + row, COL_START + col]
        if np.isnan(ndvi_timeseries).all():
            return np.full(RESULT_LENGTH, np.nan, dtype=np.float32)

        winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)
        corrected_ndvi = np.copy(ndvi_timeseries)
        corrected_ndvi[ndvi_timeseries < winter_ndvi] = winter_ndvi

        for i, (year, doy) in enumerate(METADATA):
            if np.isnan(corrected_ndvi[i]) and (doy >= 300 or doy <= 60):
                corrected_ndvi[i] = winter_ndvi

        filtered_ndvi = median_filter(corrected_ndvi, size=3)

        doy_values: list[int] = []
        ndvi_values: list[float] = []
        for i, (year, doy) in enumerate(METADATA):
            if 2002 <= year <= 2010 and not np.isnan(filtered_ndvi[i]):
                doy_values.append(int(doy))
                ndvi_values.append(float(filtered_ndvi[i]))

        doy_values = np.array(doy_values, dtype=np.float32)
        ndvi_values = np.array(ndvi_values, dtype=np.float32)
        if ndvi_values.size < 100:
            return np.full(RESULT_LENGTH, np.nan, dtype=np.float32)

        bias_guess = float(np.min(ndvi_values))
        scale_guess = float(np.max(ndvi_values) - bias_guess)
        lower_bounds = [0, 1e-5, 0, 1e-5, -np.inf, 1e-5]
        upper_bounds = [365, np.inf, 365, np.inf, np.inf, np.inf]

        best_fit: np.ndarray | None = None
        best_r2 = -np.inf
        best_cov_quality = float("nan")

        for guess in _initial_guesses(bias_guess, scale_guess):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizeWarning)
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    params, covariance = curve_fit(
                        double_logistic,
                        doy_values,
                        ndvi_values,
                        p0=guess,
                        bounds=(lower_bounds, upper_bounds),
                    )
                ndvi_fitted = double_logistic(doy_values, *params)
                r2_score = compute_r2(ndvi_values, ndvi_fitted)
                if not np.isnan(r2_score) and r2_score > best_r2:
                    best_r2 = r2_score
                    best_fit = params
                    std_errors = np.sqrt(np.diag(covariance))
                    best_cov_quality = float(np.mean(std_errors))
            except (RuntimeError, RuntimeWarning):
                continue

        if best_fit is None:
            return np.full(RESULT_LENGTH, np.nan, dtype=np.float32)

        return np.concatenate([best_fit, [best_r2, best_cov_quality]]).astype(
            np.float32
        )
    finally:
        existing_shm.close()


def _create_shared_copy(
    array: np.ndarray,
) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """Return shared memory containing ``array`` and a writable NumPy view."""

    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(shm_array, array)
    return shm, shm_array


def main() -> None:
    global METADATA, NDVI_STACK_DTYPE, NDVI_STACK_SHAPE

    if not NDVI_STACK_PATH.exists():
        raise FileNotFoundError(f"NDVI stack not found at {NDVI_STACK_PATH}")

    print(f"Loading NDVI stack from {NDVI_STACK_PATH} …")
    data = np.load(NDVI_STACK_PATH)
    ndvi_stack = data["ndvi_stack"]
    METADATA = data["metadata"]

    print(f"Loaded NDVI stack shape: {ndvi_stack.shape}")
    print(f"Metadata (first 5 entries): {METADATA[:5]}")

    NDVI_STACK_SHAPE = ndvi_stack.shape
    NDVI_STACK_DTYPE = ndvi_stack.dtype

    shm, _ = _create_shared_copy(ndvi_stack)

    num_rows = ROW_END - ROW_START
    num_cols = COL_END - COL_START
    pixel_indices = [(row, col) for row in range(num_rows) for col in range(num_cols)]

    print(
        f"Processing {len(pixel_indices):,} pixels across {num_rows} rows and {num_cols} columns …"
    )

    try:
        results = Parallel(
            n_jobs=-1,
            backend="loky",
            initializer=lower_process_priority,
        )(
            delayed(process_pixel)(row, col, shm.name)
            for row, col in tqdm(pixel_indices, desc="Processing pixels")
        )
    finally:
        shm.close()
        shm.unlink()

    ndvi_fit_all = np.array(results, dtype=np.float32).reshape(
        (num_rows, num_cols, RESULT_LENGTH)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, ndvi_fit_all=ndvi_fit_all)
    print(
        f"Saved fitted NDVI parameters and quality metrics (shape {ndvi_fit_all.shape}) "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
