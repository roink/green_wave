"""Common utilities for NDVI analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_ROOT = PROJECT_ROOT / "figure"


def ensure_script_figure_dir(script_path: Path | str) -> tuple[str, Path]:
    """Return the figure directory for a script, creating it if necessary."""

    script_path = Path(script_path)
    script_stem = script_path.stem
    figure_dir = FIGURE_ROOT / script_stem
    figure_dir.mkdir(parents=True, exist_ok=True)
    return script_stem, figure_dir


def _slugify(text: str) -> str:
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(" ", "_")
        .replace("/", "-")
        .replace(".", "p")
    )


def _coordinate_tag(lat: float, lon: float) -> str:
    return _slugify(f"lat{lat:.1f}_lon{lon:.1f}")


def _save_figure(
    fig: plt.Figure,
    description: str,
    *,
    script_stem: str,
    figure_dir: Path,
    close: bool = True,
) -> Path:
    filename = f"{script_stem}__{_slugify(description)}.png"
    path = figure_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"Saved figure to {path}")
    return path


def get_ndvi_timeseries(
    lat: float,
    lon: float,
    *,
    return_day_of_year: bool = False,
) -> tuple[Sequence[pd.Timestamp] | np.ndarray, np.ndarray]:
    """Extract an NDVI time series for the nearest pixel to ``(lat, lon)``."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NDVI stack not found at {DATA_PATH}")

    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Nearest pixel index: row={row_idx}, col={col_idx}")

    with h5py.File(DATA_PATH, "r") as h5f:
        metadata = h5f["metadata"][:]
        ndvi_timeseries = h5f["ndvi_stack"][:, row_idx, col_idx]

    dates = [
        pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata
    ]

    if return_day_of_year:
        doy = np.array([date.day_of_year for date in dates])
        return doy, ndvi_timeseries

    return dates, ndvi_timeseries


def process_ndvi(
    ndvi_timeseries: Iterable[float] | np.ndarray,
    *,
    dates: Sequence[pd.Timestamp] | None = None,
    winter_quantile: float = 0.025,
    median_size: int = 3,
    clip_below_winter: bool = True,
    fill_missing_with_winter: bool = False,
    scale: bool = False,
    scale_quantiles: tuple[float, float] = (0.025, 0.995),
    winter_fill_window: tuple[int, int] | None = (300, 60),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Apply standard NDVI preprocessing steps."""

    series = np.asarray(list(ndvi_timeseries), dtype=float)
    winter_ndvi = np.nanquantile(series, winter_quantile)

    corrected = np.copy(series)
    doy: np.ndarray | None = None
    if dates is not None:
        doy = np.array([date.day_of_year for date in dates], dtype=int)
    if fill_missing_with_winter:
        nan_mask = np.isnan(corrected)
        if doy is not None and winter_fill_window is not None:
            start, end = winter_fill_window
            if start <= end:
                winter_mask = (doy >= start) & (doy <= end)
            else:
                winter_mask = (doy >= start) | (doy <= end)
            fill_mask = nan_mask & winter_mask
            corrected[fill_mask] = winter_ndvi
            nan_mask = np.isnan(corrected)
        corrected[nan_mask] = winter_ndvi
    if clip_below_winter:
        corrected[corrected < winter_ndvi] = winter_ndvi

    filtered_input = np.nan_to_num(corrected, nan=winter_ndvi)
    filtered = median_filter(filtered_input, size=median_size)

    if not scale:
        return winter_ndvi, corrected, filtered

    lower_q, upper_q = scale_quantiles
    lower = np.nanquantile(filtered, lower_q)
    upper = np.nanquantile(filtered, upper_q)
    span = upper - lower
    if np.isclose(span, 0):
        scaled = np.zeros_like(filtered)
    else:
        scaled = (filtered - lower) / span
    return winter_ndvi, corrected, scaled


def sigmoid_double_sawtooth(
    t: np.ndarray,
    z1: float,
    d1: float,
    z2: float,
    d2: float,
    bias: float,
    scale: float,
    *,
    period: float = 365,
    return_derivative: bool = False,
):
    """Evaluate a sigmoid-wrapped double sawtooth curve."""

    z1 = z1 % period
    z2 = z2 % period
    d1 = 1 / d1
    d2 = 1 / d2
    t = np.asarray(t)

    c1, c2 = _get_crossings(z1, d1, z2, d2, period)
    val1 = _sawtooth_zp_d(c1, period, z1, d1)
    val2 = _sawtooth_zp_d(c2, period, z1, d1)

    modt = t % period
    in_first = modt < c1
    in_second = (modt >= c1) & (modt < c2)

    x_vals = np.zeros_like(t)
    if return_derivative:
        xprime_vals = np.zeros_like(t)

    if val1 < val2:
        x_vals[in_first] = _sawtooth_zp_d(t[in_first], period, z2, d2)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z1, d1)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(
            t[~(in_first | in_second)], period, z2, d2
        )
        if return_derivative:
            xprime_vals[in_first] = d2
            xprime_vals[in_second] = d1
            xprime_vals[~(in_first | in_second)] = d2
    else:
        x_vals[in_first] = _sawtooth_zp_d(t[in_first], period, z1, d1)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z2, d2)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(
            t[~(in_first | in_second)], period, z1, d1
        )
        if return_derivative:
            xprime_vals[in_first] = d1
            xprime_vals[in_second] = d2
            xprime_vals[~(in_first | in_second)] = d1

    s_vals = _logistic(x_vals)

    if return_derivative:
        dsigma_dx = s_vals * (1 - s_vals)
        ds_dt = dsigma_dx * xprime_vals
        return s_vals * scale + bias, ds_dt * scale

    return s_vals * scale + bias


def _sawtooth_zp_d(tt: np.ndarray, period: float, z: float, d: float) -> np.ndarray:
    return (((tt - z - period / 2) % period) - period / 2) * d


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _get_crossings(
    z1: float, d1: float, z2: float, d2: float, period: float = 365
) -> tuple[float, float]:
    crossings = _find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        return 0.0, period / 2
    return crossings[0], crossings[1]


def _find_sawtooth_intersections(
    z1: float, d1: float, z2: float, d2: float, period: float = 365
) -> list[float]:
    z1_mod = z1 % period
    z2_mod = z2 % period

    r = d2 / d1
    dprime = (z1_mod - z2_mod) % period

    num1 = r * dprime + (period / 2) * (1 - r)
    den = 1 - r
    T1 = num1 / den

    num2 = r * dprime + (period / 2) * (1 - 3 * r)
    T2 = num2 / den

    solutions: list[float] = []
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period / 2) + T1) % period
        solutions.append(c1)
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period / 2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions


__all__ = [
    "DATA_PATH",
    "FIGURE_ROOT",
    "PROJECT_ROOT",
    "ensure_script_figure_dir",
    "get_ndvi_timeseries",
    "process_ndvi",
    "sigmoid_double_sawtooth",
    "_coordinate_tag",
    "_save_figure",
    "_slugify",
]
