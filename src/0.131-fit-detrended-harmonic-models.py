#!/usr/bin/env python3
"""Fit semiannual harmonic models to NDVI series after removing linear trends."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
NDVI_STACK_PATH = INTERMEDIATE_DIR / "ndvi_stack_optimized.h5"
TREND_PATH = INTERMEDIATE_DIR / "ndvi_harmonic_fit_semiannual_trend.h5"
OUTPUT_PATH = INTERMEDIATE_DIR / "detrended_ndvi_harmonic_fit_semiannual.h5"

PERIOD_DAYS = 365.2422
OMEGA = 2 * np.pi / PERIOD_DAYS
BLOCK_ROWS = 16
MIN_OBSERVATIONS = 12
PARAMETER_NAMES = (
    "beta0",
    "beta1_cos1",
    "beta2_sin1",
    "beta3_cos2",
    "beta4_sin2",
)


def _pack_mask(mask_1d: np.ndarray) -> bytes:
    """Pack a boolean mask into bytes so it can be used as a dict key."""

    return np.packbits(mask_1d.astype(np.uint8)).tobytes()


def _decode_parameter_names(raw: Sequence[bytes | str] | bytes | str | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, np.bytes_)):
        return [raw.decode("utf-8")]
    if isinstance(raw, str):
        return [raw]
    array = np.asarray(raw)
    decoded: list[str] = []
    for value in array.tolist():
        if isinstance(value, (bytes, np.bytes_)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _build_time_vectors(metadata: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if metadata.ndim != 2 or metadata.shape[1] != 2:
        raise ValueError("Metadata must be shaped (N, 2) with year and day-of-year columns.")
    dates = pd.to_datetime(
        [f"{int(year)}-{int(doy):03d}" for year, doy in metadata],
        format="%Y-%j",
    )
    deltas = dates - dates[0]
    elapsed_days = deltas.total_seconds() / 86400.0
    return dates, np.asarray(elapsed_days, dtype=np.float64)


def _evaluate_columns(elapsed_days: np.ndarray) -> list[np.ndarray]:
    elapsed_days = np.asarray(elapsed_days, dtype=np.float64)
    return [
        np.ones_like(elapsed_days),
        np.cos(OMEGA * elapsed_days),
        np.sin(OMEGA * elapsed_days),
        np.cos(2 * OMEGA * elapsed_days),
        np.sin(2 * OMEGA * elapsed_days),
    ]


def _fit_group_batched(
    columns: Sequence[np.ndarray],
    series_block: np.ndarray,
    mask_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve least-squares fits for pixels that share the same valid-time mask."""

    design = np.column_stack([col[mask_1d] for col in columns])
    targets = series_block[mask_1d, :]
    n_obs = design.shape[0]
    num_params = design.shape[1]

    if n_obs < MIN_OBSERVATIONS or n_obs <= num_params or n_obs == 0:
        n_group = targets.shape[1]
        nan_params = np.full((num_params, n_group), np.nan, dtype=np.float32)
        nan_metric = np.full(n_group, np.nan, dtype=np.float32)
        zero_obs = np.zeros(n_group, dtype=np.int16)
        return nan_params, nan_metric, nan_metric, nan_metric, zero_obs

    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        n_group = targets.shape[1]
        nan_params = np.full((num_params, n_group), np.nan, dtype=np.float32)
        nan_metric = np.full(n_group, np.nan, dtype=np.float32)
        zero_obs = np.full(n_group, n_obs, dtype=np.int16)
        return nan_params, nan_metric, nan_metric, nan_metric, zero_obs

    predictions = design @ coeffs
    residual_matrix = targets - predictions
    ss_res = np.sum(residual_matrix * residual_matrix, axis=0, dtype=np.float64)
    mean_targets = np.mean(targets, axis=0, dtype=np.float64)
    ss_tot = np.sum((targets - mean_targets) ** 2, axis=0, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = 1.0 - ss_res / ss_tot
        r_squared = np.where(ss_tot <= 0, np.nan, r_squared)
        r_squared = np.clip(r_squared, -1.0, 1.0)
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / np.maximum(n_obs - num_params - 1, 1)
        adj_r_squared = np.where(np.isfinite(r_squared), adj_r_squared, np.nan)
        aic = n_obs * np.log(np.maximum(ss_res / np.maximum(n_obs, 1), np.finfo(float).tiny)) + 2 * num_params

    n_obs_array = np.full(ss_res.shape, n_obs, dtype=np.int16)
    return (
        coeffs.astype(np.float32),
        r_squared.astype(np.float32),
        adj_r_squared.astype(np.float32),
        aic.astype(np.float32),
        n_obs_array,
    )


def _derived_amplitudes_and_phases(params: np.ndarray) -> tuple[float, float, float, float]:
    amp1 = np.nan
    phase1 = np.nan
    amp2 = np.nan
    phase2 = np.nan

    coeffs1 = np.asarray(params[1:3], dtype=float)
    if np.all(np.isfinite(coeffs1)):
        amp1 = float(np.hypot(*coeffs1))
        phase1 = float((np.arctan2(coeffs1[1], coeffs1[0]) % (2 * np.pi)) / OMEGA)

    coeffs2 = np.asarray(params[3:5], dtype=float)
    if np.all(np.isfinite(coeffs2)):
        amp2 = float(np.hypot(*coeffs2))
        phase2 = float((np.arctan2(coeffs2[1], coeffs2[0]) % (2 * np.pi)) / (2 * OMEGA))

    return amp1, phase1, amp2, phase2


def main() -> None:
    if not NDVI_STACK_PATH.exists():
        raise FileNotFoundError(
            "NDVI stack not found. Expected {}.".format(NDVI_STACK_PATH)
        )
    if not TREND_PATH.exists():
        raise FileNotFoundError(
            "Harmonic trend file missing. Run 0.11-fit-harmonic-models.py first to generate {}.".format(
                TREND_PATH
            )
        )

    print(f"Loading NDVI stack from {NDVI_STACK_PATH} …")
    with h5py.File(NDVI_STACK_PATH, "r") as ndvi_file:
        ndvi_dataset = ndvi_file["ndvi_stack"]
        metadata = ndvi_file["metadata"][:]
        dates, elapsed_days = _build_time_vectors(metadata)

    with h5py.File(TREND_PATH, "r") as trend_file:
        trend_parameters = trend_file["parameters"]
        parameter_names = _decode_parameter_names(
            trend_parameters.attrs.get("parameter_names")
        )
        if parameter_names is None:
            raise KeyError("Parameter names missing from harmonic trend dataset.")
        try:
            trend_index = parameter_names.index("beta5_trend")
        except ValueError as exc:
            raise KeyError(
                "Harmonic trend dataset does not include 'beta5_trend'."
            ) from exc
        time_center = float(trend_file.attrs.get("time_center_days", float("nan")))
        if not np.isfinite(time_center):
            time_center = float(np.nanmean(elapsed_days))
            print(
                "Warning: time_center_days attribute missing; using nanmean(elapsed_days)={:.3f}.".format(
                    time_center
                )
            )
        slopes = trend_parameters[..., trend_index]
        slope_mask = np.isfinite(slopes)
        print(
            "Loaded trend slopes with shape {} (finite entries: {:,}/{:,}).".format(
                slopes.shape,
                int(np.count_nonzero(slope_mask)),
                slopes.size,
            )
        )

    n_time = elapsed_days.size
    n_rows, n_cols = slopes.shape
    columns = _evaluate_columns(elapsed_days)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as output_file, h5py.File(NDVI_STACK_PATH, "r") as ndvi_file:
        ndvi_dataset = ndvi_file["ndvi_stack"]
        metadata = ndvi_file["metadata"][:]
        if ndvi_dataset.shape[1] != n_rows or ndvi_dataset.shape[2] != n_cols:
            raise ValueError(
                "NDVI stack spatial shape {stack} does not match trend grid {grid}.".format(
                    stack=ndvi_dataset.shape[1:], grid=(n_rows, n_cols)
                )
            )

        param_ds = output_file.create_dataset(
            "parameters",
            shape=(n_rows, n_cols, len(PARAMETER_NAMES)),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols), len(PARAMETER_NAMES)),
            compression="lzf",
            fillvalue=np.nan,
        )
        param_ds.attrs["parameter_names"] = np.asarray(PARAMETER_NAMES, dtype="S")

        r2_ds = output_file.create_dataset(
            "r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        adj_ds = output_file.create_dataset(
            "adjusted_r_squared",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        aic_ds = output_file.create_dataset(
            "aic",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        nobs_ds = output_file.create_dataset(
            "num_observations",
            shape=(n_rows, n_cols),
            dtype="i2",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=0,
        )
        amp1_ds = output_file.create_dataset(
            "amplitude_annual",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        phase1_ds = output_file.create_dataset(
            "phase_annual_days",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        amp2_ds = output_file.create_dataset(
            "amplitude_semiannual",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )
        phase2_ds = output_file.create_dataset(
            "phase_semiannual_days",
            shape=(n_rows, n_cols),
            dtype="f4",
            chunks=(min(32, n_rows), min(128, n_cols)),
            compression="lzf",
            fillvalue=np.nan,
        )

        output_file.create_dataset("metadata", data=metadata, compression="lzf")
        output_file.create_dataset(
            "time_offsets_days", data=elapsed_days.astype("f4"), compression="lzf"
        )
        output_file.attrs.update(
            {
                "model_name": "semiannual_detrended",
                "include_semiannual": 1,
                "include_trend": 0,
                "period_days": PERIOD_DAYS,
                "omega": float(OMEGA),
                "time_origin": dates[0].isoformat(),
                "time_units": "days since time_origin",
                "time_center_days": float(np.nanmean(elapsed_days)),
                "detrended_from": TREND_PATH.name,
            }
        )

        print(
            "Fitting detrended harmonic model on grid of shape {} (time steps: {}).".format(
                (n_rows, n_cols), n_time
            )
        )

        time_offsets = elapsed_days[:, None, None] - time_center

        for row_start in range(0, n_rows, BLOCK_ROWS):
            row_end = min(row_start + BLOCK_ROWS, n_rows)
            block_rows = row_end - row_start
            ndvi_block = ndvi_dataset[:, row_start:row_end, :]
            slope_block = slopes[row_start:row_end, :]
            slope_block = np.where(np.isfinite(slope_block), slope_block, 0.0)
            detrended_block = ndvi_block - slope_block[None, :, :] * time_offsets

            params_block = np.full(
                (block_rows, n_cols, len(PARAMETER_NAMES)), np.nan, dtype=np.float32
            )
            r2_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            adj_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            aic_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            nobs_block = np.zeros((block_rows, n_cols), dtype=np.int16)
            amp1_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            phase1_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            amp2_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)
            phase2_block = np.full((block_rows, n_cols), np.nan, dtype=np.float32)

            for local_row in range(block_rows):
                row_series = detrended_block[:, local_row, :]
                valid = np.isfinite(row_series)

                mask_groups: dict[bytes, list[int]] = {}
                for col in range(n_cols):
                    mask_col = valid[:, col]
                    if not np.any(mask_col):
                        continue
                    mask_key = _pack_mask(mask_col)
                    mask_groups.setdefault(mask_key, []).append(col)

                for cols in mask_groups.values():
                    group_indices = np.asarray(cols, dtype=np.int32)
                    mask_1d = valid[:, group_indices[0]]
                    group_series = row_series[:, group_indices]
                    (
                        coeffs,
                        r_squared,
                        adj_r_squared,
                        aic,
                        n_obs,
                    ) = _fit_group_batched(columns, group_series, mask_1d)

                    params_block[local_row, group_indices, :] = coeffs.T
                    r2_block[local_row, group_indices] = r_squared
                    adj_block[local_row, group_indices] = adj_r_squared
                    aic_block[local_row, group_indices] = aic
                    nobs_block[local_row, group_indices] = n_obs

                    for idx, col in enumerate(group_indices):
                        amp1, phase1, amp2, phase2 = _derived_amplitudes_and_phases(
                            coeffs[:, idx]
                        )
                        amp1_block[local_row, col] = np.float32(amp1)
                        phase1_block[local_row, col] = np.float32(phase1)
                        amp2_block[local_row, col] = np.float32(amp2)
                        phase2_block[local_row, col] = np.float32(phase2)

            param_ds[row_start:row_end, :, :] = params_block
            r2_ds[row_start:row_end, :] = r2_block
            adj_ds[row_start:row_end, :] = adj_block
            aic_ds[row_start:row_end, :] = aic_block
            nobs_ds[row_start:row_end, :] = nobs_block
            amp1_ds[row_start:row_end, :] = amp1_block
            phase1_ds[row_start:row_end, :] = phase1_block
            amp2_ds[row_start:row_end, :] = amp2_block
            phase2_ds[row_start:row_end, :] = phase2_block

            print(
                f"Processed rows {row_start}-{row_end - 1} / {n_rows - 1} (block size: {block_rows})."
            )

    print(f"Saved detrended harmonic fits to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
