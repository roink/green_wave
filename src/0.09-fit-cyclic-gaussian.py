"""Fit cyclic Gaussian curves to an NDVI stack stored in HDF5 format."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/intermediate/ndvi_stack_optimized.h5"),
        help="Path to the input HDF5 file containing the NDVI stack and metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intermediate/ndvi_gaussian_fits.h5"),
        help="Destination for the fitted parameter HDF5 file.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum number of valid samples required to attempt a fit.",
    )
    return parser.parse_args()


def process_ndvi(series: np.ndarray) -> np.ndarray:
    winter_ndvi = np.nanquantile(series, 0.025)
    corrected = np.array(series, dtype=np.float32)
    corrected[np.isnan(corrected)] = winter_ndvi
    corrected[corrected < winter_ndvi] = winter_ndvi
    return median_filter(corrected, size=3)


def gaussian_cyclic(
    t: np.ndarray,
    amplitude: float,
    phase: float,
    width: float,
    offset: float,
    period: float = 365.0,
) -> np.ndarray:
    delta = ((t - phase + period / 2.0) % period) - period / 2.0
    return offset + amplitude * np.exp(-0.5 * (delta / width) ** 2)


def to_day_of_year(metadata: np.ndarray) -> np.ndarray:
    return np.array(
        [
            pd.to_datetime(f"{int(year)}-{int(day):03d}", format="%Y-%j").dayofyear
            for year, day in metadata
        ],
        dtype=np.int32,
    )


def allocate_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "A": np.full(shape, np.nan, dtype=np.float32),
        "mu": np.full(shape, np.nan, dtype=np.float32),
        "sigma": np.full(shape, np.nan, dtype=np.float32),
        "C": np.full(shape, np.nan, dtype=np.float32),
        "R2": np.full(shape, np.nan, dtype=np.float32),
    }


def fit_stack(
    stack: np.ndarray,
    doy: np.ndarray,
    min_samples: int,
) -> dict[str, np.ndarray]:
    _, nrows, ncols = stack.shape
    maps = allocate_maps((nrows, ncols))

    for i in tqdm(range(nrows), desc="Rows"):
        for j in range(ncols):
            series = stack[:, i, j]
            filtered = process_ndvi(series)
            mask = ~np.isnan(filtered) & ~np.isinf(filtered)
            if np.sum(mask) < min_samples:
                continue

            x = doy[mask]
            y = filtered[mask]
            initial_guess = [
                np.nanmax(y) - np.nanmin(y),
                x[np.argmax(y)],
                30.0,
                np.nanmin(y),
            ]
            bounds = ([0, 0, 0, -np.inf], [np.inf, 365, 365, np.inf])

            try:
                popt, _ = curve_fit(
                    gaussian_cyclic, x, y, p0=initial_guess, bounds=bounds
                )
            except RuntimeError:
                continue

            amplitude, phase, width, offset = popt
            fitted = gaussian_cyclic(x, *popt)
            residual = np.sum((y - fitted) ** 2)
            total = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - residual / total if total > 0 else np.nan

            maps["A"][i, j] = amplitude
            maps["mu"][i, j] = phase % 365
            maps["sigma"][i, j] = width
            maps["C"][i, j] = offset
            maps["R2"][i, j] = r_squared

    return maps


def save_maps(
    output_path: Path, maps: dict[str, np.ndarray], metadata: np.ndarray
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as dst:
        for key, array in maps.items():
            dst.create_dataset(key, data=array, dtype="f4", compression="lzf")
        dst.create_dataset(
            "metadata", data=metadata, dtype=metadata.dtype, compression="lzf"
        )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input HDF5 stack not found: {args.input}")

    with h5py.File(args.input, "r") as src:
        metadata = src["metadata"][:]
        ndvi_stack = src["ndvi_stack"][:]

    doy = to_day_of_year(metadata)
    print(f"Loaded NDVI stack with shape {ndvi_stack.shape} and {len(doy)} time steps.")

    maps = fit_stack(ndvi_stack, doy, args.min_samples)
    save_maps(args.output, maps, metadata)
    print(f"Gaussian fits saved to {args.output}")


if __name__ == "__main__":
    main()
