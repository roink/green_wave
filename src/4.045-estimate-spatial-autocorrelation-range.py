#!/usr/bin/env python3
"""Estimate spatial autocorrelation ranges for predictor and target fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import curve_fit

from bioclim_correlation_utils import (
    FeatureLayerSpec,
    load_bioclim_layers,
    load_feature_layers,
    load_npz_arrays,
)
from logging_setup import initialize_script_logging
from tile_sampling_utils import construct_coordinate_grid

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "detrended_ndvi_bioclim_combined.npz"
ORBITAL_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "insolation" / "orbit91"
DEFAULT_OUTPUT_PATH = INTERMEDIATE_DIR / "spatial_autocorrelation_ranges.json"

SELECTED_BIOCLIM_NUMBERS: tuple[int, ...] = (1, *range(4, 20))
INSOLATION_DAYS: tuple[int, ...] = (15, 75, 135, 195, 255, 315)
INSOLATION_FEATURE_NAMES = [f"insolation_day_{day:03d}" for day in INSOLATION_DAYS]
R2_THRESHOLD = 0.6
MIN_OBSERVATIONS = 24
DEFAULT_MAX_CELLS = 120_000
DEFAULT_PAIR_SAMPLES = 200_000
DEFAULT_BINS = 24
RANDOM_STATE = 42

# Global cap for max lag (in meters) applied inside _compute_empirical_semivariogram
_MAX_LAG_M: float | None = None


class OrbitalDataError(FileNotFoundError):
    """Raised when orbital parameter data required for insolation is missing."""


def _extract_bioclim_number(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def _select_bioclim_indices(
    names: Sequence[str], allowed_numbers: Iterable[int]
) -> tuple[list[int], list[str]]:
    allowed_set = set(allowed_numbers)
    indices: list[int] = []
    selected_names: list[str] = []
    for idx, name in enumerate(names):
        number = _extract_bioclim_number(name)
        if number is not None and number in allowed_set:
            indices.append(idx)
            selected_names.append(str(name))
    if not indices:
        raise ValueError("No bioclim variables matched the requested numbers.")
    return indices, selected_names


def _load_present_day_orbital_parameters() -> tuple[float, float, float]:
    if not ORBITAL_DATA_PATH.exists():
        raise OrbitalDataError(
            "Orbital parameter file not found at {path}. "
            "Run download_insolation_data.py to fetch insolation datasets.".format(
                path=ORBITAL_DATA_PATH
            )
        )

    data = np.loadtxt(ORBITAL_DATA_PATH, skiprows=2, usecols=(0, 1, 2, 3))
    if data.size == 0:
        raise ValueError(
            f"Orbital parameter file {ORBITAL_DATA_PATH} does not contain any rows."
        )

    kyear_column = data[:, 0]
    present_day_matches = np.where(np.isclose(kyear_column, 0.0))[0]
    if present_day_matches.size == 0:
        raise ValueError(
            "Could not locate present-day (0 kyr) orbital parameters in {path}.".format(
                path=ORBITAL_DATA_PATH
            )
        )

    idx = int(present_day_matches[0])
    ecc = float(data[idx, 1])
    long_perh = float(data[idx, 2] + 180.0)
    obliquity = float(data[idx, 3])
    return ecc, obliquity, long_perh


def _daily_insolation_for_latitudes(
    latitudes: np.ndarray, days: Sequence[int]
) -> np.ndarray:
    ecc, obliquity_deg, long_perh_deg = _load_present_day_orbital_parameters()

    latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1, 1)
    days = np.asarray(days, dtype=np.float64).reshape(1, -1)

    epsilon = np.deg2rad(obliquity_deg)
    omega = np.deg2rad(long_perh_deg)
    phi = np.deg2rad(latitudes)

    delta_lambda_m = (days - 80.0) * 2 * np.pi / 365.2422
    beta = np.sqrt(1 - ecc**2)
    lambda_m0 = -2.0 * (
        (0.5 * ecc + 0.125 * ecc**3) * (1 + beta) * np.sin(-omega)
        - 0.25 * ecc**2 * (0.5 + beta) * np.sin(-2 * omega)
        + 0.125 * ecc**3 * (1 / 3 + beta) * np.sin(-3 * omega)
    )
    lambda_m = lambda_m0 + delta_lambda_m
    lambda_true = (
        lambda_m
        + (2 * ecc - 0.25 * ecc**3) * np.sin(lambda_m - omega)
        + 1.25 * ecc**2 * np.sin(2 * (lambda_m - omega))
        + (13 / 12) * ecc**3 * np.sin(3 * (lambda_m - omega))
    )

    delta = np.arcsin(np.sin(epsilon) * np.sin(lambda_true))
    argument = -np.tan(phi) * np.tan(delta)
    argument = np.clip(argument, -1.0, 1.0)
    h0 = np.arccos(argument)

    mask_polar_day = (np.abs(phi) >= (np.pi / 2 - np.abs(delta))) & (phi * delta > 0)
    mask_polar_night = (np.abs(phi) >= (np.pi / 2 - np.abs(delta))) & (phi * delta <= 0)
    h0 = np.where(mask_polar_day, np.pi, h0)
    h0 = np.where(mask_polar_night, 0.0, h0)

    numerator = (1 + ecc * np.cos(lambda_true - omega)) ** 2
    denominator = (1 - ecc**2) ** 2
    solar_constant = 1365.0
    insolation = (
        solar_constant
        / np.pi
        * numerator
        / denominator
        * (h0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(h0))
    )
    return insolation.astype(np.float32)


def _prepare_value_groups(
    include_insolation: bool,
    bioclim_numbers: Sequence[int],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    np.ndarray,
    np.ndarray,
]:
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
        bioclim_names, bioclim_numbers
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
    latitude_grid = construct_coordinate_grid(arrays, "latitudes", (rows, cols))
    longitude_grid = construct_coordinate_grid(arrays, "longitudes", (rows, cols))

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

    flat_features = selected_stack.reshape(len(selected_names), -1).T
    features = flat_features[valid_indices]
    targets = target_matrix[valid_indices]

    latitude_valid = latitude_grid.ravel()[valid_indices]
    longitude_valid = longitude_grid.ravel()[valid_indices]

    predictor_group = {
        name: features[:, idx].astype(np.float32)
        for idx, name in enumerate(selected_names)
    }
    target_group = {
        name: targets[:, idx].astype(np.float32)
        for idx, name in enumerate(target_names)
    }

    if include_insolation:
        insolation_features = _daily_insolation_for_latitudes(
            latitude_valid, INSOLATION_DAYS
        )
        insolation_group: dict[str, np.ndarray] | None = {
            name: insolation_features[:, idx]
            for idx, name in enumerate(INSOLATION_FEATURE_NAMES)
        }
    else:
        insolation_group = None

    return predictor_group, target_group, insolation_group, latitude_valid, longitude_valid


def _compute_coordinate_scaling(latitudes: np.ndarray) -> tuple[float, float, float]:
    median_lat_deg = float(np.median(latitudes))
    median_lat_rad = np.deg2rad(median_lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(median_lat_rad)
    return median_lat_deg, m_per_deg_lat, max(m_per_deg_lon, 1e-6)


def _sample_points(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values_by_group: dict[str, dict[str, np.ndarray]],
    max_cells: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    total_points = latitudes.size
    if total_points < 2:
        raise ValueError("Need at least two valid points to estimate autocorrelation.")
    sample_size = min(max_cells, total_points)
    if sample_size < 2:
        raise ValueError("Sampling fewer than two points is not supported.")

    if sample_size == total_points:
        indices = np.arange(total_points)
    else:
        indices = rng.choice(total_points, size=sample_size, replace=False)

    lat_sampled = latitudes[indices]
    lon_sampled = longitudes[indices]

    sampled_groups: dict[str, dict[str, np.ndarray]] = {}
    for group_name, group_values in values_by_group.items():
        sampled_groups[group_name] = {
            name: values[indices].astype(np.float32)
            for name, values in group_values.items()
        }

    return lat_sampled, lon_sampled, sampled_groups


def _generate_pair_indices(
    sample_size: int, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size < 2:
        raise ValueError("Cannot generate pairs with fewer than two samples.")
    pair_a = rng.integers(0, sample_size, size=n_pairs, endpoint=False)
    pair_b = rng.integers(0, sample_size, size=n_pairs, endpoint=False)
    mask = pair_a != pair_b
    if not np.any(mask):
        raise ValueError("Pair sampling failed to generate distinct points.")
    return pair_a[mask], pair_b[mask]


def _compute_distances(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    m_per_deg_lat: float,
    m_per_deg_lon: float,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Great-circle distances via haversine (meters)
    R = 6371000.0
    phi1 = np.deg2rad(latitudes[pair_a])
    phi2 = np.deg2rad(latitudes[pair_b])
    dphi = phi2 - phi1
    dl = np.deg2rad(longitudes[pair_b] - longitudes[pair_a])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    distances = 2 * R * np.arcsin(np.sqrt(a))
    positive_mask = distances > 0
    return pair_a[positive_mask], pair_b[positive_mask], distances[positive_mask]


def _exponential_variogram_model(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    rng = max(rng, 1e-12)
    return nugget + sill * (1.0 - np.exp(-h / rng))


def _compute_empirical_semivariogram(
    distances: np.ndarray,
    semivariances: np.ndarray,
    max_bins: int,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    if distances.size == 0:
        return None

    # Optional global cap injected via CLI
    global _MAX_LAG_M
    if _MAX_LAG_M is not None:
        keep = distances <= _MAX_LAG_M
        if not np.any(keep):
            return None
        distances = distances[keep]
        semivariances = semivariances[keep]

    finite_mask = np.isfinite(distances) & np.isfinite(semivariances)
    distances = distances[finite_mask]
    semivariances = semivariances[finite_mask]
    if distances.size < 10:
        return None

    max_distance = np.percentile(distances, 95)
    if max_distance <= 0:
        max_distance = distances.max()
    if max_distance <= 0:
        return None

    bin_count = min(max_bins, max(5, distances.size // 200))
    if bin_count < 5:
        bin_count = 5
    bin_edges = np.linspace(0.0, max_distance, num=bin_count + 1, dtype=np.float64)
    if bin_edges[-1] == 0:
        return None

    sums = np.zeros(bin_count, dtype=np.float64)
    distance_sums = np.zeros(bin_count, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)

    indices = np.searchsorted(bin_edges, distances, side="right") - 1
    valid = (indices >= 0) & (indices < bin_count)
    if not np.any(valid):
        return None

    np.add.at(sums, indices[valid], semivariances[valid])
    np.add.at(distance_sums, indices[valid], distances[valid])
    np.add.at(counts, indices[valid], 1)

    mask = counts > 0
    non_empty = int(mask.sum())
    if non_empty < 3:
        return None

    gamma = sums[mask] / counts[mask]
    centers = distance_sums[mask] / counts[mask]

    return centers, gamma, float(max_distance), non_empty


def _fit_variogram(
    distances: np.ndarray,
    semivariances: np.ndarray,
    max_bins: int,
) -> dict[str, float] | None:
    binned = _compute_empirical_semivariogram(distances, semivariances, max_bins)
    if binned is None:
        return None

    centers, gamma, max_distance, bin_count = binned

    nugget0 = max(float(gamma[0]), 0.0)
    sill0 = max(float(gamma[-1] - nugget0), 1e-6)
    range0 = max(float(centers[-1] / 3.0), 1.0)

    try:
        popt, _ = curve_fit(
            _exponential_variogram_model,
            centers,
            gamma,
            p0=[nugget0, sill0, range0],
            bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, max_distance * 10.0]),
            maxfev=10_000,
        )
    except RuntimeError:
        return None

    nugget, sill, range_param = map(float, popt)
    model_values = _exponential_variogram_model(centers, nugget, sill, range_param)
    ss_res = float(np.sum((gamma - model_values) ** 2))
    ss_tot = float(np.sum((gamma - float(np.mean(gamma))) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    effective_range = 3.0 * range_param

    return {
        "nugget": nugget,
        "partial_sill": sill,
        "range_parameter_m": range_param,
        "effective_range_m": effective_range,
        "fit_r2": fit_r2,
        "bin_count": int(bin_count),
        "max_distance_m": float(max_distance),
        "fit_method": "exponential",
    }


def _fallback_variogram_range(
    distances: np.ndarray,
    semivariances: np.ndarray,
    max_bins: int,
) -> dict[str, float | None] | None:
    binned = _compute_empirical_semivariogram(distances, semivariances, max_bins)
    if binned is None:
        return None

    centers, gamma, max_distance, bin_count = binned
    nugget = float(max(gamma[0], 0.0))
    sill = float(np.max(gamma))
    partial_sill = max(sill - nugget, 0.0)
    threshold = 0.95 * sill

    meeting = np.where(gamma >= threshold)[0]
    if meeting.size > 0:
        effective_range = float(centers[meeting[0]])
    else:
        effective_range = float(centers[-1])

    if effective_range <= 0 and centers.size > 1:
        effective_range = float(centers[1])
    if effective_range <= 0:
        effective_range = float(centers[-1])

    return {
        "nugget": nugget,
        "partial_sill": partial_sill,
        "range_parameter_m": None,
        "effective_range_m": effective_range,
        "fit_r2": None,
        "bin_count": int(bin_count),
        "max_distance_m": float(max_distance),
        "fit_method": "empirical_threshold",
    }


def _estimate_for_variable(
    values: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    distances: np.ndarray,
    bins: int,
    m_per_deg_lat: float,
    m_per_deg_lon: float,
) -> dict[str, float] | None:
    diffs = values[pair_a] - values[pair_b]
    semivariances = 0.5 * (diffs.astype(np.float64) ** 2)
    fit = _fit_variogram(distances, semivariances, bins)
    if fit is None:
        fit = _fallback_variogram_range(distances, semivariances, bins)
        if fit is None:
            return None

    range_m = fit["effective_range_m"]
    tile_lat_deg = range_m / m_per_deg_lat
    tile_lon_deg = range_m / m_per_deg_lon if m_per_deg_lon > 0 else float("nan")
    recommended_deg = float(max(tile_lat_deg, tile_lon_deg))

    fit.update(
        {
            "tile_size_lat_deg": float(tile_lat_deg),
            "tile_size_lon_deg": float(tile_lon_deg),
            "recommended_tile_size_deg": recommended_deg,
        }
    )
    return fit


def _summarise_group(entries: list[dict[str, float]], key: str) -> dict[str, float] | None:
    values = [entry[key] for entry in entries if entry.get(key) is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate spatial autocorrelation ranges for predictors and targets."
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=DEFAULT_MAX_CELLS,
        help="Maximum number of valid grid cells to sample for the analysis.",
    )
    parser.add_argument(
        "--pair-samples",
        type=int,
        default=DEFAULT_PAIR_SAMPLES,
        help="Number of random point pairs to draw when estimating the variogram.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Maximum number of distance bins to use for the empirical variogram.",
    )
    parser.add_argument(
        "--bioclim-numbers",
        type=int,
        nargs="+",
        default=None,
        help="Override the default set of bioclim variable numbers to analyse.",
    )
    parser.add_argument(
        "--skip-insolation",
        action="store_true",
        help="Skip estimating autocorrelation for insolation predictors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the JSON summary.",
    )
    # === New CLI flags ===
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Restrict analysis to a bounding box (in degrees).",
    )
    parser.add_argument(
        "--max-lag-km",
        type=float,
        default=None,
        help="Cap variogram fitting to distances ≤ this many km.",
    )
    parser.add_argument(
        "--detrend",
        choices=["none", "poly2"],
        default="none",
        help="Detrend variables by a polynomial in lat/lon before variogram.",
    )
    args = parser.parse_args()

    # Wire the global max-lag cap (meters)
    global _MAX_LAG_M
    _MAX_LAG_M = None if args.max_lag_km is None else float(args.max_lag_km) * 1000.0

    rng = np.random.default_rng(args.seed)

    bioclim_numbers = (
        tuple(args.bioclim_numbers)
        if args.bioclim_numbers is not None
        else SELECTED_BIOCLIM_NUMBERS
    )
    include_insolation = not args.skip_insolation

    predictor_group, target_group, insolation_group, latitudes, longitudes = (
        _prepare_value_groups(include_insolation, bioclim_numbers)
    )

    # Optional bbox filter
    if args.bbox:
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = args.bbox
        in_box = (
            (longitudes >= LON_MIN)
            & (longitudes <= LON_MAX)
            & (latitudes >= LAT_MIN)
            & (latitudes <= LAT_MAX)
        )
        if not np.any(in_box):
            raise ValueError("BBox mask removed all points.")
        latitudes = latitudes[in_box]
        longitudes = longitudes[in_box]

        def _filter_group(group: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {k: v[in_box] for k, v in group.items()}

        predictor_group = _filter_group(predictor_group)
        target_group = _filter_group(target_group)
        if insolation_group is not None:
            insolation_group = _filter_group(insolation_group)

    value_groups: dict[str, dict[str, np.ndarray]] = {
        "predictor_bioclim": predictor_group,
        "target_harmonics": target_group,
    }
    if insolation_group is not None:
        value_groups["predictor_insolation"] = insolation_group

    print(
        "Loaded "
        f"{latitudes.size:,d} valid grid cells with {sum(len(v) for v in value_groups.values())} "
        "variables for autocorrelation analysis."
    )

    lat_sampled, lon_sampled, sampled_groups = _sample_points(
        latitudes, longitudes, value_groups, args.max_cells, rng
    )
    sample_size = lat_sampled.size
    print(f"Subsampled to {sample_size:,d} grid cells for pair sampling.")

    median_lat_deg, m_per_deg_lat, m_per_deg_lon = _compute_coordinate_scaling(lat_sampled)
    print(
        f"Median latitude: {median_lat_deg:.2f}°, metres per degree (lat, lon): "
        f"({m_per_deg_lat:.1f}, {m_per_deg_lon:.1f})."
    )

    pair_a, pair_b = _generate_pair_indices(sample_size, args.pair_samples, rng)
    pair_a, pair_b, distances = _compute_distances(
        lat_sampled,
        lon_sampled,
        m_per_deg_lat,
        m_per_deg_lon,
        pair_a,
        pair_b,
    )
    if distances.size == 0:
        raise ValueError("No non-zero pairwise distances available for variogram fitting.")
    print(
        "Prepared {pairs:,d} distinct point pairs spanning up to {max_km:.1f} km.".format(
            pairs=distances.size, max_km=float(np.max(distances)) / 1000.0
        )
    )

    # Simple optional detrend: quadratic in lat/lon
    def _detrend(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        # X = [1, lat, lon, lat^2, lat*lon, lon^2]
        X = np.column_stack(
            [np.ones_like(lat), lat, lon, lat * lat, lat * lon, lon * lon]
        ).astype(np.float64)
        beta, *_ = np.linalg.lstsq(X, values.astype(np.float64), rcond=None)
        return values - X @ beta

    variable_results: list[dict[str, float | str | int | None]] = []

    for group_name, group_values in sampled_groups.items():
        print(f"Estimating autocorrelation ranges for {group_name} ({len(group_values)} variables)...")
        for name, values in group_values.items():
            vals = values
            if args.detrend != "none":
                vals = _detrend(values, lat_sampled, lon_sampled)
            fit = _estimate_for_variable(
                vals.astype(np.float64),
                pair_a,
                pair_b,
                distances,
                args.bins,
                m_per_deg_lat,
                m_per_deg_lon,
            )
            entry: dict[str, float | str | int | None]
            if fit is None:
                print(f"  - {name}: insufficient data to fit variogram.")
                entry = {
                    "group": group_name,
                    "name": name,
                    "effective_range_m": None,
                    "recommended_tile_size_deg": None,
                    "fit_r2": None,
                }
            else:
                r2_value = fit.get("fit_r2")
                r2_display = (
                    f"{r2_value:.3f}"
                    if isinstance(r2_value, (float, np.floating)) and np.isfinite(r2_value)
                    else "n/a"
                )
                method = fit.get("fit_method", "unknown")
                print(
                    "  - {name}: range≈{range_km:.1f} km, recommended tile≈{tile_deg:.3f}° "
                    "(method={method}, R²={r2})".format(
                        name=name,
                        range_km=fit["effective_range_m"] / 1000.0,
                        tile_deg=fit["recommended_tile_size_deg"],
                        method=method,
                        r2=r2_display,
                    )
                )
                entry = {
                    "group": group_name,
                    "name": name,
                    **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in fit.items()},
                }
            variable_results.append(entry)

    def _collect(key: str, group_filter: str | None = None) -> list[float]:
        collected: list[float] = []
        for entry in variable_results:
            if group_filter is not None and entry["group"] != group_filter:
                continue
            value = entry.get(key)
            if isinstance(value, (int, float)):
                collected.append(float(value))
        return collected

    overall_ranges = _collect("effective_range_m")
    summary: Dict[str, dict[str, float] | float | int | str | list] = {
        "median_latitude_deg": median_lat_deg,
        "m_per_degree_lat": m_per_deg_lat,
        "m_per_degree_lon": m_per_deg_lon,
        "sampled_cell_count": sample_size,
        "pair_count": int(distances.size),
    }
    if overall_ranges:
        overall_array = np.asarray(overall_ranges, dtype=np.float64)
        summary["overall_effective_range_m"] = {
            "median": float(np.median(overall_array)),
            "mean": float(np.mean(overall_array)),
            "min": float(np.min(overall_array)),
            "max": float(np.max(overall_array)),
        }
        tile_sizes = _collect("recommended_tile_size_deg")
        if tile_sizes:
            tile_array = np.asarray(tile_sizes, dtype=np.float64)
            summary["overall_recommended_tile_deg"] = {
                "median": float(np.median(tile_array)),
                "mean": float(np.mean(tile_array)),
                "min": float(np.min(tile_array)),
                "max": float(np.max(tile_array)),
            }

    for group_name in sampled_groups:
        group_entries = [entry for entry in variable_results if entry["group"] == group_name]
        range_stats = _summarise_group(group_entries, "effective_range_m")
        tile_stats = _summarise_group(group_entries, "recommended_tile_size_deg")
        summary[group_name] = {
            "variable_count": len(group_entries),
            "effective_range_m": range_stats,
            "recommended_tile_size_deg": tile_stats,
        }

    output = {
        "parameters": {
            "max_cells": args.max_cells,
            "pair_samples": args.pair_samples,
            "bins": args.bins,
            "bioclim_numbers": list(bioclim_numbers),
            "include_insolation": include_insolation,
            "seed": args.seed,
            "bbox": list(args.bbox) if args.bbox else None,
            "max_lag_km": args.max_lag_km,
            "detrend": args.detrend,
        },
        "summary": summary,
        "variables": variable_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"Wrote spatial autocorrelation summary to {args.output}.")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
