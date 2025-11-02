"""Utilities for grouping grid cells into tiles and applying tile-aware bootstrapping."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from sklearn.utils import check_random_state
import sklearn.ensemble._forest as forest

__all__ = [
    "construct_coordinate_grid",
    "compute_tile_ids",
    "summarize_tile_statistics",
    "tile_bootstrap",
]


if not hasattr(forest, "_generate_sample_indices"):
    raise RuntimeError(
        "Unsupported scikit-learn version: missing _generate_sample_indices in ensemble._forest"
    )


@dataclass(frozen=True)
class _TileBootstrapState:
    """Internal state describing how samples map onto spatial tiles."""

    unique_tiles: np.ndarray
    indices_by_tile: list[np.ndarray]
    fraction: float


_ORIGINAL_GENERATE_SAMPLE_INDICES = forest._generate_sample_indices
_ACTIVE_STATE: _TileBootstrapState | None = None


def construct_coordinate_grid(
    arrays: Mapping[str, np.ndarray],
    key: str,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Construct a 2D coordinate grid from stored latitude/longitude arrays."""

    if key not in arrays:
        raise KeyError(
            "Coordinate grid missing from combined dataset. Expected '{key}' array.".format(
                key=key
            )
        )

    values = np.asarray(arrays[key], dtype=np.float32)
    values = np.squeeze(values)

    if values.shape == grid_shape:
        return values
    if values.ndim == 1:
        if values.size == grid_shape[0]:
            return np.broadcast_to(values[:, None], grid_shape)
        if values.size == grid_shape[1]:
            return np.broadcast_to(values[None, :], grid_shape)
        if values.size == grid_shape[0] * grid_shape[1]:
            return values.reshape(grid_shape)
    if values.ndim == 2 and values.shape == (1, grid_shape[1]):
        return np.broadcast_to(values, grid_shape)
    if values.ndim == 2 and values.shape == (grid_shape[0], 1):
        return np.broadcast_to(values, grid_shape)

    raise ValueError(
        "Unsupported coordinate array shape {shape}; expected {grid_shape} or broadcastable.".format(
            shape=values.shape,
            grid_shape=grid_shape,
        )
    )


def _encode_tile_pairs(lat_bins: np.ndarray, lon_bins: np.ndarray) -> tuple[np.ndarray, int, int]:
    lat_min = int(lat_bins.min())
    lon_min = int(lon_bins.min())
    lat_indices = lat_bins - lat_min
    lon_indices = lon_bins - lon_min
    lon_span = int(lon_indices.max()) + 1
    encoded = lat_indices * lon_span + lon_indices
    return encoded.astype(np.int64), lat_min, lon_min


def compute_tile_ids(
    latitude_grid: np.ndarray,
    longitude_grid: np.ndarray,
    valid_indices: np.ndarray,
    *,
    tile_size_deg: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Assign each valid sample to a spatial tile of ``tile_size_deg`` degrees."""

    lat_flat = latitude_grid.ravel()[valid_indices]
    lon_flat = longitude_grid.ravel()[valid_indices]

    lat_bins = np.floor(lat_flat / tile_size_deg).astype(np.int64)
    lon_bins = np.floor(lon_flat / tile_size_deg).astype(np.int64)
    encoded, lat_offset, lon_offset = _encode_tile_pairs(lat_bins, lon_bins)

    metadata = {
        "tile_size_degrees": float(tile_size_deg),
        "latitude_bin_offset": float(lat_offset * tile_size_deg),
        "longitude_bin_offset": float(lon_offset * tile_size_deg),
    }
    return encoded, metadata


def summarize_tile_statistics(tile_ids: np.ndarray) -> dict[str, float]:
    """Summarise how many samples fall into each spatial tile."""

    tile_ids = np.asarray(tile_ids)
    if tile_ids.size == 0:
        raise ValueError("Cannot summarise tile counts for an empty array.")

    unique_tiles, counts = np.unique(tile_ids.astype(np.int64), return_counts=True)
    return {
        "training_tile_count": int(unique_tiles.size),
        "train_tile_cell_count_min": int(counts.min()),
        "train_tile_cell_count_median": float(np.median(counts)),
        "train_tile_cell_count_max": int(counts.max()),
    }


def _build_bootstrap_state(tile_ids: np.ndarray, fraction: float) -> _TileBootstrapState:
    if not 0 < fraction <= 1:
        raise ValueError("Tile sampling fraction must lie in (0, 1].")

    tile_ids = np.asarray(tile_ids, dtype=np.int64)
    unique_tiles, inverse = np.unique(tile_ids, return_inverse=True)
    indices_by_tile = [np.flatnonzero(inverse == idx) for idx in range(unique_tiles.size)]
    return _TileBootstrapState(unique_tiles=unique_tiles, indices_by_tile=indices_by_tile, fraction=fraction)


def _tile_generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    state = _ACTIVE_STATE
    if state is None:
        return _ORIGINAL_GENERATE_SAMPLE_INDICES(random_state, n_samples, n_samples_bootstrap)

    rng = check_random_state(random_state)
    n_tiles = state.unique_tiles.size
    n_select = max(1, int(round(n_tiles * state.fraction)))
    if n_select >= n_tiles:
        selected_indices = np.arange(n_tiles)
    else:
        selected_indices = rng.choice(n_tiles, size=n_select, replace=False)

    eligible = np.concatenate([state.indices_by_tile[idx] for idx in selected_indices])
    if eligible.size == 0:
        return _ORIGINAL_GENERATE_SAMPLE_INDICES(random_state, n_samples, n_samples_bootstrap)

    if n_samples_bootstrap is None:
        draw_count = eligible.size
    else:
        draw_count = min(max(1, int(round(n_samples_bootstrap))), eligible.size)
    sampled = rng.choice(eligible, size=draw_count, replace=True)
    return sampled.astype(np.int32)


@contextmanager
def tile_bootstrap(tile_ids: Iterable[int], fraction: float):
    """Temporarily override sklearn's bootstrap sampling to operate on spatial tiles."""

    global _ACTIVE_STATE
    previous_state = _ACTIVE_STATE
    previous_function = forest._generate_sample_indices

    _ACTIVE_STATE = _build_bootstrap_state(np.asarray(tile_ids, dtype=np.int64), fraction)
    forest._generate_sample_indices = _tile_generate_sample_indices
    try:
        yield
    finally:
        forest._generate_sample_indices = previous_function
        _ACTIVE_STATE = previous_state
