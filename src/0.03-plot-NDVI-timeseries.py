#!/usr/bin/env python
"""Plot NDVI time-series for selected lat/lon locations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ndvi_analysis_utils import (
    ensure_script_figure_dir,
    get_ndvi_timeseries,
    _coordinate_tag,
    _save_figure,
)

SCRIPT_STEM, FIGURE_DIR = ensure_script_figure_dir(__file__)


def plot_ndvi(lat: float, lon: float) -> Path:
    """Create a plot of the NDVI time series and save it to disk."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    dates = np.array(dates)
    ndvi_timeseries = np.asarray(ndvi_timeseries)
    valid_mask = ~np.isnan(ndvi_timeseries)
    dates = dates[valid_mask]
    ndvi_timeseries = ndvi_timeseries[valid_mask]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        dates, ndvi_timeseries, marker="o", linestyle="-", color="green", label="NDVI"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    ax.grid(True)
    ax.legend()

    output_path = _save_figure(
        fig,
        f"{_coordinate_tag(lat, lon)}_timeseries",
        script_stem=SCRIPT_STEM,
        figure_dir=FIGURE_DIR,
    )

    return output_path


def main() -> None:
    for lat in (40.0, 50.0, 60.0):
        plot_ndvi(lat, 16.0)


if __name__ == "__main__":
    main()
