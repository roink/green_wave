#!/usr/bin/env python
"""Alternative double-logistic fitting workflow using filtered NDVI stack.

This companion script keeps the wrapped-distance formulation of the
``double_logistic`` curve, so it differs from ``0.05-...`` which instead shifts
day-of-year values prior to fitting. The comment clarifies why both workflows
remain in the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ndvi_double_logistic_utils import (
    fit_seasonal_curve,
    fit_seasonal_curve_all_years,
    plot_ndvi,
)

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
PREPROCESS_DIR = FIGURE_ROOT / "preprocessing"
FIT_SINGLE_DIR = FIGURE_ROOT / "fit-single-year"
FIT_MULTI_DIR = FIGURE_ROOT / "fit-multi-year"

for directory in (PREPROCESS_DIR, FIT_SINGLE_DIR, FIT_MULTI_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def double_logistic(
    t: np.ndarray | float,
    xmidSNDVI: float,
    scalSNDVI: float,
    xmidANDVI: float,
    scalANDVI: float,
    bias: float,
    scale: float,
) -> np.ndarray:
    tau_spring = (t - xmidSNDVI) % 365
    tau_autumn = (t - xmidANDVI) % 365
    spring = 1 / (1 + np.exp(tau_spring / scalSNDVI))
    autumn = 1 / (1 + np.exp(tau_autumn / scalANDVI))
    return bias + scale * (spring - autumn)


def main() -> None:
    plot_ndvi(60.0, 16.0, PREPROCESS_DIR)
    fit_seasonal_curve(60.0, 16.0, 2005, FIT_SINGLE_DIR, double_logistic)
    fit_seasonal_curve_all_years(60.0, 16.0, 2000, 2024, FIT_MULTI_DIR, double_logistic)


if __name__ == "__main__":
    main()
