"""Compare alternative seasonal NDVI curve fits for a single pixel.

The notebook version of this file relied on inline plotting.  This script fits
multiple analytic curves, prints the fitted parameters, and saves the resulting
figures to ``figure/Untitled1``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

SCRIPT_NAME = Path(__file__).stem
FIGURE_ROOT = Path("figure") / SCRIPT_NAME
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

DOY_VALUES = np.array(
    [
        1,
        17,
        33,
        49,
        65,
        81,
        97,
        113,
        129,
        145,
        161,
        177,
        193,
        209,
        225,
        241,
        257,
        273,
        289,
        305,
        321,
        337,
        353,
    ],
    dtype=float,
)

NDVI_VALUES = np.array(
    [
        5809,
        5809,
        2657,
        2657,
        2657,
        2780,
        6257,
        6346,
        6759,
        7396,
        8015,
        8058,
        8015,
        8058,
        7991,
        8019,
        7991,
        7410,
        7286,
        6416,
        5859,
        5859,
        1407,
    ],
    dtype=float,
)


def double_logistic(
    t: np.ndarray,
    xmid_s: float,
    scal_s: float,
    xmid_a: float,
    scal_a: float,
    bias: float,
    scale: float,
) -> np.ndarray:
    spring = 1.0 / (1.0 + np.exp((xmid_s - t) / scal_s))
    autumn = 1.0 / (1.0 + np.exp((xmid_a - t) / scal_a))
    return bias + scale * (spring - autumn)


def tanh_sine(t: np.ndarray, phase: float, bias: float, scale: float, sharpness: float) -> np.ndarray:
    angle = (t - phase) * 2.0 * np.pi / 365.0
    numerator = np.tanh(np.sin(angle) * sharpness)
    denominator = np.tanh(sharpness) if sharpness != 0 else 1.0
    return bias + scale * numerator / denominator


def sigmoid_sine(x: np.ndarray, amplitude: float, steepness: float) -> np.ndarray:
    numerator = 2.0 / (1.0 + np.exp(-steepness * np.sin(x))) - 1.0
    return amplitude * numerator


def plot_fit(
    doy: np.ndarray,
    ndvi: np.ndarray,
    x_dense: np.ndarray,
    y_dense: np.ndarray,
    title: str,
    output_name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(doy, ndvi, color="black", label="Observed NDVI", zorder=2)
    ax.plot(x_dense, y_dense, color="red", linestyle="--", label="Fitted curve", zorder=1)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    output_path = FIGURE_ROOT / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    doy_dense = np.linspace(1, 365, 365)

    # Double logistic fit -------------------------------------------------
    logistic_guess = [120, 20, 270, 20, 1000, 7000]
    logistic_bounds = ([0, 1, 180, 1, -2000, 0], [365, 200, 365, 200, 12000, 20000])
    logistic_params, _ = curve_fit(
        double_logistic,
        DOY_VALUES,
        NDVI_VALUES,
        p0=logistic_guess,
        bounds=logistic_bounds,
        maxfev=20000,
    )
    print("Double logistic parameters:")
    for name, value in zip(
        ["xmidS", "scalS", "xmidA", "scalA", "bias", "scale"], logistic_params
    ):
        print(f"  {name:>6}: {value:8.3f}")
    logistic_curve = double_logistic(doy_dense, *logistic_params)
    logistic_path = plot_fit(
        DOY_VALUES,
        NDVI_VALUES,
        doy_dense,
        logistic_curve,
        "Double Logistic Fit to NDVI",
        "double_logistic_fit.png",
    )
    print(f"Saved double logistic figure -> {logistic_path}")

    # Tanh-sine fit -------------------------------------------------------
    tanh_guess = [120, 3000, 4000, 5]
    tanh_bounds = ([0, 0, 0, 0.1], [365, 8000, 20000, 50])
    tanh_params, _ = curve_fit(
        tanh_sine,
        DOY_VALUES,
        NDVI_VALUES,
        p0=tanh_guess,
        bounds=tanh_bounds,
        maxfev=20000,
    )
    print("Tanh-sine parameters:")
    for name, value in zip(["phase", "bias", "scale", "sharpness"], tanh_params):
        print(f"  {name:>9}: {value:8.3f}")
    tanh_curve = tanh_sine(doy_dense, *tanh_params)
    tanh_path = plot_fit(
        DOY_VALUES,
        NDVI_VALUES,
        doy_dense,
        tanh_curve,
        "Tanh-Sine Fit to NDVI",
        "tanh_sine_fit.png",
    )
    print(f"Saved tanh-sine figure -> {tanh_path}")

    # Sigmoid-sine demonstration -----------------------------------------
    rng = np.random.default_rng(42)
    x_demo = np.linspace(0, 4 * np.pi, 200)
    true_a = 2.5
    true_k = 4.0
    y_demo = sigmoid_sine(x_demo, true_a, true_k) + 0.1 * rng.normal(size=len(x_demo))

    demo_guess = [1.0, 1.0]
    demo_params, _ = curve_fit(sigmoid_sine, x_demo, y_demo, p0=demo_guess, maxfev=20000)
    print("Sigmoid-sine recovered parameters:")
    print(f"  amplitude: true={true_a:.2f}, fitted={demo_params[0]:.3f}")
    print(f"  steepness: true={true_k:.2f}, fitted={demo_params[1]:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_demo, y_demo, "b.", label="Synthetic data")
    ax.plot(x_demo, sigmoid_sine(x_demo, *demo_params), "r-", label="Fit")
    ax.set_xlabel("Phase (radians)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    sigmoid_path = FIGURE_ROOT / "sigmoid_sine_demo.png"
    fig.savefig(sigmoid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sigmoid-sine demonstration figure -> {sigmoid_path}")

    print("All figures generated successfully.")
