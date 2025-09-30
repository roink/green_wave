#!/usr/bin/env python
"""Visualise the double-logistic NDVI function for different parameter guesses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def double_logistic(
    t: np.ndarray,
    xmidSNDVI: float,
    scalSNDVI: float,
    xmidANDVI: float,
    scalANDVI: float,
    bias: float,
    scale: float,
) -> np.ndarray:
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


def plot_examples() -> Path:
    t_values = np.linspace(0, 365, 366)
    bias_guess = 1000
    scale_guess = 5000

    initial_guess_1 = [120, 20, 270, 25, bias_guess, scale_guess]
    initial_guess_2 = [240, 20, 60, 25, bias_guess + scale_guess, scale_guess]
    initial_guess_3 = [240, 20, 60, 25, bias_guess + 0.5 * scale_guess, 0]

    ndvi_north = double_logistic(t_values, *initial_guess_1)
    ndvi_south = double_logistic(t_values, *initial_guess_2)
    ndvi_equatorial = double_logistic(t_values, *initial_guess_3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_values, ndvi_north, label="Northern Hemisphere", color="green")
    ax.plot(t_values, ndvi_south, label="Southern Hemisphere", color="blue")
    ax.plot(t_values, ndvi_equatorial, label="Equatorial", color="red")
    ax.set_xlabel("Day of Year (DOY)")
    ax.set_ylabel("NDVI")
    ax.set_title("Double-Logistic NDVI Function for Different Initial Guesses")
    ax.legend()
    ax.grid(True)

    output_path = FIGURE_DIR / "double-logistic-initial-guesses.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved double-logistic comparison to {output_path}")
    return output_path


def main() -> None:
    plot_examples()


if __name__ == "__main__":
    main()
