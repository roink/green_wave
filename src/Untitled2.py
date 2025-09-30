"""Visualise warped sine constructions without relying on notebook widgets."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = Path(__file__).stem
FIGURE_ROOT = Path("figure") / SCRIPT_NAME
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def g(x: np.ndarray, period: float, power: float) -> np.ndarray:
    clipped = np.clip(x, 0.0, period)
    return clipped**power / (clipped**power + (period - clipped) ** power)


def plot_power_warped_sine() -> Path:
    period = 2.0 * np.pi
    x = np.linspace(-period, 2 * period, 1000)
    p_values = [0.5, 1, 2, 5]

    fig, ax = plt.subplots(figsize=(8, 6))
    for power in p_values:
        warped_x = g(x, period, power) * period
        y = np.sin(2 * np.pi * warped_x / period)
        ax.plot(x, y, label=f"p={power}")

    ax.set_xlabel("x")
    ax.set_ylabel("sin(warped x)")
    ax.set_title("Warped sine for different power exponents")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    output_path = FIGURE_ROOT / "warped_sine_power.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_phase_modulation() -> Path:
    period = 1.0
    x = np.linspace(0, period, 1000)
    configs = [(0.5, 0.0), (0.5, 0.25), (1.0, 0.0), (1.0, 0.25)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, (power, phase_shift) in zip(axes.flat, configs):
        warped = g(x, period, power) * period
        y = np.sin(2 * np.pi * x + power * np.sin(2 * np.pi * (x - phase_shift)))
        ax.plot(x, y)
        ax.set_title(f"p={power}, phase shift={phase_shift}")
        ax.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle("Phase-modulated sine waves")
    fig.text(0.5, 0.04, "x", ha="center")
    fig.text(0.04, 0.5, "sin(warped x)", va="center", rotation="vertical")
    output_path = FIGURE_ROOT / "phase_modulation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def plot_sigmoid_modulation() -> Path:
    period = 1.0
    x = np.linspace(0, period, 1000)
    p_values = [0.25, 0.5, 0.75]
    offsets = [-1.0, 0.0, 1.0]

    fig, axes = plt.subplots(len(p_values), len(offsets), figsize=(12, 8), sharex=True, sharey=True)
    for (i, power), (j, offset) in product(enumerate(p_values), enumerate(offsets)):
        ax = axes[i, j]
        warped = g(x, period, power) * period
        y = sigmoid(np.sin(2 * np.pi * x + power * np.sin(2 * np.pi * (x - 0.25))) * 5 + offset)
        ax.plot(x, y)
        ax.set_title(f"p={power}, offset={offset}")
        ax.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle("Sigmoid-transformed warped sine waves")
    fig.text(0.5, 0.04, "x", ha="center")
    fig.text(0.04, 0.5, "sigmoid(...)", va="center", rotation="vertical")
    output_path = FIGURE_ROOT / "sigmoid_modulation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    power_path = plot_power_warped_sine()
    print(f"Saved warped sine power sweep -> {power_path}")

    phase_path = plot_phase_modulation()
    print(f"Saved phase modulation grid -> {phase_path}")

    sigmoid_path = plot_sigmoid_modulation()
    print(f"Saved sigmoid modulation grid -> {sigmoid_path}")

    print("All warped sine figures generated successfully.")
