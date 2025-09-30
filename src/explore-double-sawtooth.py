"""Explore combinations of sawtooth wave constructions and save the figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = Path(__file__).stem
FIGURE_ROOT = Path("figure") / SCRIPT_NAME
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class SawtoothConfig:
    label: str
    zero_point: float
    slope: float


def sawtooth(t: np.ndarray, period: float) -> np.ndarray:
    return np.mod(t, period)


def sawtooth_centered(t: np.ndarray, period: float, zero_point: float, slope: float = 1.0) -> np.ndarray:
    return (((t - zero_point - period / 2.0) % period) - period / 2.0) * slope


def double_sawtooth(
    t: np.ndarray,
    period: float,
    first: SawtoothConfig,
    second: SawtoothConfig,
) -> np.ndarray:
    st1 = sawtooth_centered(t, period, first.zero_point, first.slope)
    st2 = sawtooth_centered(t, period, second.zero_point, second.slope)
    crossing_start = (first.zero_point * first.slope - second.zero_point * second.slope) / (
        first.slope - second.slope
    )
    crossing_end = ((first.zero_point + period) * first.slope - second.zero_point * second.slope) / (
        first.slope - second.slope
    )
    mask = ((t % period) > crossing_start) & ((t % period) < crossing_end)
    return np.where(mask, st2, st1)


def save_plot(x: np.ndarray, ys: Iterable[np.ndarray], labels: Iterable[str], title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for series, label in zip(ys, labels):
        ax.plot(x, series, label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("f(t)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    output_path = FIGURE_ROOT / name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    period = 365.0
    t_values = np.linspace(0, 2 * period, 1000)

    base_path = save_plot(
        t_values,
        [sawtooth(t_values, period)],
        ["Sawtooth"],
        "Basic sawtooth waveform",
        "sawtooth.png",
    )
    print(f"Saved base sawtooth figure -> {base_path}")

    centered_configs = [
        SawtoothConfig("Zero point 120", 120, 1.0),
        SawtoothConfig("Zero point 120 scaled", 120, 2.0),
    ]
    for cfg in centered_configs:
        series = sawtooth_centered(t_values, period, cfg.zero_point, cfg.slope)
        path = save_plot(
            t_values,
            [series],
            [cfg.label],
            f"Centered sawtooth ({cfg.label})",
            f"centered_{int(cfg.zero_point)}_{cfg.slope:.1f}.png",
        )
        print(f"Saved centered sawtooth figure -> {path}")

    increase = SawtoothConfig("Increase", 120, 2.0)
    decrease = SawtoothConfig("Decrease", 250, -3.0)
    combined_path = save_plot(
        t_values,
        [
            sawtooth_centered(t_values, period, increase.zero_point, increase.slope),
            sawtooth_centered(t_values, period, decrease.zero_point, decrease.slope),
        ],
        [increase.label, decrease.label],
        "Contrasting sawtooth scalings",
        "scaled_comparison.png",
    )
    print(f"Saved scaled comparison figure -> {combined_path}")

    crossing = (
        (increase.zero_point * increase.slope - decrease.zero_point * decrease.slope)
        / (increase.slope - decrease.slope)
    )
    crossing_mod = (
        ((increase.zero_point + period) * increase.slope - decrease.zero_point * decrease.slope)
        / (increase.slope - decrease.slope)
    )
    print(
        "Crossing windows between increase/decrease segments: "
        f"{crossing:.2f} to {crossing_mod:.2f} (mod {period:.0f})"
    )

    double_series = double_sawtooth(t_values, period, increase, decrease)
    double_path = save_plot(
        t_values,
        [double_series],
        ["Double sawtooth"],
        "Piecewise double-sawtooth waveform",
        "double_sawtooth.png",
    )
    print(f"Saved double sawtooth figure -> {double_path}")

    print("Sawtooth exploration complete.")
