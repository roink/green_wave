"""Generate NDVI and IRG figures from fitted parameter grids.

This script replaces its original notebook export. It loads the fitted NDVI
parameters, prints basic diagnostics, and saves predicted NDVI as well as IRG
frames to dedicated sub-directories within ``figure/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def double_logistic(
    t: np.ndarray | float,
    xmid_s: np.ndarray,
    scal_s: np.ndarray,
    xmid_a: np.ndarray,
    scal_a: np.ndarray,
    bias: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Evaluate the double logistic curve used for NDVI fitting."""

    spring = 1.0 / (1.0 + np.exp((xmid_s - t) / scal_s))
    autumn = 1.0 / (1.0 + np.exp((xmid_a - t) / scal_a))
    return bias + scale * (spring - autumn)


def instantaneous_rate_of_greenup(
    t: np.ndarray | float,
    xmid_s: np.ndarray,
    scal_s: np.ndarray,
    *_: np.ndarray,
) -> np.ndarray:
    """Compute the instantaneous rate of green-up (IRG)."""

    spring = 1.0 / (1.0 + np.exp((xmid_s - t) / scal_s))
    return spring * (1.0 - spring)


def load_fit_parameters(path: Path) -> np.ndarray:
    """Load the NDVI fit parameter stack."""

    if not path.exists():
        raise FileNotFoundError(
            f"Could not locate fitted parameter archive: {path}\n"
            "Provide the path via --fit-params if the default is different."
        )
    data = np.load(path)
    if "ndvi_fit_all" not in data:
        raise KeyError("Expected 'ndvi_fit_all' array in the provided archive.")
    params = data["ndvi_fit_all"][..., :6]
    if params.ndim != 3:
        raise ValueError("Fit parameter array must have shape (rows, cols, 6).")
    return params


def save_frames(
    doys: Iterable[int],
    params: np.ndarray,
    output_dir: Path,
    generator,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
    prefix: str,
) -> None:
    """Render a grid of frames derived from the fitted parameters."""

    output_dir.mkdir(parents=True, exist_ok=True)
    param_summary = []
    for doy in doys:
        frame = generator(
            doy,
            params[..., 0],
            params[..., 1],
            params[..., 2],
            params[..., 3],
            params[..., 4],
            params[..., 5],
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_title(f"Predicted {label} {doy:03d}")
        frame_path = output_dir / f"{prefix}_{doy:03d}.png"
        fig.savefig(frame_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        param_summary.append((doy, np.nanmin(frame), np.nanmax(frame)))
        print(f"Saved {label} frame for DOY {doy:03d} -> {frame_path}")

    mins = [entry[1] for entry in param_summary]
    maxes = [entry[2] for entry in param_summary]
    print(
        f"{label} frame summary: min={np.nanmin(mins):.2f}, max={np.nanmax(maxes):.2f}"
    )


def save_parameter_map(params: np.ndarray, figure_dir: Path) -> None:
    """Persist a diagnostic map for the spring inflection point."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(params[..., 0], cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Spring inflection DOY")
    ax.set_title("xmidSNDVI parameter")
    output_path = figure_dir / "spring_inflection_map.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter diagnostic map -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-params",
        type=Path,
        default=Path("data/intermediate/ndvi_fit_params.npz"),
        help="Path to the npz archive containing the 'ndvi_fit_all' array.",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("figure"),
        help="Root directory where figures will be written.",
    )
    parser.add_argument(
        "--doy-step",
        type=int,
        default=16,
        help="Interval in days of year used when generating frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_fit_parameters(args.fit_params)
    rows, cols, bands = params.shape
    print(
        "Loaded NDVI fit parameters "
        f"({rows} x {cols} pixels, {bands} parameters per pixel)."
    )

    doys = np.arange(1, 366, args.doy_step, dtype=int)

    script_dir = args.figure_root / Path(__file__).stem
    ndvi_dir = script_dir / "predicted_ndvi"
    irg_dir = script_dir / "predicted_irg"

    save_frames(
        doys,
        params,
        ndvi_dir,
        double_logistic,
        cmap="RdYlGn",
        vmin=-2000,
        vmax=10000,
        label="NDVI",
        prefix="ndvi",
    )

    save_frames(
        doys,
        params,
        irg_dir,
        instantaneous_rate_of_greenup,
        cmap="RdYlGn",
        vmin=0,
        vmax=0.25,
        label="IRG",
        prefix="irg",
    )

    save_parameter_map(params, script_dir)
    print("All figures written successfully.")


if __name__ == "__main__":
    main()
