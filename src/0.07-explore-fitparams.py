#!/usr/bin/env python
"""Inspect and clean NDVI double-logistic fit parameters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "intermediate"
FILE_PATH = DATA_ROOT / "ndvi_fit_params.npz"
CLEANED_OUTPUT_PATH = DATA_ROOT / "ndvi_fit_params_cleaned.npz"
FIGURE_ROOT = PROJECT_ROOT / "figure" / Path(__file__).stem
MAP_DIR = FIGURE_ROOT / "maps"
PROFILE_DIR = FIGURE_ROOT / "profiles"

for directory in (MAP_DIR, PROFILE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = ["xmidSNDVI", "scalSNDVI", "xmidANDVI", "scalANDVI", "bias", "scale"]


def load_parameters() -> np.ndarray:
    data = np.load(FILE_PATH)
    ndvi_fit_params = data["ndvi_fit_all"]
    print(
        f"Loaded NDVI fit parameters with shape {ndvi_fit_params.shape} from {FILE_PATH}"
    )
    return ndvi_fit_params


def summarise_parameters(ndvi_fit_params: np.ndarray, title: str) -> None:
    print(f"\nParameter summary: {title}")
    for i, param in enumerate(PARAM_NAMES):
        param_data = ndvi_fit_params[:, :, i]
        print(
            f"  {param} -> min {np.nanmin(param_data):.2f}, max {np.nanmax(param_data):.2f}, "
            f"mean {np.nanmean(param_data):.2f}, median {np.nanmedian(param_data):.2f}, std {np.nanstd(param_data):.2f}"
        )


def apply_constraints(ndvi_fit_params: np.ndarray) -> np.ndarray:
    constraints = {
        0: (0, 365),  # xmidSNDVI
        2: (0, 365),  # xmidANDVI
        1: (1, 100),  # scalSNDVI
        3: (1, 100),  # scalANDVI
        4: (0, 10000),  # bias
        5: (0, 10000),  # scale
    }

    for idx, (lower, upper) in constraints.items():
        original_invalid = np.count_nonzero(~np.isnan(ndvi_fit_params[:, :, idx]))
        param_data = ndvi_fit_params[:, :, idx]
        param_data = np.where(
            (param_data >= lower) & (param_data <= upper), param_data, np.nan
        )
        ndvi_fit_params[:, :, idx] = param_data
        remaining = np.count_nonzero(~np.isnan(param_data))
        print(
            f"Applied bounds {lower}-{upper} to {PARAM_NAMES[idx]}: kept {remaining} / {original_invalid} values"
        )

    return ndvi_fit_params


def plot_param_map(
    ndvi_fit_params: np.ndarray, param_idx: int, title: str, cmap: str = "viridis"
) -> Path:
    param_data = ndvi_fit_params[:, :, param_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(param_data, cmap=cmap, interpolation="nearest")
    fig.colorbar(img, label=title, ax=ax)
    ax.set_title(title)

    output_path = MAP_DIR / f"{PARAM_NAMES[param_idx]}-map.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter map to {output_path}")
    return output_path


def plot_latitudinal_profile(ndvi_fit_params: np.ndarray) -> Path:
    lat_means = np.nanmean(ndvi_fit_params, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lat_means[:, 0], label="Spring Green-Up (xmidSNDVI)", color="green")
    ax.plot(lat_means[:, 2], label="Autumn Dry-Down (xmidANDVI)", color="orange")
    ax.set_xlabel("Latitude Index")
    ax.set_ylabel("Day of Year")
    ax.set_title("NDVI Phenology Across Latitude")
    ax.legend()
    ax.grid(True)

    output_path = PROFILE_DIR / "latitudinal-profile.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latitudinal profile to {output_path}")
    return output_path


def main() -> None:
    ndvi_fit_params = load_parameters()
    summarise_parameters(ndvi_fit_params, "raw values")

    ndvi_fit_params = apply_constraints(ndvi_fit_params)
    summarise_parameters(ndvi_fit_params, "after filtering")

    np.savez_compressed(CLEANED_OUTPUT_PATH, ndvi_fit_params=ndvi_fit_params)
    print(f"Saved cleaned NDVI parameters to {CLEANED_OUTPUT_PATH}")

    plot_param_map(ndvi_fit_params, 0, "Spring Green-Up (xmidSNDVI)", cmap="coolwarm")
    plot_param_map(ndvi_fit_params, 2, "Autumn Dry-Down (xmidANDVI)", cmap="coolwarm")
    plot_latitudinal_profile(ndvi_fit_params)


if __name__ == "__main__":
    main()
