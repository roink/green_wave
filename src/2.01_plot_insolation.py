#!/usr/bin/env python3
"""Generate diagnostic plots for insolation and orbital parameters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSOLATION_PATH = PROJECT_ROOT / "data" / "insolation" / "orbit91"
FIGURE_ROOT = PROJECT_ROOT / "figure"
SCRIPT_STEM = Path(__file__).stem
SCRIPT_FIGURE_DIR = FIGURE_ROOT / SCRIPT_STEM
SCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(" ", "_")
        .replace("/", "-")
        .replace(".", "p")
    )


def _save_figure(fig: plt.Figure, description: str) -> Path:
    filename = f"{SCRIPT_STEM}__{_slugify(description)}.png"
    path = SCRIPT_FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")
    return path


def unwrap(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    up = np.zeros_like(p)
    pm1 = p[0]
    up[0] = pm1
    po = 0.0
    thr = np.pi
    pi2 = 2 * np.pi
    for i in range(1, len(p)):
        cp = p[i] + po
        dp = cp - pm1
        while dp >= thr:
            po -= pi2
            dp -= pi2
        while dp <= -thr:
            po += pi2
            dp += pi2
        cp = p[i] + po
        pm1 = cp
        up[i] = cp
    return up


_INTERPOLATORS: dict[str, interp1d] | None = None
_KYEARS_FULL: np.ndarray | None = None
_ECC_FULL: np.ndarray | None = None
_EPS_FULL: np.ndarray | None = None
_OMG_FULL: np.ndarray | None = None


def _ensure_orbital_data_loaded() -> None:
    global _INTERPOLATORS, _KYEARS_FULL, _ECC_FULL, _EPS_FULL, _OMG_FULL

    if _INTERPOLATORS is not None:
        return

    if not INSOLATION_PATH.exists():
        raise FileNotFoundError(
            f"Orbital parameter file not found at {INSOLATION_PATH}"
        )

    print(f"Loading orbital parameters from {INSOLATION_PATH} …")
    ins_data = pd.read_csv(
        INSOLATION_PATH,
        sep=r"\s+",
        skiprows=2,
        usecols=[0, 1, 2, 3],
        header=None,
        names=["kyear", "ecc", "omega", "epsilon"],
    )

    kyear0 = -ins_data["kyear"].values
    ecc0 = ins_data["ecc"].values
    omega0 = ins_data["omega"].values + 180.0
    omega0u = unwrap(np.deg2rad(omega0))
    eps0 = ins_data["epsilon"].values

    f_ecc = interp1d(kyear0, ecc0, kind="cubic", fill_value="extrapolate")
    f_omega = interp1d(kyear0, omega0u, kind="cubic", fill_value="extrapolate")
    f_epsilon = interp1d(kyear0, eps0, kind="cubic", fill_value="extrapolate")

    _INTERPOLATORS = {"ecc": f_ecc, "omega": f_omega, "epsilon": f_epsilon}

    _KYEARS_FULL = np.arange(0, 50000 + 1) / 10.0
    _ECC_FULL = f_ecc(_KYEARS_FULL)
    _EPS_FULL = np.deg2rad(f_epsilon(_KYEARS_FULL))
    _OMG_FULL = f_omega(_KYEARS_FULL)

    print(f"Loaded {len(ins_data)} orbital parameter records.")


def orbital_parameters(
    kyear: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_orbital_data_loaded()
    assert _INTERPOLATORS is not None

    ecc = _INTERPOLATORS["ecc"](kyear)
    epsilon = np.deg2rad(_INTERPOLATORS["epsilon"](kyear))
    omega = _INTERPOLATORS["omega"](kyear)
    return ecc, epsilon, omega


def orbital_parameters_fast(kyear: float) -> tuple[float, float, float]:
    _ensure_orbital_data_loaded()
    assert _KYEARS_FULL is not None and _ECC_FULL is not None
    assert _EPS_FULL is not None and _OMG_FULL is not None

    idx = int(np.round(kyear * 10))
    return float(_ECC_FULL[idx]), float(_EPS_FULL[idx]), float(_OMG_FULL[idx])


def tlag(data: np.ndarray, ilag: int) -> np.ndarray:
    data = np.asarray(data)
    temp = np.tile(data, 3)
    start = len(data)
    end = start + len(data) - ilag
    return temp[start:end]


def daily_insolation_param(
    lat: float,
    day: np.ndarray,
    ecc: np.ndarray,
    obliquity: np.ndarray,
    long_perh: np.ndarray,
    day_type: int = 1,
) -> dict[str, np.ndarray]:
    ε = np.deg2rad(obliquity)
    ω = np.deg2rad(long_perh)
    φ = np.deg2rad(lat)
    day = np.asarray(day, dtype=float)

    if day_type == 1:
        Δλ_m = (day - 80.0) * 2 * np.pi / 365.2422
        β = np.sqrt(1 - ecc**2)
        λ_m0 = -2.0 * (
            (0.5 * ecc + 0.125 * ecc**3) * (1 + β) * np.sin(-ω)
            - 0.25 * ecc**2 * (0.5 + β) * np.sin(-2 * ω)
            + 0.125 * ecc**3 * (1 / 3 + β) * np.sin(-3 * ω)
        )
        λ_m = λ_m0 + Δλ_m
        λ = (
            λ_m
            + (2 * ecc - 0.25 * ecc**3) * np.sin(λ_m - ω)
            + 1.25 * ecc**2 * np.sin(2 * (λ_m - ω))
            + (13 / 12) * ecc**3 * np.sin(3 * (λ_m - ω))
        )
    elif day_type == 2:
        λ = day * 2 * np.pi / 360.0
    else:
        raise ValueError("Invalid day_type")

    So = 1365.0
    δ = np.arcsin(np.sin(ε) * np.sin(λ))
    H0 = np.arccos(-np.tan(φ) * np.tan(δ))

    mask1 = (np.abs(φ) >= (np.pi / 2 - np.abs(δ))) & (φ * δ > 0)
    mask2 = (np.abs(φ) >= (np.pi / 2 - np.abs(δ))) & (φ * δ <= 0)
    H0 = np.where(mask1, np.pi, H0)
    H0 = np.where(mask2, 0.0, H0)

    Fsw = (
        So
        / np.pi
        * (1 + ecc * np.cos(λ - ω)) ** 2
        / (1 - ecc**2) ** 2
        * (H0 * np.sin(φ) * np.sin(δ) + np.cos(φ) * np.cos(δ) * np.sin(H0))
    )

    return {
        "Fsw": Fsw,
        "ecc": ecc,
        "obliquity": obliquity,
        "long_perh": long_perh,
        "lambda": np.rad2deg(λ) % 360.0,
    }


def daily_insolation(
    kyear: float, lat: float, day: np.ndarray, day_type: int = 1, fast: bool = True
) -> dict:
    if fast:
        ecc, ε, ω = orbital_parameters_fast(kyear)
    else:
        ecc, ε, ω = orbital_parameters(kyear)

    return daily_insolation_param(
        lat,
        day,
        ecc,
        np.rad2deg(ε),
        np.rad2deg(ω),
        day_type=day_type,
    )


def annual_insolation(kyears: np.ndarray, lat: float) -> np.ndarray:
    kyears = np.atleast_1d(kyears)
    out = np.empty_like(kyears, dtype=float)
    for i, ky in enumerate(kyears):
        res = daily_insolation(ky, lat, np.arange(1, 366), day_type=1, fast=True)
        out[i] = np.mean(res["Fsw"])
    return out


def ins_march21(kyear: float, lat: float) -> np.ndarray:
    return daily_insolation(kyear, lat, np.arange(1, 366))["Fsw"]


def ins_dec21(kyear: float, lat: float) -> np.ndarray:
    r = daily_insolation(kyear, lat, np.arange(1, 366))
    lam = r["lambda"]
    shift = 355 - np.argmin(np.abs(lam - 270))
    return tlag(r["Fsw"], shift)


def ins_dec21_param(
    ecc: float, obliquity: float, long_perh: float, lat: float
) -> np.ndarray:
    r = daily_insolation_param(lat, np.arange(1, 366), ecc, obliquity, long_perh)
    shift = 355 - np.argmin(np.abs(r["lambda"] - 270))
    return tlag(r["Fsw"], shift)


def main() -> None:
    _ensure_orbital_data_loaded()

    june65 = np.array([daily_insolation(i, 65, 172)["Fsw"] for i in range(1, 5001)])
    times = -np.arange(1, 5001)
    print(f"Computed June 21 insolation at 65°N for {len(june65)} kyr.")
    print(f"Mean insolation over this period: {june65.mean():.2f} W/m²")

    fig, ax = plt.subplots()
    ax.plot(times, june65, "r-")
    ax.set_xlabel("kyr BP")
    ax.set_ylabel("Insolation (W/m²)")
    ax.set_title("June 21 Insolation at 65°N")
    ax.grid(True)
    _save_figure(fig, "june21_65N_timeseries")

    wave = june65.copy()
    wave[wave > 510] = 510
    fig, ax = plt.subplots()
    ax.plot(times, wave, "-")
    ax.axhline(june65.mean(), linestyle="--", color="black", label="Mean")
    ax.set_xlabel("kyr BP")
    ax.set_ylabel("Insolation (W/m²)")
    ax.set_title("Clipped Insolation at 65°N (June 21)")
    ax.legend()
    ax.grid(True)
    _save_figure(fig, "june21_65N_clipped")

    ecc_new = np.array([daily_insolation(i, 65, 172)["ecc"] for i in range(1, 5001)])
    obliq_new = np.array(
        [daily_insolation(i, 65, 172)["obliquity"] for i in range(1, 5001)]
    )
    perihel_new = np.array(
        [daily_insolation(i, 65, 172)["lambda"] for i in range(1, 5001)]
    )

    fig, ax = plt.subplots()
    ax.plot(times, ecc_new, "r-")
    ax.set_title("Eccentricity")
    ax.set_xlabel("kyr BP")
    ax.set_ylabel("Eccentricity")
    ax.grid(True)
    _save_figure(fig, "orbital_eccentricity")

    fig, ax = plt.subplots()
    ax.plot(times, obliq_new, "b-")
    ax.set_title("Obliquity")
    ax.set_xlabel("kyr BP")
    ax.set_ylabel("Degrees")
    ax.grid(True)
    _save_figure(fig, "orbital_obliquity")

    fig, ax = plt.subplots()
    ax.plot(times, perihel_new, "k-")
    ax.set_title("Solar Longitude λ")
    ax.set_xlabel("kyr BP")
    ax.set_ylabel("Degrees")
    ax.grid(True)
    _save_figure(fig, "orbital_solar_longitude")

    threekBP65 = np.array([daily_insolation(0, 65, i)["Fsw"] for i in range(1, 365)])
    fifteenBP65 = np.array([daily_insolation(12, 65, i)["Fsw"] for i in range(1, 365)])
    thirtyBP65 = np.array([daily_insolation(5, 65, i)["Fsw"] for i in range(1, 365)])
    days = np.arange(1, 365)

    fig, ax = plt.subplots()
    ax.plot(days, threekBP65, "r-", label="0 kyr BP")
    ax.plot(days, fifteenBP65, "b-", label="12 kyr BP")
    ax.plot(days, thirtyBP65, "g-", label="5 kyr BP")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Daily Insolation (W/m²)")
    ax.set_title("Daily Insolation at 65°N")
    ax.legend()
    ax.grid(True)
    _save_figure(fig, "daily_insolation_65N_comparison")


if __name__ == "__main__":
    main()
