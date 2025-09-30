#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# In[4]:


# === unwrap function (phase‐unwrap for perihelion longitude) ===
def unwrap(p):
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

# === Load and interpolate orbital parameters ===
_ins_data = pd.read_csv(
    '/work/pschluet/green_wave/data/insolation/orbit91',
    sep=r'\s+',            # regex for any amount of whitespace
    skiprows=2,            # skip the “INSOLATION” header and the column‐names line
    usecols=[0, 1, 2, 3],  # only the first four fields: kyear, ecc, omega, obliquity
    header=None,
    names=['kyear', 'ecc', 'omega', 'epsilon']
)

_kyear0 = -_ins_data['kyear'].values
_ecc0    =  _ins_data['ecc'].values
_omega0  =  _ins_data['omega'].values + 180.0         # R adds 180°
_omega0u =  unwrap(np.deg2rad(_omega0))              # convert to radians & unwrap
_eps0    =  _ins_data['epsilon'].values              # in degrees

# cubic spline interpolators (extrapolate beyond ends if needed)
_f_ecc     = interp1d(_kyear0, _ecc0,    kind='cubic', fill_value='extrapolate')
_f_omega   = interp1d(_kyear0, _omega0u, kind='cubic', fill_value='extrapolate')
_f_epsilon = interp1d(_kyear0, _eps0,    kind='cubic', fill_value='extrapolate')


# In[5]:


def orbital_parameters(kyear):
    """
    Return (ecc, epsilon_rad, omega_rad) for any kyear (kyr BP).
    """
    ecc     = _f_ecc(kyear)
    epsilon = np.deg2rad(_f_epsilon(kyear))
    omega   = _f_omega(kyear)   # already in radians
    return ecc, epsilon, omega

# Pre‐compute global arrays for fast lookup (every 0.1 kyr from 0 to 5 Myr)
_kyears_full = np.arange(0, 50000 + 1) / 10.0
_ecc_full, _eps_full, _omg_full = orbital_parameters(_kyears_full)

def orbital_parameters_fast(kyear):
    """
    Fast lookup of (ecc, epsilon_rad, omega_rad) assuming kyear*10 is integer.
    """
    idx = int(np.round(kyear * 10))
    return _ecc_full[idx], _eps_full[idx], _omg_full[idx]

def tlag(data, ilag):
    """
    Replicate R's tlag: tile data 3×, then take slice [365:730-ilag] (0‐based).
    """
    data = np.asarray(data)
    temp = np.tile(data, 3)
    start = len(data)
    end   = start + len(data) - ilag
    return temp[start:end]

def daily_insolation_param(lat, day, ecc, obliquity, long_perh, day_type=1):
    """
    Translated from R's daily_insolation_param.
    lat in degrees, day scalar or array,
    ecc scalar or array, obliquity & long_perh in degrees.
    Returns dict with keys 'Fsw','ecc','obliquity','long_perh','lambda'.
    """
    ε = np.deg2rad(obliquity)
    ω = np.deg2rad(long_perh)
    φ = np.deg2rad(lat)
    day = np.asarray(day, dtype=float)

    # solar longitude λ
    if day_type == 1:
        Δλ_m = (day - 80.0) * 2*np.pi / 365.2422
        β    = np.sqrt(1 - ecc**2)
        λ_m0 = -2.0 * (
            (0.5*ecc + 0.125*ecc**3)*(1+β)*np.sin(-ω)
            - 0.25*ecc**2*(0.5+β)*np.sin(-2*ω)
            + 0.125*ecc**3*(1/3+β)*np.sin(-3*ω)
        )
        λ_m  = λ_m0 + Δλ_m
        λ    = (
            λ_m
            + (2*ecc - 0.25*ecc**3)*np.sin(λ_m - ω)
            + 1.25*ecc**2*np.sin(2*(λ_m - ω))
            + (13/12)*ecc**3*np.sin(3*(λ_m - ω))
        )
    elif day_type == 2:
        λ = day * 2*np.pi/360.0
    else:
        raise ValueError("Invalid day_type")

    So = 1365.0
    δ  = np.arcsin(np.sin(ε)*np.sin(λ))
    H0 = np.arccos(-np.tan(φ)*np.tan(δ))

    # no sunrise / no sunset adjustments
    mask1 = (np.abs(φ) >= (np.pi/2 - np.abs(δ))) & (φ*δ > 0)
    mask2 = (np.abs(φ) >= (np.pi/2 - np.abs(δ))) & (φ*δ <= 0)
    H0 = np.where(mask1, np.pi, H0)
    H0 = np.where(mask2, 0.0, H0)

    Fsw = (
        So/np.pi
        * (1 + ecc*np.cos(λ - ω))**2
        / (1 - ecc**2)**2
        * (H0*np.sin(φ)*np.sin(δ) + np.cos(φ)*np.cos(δ)*np.sin(H0))
    )

    return {
        'Fsw':        Fsw,
        'ecc':        ecc,
        'obliquity':  obliquity,
        'long_perh':  long_perh,
        'lambda':     np.rad2deg(λ) % 360.0
    }

def daily_insolation(kyear, lat, day, day_type=1, fast=True):
    """
    Translated from R's daily_insolation.
    kyear scalar or array, lat in degrees,
    day scalar or array of days 1–365.24 or solar longitudes.
    """
    if fast:
        ecc, ε, ω = orbital_parameters_fast(kyear)
    else:
        ecc, ε, ω = orbital_parameters(kyear)

    # reuse daily_insolation_param by converting back to degrees
    return daily_insolation_param(
        lat, day,
        ecc, np.rad2deg(ε), np.rad2deg(ω),
        day_type=day_type
    )

def annual_insolation(kyears, lat):
    """
    Equivalent to R's annual_insolation: mean over days 1–365.
    """
    kyears = np.atleast_1d(kyears)
    out = np.empty_like(kyears, dtype=float)
    for i, ky in enumerate(kyears):
        res = daily_insolation(ky, lat, np.arange(1,366), day_type=1, fast=True)
        out[i] = np.mean(res['Fsw'])
    return out

def ins_march21(kyear, lat):
    """Daily Fsw for days 1–365, March‐21 calendar."""
    return daily_insolation(kyear, lat, np.arange(1,366))['Fsw']

def ins_dec21(kyear, lat):
    """Align so that day=1 corresponds to Dec‐21 solstice."""
    r = daily_insolation(kyear, lat, np.arange(1,366))
    lam = r['lambda']
    shift = 355 - np.argmin(np.abs(lam - 270))
    return tlag(r['Fsw'], shift)

def ins_dec21_param(ecc, obliquity, long_perh, lat):
    r = daily_insolation_param(lat, np.arange(1,366), ecc, obliquity, long_perh)
    shift = 355 - np.argmin(np.abs(r['lambda'] - 270))
    return tlag(r['Fsw'], shift)


# In[6]:


# June 21 at 65°N over last 5 Myr
june65 = np.array([daily_insolation(i, 65, 172)['Fsw'] for i in range(1,5001)])
times = -np.arange(1,5001)

plt.figure()
plt.plot(times, june65, 'r-')
plt.xlabel('kyr BP')
plt.ylabel('Insolation (W/m²)')
plt.title('June 21 at 65°N')
plt.show()


# In[7]:


# Clip wave at 510, plot mean line
wave = june65.copy()
wave[wave > 510] = 510
plt.figure()
plt.plot(times, wave, '-')
plt.axhline(june65.mean(), linestyle='--')
plt.title('Clipped Insolation, 65°N on June 21')
plt.show()


# In[8]:


# Plot orbital parameters
ecc_new       = np.array([daily_insolation(i,65,172)['ecc'] for i in range(1,5001)])
obliq_new     = np.array([daily_insolation(i,65,172)['obliquity'] for i in range(1,5001)])
perihel_new   = np.array([daily_insolation(i,65,172)['lambda'] for i in range(1,5001)])
plt.figure(); plt.plot(times, ecc_new, 'r-'); plt.title('Eccentricity'); plt.show()
plt.figure(); plt.plot(times, obliq_new, 'b-'); plt.title('Obliquity (deg)'); plt.show()
plt.figure(); plt.plot(times, perihel_new, 'k-'); plt.title('Solar Longitude λ'); plt.show()


# In[16]:


# June 21 at 65°N over last 5 Myr
threekBP65 = np.array([daily_insolation(0, 65, i)['Fsw'] for i in range(1,365)])
thirtyBP65 = np.array([daily_insolation(5, 65, i)['Fsw'] for i in range(1,365)])

fifteenBP65 = np.array([daily_insolation(12, 65, i)['Fsw'] for i in range(1,365)])

times = np.arange(1,365)

plt.figure()
plt.plot(times, threekBP65, 'r-')
plt.plot(times, fifteenBP65, 'b-')
plt.plot(times, thirtyBP65, 'g-')
plt.xlabel('kyr BP')
plt.ylabel('Daily Insolation (W/m²)')
plt.title('June 21 at 65°N')
plt.show()


# In[ ]:




