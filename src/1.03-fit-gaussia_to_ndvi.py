#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

# Path to HDF5 file
directory = "/net/sno/pschluet/green_wave/data/intermediate"
hdf5_file = os.path.join(directory, "ndvi_stack_optimized.h5")

# -------------------------------
# Function to extract NDVI time series for a given lat/lon
# -------------------------------
def get_ndvi_timeseries(lat, lon):
    """Extract NDVI time series from HDF5 for a given latitude & longitude."""
    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Nearest pixel index: row={row_idx}, col={col_idx}")

    with h5py.File(hdf5_file, "r") as h5f:
        metadata = h5f["metadata"][:]  # (year, doy)
        ndvi_stack = h5f["ndvi_stack"][:, row_idx, col_idx]

    dates = [pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata]
    return np.array([d.day_of_year for d in dates]), ndvi_stack

# -------------------------------
# Function to process NDVI time series
# -------------------------------
def process_ndvi(doy, ndvi_timeseries):
    """Apply winter correction and moving median filter to NDVI time series."""
    winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)
    corrected = np.copy(ndvi_timeseries)
    corrected[np.isnan(corrected)] = winter_ndvi
    corrected[corrected < winter_ndvi] = winter_ndvi

    filtered = median_filter(corrected, size=3)
    #filtered = (filtered - np.nanquantile(filtered, 0.025))
    #filtered = filtered / np.nanquantile(filtered, 0.995)
    return winter_ndvi, corrected, filtered

# -------------------------------
# Periodic Gaussian (wrapped) definition
# -------------------------------
def gaussian_cyclic(t, A, mu, sigma, C, period=365):
    """
    Periodic (wrapped) Gaussian function on [0, period).
    t : array-like days of year
    A : amplitude
    mu : peak location (day of year)
    sigma : standard deviation
    C : baseline offset
    """
    # compute minimal distance on circle
    delta = ((t - mu + period/2) % period) - period/2
    return C + A * np.exp(-0.5 * (delta / sigma)**2)

# -------------------------------
# Function to calculate Gaussian fit and R-squared
# -------------------------------
def calculate_gaussian_fit(lat, lon):
    """Calculate cyclic Gaussian fit parameters for NDVI at a given location.

    Returns:
        popt : array, fitted parameters [A, mu, sigma, C]
        r_squared : float, coefficient of determination
    """
    # load and process
    doy, ndvi = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(doy, ndvi)

    # mask invalid
    mask = ~np.isnan(filtered_ndvi) & ~np.isinf(filtered_ndvi)
    x = doy[mask]
    y = filtered_ndvi[mask]

    # initial guesses
    A0 = np.nanmax(y) - np.nanmin(y)
    mu0 = x[np.argmax(y)]
    sigma0 = 30
    C0 = np.nanmin(y)
    p0 = [A0, mu0, sigma0, C0]
    bounds = ([0, -365, 0, -np.inf], [np.inf, 365, 365, np.inf])

    # fit
    try:
        popt, pcov = curve_fit(gaussian_cyclic, x, y, p0=p0, bounds=bounds)
    except RuntimeError as e:
        print("Gaussian fit failed:", e)
        return None, None

    [A_opt, mu_opt, sigma_opt, C_opt] = popt
    popt = [A_opt, mu_opt % 365, sigma_opt, C_opt]
    # compute R-squared
    y_fit = gaussian_cyclic(x, *popt)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res/ss_tot

    return popt, r_squared

# -------------------------------
# Function to plot data and Gaussian fit
# -------------------------------
def plot_gaussian_fit(lat, lon, popt, r_squared=None):
    """Plot filtered NDVI and fitted cyclic Gaussian at a given location."""
    # reload data
    doy, ndvi = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(doy, ndvi)

    # unpack params
    A_fit, mu_fit, sigma_fit, C_fit = popt

    # scatter
    plt.figure(figsize=(12,6))
    plt.scatter(doy, filtered_ndvi, color='grey', alpha=0.4, label='Filtered NDVI')

    # fit curve
    t_fit = np.linspace(1, 365, 1000)
    y_fit = gaussian_cyclic(t_fit, *popt)
    plt.plot(t_fit, y_fit, 'r--', lw=2, label='Gaussian Fit')
    plt.axvline(mu_fit, color='blue', ls=':', lw=2, label='Peak Day')

    # labels and title
    plt.xlabel('Day of Year')
    plt.ylabel('Normalized NDVI')
    title = f'Gaussian Fit (cyclic) at ({lat}°N, {lon}°E)'
    if r_squared is not None:
        title += f' — $R^2$={r_squared:.3f}'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


coords = [(60, 16), (50, 16), (30, 16)]
for lat, lon in coords:
    print(f"\nCalculating fit at ({lat}, {lon})")
    popt, r2 = calculate_gaussian_fit(lat, lon)
    if popt is not None:
        print(f"Fitted params: A={popt[0]:.3f}, mu={popt[1]:.1f}, sigma={popt[2]:.1f}, C={popt[3]:.3f}")
        print(f"R-squared: {r2:.3f}")
        plot_gaussian_fit(lat, lon, popt, r2)
    else:
        print("Fit failed for this location.")


# In[ ]:




