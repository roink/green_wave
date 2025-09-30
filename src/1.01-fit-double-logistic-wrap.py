#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit


# In[2]:


# Path to HDF5 file
hdf5_file = "/work/pschluet/green_wave/data/intermediate/ndvi_stack_optimized.h5"


# In[3]:


# -------------------------------
# Function to extract NDVI time series for a given lat/lon
# -------------------------------
def get_ndvi_timeseries(lat, lon):
    """Extract NDVI time series from HDF5 for a given latitude & longitude."""
    
    # Convert latitude & longitude to row/column indices
    row_idx = int((90 - lat) / 0.05)  # Convert latitude to row
    col_idx = int((lon + 180) / 0.05)  # Convert longitude to column
    print(f"Nearest pixel index: row={row_idx}, col={col_idx}")

    # Open HDF5 file and read metadata and NDVI time series
    with h5py.File(hdf5_file, "r") as h5f:
        metadata = h5f["metadata"][:]  # Load (year, doy)
        ndvi_timeseries = h5f["ndvi_stack"][:, row_idx, col_idx]  # Load only the required pixel

    # Convert metadata (year, DOY) to actual dates
    dates = [pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata]

    return dates, ndvi_timeseries


# In[4]:


# -------------------------------
# Function to process NDVI time series
# -------------------------------
def process_ndvi(dates, ndvi_timeseries):
    """Apply winter correction and moving median filter to NDVI time series."""

    # Compute 2.5% quantile (winter baseline)
    winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)

    # Apply winter NDVI correction (clip values below threshold)
    corrected_ndvi = np.copy(ndvi_timeseries)
    corrected_ndvi[ndvi_timeseries < winter_ndvi] = winter_ndvi

   # # Replace missing values in winter period (DOY 300-365 & 1-60)
   # for i, date in enumerate(dates):
   #     doy = date.day_of_year
   #     if np.isnan(corrected_ndvi[i]) and (doy >= 300 or doy <= 60):
   #         corrected_ndvi[i] = winter_ndvi

    # Apply Moving Median Filter (window size = 3)
    filtered_ndvi = median_filter(corrected_ndvi, size=3)

    return winter_ndvi, corrected_ndvi, filtered_ndvi


# In[5]:


# -------------------------------
# Function to plot NDVI time series
# -------------------------------
def plot_ndvi(lat, lon):
    """Extract, process, and plot NDVI time series for a given location."""
    
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    winter_ndvi, corrected_ndvi, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    plt.figure(figsize=(12, 6))

    # Raw NDVI
    plt.plot(dates, ndvi_timeseries, marker="o", linestyle="-", color="gray", alpha=0.6, label="Raw NDVI")

    # Winter NDVI baseline
    plt.axhline(y=winter_ndvi, color="blue", linestyle="--", label=f"Winter NDVI ({winter_ndvi:.3f})")

    # Corrected NDVI (baseline applied)
    plt.plot(dates, corrected_ndvi, marker="o", linestyle="-", color="green", label="Corrected NDVI")

    # Filtered NDVI
    plt.plot(dates, filtered_ndvi, marker="o", linestyle="-", color="red", label="Filtered NDVI (Moving Median)")

    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[6]:


plot_ndvi(60, 16)


# In[7]:


plot_ndvi(50, 16)


# In[44]:


def sigmoid_double_sawtooth(
    t, z1, d1, z2, d2, bias, scale, period=365,return_derivative=False
):
    """
    Compute s(t) = sigmoid(double_sawtooth(t)) in one pass.
    Optionally also return ds/dt = d/dt [sigmoid(double_sawtooth(t))].

    Parameters
    ----------
    t : array-like
        Time points at which to evaluate.
    period : float
        The period (e.g. 365).
    z1, d1 : float
        Offset & slope for the increasing sawtooth (st1).
    z2, d2 : float
        Offset & slope for the decreasing sawtooth (st2).
    return_derivative : bool, optional
        If True, also return ds/dt in addition to s(t).

    Returns
    -------
    s_values : ndarray
        The values of sigmoid(double_sawtooth(t)).
    ds_dt_values : ndarray (only if return_derivative=True)
        The derivative d/dt [sigmoid(double_sawtooth(t))].
    """
    z1 = z1 % 365
    z2 = z2 % 365
    d1 = 1/d1
    d2 = 1/d2
    t = np.asarray(t)

    # 1) Find the two crossing times in [0, period)
    c1, c2 = _get_crossings(z1, d1, z2, d2, period)

    # 2) Evaluate st1, st2 at the crossing times c1, c2 to see which is "lowest" or "highest"
    val1 = _sawtooth_zp_d(c1, period, z1, d1)
    val2 = _sawtooth_zp_d(c2, period, z1, d1)

    # We'll define the piecewise intervals:
    modt = t % period
    in_first  = (modt < c1)
    in_second = (modt >= c1) & (modt < c2)
    # The complement is the third interval: modt >= c2

    # Pre-allocate outputs
    x_vals = np.zeros_like(t)  # double_sawtooth(t)
    if return_derivative:
        xprime_vals = np.zeros_like(t)  # piecewise slope d1 or d2

    # 3) Fill in piecewise segments
    # If val1 < val2 => c1 is 'lowest crossing', c2 is 'highest crossing',
    # => st1 used on [c1,c2), st2 on [0,c1) & [c2,P).
    # Otherwise st2 used on [c1,c2), st1 on outside.
    if val1 < val2:
        # st2 in [0,c1) and [c2,P), st1 in [c1,c2)
        x_vals[in_first]  = _sawtooth_zp_d(t[in_first],  period, z2, d2)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z1, d1)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(t[~(in_first | in_second)], period, z2, d2)

        if return_derivative:
            xprime_vals[in_first]  = d2
            xprime_vals[in_second] = d1
            xprime_vals[~(in_first | in_second)] = d2
    else:
        # st1 in [0,c1) and [c2,P), st2 in [c1,c2)
        x_vals[in_first]  = _sawtooth_zp_d(t[in_first],  period, z1, d1)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z2, d2)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(t[~(in_first | in_second)], period, z1, d1)

        if return_derivative:
            xprime_vals[in_first]  = d1
            xprime_vals[in_second] = d2
            xprime_vals[~(in_first | in_second)] = d1

    # 4) Sigmoid
    s_vals = _logistic(x_vals)

    if return_derivative:
        # dsigma/dx = s * (1 - s)
        dsigma_dx = s_vals * (1 - s_vals)
        # By chain rule: d/dt [sigma(x(t))] = dsigma/dx * dx/dt
        ds_dt = dsigma_dx * xprime_vals
        return s_vals*scale+bias, ds_dt*scale

    else:
        # Just return the sigmoid
        return s_vals*scale+bias


# --- Small helpers (underscored to show they are "private") ---
def _sawtooth_zp_d(tt, period, z, d):
    """Compute the standard sawtooth at times tt, with offset z, slope d."""
    return (((tt - z - period/2) % period) - period/2) * d

def _logistic(x):
    """Sigmoid."""
    return 1 / (1 + np.exp(-x))

def _get_crossings(z1, d1, z2, d2, period=365):
    """
    Return the two crossing times c1, c2 in ascending order.
    If fewer than 2, just pick c1=0, c2=period/2 to avoid errors.
    """
    crossings = _find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        return (0.0, period / 2)
    return crossings[0], crossings[1]

def _find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """Analytic intersection solver, same logic as before."""
    z1_mod = z1 % period
    z2_mod = z2 % period

    r = d2 / d1
    dprime = (z1_mod - z2_mod) % period

    # T1
    num1 = r*dprime + (period/2)*(1 - r)
    den = (1 - r)
    T1 = num1 / den

    # T2
    num2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = num2 / den

    solutions = []
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period/2) + T1) % period
        solutions.append(c1)
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period/2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions


# In[62]:


# -------------------------------
# Function to fit sigmoid_double_sawtooth to NDVI data
# -------------------------------
def fit_sigmoid_double_sawtooth(lat, lon):
    """Fit sigmoid_double_sawtooth function to NDVI time series at given location."""
    
    # Extract and process NDVI data
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    winter_ndvi, corrected_ndvi, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    # Convert dates to numeric values (days of year)
    doy = np.array([date.day_of_year for date in dates])

    # Initial parameter guesses: (z1, d1, z2, d2, bias, scale)
    
    


    # Interpolate NaN values
    valid_mask = ~np.isnan(filtered_ndvi) & ~np.isinf(filtered_ndvi)
    doy_valid = doy[valid_mask]
    ndvi_valid = filtered_ndvi[valid_mask]
    
    initial_guess = [100, 50, 250, -50, np.mean(ndvi_valid), np.ptp(ndvi_valid)]

    # Fit the model
    try:
        params, _ = curve_fit(
            sigmoid_double_sawtooth, doy_valid, ndvi_valid,
            p0=initial_guess, bounds=([-np.inf, 0, -np.inf, -np.inf, -np.inf, 0.00001], [np.inf, np.inf, np.inf, 0, np.inf, np.inf])
        )
        print("Optimized Parameters:", params)
    except RuntimeError as e:
        print("Fit failed:", e)
        return None

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(doy, filtered_ndvi, label="Filtered NDVI", color="red", alpha=0.6)
    t_fit = np.linspace(1, 365, 1000)
    plt.plot(t_fit, sigmoid_double_sawtooth(t_fit, *params), label="Fitted Curve", color="blue")
    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Fit at ({lat}°N, {lon}°E)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return params

# Example usage
fit_params = fit_sigmoid_double_sawtooth(60, 16)


# In[63]:


fit_params = fit_sigmoid_double_sawtooth(61, 16)


# In[64]:


fit_params = fit_sigmoid_double_sawtooth(59, 16)


# In[65]:


fit_params = fit_sigmoid_double_sawtooth(60, 15)


# In[66]:


fit_params = fit_sigmoid_double_sawtooth(50, 16)


# In[67]:


fit_params = fit_sigmoid_double_sawtooth(40, 16)


# In[68]:


fit_params = fit_sigmoid_double_sawtooth(30, 16)


# In[69]:


fit_params = fit_sigmoid_double_sawtooth(20, 16)


# In[70]:


fit_params = fit_sigmoid_double_sawtooth(-33, 27)


# In[ ]:




