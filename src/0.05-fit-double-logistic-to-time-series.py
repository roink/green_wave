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

    # Replace missing values in winter period (DOY 300-365 & 1-60)
    for i, date in enumerate(dates):
        doy = date.day_of_year
        if np.isnan(corrected_ndvi[i]) and (doy >= 300 or doy <= 60):
            corrected_ndvi[i] = winter_ndvi

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


# -------------------------------
# Double-Logistic Function for Seasonal Fitting
# -------------------------------
def double_logistic(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """
    Double-logistic function for fitting annual NDVI profiles.
    t: Day of year (DOY)
    xmidSNDVI, scalSNDVI: Spring inflection point & scale
    xmidANDVI, scalANDVI: Autumn inflection point & scale
    bias: Baseline NDVI
    scale: Scaling factor to adjust NDVI amplitude
    """
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


# In[7]:


# -------------------------------
# Function to Fit Seasonal NDVI Curve
# -------------------------------
def fit_seasonal_curve(lat, lon, selected_year):
    """Fit a double-logistic curve to NDVI time series for a given year."""
    
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    # Create a mask for the selected year
    year_mask = [date.year == selected_year for date in dates]

    # Extract DOY and NDVI values for the selected year
    doy_values = np.array([date.day_of_year for i, date in enumerate(dates) if year_mask[i]])
    ndvi_values = np.array([filtered_ndvi[i] for i in range(len(filtered_ndvi)) if year_mask[i]])

    # Remove NaNs (curve fitting requires finite values)
    valid_mask = ~np.isnan(ndvi_values)
    doy_values = doy_values[valid_mask]
    ndvi_values = ndvi_values[valid_mask]

    # Initial parameter guesses
    initial_guess = [120, 20, 270, 25, np.min(ndvi_values), np.max(ndvi_values) - np.min(ndvi_values)]

    # Fit the double-logistic function
    params, _ = curve_fit(double_logistic, doy_values, ndvi_values, p0=initial_guess)

    # Generate fitted curve
    doy_full = np.arange(1, 366)  # Full year for smooth plotting
    ndvi_fitted = double_logistic(doy_full, *params)

    # Plot fitted curve
    plt.figure(figsize=(10, 5))
    plt.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
    plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")
    plt.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    plt.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")

    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Seasonal Curve Fitting - {selected_year} at ({lat}°N, {lon}°E)")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[8]:


plot_ndvi(60.0, 16)  # NDVI at 60°N, 16°E


# In[9]:


fit_seasonal_curve(60.0, 16, selected_year=2005)  # Fit seasonal cycle for 2005


# In[10]:


# -------------------------------
# Function to Fit Seasonal NDVI Curve for Multiple Years (2002-2010)
# -------------------------------
def fit_seasonal_curve_all_years(lat, lon, start_year=2002, end_year=2010, save = False):
    """Fit a double-logistic curve to NDVI time series for multiple years."""
    
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    doy_values = np.array([date.day_of_year for i, date in enumerate(dates) if start_year <= date.year <= end_year])
    ndvi_values = np.array([filtered_ndvi[i] for i in range(len(filtered_ndvi)) if start_year <= dates[i].year <= end_year])

    # Remove NaNs
    valid_mask = ~np.isnan(ndvi_values)
    doy_values = doy_values[valid_mask]
    ndvi_values = ndvi_values[valid_mask]

    # Fit the function
    initial_guess = [120, 20, 270, 25, np.min(ndvi_values), np.max(ndvi_values) - np.min(ndvi_values)]
    params, _ = curve_fit(double_logistic, doy_values, ndvi_values, p0=initial_guess)

    # Generate fitted curve
    doy_full = np.arange(1, 366)
    ndvi_fitted = double_logistic(doy_full, *params)

    # Plot fitted curve
    plt.figure(figsize=(10, 5))
    plt.scatter(doy_values, ndvi_values, color="black", alpha=0.3, label="Observed NDVI")
    plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", linewidth=2, label="Fitted Double-Logistic Curve")
    plt.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    plt.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")

    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Seasonal Curve Fitting ({start_year}-{end_year}) at ({lat}°N, {lon}°E)")
    plt.legend()
    plt.grid(True)
        
    if save:
        # Save the figure
        output_path = "../figures/double_logistic_fit"
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')                            
                                 
    plt.show()


# In[11]:


fit_seasonal_curve_all_years(60.0, 16, 2000, 2024, save = True)


# In[12]:


fit_seasonal_curve_all_years(50.0, 16, 2000, 2024)


# In[13]:


fit_seasonal_curve_all_years(40.0, 16, 2000, 2024)


# In[14]:


fit_seasonal_curve_all_years(30.0, 16, 2000, 2024)


# In[15]:


fit_seasonal_curve_all_years(20.0, 16, 2000, 2024)


# In[16]:


def shift_doy(doy_values, peak_doy, center_doy=183):
    """Shift DOY values such that the peak DOY is at the center of the year."""
    shifted_doy = (doy_values - peak_doy + center_doy) % 365
    return shifted_doy


def fit_seasonal_curve_transformed(lat, lon, start_year=2002, end_year=2024):
    """Fit a double-logistic curve using transformed time for better seasonal alignment."""
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)
    
    # Extract data for selected years
    mask = np.array([start_year <= date.year <= end_year for date in dates])
    dates = np.array(dates)[mask]
    ndvi_values = np.array(filtered_ndvi)[mask]
    doy_values = np.array([date.day_of_year for date in dates])
    
    # Determine peak DOY
    peak_doy = doy_values[np.nanargmax(ndvi_values)]
    
    # Transform time
    doy_shifted = shift_doy(doy_values, peak_doy)
    
    # Fit the double-logistic function
    initial_guess = [120, 20, 270, 25, np.min(ndvi_values), np.max(ndvi_values) - np.min(ndvi_values)]
    params, _ = curve_fit(double_logistic, doy_shifted, ndvi_values, p0=initial_guess,maxfev = 14000)
    
    # Generate fitted curve in transformed time
    doy_full = np.arange(1, 366)
    ndvi_fitted_shifted = double_logistic(doy_full, *params)
    
    # Revert the transformation for visualization
    doy_original = shift_doy(doy_full, -peak_doy)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(doy_values, ndvi_values, color="black", alpha=0.3, label="Observed NDVI")
    plt.plot(doy_original[:-1], ndvi_fitted_shifted[:-1], color="red", linestyle="--", linewidth=2, label="Fitted Double-Logistic Curve")
    plt.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    plt.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")
    
    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Seasonal Curve Fitting with Time Shift ({start_year}-{end_year}) at ({lat}°N, {lon}°E)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
fit_seasonal_curve_transformed(60.0, 16, 2000, 2024)


# In[17]:


fit_seasonal_curve_transformed(50.0, 16, 2000, 2024)


# In[18]:


fit_seasonal_curve_transformed(40.0, 16, 2000, 2024)


# In[19]:


fit_seasonal_curve_transformed(30.0, 16, 2000, 2024)


# In[ ]:




