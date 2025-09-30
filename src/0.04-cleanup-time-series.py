#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


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
def plot_ndvi(lat, lon, save = False):
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
    
    if save:
        # Save the figure
        output_path = "../figures/ndvi_preprocessing_steps"
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')
    
    plt.show()


# In[6]:


# -------------------------------
# Plot NDVI Time Series for Given Locations
# -------------------------------
plot_ndvi(40.0, 16)  # NDVI at 40°N, 16°E


# In[7]:


plot_ndvi(50.0, 16)  # NDVI at 50°N, 16°E


# In[8]:


plot_ndvi(60.0, 16, save = True)  # NDVI at 60°N, 16°E


# In[9]:


# -------------------------------
# Compare NDVI Across Years
# -------------------------------
def compare_yearly_ndvi(lat, lon, year_start=2002, year_end=2010):
    """Compare yearly NDVI patterns for a given latitude & longitude."""
    
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    ndvi_by_year = {}
    for i, date in enumerate(dates):
        year, doy = date.year, date.day_of_year
        if year_start <= year <= year_end:
            if year not in ndvi_by_year:
                ndvi_by_year[year] = np.full(366, np.nan)  # Space for leap years
            ndvi_by_year[year][doy - 1] = filtered_ndvi[i]  # Store NDVI by DOY

    # Plot yearly comparison
    plt.figure(figsize=(12, 6))
    for year, ndvi_values in ndvi_by_year.items():
        mask = ~np.isnan(ndvi_values)
        if np.any(mask):
            plt.plot(np.arange(1, 367)[mask], ndvi_values[mask], label=f"{year}", alpha=0.7)

    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.title(f"Filtered NDVI Time Series ({lat}°N, {lon}°E) - Yearly Comparison")
    plt.legend(title="Year", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    # Save the figure
    output_path = "../figures/mulitiyear_ndvi"
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')
    
    plt.show()


# In[10]:


# Compare NDVI for 60°N, 16°E across years 2002-2010
compare_yearly_ndvi(60.0, 16, year_start=2000, year_end=2024)

