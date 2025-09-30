#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Path to HDF5 file
hdf5_file = "/work/pschluet/green_wave/data/intermediate/ndvi_stack_optimized.h5"

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

    # Remove NaNs for plotting clarity
    valid_mask = ~np.isnan(ndvi_timeseries)
    ndvi_timeseries = ndvi_timeseries[valid_mask]
    valid_dates = metadata[valid_mask]  # Keep only valid time steps

    # Convert metadata (year, DOY) to actual dates
    dates = [pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in valid_dates]

    return dates, ndvi_timeseries


# In[3]:


# -------------------------------
# Function to plot NDVI time series
# -------------------------------
def plot_ndvi(lat, lon):
    """Plot the NDVI time series for a given latitude & longitude."""
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, ndvi_timeseries, marker="o", linestyle="-", color="green", label="NDVI")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[4]:


# -------------------------------
# Plot NDVI Time Series for Given Locations
# -------------------------------
plot_ndvi(40.0, 16)  # NDVI at 40°N, 16°E


# In[5]:


plot_ndvi(50.0, 16)  # NDVI at 50°N, 16°E


# In[6]:


plot_ndvi(60.0, 16)  # NDVI at 60°N, 16°E

