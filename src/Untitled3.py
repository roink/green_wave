#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
Script to iterate over all spatial cells in an HDF5 NDVI stack,
fit a cyclic Gaussian to each pixel time series, and save
the fitted parameters and R² map to a new HDF5 file using
float32 precision and LZF compression.
"""
import os
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
directory = "/net/sno/pschluet/green_wave/data/intermediate"
input_file = os.path.join(directory, "ndvi_stack_optimized.h5")
output_file = os.path.join(directory, "ndvi_gaussian_fits.h5")

# -------------------------------
# Open input file and read metadata and stack
# -------------------------------
src = h5py.File(input_file, 'r')
metadata = src['metadata'][:]  # (N,2): year, doy
ndvi_stack = src['ndvi_stack']  # dataset
T, nrows, ncols = ndvi_stack.shape

# Convert metadata to day-of-year array once
doy = np.array([
    pd.to_datetime(f"{year}-{int(d):03d}", format="%Y-%j").dayofyear
    for year, d in metadata
], dtype=np.int32)

# -------------------------------
# Define helper functions
# -------------------------------
def process_ndvi(series):
    winter_ndvi = np.nanquantile(series, 0.025)
    corr = np.array(series, dtype=np.float32)
    corr[np.isnan(corr)] = winter_ndvi
    corr[corr < winter_ndvi] = winter_ndvi
    return median_filter(corr, size=3)


def gaussian_cyclic(t, A, mu, sigma, C, period=365):
    delta = ((t - mu + period/2) % period) - period/2
    return C + A * np.exp(-0.5 * (delta / sigma)**2)

# -------------------------------
# Allocate result maps in memory
# -------------------------------
shape = (nrows, ncols)
A_map     = np.full(shape, np.nan, dtype=np.float32)
mu_map    = np.full(shape, np.nan, dtype=np.float32)
sigma_map = np.full(shape, np.nan, dtype=np.float32)
C_map     = np.full(shape, np.nan, dtype=np.float32)
r2_map    = np.full(shape, np.nan, dtype=np.float32)

# -------------------------------
# Loop over all cells and fit
# -------------------------------
for i in tqdm(range(nrows), desc='Rows'):
    for j in range(ncols):
        series = ndvi_stack[:, i, j]
        filt = process_ndvi(series)
        mask = ~np.isnan(filt) & ~np.isinf(filt)
        if np.sum(mask) < 5:
            continue
        x = doy[mask]
        y = filt[mask]

        p0 = [np.nanmax(y) - np.nanmin(y), x[np.argmax(y)], 30.0, np.nanmin(y)]
        bounds = ([0, 0, 0, -np.inf], [np.inf, 365, 365, np.inf])
        try:
            popt, _ = curve_fit(gaussian_cyclic, x, y, p0=p0, bounds=bounds)
        except RuntimeError:
            continue
        A_opt, mu_opt, sigma_opt, C_opt = popt

        # compute R-squared
        y_fit = gaussian_cyclic(x, *popt)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        A_map[i,j]     = A_opt
        mu_map[i,j]    = mu_opt % 365
        sigma_map[i,j] = sigma_opt
        C_map[i,j]     = C_opt
        r2_map[i,j]    = r2

# close input file
src.close()

# -------------------------------
# Save maps to HDF5 with LZF compression
# -------------------------------
with h5py.File(output_file, 'w') as dst:
    dst.create_dataset('A',     data=A_map,     dtype='f4', compression='lzf')
    dst.create_dataset('mu',    data=mu_map,    dtype='f4', compression='lzf')
    dst.create_dataset('sigma', data=sigma_map, dtype='f4', compression='lzf')
    dst.create_dataset('C',     data=C_map,     dtype='f4', compression='lzf')
    dst.create_dataset('R2',    data=r2_map,    dtype='f4', compression='lzf')
    dst.create_dataset('metadata', data=metadata, dtype=metadata.dtype, compression='lzf')

print(f"Gaussian fits saved to {output_file}")


# In[ ]:




