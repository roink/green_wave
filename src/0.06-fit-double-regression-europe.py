#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import median_filter
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from scipy.stats import linregress
import warnings


# In[2]:


# Path to save output
output_path = "/work/pschluet/green_wave/data/intermediate/ndvi_fit_params.npz"

# Define bounding box for Europe (pixel indices)
row_start, row_end = 320, 1198
col_start, col_end = 3335, 4553

# Initialize array for fit parameters
num_rows = row_end - row_start
num_cols = col_end - col_start
num_params = 6  # Number of fitted parameters
ndvi_fit_params = np.full((num_rows, num_cols, 6), np.nan, dtype=np.float32)  # 6 fit parameters


# In[3]:


def double_logistic(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """
    Double-logistic function for fitting NDVI profiles.
    """
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


# In[4]:


# Load NDVI stack
data = np.load("/work/pschluet/green_wave/data/intermediate/ndvi_stack_filtered.npz")

ndvi_stack = data["ndvi_stack"]
metadata = data["metadata"]

print("Loaded NDVI stack shape:", ndvi_stack.shape)  # (time, 3600, 7200)
print("Metadata (first 5 entries):", metadata[:5])  # [(year, doy), ...]


# In[5]:


# Create shared memory for NDVI stack
ndvi_stack_shape = ndvi_stack.shape
ndvi_stack_dtype = ndvi_stack.dtype
shm = shared_memory.SharedMemory(create=True, size=ndvi_stack.nbytes)
ndvi_stack_shared = np.ndarray(ndvi_stack_shape, dtype=ndvi_stack_dtype, buffer=shm.buf)
np.copyto(ndvi_stack_shared, ndvi_stack)  # Copy data to shared memory


# In[6]:


def compute_r2(y_true, y_pred):
    """ Compute R² score to evaluate goodness-of-fit """
    mask = ~np.isnan(y_true)
    if np.sum(mask) < 5:
        return np.nan  # Not enough points

    slope, intercept, r_value, _, _ = linregress(y_true[mask], y_pred[mask])
    return r_value ** 2


# In[7]:


num_params = 6  # Number of fitted parameters

# Define new output shape
num_metrics = 2  # [fit_quality, fit_status]
ndvi_fit_metrics = np.full((num_rows, num_cols, num_metrics), np.nan, dtype=np.float32)  # R² and status

result_length = num_params + num_metrics

# Function to process a single pixel (row, col)
def process_pixel(row, col, shm_name):
    """Fit the double-logistic function for one pixel using multiple initial guesses.
    Returns a vector of length 8: [6 fitted parameters, R², covariance quality]."""
    # Reconnect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    ndvi_stack_shared = np.ndarray(ndvi_stack_shape, dtype=ndvi_stack_dtype, buffer=existing_shm.buf)
    
    # Extract NDVI time series for this pixel
    ndvi_timeseries = ndvi_stack_shared[:, row_start + row, col_start + col]
    if np.isnan(ndvi_timeseries).all():
        return np.full(result_length, np.nan)
    
    # Apply winter NDVI correction
    winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)
    corrected_ndvi = np.copy(ndvi_timeseries)
    corrected_ndvi[ndvi_timeseries < winter_ndvi] = winter_ndvi
    
    # Replace missing winter values (DOY >=300 or <=60)
    for i, (year, doy) in enumerate(metadata):
        if np.isnan(corrected_ndvi[i]) and (doy >= 300 or doy <= 60):
            corrected_ndvi[i] = winter_ndvi
            
    # Apply a moving median filter (window size = 3)
    filtered_ndvi = median_filter(corrected_ndvi, size=3)
    
    # Extract NDVI values for selected years (2002-2010)
    doy_values, ndvi_values = [], []
    for i, (year, doy) in enumerate(metadata):
        if 2002 <= year <= 2010 and not np.isnan(filtered_ndvi[i]):
            doy_values.append(doy)
            ndvi_values.append(filtered_ndvi[i])
    
    doy_values = np.array(doy_values)
    ndvi_values = np.array(ndvi_values)
    if len(ndvi_values) < 100:
        return np.full(result_length, np.nan)
    
    # Define candidate initial guesses
    bias_guess = np.min(ndvi_values)
    scale_guess = np.max(ndvi_values) - np.min(ndvi_values)
    initial_guess_1 = [120, 20, 270, 25, bias_guess, scale_guess]  # typical northern hemisphere
    initial_guess_2 = [240, 20, 60, 25, bias_guess+scale_guess, scale_guess]   # possible southern pattern
    initial_guess_3 = [240, 20, 60, 25, bias_guess+0.5*scale_guess, 1] # southern equitorial 
    initial_guess_4 = [120, 20, 270, 25, bias_guess+0.5*scale_guess, 1] # northern equitorial
    candidate_guesses = [initial_guess_1, initial_guess_2, initial_guess_3, initial_guess_4]

    # Define parameter bounds
    lower_bounds = [0, 1e-5, 0, 1e-5, -np.inf, 1e-5]  # Lower bounds
    upper_bounds = [365, np.inf, 365, np.inf, np.inf, np.inf]  # Upper bounds
    
    best_fit = None
    best_r2 = -np.inf
    best_cov_quality = np.nan
    
    # Try each candidate initial guess.
    for guess in candidate_guesses:
        try:
            with warnings.catch_warnings():
                # Ignore OptimizeWarning but treat RuntimeWarning as error
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                warnings.filterwarnings("error", category=RuntimeWarning)
                params, covariance = curve_fit(double_logistic, doy_values, ndvi_values, p0=guess, bounds=(lower_bounds, upper_bounds))
            ndvi_fitted = double_logistic(doy_values, *params)
            r2_score = compute_r2(ndvi_values, ndvi_fitted)
            if (not np.isnan(r2_score)) and (r2_score > best_r2):
                best_r2 = r2_score
                best_fit = params
                std_errors = np.sqrt(np.diag(covariance))
                best_cov_quality = np.mean(std_errors)
        except (RuntimeError, RuntimeWarning):
            continue
    
    if best_fit is None:
        return np.full(result_length, np.nan)
    
    return np.concatenate([best_fit, [best_r2, best_cov_quality]])



# In[8]:


# Flatten row/col indices for parallel execution
all_pixel_indices = [(row, col) for row in range(num_rows) for col in range(num_cols)]

# Run parallelized fitting
all_pixel_indices = [(row, col) for row in range(num_rows) for col in range(num_cols)]
num_jobs = -1  # Use all CPU cores

results = Parallel(n_jobs=num_jobs, backend="loky")(
    delayed(process_pixel)(row, col, shm.name) for row, col in tqdm(all_pixel_indices, desc="Processing pixels")
)


# In[9]:


# Reshape results into a 3D array: (num_rows, num_cols, 8)
ndvi_fit_all = np.array(results).reshape((num_rows, num_cols, result_length))

# --- Save the results as a single file ---
save_path = "/work/pschluet/green_wave/data/intermediate/ndvi_fit_params.npz"
np.savez_compressed(save_path, ndvi_fit_all=ndvi_fit_all)

# Cleanup shared memory
shm.close()
shm.unlink()

print(f" Saved fitted NDVI parameters and quality metrics (shape {ndvi_fit_all.shape}) to {save_path}")


# In[ ]:




