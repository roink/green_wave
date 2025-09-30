#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Path to the saved file
file_path = "/work/pschluet/green_wave/data/intermediate/ndvi_fit_params.npz"

# Load the data
data = np.load(file_path)


# In[3]:


data


# In[4]:


ndvi_fit_params = data["ndvi_fit_all"]  # Shape: (rows, cols, 6)

# Check the shape
print("Loaded NDVI fit parameters shape:", ndvi_fit_params.shape)


# In[5]:


param_names = ["xmidSNDVI", "scalSNDVI", "xmidANDVI", "scalANDVI", "bias", "scale"]

# Compute basic statistics
for i, param in enumerate(param_names):
    param_data = ndvi_fit_params[:, :, i]
    valid_values = param_data[~np.isnan(param_data)]  # Exclude NaNs

    print(f"\n {param}:")
    print(f"  Min:  {np.nanmin(param_data):.2f}")
    print(f"  Max:  {np.nanmax(param_data):.2f}")
    print(f"  Mean: {np.nanmean(param_data):.2f}")
    print(f"  Median:  {np.nanmedian(param_data):.2f}")
    print(f"  Std:  {np.nanstd(param_data):.2f}")


# In[6]:


# Define parameter constraints
xmid_min, xmid_max = 0, 365
scal_min, scal_max = 1, 100
bias_min, bias_max = 0, 10000
scale_min, scale_max = 0, 10000

# Apply filtering
ndvi_fit_params[:, :, 0] = np.where(
    (ndvi_fit_params[:, :, 0] >= xmid_min) & (ndvi_fit_params[:, :, 0] <= xmid_max),
    ndvi_fit_params[:, :, 0],
    np.nan
)

ndvi_fit_params[:, :, 2] = np.where(
    (ndvi_fit_params[:, :, 2] >= xmid_min) & (ndvi_fit_params[:, :, 2] <= xmid_max),
    ndvi_fit_params[:, :, 2],
    np.nan
)

ndvi_fit_params[:, :, 1] = np.where(
    (ndvi_fit_params[:, :, 1] >= scal_min) & (ndvi_fit_params[:, :, 1] <= scal_max),
    ndvi_fit_params[:, :, 1],
    np.nan
)

ndvi_fit_params[:, :, 3] = np.where(
    (ndvi_fit_params[:, :, 3] >= scal_min) & (ndvi_fit_params[:, :, 3] <= scal_max),
    ndvi_fit_params[:, :, 3],
    np.nan
)

ndvi_fit_params[:, :, 4] = np.where(
    (ndvi_fit_params[:, :, 4] >= bias_min) & (ndvi_fit_params[:, :, 4] <= bias_max),
    ndvi_fit_params[:, :, 4],
    np.nan
)

ndvi_fit_params[:, :, 5] = np.where(
    (ndvi_fit_params[:, :, 5] >= scale_min) & (ndvi_fit_params[:, :, 5] <= scale_max),
    ndvi_fit_params[:, :, 5],
    np.nan
)


# In[7]:


# Compute basic statistics
for i, param in enumerate(param_names):
    param_data = ndvi_fit_params[:, :, i]
    valid_values = param_data[~np.isnan(param_data)]  # Exclude NaNs

    print(f"\n {param}:")
    print(f"  Min:  {np.nanmin(param_data):.2f}")
    print(f"  Max:  {np.nanmax(param_data):.2f}")
    print(f"  Mean: {np.nanmean(param_data):.2f}")
    print(f"  Median:  {np.nanmedian(param_data):.2f}")
    print(f"  Std:  {np.nanstd(param_data):.2f}")


# In[8]:


# Save cleaned data
cleaned_output_path = "/work/pschluet/green_wave/data/intermediate/ndvi_fit_params_cleaned.npz"
np.savez_compressed(cleaned_output_path, ndvi_fit_params=ndvi_fit_params)

print(f"Saved cleaned NDVI parameters to {cleaned_output_path}")


# In[9]:


import matplotlib.pyplot as plt

# Define function to plot parameter maps
def plot_param(param_idx, title, cmap="viridis"):
    param_data = ndvi_fit_params[:, :, param_idx]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(param_data, cmap=cmap, interpolation="nearest")
    plt.colorbar(label=title)
    plt.title(title)
    plt.show()

# Plot green-up timing
plot_param(0, "Spring Green-Up (xmidSNDVI)", cmap="coolwarm")

# Plot autumn senescence timing
plot_param(2, "Autumn Dry-Down (xmidANDVI)", cmap="coolwarm")


# In[10]:


lat_means = np.nanmean(ndvi_fit_params, axis=1)  # Average over longitudes

plt.figure(figsize=(10, 5))
plt.plot(lat_means[:, 0], label="Spring Green-Up (xmidSNDVI)", color="green")
plt.plot(lat_means[:, 2], label="Autumn Dry-Down (xmidANDVI)", color="orange")

plt.xlabel("Latitude Index")
plt.ylabel("Day of Year")
plt.title("NDVI Phenology Across Latitude")
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


# Plot green-up timing
plot_param(6, "R-squared", cmap="coolwarm")

# Plot autumn senescence timing
plot_param(7, "sqrt(coVar)", cmap="coolwarm")


# In[ ]:




