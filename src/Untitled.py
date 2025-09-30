#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


# Define the double logistic function
def double_logistic(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """Double-logistic function for fitting NDVI profiles."""
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


# In[14]:


# Define the double logistic function
def IRG(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """Double-logistic function for fitting NDVI profiles."""
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    return  (spring * (1-spring))


# In[4]:


# Load the fitted parameters (shape: num_rows x num_cols x 6)
fit_params_path = "/work/pschluet/green_wave/data/intermediate/ndvi_fit_params.npz"
data = np.load(fit_params_path)
ndvi_fit_params = data["ndvi_fit_all"][..., :6]  # Extract only the 6 fit parameters

# Define the time steps
time_steps = np.arange(1, 366, 16)  # [1, 17, 33, ..., 353]

# Define output directories
europe_frames_dir = "/work/pschluet/green_wave/data/intermediate/predicted-NDVI-europe-frames"
os.makedirs(europe_frames_dir, exist_ok=True)

# Define colormap limits
vmin, vmax = -2000, 10000


# In[5]:


# Loop through time steps to generate maps
for doy in time_steps:
    # Compute NDVI using fitted parameters
    ndvi_predicted = double_logistic(
        doy,
        ndvi_fit_params[..., 0],  # xmidSNDVI
        ndvi_fit_params[..., 1],  # scalSNDVI
        ndvi_fit_params[..., 2],  # xmidANDVI
        ndvi_fit_params[..., 3],  # scalANDVI
        ndvi_fit_params[..., 4],  # bias
        ndvi_fit_params[..., 5],  # scale
    )

    # GLOBAL NDVI PLOT
    plt.figure(figsize=(10, 6))
    plt.imshow(ndvi_predicted, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(label="NDVI Value")
    plt.title(f"Predicted NDVI {doy:03d}")
    
    # Save global frame
    global_frame_path = os.path.join(europe_frames_dir, f"ndvi_europe_pred_{doy:03d}.png")
    plt.savefig(global_frame_path, dpi=300, bbox_inches='tight')
    plt.close()


print("Predicted NDVI frames saved successfully!")


# In[12]:


# Define the time steps
time_steps = np.arange(1, 366, 16)  # [1, 17, 33, ..., 353]
# Define output directories
europe_IRG_dir = "/work/pschluet/green_wave/data/intermediate/predicted-IRG-europe-frames"
os.makedirs(europe_frames_dir, exist_ok=True)

# Define colormap limits
vmin, vmax = -2000, 10000


# In[16]:


# Define colormap limits
vmin, vmax = 0, 0.25
# Loop through time steps to generate maps
for doy in time_steps:
    # Compute IRG using fitted parameters
    irg_predicted = IRG(
        doy,
        ndvi_fit_params[..., 0],  # xmidSNDVI
        ndvi_fit_params[..., 1],  # scalSNDVI
        ndvi_fit_params[..., 2],  # xmidANDVI
        ndvi_fit_params[..., 3],  # scalANDVI
        ndvi_fit_params[..., 4],  # bias
        ndvi_fit_params[..., 5],  # scale
    )
    print(f"  Min:  {np.nanmin(irg_predicted):.2f}")
    print(f"  Max:  {np.nanmax(irg_predicted):.2f}")

    # GLOBAL NDVI PLOT
    plt.figure(figsize=(10, 6))
    plt.imshow(irg_predicted, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(label="IRG Value")
    plt.title(f"Predicted IRG {doy:03d}")
    
    # Save global frame
    global_frame_path = os.path.join(europe_IRG_dir, f"IRG_europe_pred_{doy:03d}.png")
    plt.savefig(global_frame_path, dpi=300, bbox_inches='tight')
    plt.close()


print("Predicted NDVI frames saved successfully!")


# In[17]:


plt.figure(figsize=(10, 6))
plt.imshow(ndvi_fit_params[..., 0], cmap="RdYlGn")
plt.colorbar(label="Spring day")
plt.title(f"Predicted NDVI {doy:03d}")



# In[ ]:




