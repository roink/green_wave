#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import re
import numpy as np
import h5py
from pyhdf.SD import SD, SDC
from tqdm.notebook import tqdm


# In[2]:


# Path to raw HDF files
data_path = "/data/hescor/pschluet/green_wave/data/NDVI"
hdf_files = sorted(glob.glob(os.path.join(data_path, "MOD13C1.A*.hdf")))

# Get spatial dimensions from first file
first_file = SD(hdf_files[0], SDC.READ)
ndvi_shape = first_file.select("CMG 0.05 Deg 16 days NDVI")[:].shape  # (3600, 7200)
first_file.end()

# Total time steps (number of HDF files)
num_timesteps = len(hdf_files)

# Output HDF5 file path
hdf5_output_path = "/work/pschluet/green_wave/data/intermediate/ndvi_stack_filtered_all.h5"



# In[3]:


# Function to extract date from filename
def extract_date(filename):
    match = re.search(r"A(\d{4})(\d{3})", filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # (year, doy)
    return None, None


# In[4]:


# Create HDF5 file with chunked datasets
with h5py.File(hdf5_output_path, "w") as h5f:
    # Create NDVI dataset with chunking for better performance
    dset_ndvi = h5f.create_dataset(
        "ndvi_stack",
        shape=(num_timesteps, *ndvi_shape),
        dtype=np.float32,  # Use reduced precision to save memory
        chunks=(1, ndvi_shape[0], ndvi_shape[1]),  # Chunk by time step
        compression="lzf"
    )

    # Create metadata dataset (year, doy)
    dset_metadata = h5f.create_dataset(
        "metadata",
        shape=(num_timesteps, 2),  # (year, doy)
        dtype=np.int32
    )

    # Process each file and store in HDF5
    for i, file_path in tqdm(enumerate(hdf_files), total=len(hdf_files), desc="Processing HDF files"):
        year, doy = extract_date(file_path)
        if year is None:
            continue  # Skip if date not found

        # Store metadata
        dset_metadata[i] = (year, doy)

        # Open HDF file
        hdf = SD(file_path, SDC.READ)

        # Read NDVI data
        try:
            ndvi_data = hdf.select("CMG 0.05 Deg 16 days NDVI")[:]
            ndvi_reliability = hdf.select("CMG 0.05 Deg 16 days pixel reliability")[:]
        except Exception as e:
            print(f"Could not read NDVI or reliability from {file_path}: {e}")
            continue

        # Convert fill values (-3000) to NaN
        ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)

        # Apply reliability filter: set to NaN where reliability is <0 (no data) or >2 (snow/ice)
        mask = (ndvi_reliability < 0) | (ndvi_reliability > 2)
        ndvi_data[mask] = np.nan

        # Store in HDF5 dataset
        dset_ndvi[i, :, :] = ndvi_data

print(f"Filtered NDVI stack saved at {hdf5_output_path}")


# In[ ]:




