#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


hdf5_file = "/mnt/ramdisk/ndvi_stack_filtered_all.h5"


# In[3]:


with h5py.File(hdf5_file, "r") as h5f:
    print(h5f["ndvi_stack"].chunks)  # Check chunk size
    print(h5f["ndvi_stack"].compression)  # Check compression type


# In[4]:


# Original file and new optimized file
old_hdf5_file = "/mnt/ramdisk/ndvi_stack_filtered_all.h5"
new_hdf5_file = "/mnt/ramdisk/ndvi_stack_optimized.h5"

# Define new chunk size
new_chunk_size = (1, 256, 256)  # Tune for best performance

with h5py.File(old_hdf5_file, "r") as old_h5f, h5py.File(new_hdf5_file, "w") as new_h5f:
    # Copy metadata without changes
    for key in old_h5f.keys():
        if key != "ndvi_stack":  # Don't copy NDVI stack yet
            old_h5f.copy(key, new_h5f)

    # Read NDVI data
    old_data = old_h5f["ndvi_stack"]

    # Create a new dataset with smaller chunks
    new_ds = new_h5f.create_dataset(
        "ndvi_stack",
        shape=old_data.shape,
        dtype=old_data.dtype,
        chunks=new_chunk_size,
        compression="lzf"
    )

    # Copy data in chunks to avoid memory issues
    for i in range(old_data.shape[0]):  # Iterate over time steps
        new_ds[i, :, :] = old_data[i, :, :]  # Copy slice-by-slice

print(f"Optimized HDF5 saved to {new_hdf5_file}")


# In[6]:


old_file = "/mnt/ramdisk/ndvi_stack_filtered_all.h5"
new_file = "/mnt/ramdisk/ndvi_stack_optimized.h5"

with h5py.File(old_file, "r") as old_h5, h5py.File(new_file, "r") as new_h5:
    print("Old shape:", old_h5["ndvi_stack"].shape)
    print("New shape:", new_h5["ndvi_stack"].shape)
    print("Old dtype:", old_h5["ndvi_stack"].dtype)
    print("New dtype:", new_h5["ndvi_stack"].dtype)
    
    # Compare random samples
    idx = np.random.randint(0, old_h5["ndvi_stack"].shape[0])
    print("Sample comparison:", np.allclose(old_h5["ndvi_stack"][idx], new_h5["ndvi_stack"][idx], equal_nan=True))


# In[7]:


# Original file and new optimized file
old_hdf5_file = "/mnt/ramdisk/ndvi_stack_filtered_all.h5"
new_hdf5_file = "/mnt/ramdisk/ndvi_stack_optimized_small_chunks.h5"

# Define new chunk size
new_chunk_size = (1, 36, 72)  # Tune for best performance

with h5py.File(old_hdf5_file, "r") as old_h5f, h5py.File(new_hdf5_file, "w") as new_h5f:
    # Copy metadata without changes
    for key in old_h5f.keys():
        if key != "ndvi_stack":  # Don't copy NDVI stack yet
            old_h5f.copy(key, new_h5f)

    # Read NDVI data
    old_data = old_h5f["ndvi_stack"]

    # Create a new dataset with smaller chunks
    new_ds = new_h5f.create_dataset(
        "ndvi_stack",
        shape=old_data.shape,
        dtype=old_data.dtype,
        chunks=new_chunk_size,
        compression="lzf"
    )

    # Copy data in chunks to avoid memory issues
    for i in range(old_data.shape[0]):  # Iterate over time steps
        new_ds[i, :, :] = old_data[i, :, :]  # Copy slice-by-slice

print(f"Optimized HDF5 saved to {new_hdf5_file}")


# In[8]:


# Original file and new optimized file
old_hdf5_file = "/mnt/ramdisk/ndvi_stack_filtered_all.h5"
new_hdf5_file = "/mnt/ramdisk/ndvi_stack_optimized_intermediate_chunks.h5"

# Define new chunk size
new_chunk_size = (1, 360, 360)  # Tune for best performance

with h5py.File(old_hdf5_file, "r") as old_h5f, h5py.File(new_hdf5_file, "w") as new_h5f:
    # Copy metadata without changes
    for key in old_h5f.keys():
        if key != "ndvi_stack":  # Don't copy NDVI stack yet
            old_h5f.copy(key, new_h5f)

    # Read NDVI data
    old_data = old_h5f["ndvi_stack"]

    # Create a new dataset with smaller chunks
    new_ds = new_h5f.create_dataset(
        "ndvi_stack",
        shape=old_data.shape,
        dtype=old_data.dtype,
        chunks=new_chunk_size,
        compression="lzf"
    )

    # Copy data in chunks to avoid memory issues
    for i in range(old_data.shape[0]):  # Iterate over time steps
        new_ds[i, :, :] = old_data[i, :, :]  # Copy slice-by-slice

print(f"Optimized HDF5 saved to {new_hdf5_file}")


# In[ ]:




