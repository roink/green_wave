#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyhdf.SD import SD, SDC

file_path = "/data/hescor/pschluet/green_wave/data/NDVI/MOD13C1.A2009241.061.2021141172023.hdf"

# Open the HDF file
hdf = SD(file_path, SDC.READ)

# List available datasets
datasets = hdf.datasets()
print("Datasets in file:")
for key, value in datasets.items():
    print(f"{key}: {value}")

# Print global attributes (metadata)
attrs = hdf.attributes()
print("\nMetadata:")
for attr, value in attrs.items():
    print(f"{attr}: {value}")


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

# Open HDF4 file
hdf = SD(file_path, SDC.READ)

# Check dataset names
datasets = hdf.datasets()
print("Datasets:", datasets.keys())

# Select NDVI dataset (may need adjustment)
ndvi_data = hdf.select("CMG 0.05 Deg 16 days NDVI")[:]

# Handle fill values (-3000 is often used in MODIS)
ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)

# Plot NDVI
plt.figure(figsize=(10, 6))
plt.imshow(ndvi_data, cmap="RdYlGn", vmin=-2000, vmax=10000)
plt.colorbar(label="NDVI Value")
plt.title("MODIS NDVI, 2009 day 241")

# Save the figure
output_path = "../figures/global_ndvi_sample"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

plt.show()


# In[3]:


# Select NDVI dataset (may need adjustment)
ndvi_quality = hdf.select("CMG 0.05 Deg 16 days pixel reliability")[:]


# In[4]:


import numpy as np
from collections import Counter

# Extract unique values and their frequencies
unique_values, counts = np.unique(ndvi_quality, return_counts=True)

# Print unique values with their counts
print("Unique values in the quality dataset and their frequencies:")
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")


# In[5]:


# Handle fill values (-3000 is often used in MODIS)

# Plot NDVI
plt.figure(figsize=(10, 6))
plt.imshow(ndvi_quality, cmap="RdYlGn", )
plt.colorbar(label="NDVI Quality")
plt.title("NDVI Quality, 2009 day 241")

# Save the figure
output_path = "../figures/data_quality_sample"
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.eps", format='eps', bbox_inches='tight')

plt.show()


# In[ ]:




