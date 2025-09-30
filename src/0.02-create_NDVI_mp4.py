#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing
import glob


# In[11]:


# -------------------------------
# Define paths and create folders
# -------------------------------

# Path to the pre-processed HDF5 file
hdf5_file = "/work/pschluet/green_wave/data/intermediate/ndvi_stack_filtered_all.h5"

# Directories to save global and Europe frames
global_frames_dir = "/work/pschluet/green_wave/data/intermediate/NDVI-global-frames"
europe_frames_dir = "/work/pschluet/green_wave/data/intermediate/NDVI-europe-frames"
os.makedirs(global_frames_dir, exist_ok=True)
os.makedirs(europe_frames_dir, exist_ok=True)


# In[3]:


# -------------------------------
# Define Europe bounding box indices (pixels)
# -------------------------------
row_start, row_end = 320, 1198
col_start, col_end = 3335, 4553


# In[4]:


# -------------------------------
# Load metadata and get dataset shape
# -------------------------------
with h5py.File(hdf5_file, "r") as h5f:
    num_timesteps = h5f["ndvi_stack"].shape[0]  # Number of time steps
    metadata = h5f["metadata"][:]  # Load metadata array (year, doy)


# In[5]:


def process_timestep(i):
    """Generate and save NDVI plots for a given time index."""
    # Import gc in worker (if not already imported)
    import gc

    # Open the HDF5 file in each worker process
    with h5py.File(hdf5_file, "r") as h5f:
        ndvi_stack = h5f["ndvi_stack"]
        year, doy = metadata[i]  # Get year, doy from pre-loaded metadata
        ndvi_data = ndvi_stack[i, :, :]  # Read one time step

    # ----- Global NDVI Plot -----
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(ndvi_data, cmap="RdYlGn", vmin=-2000, vmax=10000)
    plt.colorbar(img, label="NDVI Value", ax=ax)
    ax.set_title(f"NDVI Value {year} day {doy:03d}")
    global_frame_path = os.path.join(global_frames_dir, f"ndvi_global_{year}_{doy:03d}.png")
    plt.savefig(global_frame_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure

    # ----- Europe NDVI Plot (Cropped) -----
    europe_ndvi = ndvi_data[row_start:row_end, col_start:col_end]
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(europe_ndvi, cmap="RdYlGn", vmin=-2000, vmax=10000)
    plt.colorbar(img, label="NDVI Value", ax=ax)
    ax.set_title(f"NDVI Value Europe {year} day {doy:03d}")
    europe_frame_path = os.path.join(europe_frames_dir, f"ndvi_europe_{year}_{doy:03d}.png")
    plt.savefig(europe_frame_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure

    # Delete large arrays and force garbage collection
    del ndvi_data, europe_ndvi
    gc.collect()

    return f"Processed {year}-{doy}"


# In[ ]:


# Use fork (on Unix) and recycle each worker after one task
multiprocessing.set_start_method("fork")
num_workers = min(multiprocessing.cpu_count(), num_timesteps)

# Setting maxtasksperchild=1 forces a worker to exit after one task,
# which can help avoid long-lived memory leaks.
with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
    results = list(tqdm(pool.imap_unordered(process_timestep, range(num_timesteps)),
                        total=num_timesteps,
                        desc="Generating Frames"))
print("Frames saved successfully!")


# In[14]:


# -------------------------------
# Create videos from the generated frames
# -------------------------------

def create_video(frames_dir, output_path, fps=10):
    """Create an MP4 video from PNG frames in a directory."""
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        print(f"No frames found in {frames_dir}, skipping video creation.")
        return

    # Read first frame to get video size
    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_file in tqdm(frame_files, desc=f"Creating video: {output_path}"):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {output_path}")

# Create videos
create_video(global_frames_dir, "/work/pschluet/green_wave/data/intermediate/NDVI-global.mp4")
create_video(europe_frames_dir, "/work/pschluet/green_wave/data/intermediate/NDVI-europe.mp4")


# In[ ]:




