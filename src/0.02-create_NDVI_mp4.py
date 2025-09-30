#!/usr/bin/env python
"""Generate NDVI frame sequences and assemble MP4 videos."""

from __future__ import annotations

import glob
import multiprocessing
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_FILE = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_ROOT = PROJECT_ROOT / "figure" / Path(__file__).stem
GLOBAL_FRAMES_DIR = FIGURE_ROOT / "frames" / "global"
EUROPE_FRAMES_DIR = FIGURE_ROOT / "frames" / "europe"
VIDEO_DIR = FIGURE_ROOT / "video"

ROW_START, ROW_END = 320, 1198
COL_START, COL_END = 3335, 4553

metadata: np.ndarray | None = None


def ensure_directories() -> None:
    """Ensure the output directories exist."""

    for directory in (GLOBAL_FRAMES_DIR, EUROPE_FRAMES_DIR, VIDEO_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_metadata() -> tuple[np.ndarray, int]:
    """Load dataset metadata and return it along with the timestep count."""

    with h5py.File(HDF5_FILE, "r") as h5f:
        metadata_array = h5f["metadata"][:]
        num_timesteps = h5f["ndvi_stack"].shape[0]
    print(f"Loaded metadata for {num_timesteps} time steps from {HDF5_FILE}")
    return metadata_array, num_timesteps


def process_timestep(index: int) -> str:
    """Generate both global and European NDVI frames for the requested index."""

    assert metadata is not None, "Metadata must be loaded before processing frames"

    with h5py.File(HDF5_FILE, "r") as h5f:
        ndvi_stack = h5f["ndvi_stack"]
        year, doy = metadata[index]
        ndvi_data = ndvi_stack[index, :, :]

    figure_kwargs = dict(figsize=(10, 6))

    fig, ax = plt.subplots(**figure_kwargs)
    img = ax.imshow(ndvi_data, cmap="RdYlGn", vmin=-2000, vmax=10000)
    fig.colorbar(img, label="NDVI Value", ax=ax)
    ax.set_title(f"NDVI Value {year} day {doy:03d}")
    global_frame_path = GLOBAL_FRAMES_DIR / f"ndvi_global_{year}_{doy:03d}.png"
    fig.savefig(global_frame_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    europe_ndvi = ndvi_data[ROW_START:ROW_END, COL_START:COL_END]
    fig, ax = plt.subplots(**figure_kwargs)
    img = ax.imshow(europe_ndvi, cmap="RdYlGn", vmin=-2000, vmax=10000)
    fig.colorbar(img, label="NDVI Value", ax=ax)
    ax.set_title(f"NDVI Value Europe {year} day {doy:03d}")
    europe_frame_path = EUROPE_FRAMES_DIR / f"ndvi_europe_{year}_{doy:03d}.png"
    fig.savefig(europe_frame_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return f"Processed {year}-{doy:03d}"


def create_video(frames_dir: Path, output_path: Path, fps: int = 10) -> None:
    """Create a video from the frames stored in ``frames_dir``."""

    frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    if not frame_files:
        print(f"No frames found in {frames_dir}, skipping video creation.")
        return

    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc=f"Creating video: {output_path.name}"):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {output_path}")


def generate_frames(num_timesteps: int) -> None:
    """Generate frames for all available time steps using multiprocessing."""

    multiprocessing.set_start_method("fork", force=True)
    worker_count = min(multiprocessing.cpu_count(), num_timesteps)

    with multiprocessing.Pool(processes=worker_count, maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_timestep, range(num_timesteps)),
            total=num_timesteps,
            desc="Generating Frames",
        ):
            pass

    print(f"Finished generating {num_timesteps} frames per region.")


def main() -> None:
    ensure_directories()
    global metadata
    metadata, num_timesteps = load_metadata()
    generate_frames(num_timesteps)

    create_video(
        GLOBAL_FRAMES_DIR,
        VIDEO_DIR / "NDVI-global.mp4",
    )
    create_video(
        EUROPE_FRAMES_DIR,
        VIDEO_DIR / "NDVI-europe.mp4",
    )


if __name__ == "__main__":
    main()
