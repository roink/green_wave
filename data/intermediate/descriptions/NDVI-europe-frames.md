# NDVI-europe-frames/

## Overview
- **Type:** Directory of PNG frame sequences cropped to the European subset of the global NDVI stack.
- **Purpose:** Intermediate renderings used to assemble animations or inspect seasonal evolution visually.
- **Source script:** `src/0.02-create_NDVI_mp4.py` (`logs/0.02-create_NDVI_mp4.log`).

## Layout
- Files follow the pattern `ndvi_europe_{year}_{doy:03d}.png` and span all 574 MOD13C1 timesteps between 2000-02-18 and 2015-12-31.
- Each frame depicts a `878 × 1218` pixel window corresponding to NDVI rows `[320, 1198)` and columns `[3335, 4553)` (bounding Europe).
- Colour mapping: Matplotlib `RdYlGn`, fixed range `[-2000, 10000]`, with an accompanying colourbar baked into each PNG.
- Companion global frames (`ndvi_global_{year}_{doy:03d}.png`) are generated in parallel under `figure/0.02-create_NDVI_mp4/frames/global/`.

## Regeneration
```bash
python src/0.02-create_NDVI_mp4.py
```
The script reads `ndvi_stack_optimized.h5`, renders the frame pairs using multiprocessing, and (optionally) encodes MP4 videos from the PNG sequences.

## Downstream consumers
- The rendered PNGs feed the MP4 videos stored in `figure/0.02-create_NDVI_mp4/video/`.
- Individual frames can be sampled for documentation or quality-control checks before animation.
