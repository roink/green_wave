# Data directory

This repository keeps all datasets under the project-local `data/` tree so that
raw inputs, intermediate artefacts, and publishable outputs stay separated while
remaining portable with the codebase.

## Subdirectories

- `raw/` – Direct downloads or external deliveries. Files in this folder are
  treated as read-only provenance data (e.g., MODIS MOD13C1 scenes). Scripts
  should never overwrite them.
- `intermediate/` – Rebuildable products such as filtered NDVI stacks or cached
  analysis grids. Pipelines may delete and recreate these files as needed.
- `finished/` – Curated exports meant for distribution (figures, tables, or
  compact parameter sets). These should change rarely and only after review.

## Usage guidelines

- The download helper `src/download_NDVI.sh` ensures `raw/NDVI/` exists and only
  fetches missing MODIS scenes, preventing duplicate transfers.
- Processing scripts resolve data paths relative to the repository root so the
  workflows run consistently on any machine that has the repository checked out
  alongside this directory.
- Large binary files under `data/` are ignored by Git; use this README to orient
  collaborators to the expected layout and regeneration strategy.
