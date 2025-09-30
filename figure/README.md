# Figure Output Guidelines

Each analysis script should save its figures inside the `figure/` directory using the following conventions:

- Create or reuse a sub-directory named after the script's filename without the extension (for example, `figure/1.03-fit-gaussia_to_ndvi`).
- Save each image with the pattern `<script-stem>__<descriptive-name>.png` so the producing script can be identified immediately. The descriptive portion should be short, lowercase, and use hyphens or underscores instead of spaces (for example, `1.03-fit-gaussia_to_ndvi__lat60p0_lon16p0_gaussian_fit.png`).
- When multiple figures are produced for a single script run, incrementally vary the descriptive part to make each file unique.

Following this layout keeps figures organized and makes it clear which script generated each asset.
