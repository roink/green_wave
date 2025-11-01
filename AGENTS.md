# Repository-wide agent guidelines

- Consult the script logs in the `logs/` directory when you need high-level information about available datasets. The logs mirror the console output of every data-processing script and describe the datasets they touch. Prefer summarising or citing the log contents instead of re-downloading raw data.
- Ensure any runnable script continues to emit meaningful, descriptive console output so that the generated logs remain informative for downstream agents.
- Keep log files deterministic: never add timestamps (to file names or contents) and always overwrite an existing log on each run.
- When modifying or creating scripts, initialise mirrored logging by calling `initialize_script_logging(__file__)` from `logging_setup.py` (for Python) or by using the established `tee` pattern in shell scripts so that console output is captured to `logs/<script>.log` with start/finish markers.
- Consult the dataset summaries in `data/intermediate/descriptions/` before regenerating intermediate artefacts; each Markdown file documents the shape, provenance, and consumers of its corresponding data product.
