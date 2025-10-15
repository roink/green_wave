# Agent instructions

- Use the log files under `logs/` to understand available datasets and their
  preprocessing status. Every analytics script mirrors its console output into
  timestamped log files, so those logs often include dataset statistics and
  descriptive summaries gathered at runtime.
- Every runnable script calls `configure_logging(__file__)` from
  `src/green_wave_logging.py` near the top of the module. Keep that call (or an
  equivalent alias) when editing existing files, and add it to new entry points
  so console output continues to be mirrored into logs.
- The lightweight bootstrap in `sitecustomize.py` exists for scenarios where
  Python automatically imports the module; avoid removing it, but prefer the
  explicit `configure_logging` call pattern for new scripts.
- If a change affects how datasets are produced or transformed, emit concise
  human-readable descriptions to stdout—those statements will also be captured
  in the logs and become available to downstream coding agents.
