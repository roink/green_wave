"""Automatic logging bootstrap for standalone script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    try:
        import importlib

        spec = importlib.util.spec_from_file_location(
            "green_wave_logging",
            repo_root / "src" / "green_wave_logging.py",
        )
        if spec and spec.loader:
            module = sys.modules.get("green_wave_logging")
            if module is None:
                module = importlib.util.module_from_spec(spec)
                sys.modules["green_wave_logging"] = module
                spec.loader.exec_module(module)
            elif hasattr(module, "__spec__") and module.__spec__ is spec:
                # Module already initialised in this interpreter; reuse it.
                spec.loader.exec_module(module)
            if hasattr(module, "configure_logging"):
                module.configure_logging()
    except Exception:  # pragma: no cover - logging must never block script start
        # Avoid breaking user workflows if logging fails to initialise.
        return


_bootstrap()
