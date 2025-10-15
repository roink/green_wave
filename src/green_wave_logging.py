"""Utilities for mirroring console output into version-controlled logs."""

from __future__ import annotations

import atexit
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

_CONFIGURED = False


def _should_skip_logging() -> bool:
    return os.environ.get("GREEN_WAVE_DISABLE_AUTOMATIC_LOGGING") == "1"


def _slugify_script_name(script_path: str | None) -> str:
    if not script_path:
        return "interactive"
    name = script_path.replace(os.sep, "_").replace("/", "_")
    stem = Path(name).stem
    return stem or "interactive"


def _data_tree_snapshot(repo_root: Path) -> dict:
    data_dir = repo_root / "data"
    summary: dict[str, dict[str, object]] = {}

    for sub in ("raw", "intermediate", "finished"):
        folder = data_dir / sub
        if not folder.exists():
            continue

        file_count = 0
        total_bytes = 0
        sample_files: list[str] = []

        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
            if len(sample_files) < 5:
                sample_files.append(str(path.relative_to(repo_root)))

        summary[sub] = {
            "files": file_count,
            "size_mb": round(total_bytes / (1024 * 1024), 3),
            "samples": sample_files,
        }

    return summary


class _StreamTee(io.TextIOBase):
    """Mirror writes to the original stream and one or more secondary streams."""

    def __init__(self, primary: io.TextIOBase, mirrors: Iterable[io.TextIOBase]):
        self._primary = primary
        self._mirrors = tuple(mirrors)

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        written = self._primary.write(s)
        for stream in self._mirrors:
            stream.write(s)
        return written

    def flush(self) -> None:  # type: ignore[override]
        self._primary.flush()
        for stream in self._mirrors:
            stream.flush()

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._primary, "errors", None)

    def isatty(self) -> bool:  # type: ignore[override]
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:  # type: ignore[override]
        return getattr(self._primary, "fileno", lambda: -1)()

    def writable(self) -> bool:  # type: ignore[override]
        return True


def configure_logging(script_path: str | None = None) -> Path | None:
    """Mirror console output to a timestamped log file.

    Parameters
    ----------
    script_path:
        Optional hint about the current script. When omitted, ``sys.argv[0]`` is
        used.
    """

    global _CONFIGURED

    if _CONFIGURED or _should_skip_logging():
        return None

    repo_root = Path(__file__).resolve().parents[1]
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)

    script_slug = _slugify_script_name(script_path or sys.argv[0])
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{script_slug}-{timestamp}.log"

    log_file = log_path.open("w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    header = {
        "timestamp_utc": timestamp,
        "script": script_path or sys.argv[0],
        "working_directory": os.getcwd(),
        "data_snapshot": _data_tree_snapshot(repo_root),
    }
    log_file.write("# Green Wave execution log\n")
    log_file.write(json.dumps(header, indent=2, sort_keys=True))
    log_file.write("\n\n")
    log_file.flush()

    sys.stdout = _StreamTee(original_stdout, (log_file,))
    sys.stderr = _StreamTee(original_stderr, (log_file,))

    def _close_log() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(_close_log)
    _CONFIGURED = True
    return log_path
