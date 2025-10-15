"""Utilities for mirroring script output to deterministic log files."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path
from typing import Iterable, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"

_log_file_path: Path | None = None


class TeeStream:
    """A file-like object that duplicates writes to multiple streams."""

    def __init__(self, streams: Iterable[TextIO]) -> None:
        self._streams = list(streams)
        self.encoding = getattr(self._streams[0], "encoding", "utf-8") if self._streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except ValueError:
                # Streams may already be closed during interpreter shutdown.
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except ValueError:
                # Ignore already-closed streams when flushing at exit.
                continue

    def isatty(self) -> bool:  # pragma: no cover - passthrough behaviour
        return False


_defused = False


def initialize_script_logging(script_path: str | Path) -> Path:
    """Mirror stdout/stderr to a log file for the executing script."""

    global _defused
    global _log_file_path
    if _defused and _log_file_path is not None:
        return _log_file_path

    script_name = Path(script_path).stem
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / f"{script_name}.log"

    log_file = log_file_path.open("w", encoding="utf-8")

    stdout_stream = TeeStream([sys.__stdout__, log_file])
    stderr_stream = TeeStream([sys.__stderr__, log_file])
    sys.stdout = stdout_stream  # type: ignore[assignment]
    sys.stderr = stderr_stream  # type: ignore[assignment]

    start_message = f"=== RUN START ({script_name}) ==="
    stdout_stream.write(start_message + "\n")
    stdout_stream.flush()

    def finish() -> None:
        finish_message = f"=== RUN FINISH ({script_name}) ==="
        stdout_stream.write(finish_message + "\n")
        stdout_stream.flush()
        if not log_file.closed:
            log_file.close()

    atexit.register(finish)
    _defused = True
    _log_file_path = log_file_path
    return log_file_path
