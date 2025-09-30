"""Utilities for adjusting the scheduling priority of processes."""

from __future__ import annotations

import os
from typing import Final

__all__ = ["lower_process_priority"]


# Constants defined in the Windows API for process priority management. Keeping
# them module-level avoids re-allocating them for every call while still letting
# the import of ``ctypes`` remain lazy (Windows-only).
_PROCESS_SET_INFORMATION: Final[int] = 0x0200
_BELOW_NORMAL_PRIORITY_CLASS: Final[int] = 0x00004000


def lower_process_priority() -> None:
    """Best-effort attempt to reduce the current process priority.

    On POSIX systems the function increases the niceness value to push the
    process toward the lowest priority. On Windows the below-normal process
    class is used so that CPU time is yielded to foreground tasks. Any errors
    are swallowed because priority management is opportunistic and should not
    interrupt the workload.
    """

    try:
        if os.name == "posix" and hasattr(os, "nice"):
            # ``os.nice`` returns the new niceness while modifying the priority
            # of the current process in-place.
            os.nice(19)
        elif os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, os.getpid())
            if handle:
                kernel32.SetPriorityClass(handle, _BELOW_NORMAL_PRIORITY_CLASS)
                kernel32.CloseHandle(handle)
    except Exception as exc:  # pragma: no cover - best effort behaviour
        print(f"Warning: unable to lower process priority: {exc}")
