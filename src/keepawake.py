"""
ClinScope — keep Windows awake for the duration of a long job.

Background training on this machine has been silently killed mid-run. Event-log
forensics (2026-06) showed the real culprit was a *reboot* (Windows Update
auto-restart / a manual "update & restart"), NOT sleep — pause Windows Update
before long runs to stop that. This helper covers the secondary risk: a single
GPU box being lost to system sleep. It asserts ES_SYSTEM_REQUIRED so Windows
won't sleep / idle-suspend while the process runs, independent of the active
power plan (the Windows equivalent of macOS `caffeinate`).

Deliberately does NOT set ES_DISPLAY_REQUIRED, so the monitor can still turn off
on idle — turning the display off does not kill the process. The assertion is
cleared automatically when the process exits. Best-effort: a no-op on non-Windows
or if the call fails.
"""
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> bool:
    """Block system sleep until this process exits. Returns True if armed."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[keep-awake] system sleep blocked for this run "
              "(monitor may still sleep; does NOT stop a Windows Update reboot)")
        return True
    except Exception as e:  # ctypes missing, non-Windows kernel, etc.
        print(f"[keep-awake] could not arm ({e}); continuing without it")
        return False
