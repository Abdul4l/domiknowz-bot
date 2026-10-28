"""
Minimal ADB screen capture for BlueStacks, used to feed domino_vision.py.

Usage:
    from adb_capture import grab_frame
    frame = grab_frame()          # BGR np.ndarray, ready for DominoDetector.detect()
"""

import subprocess
import numpy as np
import cv2

ADB_SERIAL = "127.0.0.1:5555"  # e.g. "127.0.0.1:5555" — set if you have multiple devices/emulators


def _adb(*args: str) -> bytes:
    cmd = ["adb"]
    if ADB_SERIAL:
        cmd += ["-s", ADB_SERIAL]
    cmd += list(args)
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def grab_frame() -> np.ndarray:
    """Captures the current BlueStacks screen via `adb exec-out screencap` and
    returns it as a BGR OpenCV image (no temp files, no PNG round-trip to disk)."""
    png_bytes = _adb("exec-out", "screencap", "-p")
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode screenshot from adb — check ADB_SERIAL / connection.")
    return frame


def tap(x: int, y: int) -> None:
    """Taps the device screen at (x, y) in screencap pixel coordinates.
    BlueStacks' screencap resolution normally matches its reported display
    size (`adb shell wm size`), so these coordinates should line up 1:1
    with touch input — verify once with a known on-screen button before
    relying on it, and add a scale factor here if your setup differs."""
    _adb("shell", "input", "tap", str(int(x)), str(int(y)))


def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 150) -> None:
    _adb("shell", "input", "swipe", str(int(x1)), str(int(y1)),
         str(int(x2)), str(int(y2)), str(int(duration_ms)))


if __name__ == "__main__":
    img = grab_frame()
    cv2.imwrite("screen_capture.png", img)
    print(f"Saved {img.shape[1]}x{img.shape[0]} screenshot to screen_capture.png")
