"""Post-download checks so KOA sync cannot report success without a real file."""

from __future__ import annotations

import os
import shutil
from typing import Any

__all__ = [
    "MIN_CALIBRATION_BYTES",
    "ensure_calibration_on_disk",
    "normalize_downloaded_calibration_path",
]

MIN_CALIBRATION_BYTES = 2880


def normalize_downloaded_calibration_path(
    downloaded_path: str | None,
    expected_path: str,
) -> str:
    """Ensure the on-disk file lives at ``expected_path`` (data_dir / DB filename)."""
    expected = os.path.abspath(os.path.expanduser(expected_path))
    if downloaded_path is None:
        return expected
    downloaded = os.path.abspath(os.path.expanduser(downloaded_path))
    if downloaded == expected:
        return expected
    if not os.path.isfile(downloaded):
        return expected
    os.makedirs(os.path.dirname(expected), exist_ok=True)
    if os.path.isfile(expected) and downloaded != expected:
        try:
            os.remove(expected)
        except OSError:
            pass
    shutil.move(downloaded, expected)
    return expected


def ensure_calibration_on_disk(
    path: str | None,
    *,
    expected_filename: str | None = None,
    cal_record: dict[str, Any] | None = None,
    min_bytes: int = MIN_CALIBRATION_BYTES,
) -> str:
    """Raise if ``path`` is missing or too small to be a real calibration FITS."""
    if path is None:
        name = expected_filename or (cal_record or {}).get("filename") or "calibration"
        raise FileNotFoundError(f"{name} does not exist on disk")
    resolved = os.path.abspath(os.path.expanduser(path))
    name = expected_filename or os.path.basename(resolved)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{name} does not exist on disk: {resolved}")
    size = os.path.getsize(resolved)
    if int(size) < int(min_bytes):
        raise RuntimeError(
            f"{name} is too small ({size} bytes < {min_bytes}); "
            "likely an empty/truncated download stub"
        )
    return resolved
