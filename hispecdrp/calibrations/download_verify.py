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

# Minimal FITS primary HDU is one 2880-byte block; allow small headers but reject empty stubs.
MIN_CALIBRATION_BYTES = 2880


def normalize_downloaded_calibration_path(
    downloaded_path: str | None,
    expected_path: str,
) -> str:
    """
    Ensure the on-disk file lives at ``expected_path`` (``data_dir`` / DB filename).

    ``koa_middleware`` returns whatever basename is inside the KOA zip. If that differs
    from the DB ``filename``, later ``calibration_file_in_cache`` / Stage-0 open look
    under the DB name and raise ``FileNotFoundError`` even though a download "succeeded".
    """
    expected_path = os.path.abspath(os.path.expanduser(expected_path))
    if not downloaded_path:
        return expected_path
    downloaded_path = os.path.abspath(os.path.expanduser(downloaded_path))
    if downloaded_path == expected_path:
        return expected_path
    if os.path.isfile(downloaded_path):
        os.makedirs(os.path.dirname(expected_path) or ".", exist_ok=True)
        if os.path.isfile(expected_path):
            os.remove(expected_path)
        shutil.move(downloaded_path, expected_path)
    return expected_path


def ensure_calibration_on_disk(
    path: str | None,
    *,
    expected_filename: str | None = None,
    min_bytes: int = MIN_CALIBRATION_BYTES,
    cal_record: dict[str, Any] | None = None,
) -> str:
    """
    Raise ``FileNotFoundError`` / ``RuntimeError`` unless ``path`` is a real cal file.

    Parameters
    ----------
    path
        Local path returned by ``store.get_calibration`` / download.
    expected_filename
        Basename that must match ``path`` (usually the DB ``filename``).
    min_bytes
        Reject empty or truncated downloads (default one FITS block).
    cal_record
        Optional metadata for error messages.
    """
    label = expected_filename or (cal_record or {}).get("filename") or path or "?"
    if not path:
        raise FileNotFoundError(
            f"KOA calibration {label!r}: download returned no local path "
            f"(record id={(cal_record or {}).get('id')!r})."
        )
    path = os.path.abspath(os.path.expanduser(path))
    if expected_filename and os.path.basename(path) != os.path.basename(expected_filename):
        raise FileNotFoundError(
            f"KOA calibration {label!r}: download path basename {os.path.basename(path)!r} "
            f"does not match expected {os.path.basename(expected_filename)!r} "
            f"(full path={path})."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"KOA calibration {label!r}: reported path does not exist on disk: {path}. "
            "Authentication / caldb metadata sync can succeed while the FITS download "
            "fails or lands under a different name; re-run with a writable "
            "KOA_CALIBRATION_CACHE and check network access to the Keck calibrations API."
        )
    size = os.path.getsize(path)
    if size < int(min_bytes):
        raise RuntimeError(
            f"KOA calibration {label!r}: file at {path} is only {size} bytes "
            f"(need >= {min_bytes}). Treating as a failed / truncated download."
        )
    return path
