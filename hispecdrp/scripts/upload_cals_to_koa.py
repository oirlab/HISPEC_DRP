#!/usr/bin/env python
"""
Add calibration files from a target directory to the local calDB, then upload them to KOA.

Reads FITS calibration files from a given directory, registers each with an
HISPECCalibrationStore (so they are copied into the cache and added to the local
SQLite DB), then uses the store's sync_records_to_remote() to upload local-only
entries to the Keck Observatory Archive (KOA) via koa_middleware.

Requires KOA_CALIBRATION_CACHE to be set (or it will use the default cache).
Requires remote authentication for upload (see KOA Middleware docs).

Usage:
    python -m hispecdrp.scripts.upload_cals_to_koa /path/to/calibrations [--origin LOCAL] [--no-upload] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

from hispecdrp.calibrations import HISPECCalibrationStore
from hispecdrp.utils.logging import LogConfig, ROOT_LOGGER

logger = logging.getLogger(__name__)


def _ensure_cache_env() -> None:
    """Set KOA_CALIBRATION_CACHE to a default if not already set."""
    if os.environ.get("KOA_CALIBRATION_CACHE"):
        return
    try:
        from astropy.utils.data import get_cache_dir
        cache_root = get_cache_dir()
    except Exception:
        cache_root = os.path.expanduser("~/.cache")
    default_cache = os.path.join(cache_root, "hispecdrp_calibration_cache")
    os.environ["KOA_CALIBRATION_CACHE"] = default_cache + os.sep
    os.makedirs(default_cache, exist_ok=True)
    logger.info(
        "KOA_CALIBRATION_CACHE not set; using default: %s",
        os.environ["KOA_CALIBRATION_CACHE"],
    )


def run(
    target_dir: str,
    *,
    origin: str = "LOCAL",
    upload: bool = True,
    dry_run: bool = False,
    instrument_name: str | None = None,
) -> None:
    """
    Add FITS files from target_dir to the store and optionally upload to KOA.

    Parameters
    ----------
    target_dir : str
        Directory containing calibration FITS files to add.
    origin : str, optional
        Origin to assign to registered calibrations (e.g. 'LOCAL', 'KOA').
        Default is 'LOCAL'.
    upload : bool, optional
        If True, call sync_records_to_remote() after registering files.
        Default is True.
    dry_run : bool, optional
        If True, do not register or upload; only list files that would be processed.
        Default is False.
    instrument_name : str or None, optional
        Instrument name for the store. If None, uses env or HISPEC default.
    """
    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"Target directory does not exist: {target_dir}")

    fits_files = sorted(glob.glob(os.path.join(target_dir, "*.fits")))
    if not fits_files:
        logger.warning("No *.fits files found in %s", target_dir)
        return

    if dry_run:
        logger.info("Dry run: would process %d file(s) from %s", len(fits_files), target_dir)
        for fp in fits_files:
            logger.info("  %s", os.path.basename(fp))
        return

    _ensure_cache_env()

    with HISPECCalibrationStore(
        instrument_name=instrument_name,
        connect_remote=upload,
    ) as store:
        registered = []
        for filepath in fits_files:
            try:
                local_path, cal_record = store.register_calibration(
                    filepath,
                    origin=origin,
                    new_version=False,
                )
                if cal_record is not None:
                    registered.append(cal_record)
                    logger.info("Registered: %s -> %s", os.path.basename(filepath), local_path)
                else:
                    logger.debug("Skipped (already in cache): %s", filepath)
            except Exception as e:
                logger.exception("Failed to register %s: %s", filepath, e)

        if not registered:
            logger.info("No new calibrations registered.")
            return

        if upload and store.remote_db is not None:
            try:
                uploaded = store.sync_records_to_remote(mode="id")
                logger.info("Uploaded %d calibration(s) to KOA.", len(uploaded))
                for rec in uploaded:
                    logger.info("  %s", rec.get("filename") or rec.get("cal_id") or rec)
            except Exception as e:
                logger.exception("Upload to KOA failed: %s", e)
        elif upload and store.remote_db is None:
            logger.warning("Remote DB not available; skipping upload. Set connect_remote=True and check auth.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add calibration files from a directory to the local calDB and upload to KOA.",
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Directory containing calibration FITS files (default: current directory).",
    )
    parser.add_argument(
        "--origin",
        "-o",
        default="LOCAL",
        help="Origin to assign to registered calibrations (default: LOCAL).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only add to local store; do not sync to remote KOA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be processed; do not register or upload.",
    )
    parser.add_argument(
        "--instrument-name",
        default=None,
        help="Instrument name for the store (default: from env or HISPEC).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    config = LogConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        enable_stdout=True,
        enable_file=False,
    )
    config.apply([ROOT_LOGGER])  # only root; script logger propagates, so no duplicate messages

    run(
        args.target_dir,
        origin=args.origin,
        upload=not args.no_upload,
        dry_run=args.dry_run,
        instrument_name=args.instrument_name or None,
    )


if __name__ == "__main__":
    main()
