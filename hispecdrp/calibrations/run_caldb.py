#!/usr/bin/env python
"""
Sync calibration metadata from KOA and download missing calibration files.

This script uses the HISPECCalibrationStore to sync calibration records from
the remote Keck Observatory Archive (KOA) into the local SQLite database,
download any calibration files that are in the DB but missing from the local
cache, and register calibration files that exist in the cache but are not yet
in the DB. At the end of each run it prints a summary of all files in the
local calibration store.

The calibration cache and local database path is taken from the
KOA_CALIBRATION_CACHE environment variable. If unset, it defaults to
<astropy cache>/hispecdrp_calibration_cache/.

See docs/source/installation.rst and
https://oirlab.github.io/KOA_Middleware/installation.html for setup and
authentication.
"""

from __future__ import annotations

# Allow running as script: python hispecdrp/calibrations/run_caldb.py (relative imports need package context)
if __name__ == "__main__" and __package__ is None:
    import sys
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "hispecdrp.calibrations"

import argparse
import logging
import os
import sys

from .. import datamodels
from ..utils.logging import ROOT_LOGGER, setup_default_logging
from . import (
    HISPECCalibrationStore,
    DQSelector,
    SaturationSelector,
    GainSelector,
    DetectorFlatSelector,
    BiasSelector,
    DarkSelector,
    ReadNoiseSelector,
)

__all__ = ["run", "main", "get_master_calibrations"]

logger = logging.getLogger(__name__)


# Metadata keys to show in the local store summary (skip internal/id-like keys)
_SUMMARY_META_KEYS = (
    "filename",
    "cal_type",
    "datetime_obs",
    "cal_version",
    "origin",
    "spectrograph",
    "master_cal",
)


def _print_local_store_summary(store: HISPECCalibrationStore) -> None:
    """
    Print a summary of each file in the local calibration store.

    Queries the store's local database and logs each calibration's filename,
    type, and selected metadata (e.g. datetime_obs, cal_version, origin).
    If the store is empty or the query fails, logs a short message instead.

    Parameters
    ----------
    store : HISPECCalibrationStore
        The open calibration store whose local DB should be summarized.

    Returns
    -------
    None
    """
    try:
        # Query local DB directly; store.query() can return list, single dict, or None
        raw = store.local_db.query() if store.local_db else None
        if raw is None:
            records = []
        elif isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = [raw]
        else:
            records = []
    except Exception as e:
        logger.warning("Could not query local store for summary: %s", e)
        return
    if not records:
        logger.info("Local calibration store is empty.")
        return
    logger.info("Local calibration store summary (%d file(s)):", len(records))
    for i, rec in enumerate(records, 1):
        name = rec.get("filename") or rec.get("id") or "?"
        cal_type = rec.get("cal_type") or "?"
        meta_parts = [
            f"{k}={rec.get(k)}"
            for k in _SUMMARY_META_KEYS
            if k not in ("filename", "cal_type") and rec.get(k) is not None
        ]
        meta_str = ", ".join(meta_parts) if meta_parts else "(no extra metadata)"
        logger.info("  [%d] %s  type=%s  %s", i, name, cal_type, meta_str)


def _ensure_cache_env() -> None:
    """
    Set KOA_CALIBRATION_CACHE to a default if not already set.

    If the environment variable is unset, sets it to a directory under the
    astropy cache (or ~/.cache if that is unavailable) and creates the
    directory. Logs the chosen path when the default is used.

    Returns
    -------
    None
    """
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


def get_master_calibrations(
    meta: dict | None = None,
    *,
    instrument_name: str | None = None,
    instrument_era: str = "0.0.1",
    spectrograph: str = "RSPEC",
    mjd_start: float | None = None,
    datetime_obs: str | None = None,
    connect_remote: bool = True,
    sync_first: bool = True,
) -> dict[str, datamodels.CalibrationModel]:
    """
    Fetch master calibration models from the calibration store for the given meta.

    Uses the same store and selectors as :func:`run`: syncs from remote (if
    ``sync_first``), then selects and downloads the appropriate DQ, saturation,
    gain, detector flat, bias, dark, and read noise calibrations for the
    observation time and spectrograph. Each calibration file is downloaded if
    not already in the cache.

    Parameters
    ----------
    meta : dict or None, optional
        Metadata dict with keys ``instrument_era``, ``spectrograph``,
        ``mjd_start`` (used by selectors). If None, built from the keyword
        arguments below. Default is None.
    instrument_name : str or None, optional
        Instrument name for the store. Used when meta is None. Default is None.
    instrument_era : str, optional
        Instrument era for selection. Used when meta is None. Default is "0.0.1".
    spectrograph : str, optional
        Spectrograph (e.g. 'RSPEC', 'BSPEC'). Used when meta is None.
        Default is "RSPEC".
    mjd_start : float or None, optional
        MJD start time for selection. Used when meta is None. If both
        mjd_start and datetime_obs are None, uses current time. Default is None.
    datetime_obs : str or None, optional
        ISO datetime for observation. Used when meta is None to set mjd_start
        if mjd_start is not provided. Default is None.
    connect_remote : bool, optional
        Whether to connect to remote KOA. Default is True.
    sync_first : bool, optional
        If True, sync records from remote and add local cache files to the DB
        before selecting. Default is True.

    Returns
    -------
    dict
        Dictionary with keys: ``dq``, ``saturation``, ``gain``, ``detflat``,
        ``bias``, ``dark``, ``rn``. Each value is the corresponding
        calibration data model (opened from the store).

    Raises
    ------
    ValueError
        If a required calibration has no candidate in the store.
    """
    _ensure_cache_env()
    logger.info(
        "Fetching master calibrations (instrument_era=%s, spectrograph=%s, sync_first=%s).",
        instrument_era if meta is None else meta.get("instrument_era"),
        spectrograph if meta is None else meta.get("spectrograph"),
        sync_first,
    )

    if meta is None:
        if mjd_start is None and datetime_obs is not None:
            import astropy.time as at
            mjd_start = at.Time(datetime_obs, format="isot", scale="utc").mjd
        if mjd_start is None:
            import astropy.time as at
            mjd_start = at.Time.now().mjd
        meta = {
            "instrument_era": instrument_era,
            "spectrograph": spectrograph,
            "mjd_start": mjd_start,
        }

    # Store expects a DataModel-like input (with .meta) for select_and_get_calibration
    if not isinstance(meta, datamodels.DataModel):
        raw_name = instrument_name or os.getenv("HISPECDRP_INSTRUMENT_NAME", "HISPEC")
        instrument_name_meta = raw_name.upper() if isinstance(raw_name, str) else "HISPEC"
        input_model = datamodels.L0Model(
            meta={**meta, "instrument_name": instrument_name_meta},
            meta_only=True,
        )
        input_for_store = input_model
    else:
        input_for_store = meta

    selectors = [
        ("dq", DQSelector()),
        ("saturation", SaturationSelector()),
        ("gain", GainSelector()),
        ("detflat", DetectorFlatSelector()),
        ("bias", BiasSelector()),
        ("dark", DarkSelector()),
        ("rn", ReadNoiseSelector()),
    ]

    with HISPECCalibrationStore(
        instrument_name=instrument_name,
        connect_remote=connect_remote,
        sync_on_init=False,
    ) as store:
        if sync_first and not (connect_remote and store.remote_db is None):
            try:
                if connect_remote and store.remote_db is not None:
                    store.sync_records_from_remote(mode="id")
                store.sync_local_db_from_local_files()
            except Exception as e:
                logger.warning("Sync before get_master_calibrations failed: %s", e)

        result = {}
        for key, selector in selectors:
            try:
                # Selector expects subscriptable meta (e.g. meta["instrument_era"]); pass .meta not the model
                filepath, _ = store.select_and_get_calibration(input_for_store.meta, selector=selector)
                result[key] = datamodels.open(filepath)
                logger.debug("Loaded master calibration %s from %s", key, filepath)
            except Exception as e:
                raise ValueError(
                    f"Could not get master calibration for {key!r} (instrument_era={meta.get('instrument_era')!r}, "
                    f"spectrograph={meta.get('spectrograph')!r}, mjd_start={meta.get('mjd_start')}). "
                    f"Run run_caldb to sync/download calibrations, or use simulated calibrations. Error: {e}"
                ) from e

    logger.info("Loaded %d master calibration(s).", len(result))
    return result


def run(
    *,
    instrument_name: str | None = None,
    connect_remote: bool = True,
    sync_only: bool = False,
    download_missing: bool = True,
    cal_type: str | None = None,
    sync_mode: str = "id",
    dry_run: bool = False,
) -> None:
    """
    Sync the calibration store from remote, download missing files, and print a summary.

    Ensures KOA_CALIBRATION_CACHE is set, then opens a HISPECCalibrationStore and
    (1) optionally syncs calibration metadata from the remote KOA into the local DB,
    (2) optionally downloads calibration files that are in the DB but not in the cache,
    (3) registers any FITS files in the cache that are not yet in the DB,
    (4) prints a summary of all calibrations in the local store.

    Parameters
    ----------
    instrument_name : str or None, optional
        Instrument name (e.g. 'hispec'). If None, uses HISPECDRP_INSTRUMENT_NAME
        or the package default.
    connect_remote : bool, optional
        Whether to connect to the remote KOA. If True, sync from remote is attempted
        when remote_db is available. Default is True.
    sync_only : bool, optional
        If True, only sync metadata from remote and skip downloading missing files.
        Default is False.
    download_missing : bool, optional
        If True (and not sync_only), download calibrations that are in the local DB
        but missing from the cache. Default is True.
    cal_type : str or None, optional
        If set, restrict downloads to this calibration type (e.g. 'dark', 'flat').
        Default is None (all types).
    sync_mode : str, optional
        Mode for sync_records_from_remote: 'id' or 'last_updated'. Default is 'id'.
    dry_run : bool, optional
        If True, log intended actions without syncing or downloading. Default is False.

    Returns
    -------
    None
    """
    _ensure_cache_env()

    with HISPECCalibrationStore(
        instrument_name=instrument_name,
        connect_remote=connect_remote,
    ) as store:
        if dry_run:
            logger.info("Dry run: no changes will be made.")

        # Sync metadata from remote into local DB (requires remote_db to be initialized)
        if connect_remote and store.remote_db is not None:
            logger.info("Syncing calibration records from remote (mode=%s)...", sync_mode)
            if not dry_run:
                added = store.sync_records_from_remote(mode=sync_mode)
                logger.info("Synced %d calibration record(s) from remote.", len(added))
            else:
                logger.info("(Dry run: skipping sync_records_from_remote)")
        elif connect_remote and store.remote_db is None:
            logger.warning(
                "Remote connection not available (remote_db is None). "
                "Skipping sync from remote; only local cache will be used."
            )
        else:
            logger.info("Skipping remote sync (connect_remote=False).")

        if sync_only:
            logger.info("Sync only: skipping download of missing files.")
        else:
            # Find calibrations that are in the local DB but not in the cache
            try:
                missing = store.get_missing_local_files()
            except AttributeError as e:
                if "'NoneType' object has no attribute 'query'" in str(e):
                    logger.warning(
                        "Could not list missing files (remote DB not available). Skipping download step."
                    )
                    missing = []
                else:
                    raise
            if cal_type is not None:
                missing = [m for m in missing if m.get("cal_type") == cal_type]
            logger.info("Found %d calibration(s) in DB but missing from cache.", len(missing))

            if missing:
                if dry_run:
                    for rec in missing[:10]:
                        logger.info("  Would download: %s (id=%s)", rec.get("filename"), rec.get("id"))
                    if len(missing) > 10:
                        logger.info("  ... and %d more.", len(missing) - 10)
                else:
                    # Download each missing calibration (get_calibration downloads if not cached)
                    n_ok = 0
                    n_fail = 0
                    for i, rec in enumerate(missing, 1):
                        cal_id = rec.get("id")
                        filename = rec.get("filename", "?")
                        try:
                            local_path, _ = store.get_calibration(cal=rec)
                            size = (
                                os.path.getsize(local_path)
                                if local_path and os.path.isfile(local_path)
                                else None
                            )
                            logger.info(
                                "[%d/%d] Verified: %s -> %s (%s bytes)",
                                i,
                                len(missing),
                                filename,
                                local_path,
                                size,
                            )
                            n_ok += 1
                        except Exception as e:
                            n_fail += 1
                            logger.warning(
                                "[%d/%d] Failed to get calibration %s: %s",
                                i,
                                len(missing),
                                filename,
                                e,
                            )
                    still = store.get_missing_local_files() or []
                    if cal_type is not None:
                        still = [m for m in still if m.get("cal_type") == cal_type]
                    if n_fail or still:
                        raise RuntimeError(
                            f"KOA download incomplete: {n_ok} verified, {n_fail} failed, "
                            f"{len(still)} still missing from cache. "
                            "Metadata sync is not enough — FITS files must exist under "
                            "KOA_CALIBRATION_CACHE."
                        )

        # Add any files currently in the local cache that are not yet in the cal DB
        if not dry_run:
            try:
                cals_added = store.sync_local_db_from_local_files()
                if cals_added:
                    logger.info("Added %d file(s) from local cache to calibration DB.", len(cals_added))
            except Exception as e:
                logger.warning("Could not sync local cache files into cal DB: %s", e)

        # Summary of all files in the local calibration store
        _print_local_store_summary(store)


def main() -> None:
    """
    Parse command-line arguments and run the calibration DB sync.

    Configures logging via :func:`hispecdrp.utils.logging.setup_default_logging`, then calls
    :func:`run` with the parsed options. See the script help (e.g.
    ``python -m hispecdrp.calibrations.run_caldb --help``) for available arguments.

    Returns
    -------
    None
    """
    setup_default_logging()
    logging.getLogger(ROOT_LOGGER).setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="Sync HISPEC calibration DB from KOA and download missing calibration files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instrument-name",
        default=None,
        help="Instrument name (e.g. hispec). Default: HISPECDRP_INSTRUMENT_NAME or package default.",
    )
    parser.add_argument(
        "--no-remote",
        action="store_true",
        dest="no_remote",
        help="Do not connect to remote KOA; only use local cache and DB.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync metadata from remote; do not download missing files.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        dest="no_download",
        help="Do not download missing files (only sync metadata if --sync-only is not set).",
    )
    parser.add_argument(
        "--cal-type",
        default=None,
        help="If set, only download missing calibrations of this type (e.g. dark, flat, bias).",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("id", "last_updated"),
        default="id",
        help="Mode for sync_records_from_remote: 'id' or 'last_updated'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without syncing or downloading.",
    )
    args = parser.parse_args()

    run(
        instrument_name=args.instrument_name,
        connect_remote=not args.no_remote,
        sync_only=args.sync_only,
        download_missing=not args.no_download,
        cal_type=args.cal_type,
        sync_mode=args.sync_mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
