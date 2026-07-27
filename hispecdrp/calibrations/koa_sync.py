"""Helpers to sync KOA calibration metadata and download missing cache files."""

from __future__ import annotations

import logging
import os
from typing import Any

from .download_verify import ensure_calibration_on_disk
from .hispec_store import HISPECCalibrationStore

logger = logging.getLogger(__name__)

__all__ = ["verify_koa_calibrations"]


def verify_koa_calibrations(
    spectrograph: str = "BSPEC",
    *,
    connect_remote: bool = True,
    sync_remote: bool = True,
    download_missing: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    """
    Sync KOA catalog metadata and download missing calibration files for one spectrograph.

    Uses ``KOA_CALIBRATION_CACHE`` for the local cache. When ``connect_remote`` /
    ``sync_remote`` are True, also requires ``KECK_OBSERVER_EMAIL`` and
    ``KECK_OBSERVER_PASSWORD`` for remote KOA access.

    Each download is verified on disk (exists, non-trivial size, expected basename)
    before it is counted as successful. Metadata sync alone is not treated as a
    successful file download.

    Parameters
    ----------
    spectrograph : str, optional
        Spectrograph to retain / download (``'BSPEC'`` or ``'RSPEC'``). Default ``'BSPEC'``.
    connect_remote : bool, optional
        Open a remote KOA connection. Default True.
    sync_remote : bool, optional
        Sync remote catalog IDs into the local SQLite DB. Default True.
    download_missing : bool, optional
        Download calibrations listed in the local DB but missing from the cache,
        filtered to ``spectrograph``. Default True.
    require_complete : bool, optional
        If True (default), raise ``RuntimeError`` when any download fails or files
        remain missing after the pass. Set False to only return the summary.

    Returns
    -------
    dict
        Summary with keys ``spectrograph``, ``n_records``, ``cal_types``,
        ``n_missing``, ``n_downloaded``, ``n_failed``, ``downloaded``, ``failed``,
        ``verified_paths``, ``still_missing``.
    """
    spec = str(spectrograph).strip().upper()
    if spec not in ("BSPEC", "RSPEC"):
        raise ValueError(f"spectrograph must be BSPEC or RSPEC, got {spectrograph!r}")

    downloaded: list[str] = []
    verified_paths: list[str] = []
    failed: list[dict[str, str]] = []
    still_missing: list[str] = []

    with HISPECCalibrationStore(connect_remote=connect_remote, sync_on_init=False) as store:
        if sync_remote:
            if not connect_remote or store.remote_db is None:
                logger.warning(
                    "Skipping remote sync (connect_remote=%s, remote_db=%s).",
                    connect_remote,
                    getattr(store, "remote_db", None),
                )
            else:
                added = store.sync_records_from_remote(mode="id")
                logger.info(
                    "Synced %d KOA calibration record(s) into the local caldb.",
                    len(added or []),
                )

        missing: list[dict] = []
        if download_missing:
            missing = [
                r
                for r in (store.get_missing_local_files() or [])
                if str(r.get("spectrograph", "")).upper() == spec
            ]
            if missing:
                logger.info(
                    "Found %d %s calibration file(s) in the local DB but missing from "
                    "KOA_CALIBRATION_CACHE=%s",
                    len(missing),
                    spec,
                    os.environ.get("KOA_CALIBRATION_CACHE", "<unset>"),
                )
            for i, rec in enumerate(missing, 1):
                filename = str(rec.get("filename") or "?")
                try:
                    path, _ = store.get_calibration(cal=rec)
                    path = ensure_calibration_on_disk(
                        path,
                        expected_filename=filename if filename != "?" else None,
                        cal_record=rec,
                    )
                    downloaded.append(filename)
                    verified_paths.append(path)
                    size = os.path.getsize(path)
                    logger.info(
                        "[%d/%d] Verified download %s -> %s (%d bytes)",
                        i,
                        len(missing),
                        filename,
                        path,
                        size,
                    )
                except Exception as e:
                    failed.append({"filename": filename, "error": str(e)})
                    logger.warning(
                        "[%d/%d] Failed to download/verify %s: %s",
                        i,
                        len(missing),
                        filename,
                        e,
                    )

            # Re-scan so we catch zip-basename mismatches and silent no-ops.
            still_missing = [
                str(r.get("filename") or r.get("id") or "?")
                for r in (store.get_missing_local_files() or [])
                if str(r.get("spectrograph", "")).upper() == spec
            ]

        rows = store.local_db.query() or []
        if not isinstance(rows, list):
            rows = [rows]
        spec_rows = [r for r in rows if str(r.get("spectrograph", "")).upper() == spec]
        cal_types = sorted({r.get("cal_type") for r in spec_rows if r.get("cal_type")})

    summary = {
        "spectrograph": spec,
        "n_records": len(spec_rows),
        "cal_types": cal_types,
        "n_missing": len(missing),
        "n_downloaded": len(downloaded),
        "n_failed": len(failed),
        "downloaded": downloaded,
        "failed": failed,
        "verified_paths": verified_paths,
        "still_missing": still_missing,
    }
    logger.info(
        "%s caldb: %d record(s); downloaded %d/%d missing file(s) (%d failed); "
        "still_missing=%d; types=%s",
        spec,
        summary["n_records"],
        summary["n_downloaded"],
        summary["n_missing"],
        summary["n_failed"],
        len(still_missing),
        summary["cal_types"],
    )
    if require_complete and (failed or still_missing):
        raise RuntimeError(
            f"KOA calibration download incomplete for {spec}: "
            f"{len(failed)} failed, {len(still_missing)} still missing from cache "
            f"({still_missing[:5]}{'...' if len(still_missing) > 5 else ''}). "
            "Check KECK_OBSERVER_* credentials, network access to the Keck calibrations "
            "API, and that KOA_CALIBRATION_CACHE is writable."
        )
    return summary
