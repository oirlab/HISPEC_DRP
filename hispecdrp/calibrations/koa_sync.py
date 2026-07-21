"""Helpers to sync KOA calibration metadata and download missing cache files."""

from __future__ import annotations

import logging
from typing import Any

from .hispec_store import HISPECCalibrationStore

logger = logging.getLogger(__name__)

__all__ = ["verify_koa_calibrations"]


def verify_koa_calibrations(
    spectrograph: str = "BSPEC",
    *,
    connect_remote: bool = True,
    sync_remote: bool = True,
    download_missing: bool = True,
) -> dict[str, Any]:
    """
    Sync KOA catalog metadata and download missing calibration files for one spectrograph.

    Uses ``KOA_CALIBRATION_CACHE`` for the local cache. When ``connect_remote`` /
    ``sync_remote`` are True, also requires ``KECK_OBSERVER_EMAIL`` and
    ``KECK_OBSERVER_PASSWORD`` for remote KOA access.

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

    Returns
    -------
    dict
        Summary with keys ``spectrograph``, ``n_records``, ``cal_types``,
        ``n_missing``, ``n_downloaded``, ``n_failed``, ``downloaded``, ``failed``.
    """
    spec = str(spectrograph).strip().upper()
    if spec not in ("BSPEC", "RSPEC"):
        raise ValueError(f"spectrograph must be BSPEC or RSPEC, got {spectrograph!r}")

    downloaded: list[str] = []
    failed: list[dict[str, str]] = []

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
            for i, rec in enumerate(missing, 1):
                filename = str(rec.get("filename") or "?")
                try:
                    path, _ = store.get_calibration(cal=rec)
                    downloaded.append(filename)
                    logger.info(
                        "[%d/%d] Downloaded %s -> %s",
                        i,
                        len(missing),
                        filename,
                        path,
                    )
                except Exception as e:
                    failed.append({"filename": filename, "error": str(e)})
                    logger.warning(
                        "[%d/%d] Failed to download %s: %s",
                        i,
                        len(missing),
                        filename,
                        e,
                    )

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
    }
    logger.info(
        "%s caldb: %d record(s); downloaded %d/%d missing file(s) (%d failed); types=%s",
        spec,
        summary["n_records"],
        summary["n_downloaded"],
        summary["n_missing"],
        summary["n_failed"],
        summary["cal_types"],
    )
    return summary
