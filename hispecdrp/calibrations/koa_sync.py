"""Helpers to sync KOA calibration metadata and download missing cache files."""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from .download_verify import ensure_calibration_on_disk
from .hispec_store import HISPECCalibrationStore

logger = logging.getLogger(__name__)

__all__ = [
    "verify_koa_calibrations",
    "seed_koa_calibration_cache",
    "ensure_koa_calibrations",
]



def verify_koa_calibrations(
    spectrograph: str = "BSPEC",
    *,
    connect_remote: bool = True,
    sync_remote: bool = True,
    download_missing: bool = True,
    require_complete: bool = True,
    ignore_missing_cal_types: list[str] | tuple[str, ...] | None = None,
    fallback_local: bool = True,
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
    ignore_missing_cal_types : list of str or None, optional
        Calibration types that may be absent from the cache without failing
        ``require_complete`` (and are skipped for download attempts). Useful for
        simulation notebooks that register a packaged ``lfc_wavesol`` instead of
        the remote geometric product. Example: ``["lfc_wavesol"]``.
    fallback_local : bool, optional
        If True (default) and a remote connect/sync fails with a recoverable
        auth/network/API error, retry once with ``connect_remote=False`` and
        ``sync_remote=False`` so notebooks need no try/except boilerplate.

    Returns
    -------
    dict
        Summary with keys ``spectrograph``, ``n_records``, ``cal_types``,
        ``n_missing``, ``n_downloaded``, ``n_failed``, ``downloaded``, ``failed``,
        ``verified_paths``, ``still_missing``, ``ignored_missing``.
        On local fallback, also ``remote_mode='local_fallback'`` and ``remote_error``.
    """
    try:
        summary = _verify_koa_calibrations_once(
            spectrograph,
            connect_remote=connect_remote,
            sync_remote=sync_remote,
            download_missing=download_missing,
            require_complete=require_complete,
            ignore_missing_cal_types=ignore_missing_cal_types,
        )
        summary.setdefault("remote_mode", "remote" if connect_remote else "local")
        return summary
    except (RuntimeError, ValueError, OSError) as exc:
        if not (
            fallback_local
            and connect_remote
            and _is_recoverable_remote_error(exc)
        ):
            raise
        logger.warning(
            "Remote KOA sync/login failed (%s: %s); "
            "retrying with local cache only.",
            type(exc).__name__,
            str(exc).splitlines()[0] if str(exc) else exc,
        )
        summary = _verify_koa_calibrations_once(
            spectrograph,
            connect_remote=False,
            sync_remote=False,
            download_missing=download_missing,
            require_complete=require_complete,
            ignore_missing_cal_types=ignore_missing_cal_types,
        )
        summary["remote_mode"] = "local_fallback"
        summary["remote_error"] = f"{type(exc).__name__}: {exc}"
        return summary


def _verify_koa_calibrations_once(
    spectrograph: str = "BSPEC",
    *,
    connect_remote: bool = True,
    sync_remote: bool = True,
    download_missing: bool = True,
    require_complete: bool = True,
    ignore_missing_cal_types: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Single-pass KOA verify (no local fallback). See :func:`verify_koa_calibrations`."""
    spec = str(spectrograph).strip().upper()
    if spec not in ("BSPEC", "RSPEC"):
        raise ValueError(f"spectrograph must be BSPEC or RSPEC, got {spectrograph!r}")

    ignore_types = {
        str(t).strip().lower()
        for t in (ignore_missing_cal_types or ())
        if str(t).strip()
    }

    def _ignored(rec: dict) -> bool:
        return str(rec.get("cal_type", "")).strip().lower() in ignore_types

    downloaded: list[str] = []
    verified_paths: list[str] = []
    failed: list[dict[str, str]] = []
    still_missing: list[str] = []
    ignored_missing: list[str] = []

    with HISPECCalibrationStore(connect_remote=connect_remote, sync_on_init=False) as store:
        if sync_remote:
            if not connect_remote or store.remote_db is None:
                logger.warning(
                    "Skipping remote sync (connect_remote=%s, remote_db=%s).",
                    connect_remote,
                    getattr(store, "remote_db", None),
                )
            else:
                try:
                    added = store.sync_records_from_remote(mode="id")
                except RuntimeError as exc:
                    # Belt-and-suspenders: store soft-fails most remote 404s, but
                    # keep verify usable when the Keck catalog API is down.
                    msg = str(exc)
                    if "Failed to query metadata" not in msg and "404" not in msg:
                        raise
                    logger.warning(
                        "Remote KOA metadata sync failed (%s); "
                        "continuing with local caldb only.",
                        msg.splitlines()[0] if msg else exc,
                    )
                    added = []
                logger.info(
                    "Synced %d KOA calibration record(s) into the local caldb.",
                    len(added or []),
                )

        missing: list[dict] = []
        if download_missing:
            missing_all = [
                r
                for r in (store.get_missing_local_files() or [])
                if str(r.get("spectrograph", "")).upper() == spec
            ]
            ignored_missing = [
                str(r.get("filename") or r.get("id") or "?")
                for r in missing_all
                if _ignored(r)
            ]
            missing = [r for r in missing_all if not _ignored(r)]
            if ignored_missing:
                logger.info(
                    "Ignoring %d missing %s calibration(s) of type(s) %s: %s",
                    len(ignored_missing),
                    spec,
                    sorted(ignore_types),
                    ignored_missing[:5],
                )
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
            still_missing_recs = [
                r
                for r in (store.get_missing_local_files() or [])
                if str(r.get("spectrograph", "")).upper() == spec and not _ignored(r)
            ]
            still_missing = [
                str(r.get("filename") or r.get("id") or "?")
                for r in still_missing_recs
            ]
            # Refresh ignored list from the full re-scan.
            ignored_missing = [
                str(r.get("filename") or r.get("id") or "?")
                for r in (store.get_missing_local_files() or [])
                if str(r.get("spectrograph", "")).upper() == spec and _ignored(r)
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
        "ignored_missing": ignored_missing,
    }
    logger.info(
        "%s caldb: %d record(s); downloaded %d/%d missing file(s) (%d failed); "
        "still_missing=%d; ignored_missing=%d; types=%s",
        spec,
        summary["n_records"],
        summary["n_downloaded"],
        summary["n_missing"],
        summary["n_failed"],
        len(still_missing),
        len(ignored_missing),
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

def _koa_cache_fit_count(cache_dir: str | os.PathLike[str]) -> int:
    root = os.path.join(str(cache_dir), "calibrations")
    if not os.path.isdir(root):
        return 0
    n = 0
    for _dirpath, _dirnames, filenames in os.walk(root):
        n += sum(1 for name in filenames if name.lower().endswith(".fits"))
    return n


def _koa_cache_has_db(cache_dir: str | os.PathLike[str]) -> bool:
    cache_dir = str(cache_dir)
    for name in (
        os.path.join(cache_dir, "database", "hispec_calibrations.db"),
        os.path.join(cache_dir, "hispec_calibrations.db"),
    ):
        if os.path.isfile(name):
            return True
    return False


def _discover_koa_cache_sources(
    dest: str,
    *,
    extra: Sequence[str | os.PathLike[str]] | None = None,
) -> list[str]:
    """Candidate seed caches, richest first (by FITS count), excluding ``dest``."""
    dest_abs = os.path.abspath(dest)
    cands: list[str] = []

    def _add(path: str | os.PathLike[str] | None) -> None:
        if not path:
            return
        p = os.path.abspath(os.path.expanduser(str(path)))
        if p == dest_abs or not os.path.isdir(p):
            return
        if p not in cands:
            cands.append(p)

    for path in extra or ():
        _add(path)

    env_seed = (os.environ.get("HISPEC_KOA_CACHE_SEED") or "").strip()
    if env_seed:
        _add(env_seed)

    home = os.path.expanduser("~")
    try:
        for name in os.listdir(home):
            if not name.startswith("hispec"):
                continue
            _add(os.path.join(home, name, "cals", "KOA_CALIBRATION_CACHE"))
    except OSError:
        pass

    # Rank by FITS count then DB presence.
    ranked = sorted(
        cands,
        key=lambda p: (_koa_cache_fit_count(p), _koa_cache_has_db(p)),
        reverse=True,
    )
    return [p for p in ranked if _koa_cache_fit_count(p) > 0 or _koa_cache_has_db(p)]


def seed_koa_calibration_cache(
    dest: str | os.PathLike[str] | None = None,
    *,
    sources: Sequence[str | os.PathLike[str]] | None = None,
    min_fits: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """
    Copy a populated KOA cache into ``dest`` when the destination is empty/thin.

    Parameters
    ----------
    dest
        Target ``KOA_CALIBRATION_CACHE`` directory. Defaults to the current
        ``KOA_CALIBRATION_CACHE`` environment variable.
    sources
        Optional explicit seed directories (tried first). Otherwise discovers
        ``~/hispec*/cals/KOA_CALIBRATION_CACHE`` and ``HISPEC_KOA_CACHE_SEED``.
    min_fits
        Destination is considered populated once it has at least this many
        FITS files under ``calibrations/``.
    force
        If True, replace the destination from the richest source even when
        ``dest`` already meets ``min_fits``.

    Returns
    -------
    dict
        ``dest``, ``seeded``, ``source``, ``n_fits_before``, ``n_fits_after``,
        ``candidates``.
    """
    import shutil

    dest_path = (str(dest).strip() if dest is not None else "") or (
        os.environ.get("KOA_CALIBRATION_CACHE") or ""
    ).strip()
    if not dest_path:
        raise ValueError(
            "dest or KOA_CALIBRATION_CACHE must be set to seed a KOA cache."
        )
    dest_path = os.path.abspath(os.path.expanduser(dest_path))
    os.makedirs(dest_path, exist_ok=True)
    os.environ["KOA_CALIBRATION_CACHE"] = dest_path

    n_before = _koa_cache_fit_count(dest_path)
    candidates = _discover_koa_cache_sources(dest_path, extra=sources)
    result: dict[str, Any] = {
        "dest": dest_path,
        "seeded": False,
        "source": None,
        "n_fits_before": n_before,
        "n_fits_after": n_before,
        "candidates": candidates,
    }
    if not force and n_before >= int(min_fits) and _koa_cache_has_db(dest_path):
        logger.info(
            "KOA cache already populated (%d FITS) at %s; skip seed.",
            n_before,
            dest_path,
        )
        return result
    if not candidates:
        logger.warning(
            "No KOA cache seed found for dest=%s (set HISPEC_KOA_CACHE_SEED "
            "or pass sources=...).",
            dest_path,
        )
        return result

    source = candidates[0]
    if os.path.abspath(source) == dest_path:
        return result

    logger.info(
        "Seeding KOA cache %s <- %s (%d FITS in source).",
        dest_path,
        source,
        _koa_cache_fit_count(source),
    )
    # Copy tree contents into dest (preserve existing unrelated files).
    for name in os.listdir(source):
        src_item = os.path.join(source, name)
        dst_item = os.path.join(dest_path, name)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        elif os.path.isfile(src_item):
            os.makedirs(os.path.dirname(dst_item) or dest_path, exist_ok=True)
            shutil.copy2(src_item, dst_item)

    n_after = _koa_cache_fit_count(dest_path)
    result.update(
        seeded=True,
        source=source,
        n_fits_after=n_after,
    )
    logger.info(
        "Seeded KOA cache: %d -> %d FITS under %s",
        n_before,
        n_after,
        dest_path,
    )
    return result


def _is_recoverable_remote_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = (
        "failed to query metadata",
        "404",
        "failed login",
        "incorrect password",
        "incorrect password or email",
        "must be set as environment variables",
        "keck_observer",
        "connection",
        "timeout",
        "timed out",
        "name or service not known",
        "nodename nor servname",
        "temporary failure",
        "remote",
    )
    return any(m in msg for m in markers)


def ensure_koa_calibrations(
    spectrograph: str = "BSPEC",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    seed_sources: Sequence[str | os.PathLike[str]] | None = None,
    ignore_missing_cal_types: list[str] | tuple[str, ...] | None = None,
    require_complete: bool = True,
    connect_remote: bool = True,
) -> dict[str, Any]:
    """
    Seed the local KOA cache if thin, then verify/download calibrations.

    Tries a remote sync first. On auth / network / API failures, falls back to
    the local cache only (after seeding from another cache when needed).
    """
    if ignore_missing_cal_types is None:
        ignore_missing_cal_types = ["lfc_wavesol"]

    if cache_dir is not None:
        cache_path = os.path.abspath(os.path.expanduser(str(cache_dir)))
        os.makedirs(cache_path, exist_ok=True)
        os.environ["KOA_CALIBRATION_CACHE"] = cache_path
    elif not (os.environ.get("KOA_CALIBRATION_CACHE") or "").strip():
        raise ValueError(
            "KOA_CALIBRATION_CACHE is unset; pass cache_dir= or set the env var."
        )

    seed_info = seed_koa_calibration_cache(
        os.environ["KOA_CALIBRATION_CACHE"],
        sources=seed_sources,
    )

    summary = verify_koa_calibrations(
        spectrograph,
        connect_remote=bool(connect_remote),
        sync_remote=bool(connect_remote),
        ignore_missing_cal_types=ignore_missing_cal_types,
        require_complete=require_complete,
        fallback_local=bool(connect_remote),
    )
    summary["seed"] = seed_info
    return summary
