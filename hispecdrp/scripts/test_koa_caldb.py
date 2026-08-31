#!/usr/bin/env python
"""
Test KOA calibration download and caldb association with dummy BSPEC / RSPEC L0 files.

Sets up ``KOA_CALIBRATION_CACHE`` under ``test_data/`` (by default), syncs calibration
metadata from KOA, creates the least data needed for selector matching (tiny L0 rate
maps), then associates every Stage 0 / Stage 1 caldb role that can be selected for
each spectrograph.

Caldb roles exercised (per spectrograph):

* Stage 0: dq, saturation, gain, detflat, bias, dark, rn, nonlin
* Stage 1: fiberflat (master covering fibers 1–3), trace_pos, trace_prof,
  wavesol (``lfc_wavesol``), and ``sim_wavesol`` if present
* Extra: ref_etalon

Per-fiber fiberflat *rate* FITS (``..._rate-FIBERFLAT-FIB{n}.fits``) are science-like
L0 inputs to ``ProcessFiberflatsPipeline``, not caldb products; the master
``fiberflat`` calibration is what the store associates.

Requires KOA Middleware authentication for remote sync/download (see
``docs/source/installation.rst``). Use ``--no-remote`` to exercise only an existing
local cache.

Usage (from the repository root, ``hispec`` conda env)::

    python -m hispecdrp.scripts.test_koa_caldb
    python -m hispecdrp.scripts.test_koa_caldb --output-dir test_data/koa_caldb_test --no-remote
    python -m hispecdrp.scripts.test_koa_caldb --skip-bulk-download
    python -m hispecdrp.scripts.test_koa_caldb --redo-download
    python -m hispecdrp.scripts.test_koa_caldb --mjd-start 61210 --dry-run

By default, after syncing KOA metadata the script downloads every calibration
listed in the local DB but missing from the cache. Use ``--redo-download`` to
delete cached FITS and re-fetch them. Use ``--skip-bulk-download`` to only
download files when they are selected for the dummy L0 products.

Products land under ``test_data/koa_caldb_test/`` by default (dummy L0 FITS,
``cals/KOA_CALIBRATION_CACHE/``, and ``cals/koa_caldb_test_report.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from hispecdrp import datamodels
from hispecdrp.calibrations import (
    BiasSelector,
    DarkSelector,
    DetectorFlatSelector,
    DQSelector,
    FiberFlatSelector,
    GainSelector,
    HISPECCalibrationStore,
    NonlinearCorrectionSelector,
    ReadNoiseSelector,
    ReferenceEtalonSelector,
    SaturationSelector,
    TracePosSelector,
    TraceProfSelector,
    WavelengthSolutionSelector,
)
from hispecdrp.simulations.e2e_support import (
    calibration_dir,
    ensure_output_directory,
    repository_root,
)
from hispecdrp.driver.logging import setup_default_logging

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENT_ERA = "0.0.1"
DEFAULT_MJD_START = 61210.0  # same fixed epoch as E2E sims
DEFAULT_SHAPE = (2, 2)  # minimal arrays; selectors only need metadata

# (report key, selector factory). Order matches Stage 0 then Stage 1 docs.
_CAL_ROLES: tuple[tuple[str, Any], ...] = (
    ("dq", DQSelector),
    ("saturation", SaturationSelector),
    ("gain", GainSelector),
    ("detflat", DetectorFlatSelector),
    ("bias", BiasSelector),
    ("dark", DarkSelector),
    ("rn", ReadNoiseSelector),
    ("nonlin", NonlinearCorrectionSelector),
    ("fiberflat", FiberFlatSelector),
    ("trace_pos", TracePosSelector),
    ("trace_prof", TraceProfSelector),
    ("wavesol", WavelengthSolutionSelector),  # default cal_type=lfc_wavesol
    ("sim_wavesol", lambda: WavelengthSolutionSelector(cal_type="sim_wavesol")),
    ("ref_etalon", ReferenceEtalonSelector),
)


@dataclass
class AssociationResult:
    """One caldb association attempt for a spectrograph."""

    key: str
    status: str  # "ok" | "missing" | "error"
    filepath: str | None = None
    cal_id: str | None = None
    cal_type: str | None = None
    filename: str | None = None
    message: str | None = None


@dataclass
class SpectrographReport:
    spectrograph: str
    dummy_path: str
    instrument_era: str
    mjd_start: float
    associations: list[AssociationResult] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for a in self.associations if a.status == "ok")

    @property
    def n_missing(self) -> int:
        return sum(1 for a in self.associations if a.status == "missing")


def _default_output_dir() -> str:
    """Isolate caldb test products under ``test_data/koa_caldb_test/``."""
    return str(repository_root() / "test_data" / "koa_caldb_test")


def _pin_koa_cache_under_output(output_dir: str, *, force: bool = True) -> str:
    """
    Point ``KOA_CALIBRATION_CACHE`` at ``<output_dir>/cals/KOA_CALIBRATION_CACHE``.

    When ``force`` is True (default for this smoke test), any existing env value
    is overridden so the run is isolated under ``output_dir``.
    """
    cal_root = calibration_dir(output_dir)
    koa_path = os.path.join(cal_root, "KOA_CALIBRATION_CACHE")
    os.makedirs(koa_path, exist_ok=True)
    if force or not (os.environ.get("KOA_CALIBRATION_CACHE") or "").strip():
        os.environ["KOA_CALIBRATION_CACHE"] = koa_path
        logger.info("KOA_CALIBRATION_CACHE -> %s", koa_path)
    else:
        logger.info(
            "Using existing KOA_CALIBRATION_CACHE=%s",
            os.environ["KOA_CALIBRATION_CACHE"],
        )
    return os.environ["KOA_CALIBRATION_CACHE"]


def make_dummy_l0(
    spectrograph: str,
    output_dir: str,
    *,
    mjd_start: float = DEFAULT_MJD_START,
    instrument_era: str = DEFAULT_INSTRUMENT_ERA,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> tuple[str, datamodels.L0Model]:
    """
    Build and save the least L0 product needed for caldb matching.

    Only ``instrument_era``, ``spectrograph``, and ``mjd_start`` are used by
    selectors; a tiny zero rate map is written so the file is a valid FITS L0.
    Files are named ``dummy_{BSPEC|RSPEC}_L0.fits`` to avoid colliding with
    KOA-style E2E products in ``test_data/``.
    """
    spec = spectrograph.upper()
    if spec not in ("BSPEC", "RSPEC"):
        raise ValueError(f"spectrograph must be BSPEC or RSPEC, got {spectrograph!r}")

    model = datamodels.L0Model(
        meta={
            "instrument_name": "HISPEC",
            "instrument_era": instrument_era,
            "spectrograph": spec,
            "mjd_start": float(mjd_start),
        }
    )
    ny, nx = shape
    model.rate = np.zeros((ny, nx), dtype=np.float32)
    model.rate_err = np.zeros((ny, nx), dtype=np.float32)
    model.rate_var_poisson = np.zeros((ny, nx), dtype=np.float32)
    model.rate_var_rnoise = np.zeros((ny, nx), dtype=np.float32)
    model.dq = np.zeros((ny, nx), dtype=np.uint32)
    model.mid_time = np.full((ny, nx), float(mjd_start), dtype=np.float64)

    out_dir = ensure_output_directory(output_dir)
    filename = f"dummy_{spec}_L0.fits"
    filepath = model.save(output_dir=out_dir, filename=filename)
    logger.info(
        "Wrote dummy %s L0 (%dx%d): %s  (era=%s, mjd_start=%s)",
        spec,
        ny,
        nx,
        filepath,
        instrument_era,
        mjd_start,
    )
    return filepath, model


def sync_caldb_from_scratch(
    store: HISPECCalibrationStore,
    *,
    connect_remote: bool,
    download_all_missing: bool,
    redo_download: bool = False,
    dry_run: bool,
) -> None:
    """Sync KOA metadata into the local DB; optionally download every missing file."""
    if dry_run:
        logger.info("Dry run: skipping remote sync / cache indexing / bulk download.")
        return

    if connect_remote and store.remote_db is not None:
        try:
            added = store.sync_records_from_remote(mode="id")
            logger.info("Synced %d calibration record(s) from remote KOA.", len(added or []))
        except Exception as e:
            logger.warning("KOA remote metadata sync failed: %s", e)
    elif connect_remote:
        logger.warning(
            "Remote KOA connection unavailable (remote_db is None). "
            "Continuing with local cache only."
        )
    else:
        logger.info("Skipping remote sync (--no-remote).")

    # Index FITS already present in the cache (skip when empty to avoid stub DB issues).
    data_dir = getattr(store, "data_dir", None) or ""
    cached_fits = []
    if data_dir and os.path.isdir(data_dir):
        cached_fits = [
            f
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
            and not f.startswith(".")
            and f.lower().endswith((".fits", ".fit"))
        ]
    if cached_fits:
        try:
            indexed = store.sync_records_from_cached_files()
            if indexed:
                logger.info(
                    "Indexed %d file(s) already present in the local KOA cache.",
                    len(indexed),
                )
        except Exception as e:
            logger.warning("Local cache indexing failed: %s", e)
    else:
        logger.info("Local KOA cache has no FITS files yet; skipping cache indexing.")

    if not download_all_missing and not redo_download:
        return

    try:
        if redo_download:
            records = list(store.local_db.rows_where()) if store.local_db else []
            removed = 0
            for rec in records:
                local_path = store._get_local_filepath(rec)
                if local_path and os.path.isfile(local_path):
                    os.remove(local_path)
                    removed += 1
            logger.info(
                "Redo download: removed %d cached file(s); re-downloading %d record(s).",
                removed,
                len(records),
            )
            to_download = records
        else:
            to_download = store.get_missing_local_files() or []
    except Exception as e:
        logger.warning("Could not list KOA files for download: %s", e)
        return

    if not to_download:
        logger.info("No KOA calibration files to download.")
        return

    logger.info("Downloading %d KOA calibration file(s).", len(to_download))
    for i, rec in enumerate(to_download, 1):
        filename = rec.get("filename", "?")
        try:
            local_path, _ = store.get_calibration(cal=rec)
            logger.info("[%d/%d] Downloaded: %s -> %s", i, len(to_download), filename, local_path)
        except Exception as e:
            logger.warning("[%d/%d] Failed to download %s: %s", i, len(to_download), filename, e)


def associate_calibrations(
    store: HISPECCalibrationStore | None,
    model: datamodels.L0Model,
    *,
    dry_run: bool = False,
) -> list[AssociationResult]:
    """Select (and download if needed) every known caldb role for ``model``."""
    results: list[AssociationResult] = []
    meta = model.meta
    for key, selector_factory in _CAL_ROLES:
        if dry_run or store is None:
            results.append(
                AssociationResult(
                    key=key,
                    status="missing",
                    message="dry-run: selection not executed",
                )
            )
            continue
        selector = selector_factory()
        try:
            filepath, rec = store.select_and_get_calibration(meta, selector=selector)
            rec = rec or {}
            results.append(
                AssociationResult(
                    key=key,
                    status="ok",
                    filepath=filepath,
                    cal_id=rec.get("id") or rec.get("cal_id"),
                    cal_type=rec.get("cal_type"),
                    filename=rec.get("filename") or (os.path.basename(filepath) if filepath else None),
                )
            )
            logger.info(
                "  %-12s OK  %s",
                key,
                filepath or rec.get("filename"),
            )
        except Exception as e:
            msg = str(e)
            lower = msg.lower()
            # No candidate in DB (or koa_middleware NoneType bug before our store guard).
            if (
                "no matching calibration" in lower
                or "not subscriptable" in lower
                or "no candidate" in lower
                or ("candidate" in lower and "no" in lower)
            ):
                status = "missing"
                if "not subscriptable" in lower:
                    msg = (
                        "No matching calibration in local caldb "
                        f"(selector={type(selector).__name__})"
                    )
            elif "no such column" in lower:
                msg = (
                    f"{msg} (local caldb schema incomplete — sync from remote KOA "
                    "or register calibrations into this cache first)"
                )
                status = "error"
            else:
                status = "error"
            results.append(
                AssociationResult(key=key, status=status, message=msg)
            )
            logger.warning("  %-12s %s  %s", key, status.upper(), msg)
    return results


def _write_report(report_path: str, reports: list[SpectrographReport], extra: dict[str, Any]) -> None:
    payload = {
        **extra,
        "spectrographs": [
            {
                "spectrograph": r.spectrograph,
                "dummy_path": r.dummy_path,
                "instrument_era": r.instrument_era,
                "mjd_start": r.mjd_start,
                "n_ok": r.n_ok,
                "n_missing": r.n_missing,
                "associations": [asdict(a) for a in r.associations],
            }
            for r in reports
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote association report: %s", report_path)


def run(
    *,
    output_dir: str | None = None,
    instrument_era: str = DEFAULT_INSTRUMENT_ERA,
    mjd_start: float = DEFAULT_MJD_START,
    connect_remote: bool = True,
    download_all_missing: bool = True,
    redo_download: bool = False,
    dry_run: bool = False,
    instrument_name: str | None = None,
    force_cache_under_output: bool = True,
) -> list[SpectrographReport]:
    """
    Create dummy BSPEC/RSPEC L0 files and associate all available KOA calibrations.

    Parameters
    ----------
    output_dir
        Directory for dummy L0 FITS, the KOA cache (if unset in the environment),
        and the JSON report. Defaults to ``<repo>/test_data/koa_caldb_test``.
    instrument_era, mjd_start
        Metadata used for selector matching (and written into the dummy files).
    connect_remote
        If True, sync metadata from remote KOA when credentials are available.
    download_all_missing
        If True (default), after syncing metadata download every calibration
        listed in the local DB but missing from the cache.
    redo_download
        If True, delete cached FITS for local-DB records and re-download them.
    dry_run
        If True, create dummy files and list intended associations without
        contacting KOA or selecting calibrations.
    instrument_name
        Store instrument name override (default HISPEC / env).
    force_cache_under_output
        If True, set ``KOA_CALIBRATION_CACHE`` under ``output_dir`` even when
        already defined in the environment (isolated from-scratch test).
    """
    out_dir = ensure_output_directory(output_dir or _default_output_dir())
    cal_dir = calibration_dir(out_dir)
    cache_path = _pin_koa_cache_under_output(out_dir, force=force_cache_under_output)
    logger.info("Output directory: %s", out_dir)
    logger.info("KOA_CALIBRATION_CACHE: %s", cache_path)

    reports: list[SpectrographReport] = []
    dummy_paths: dict[str, str] = {}

    for spec in ("BSPEC", "RSPEC"):
        path, _ = make_dummy_l0(
            spec,
            out_dir,
            mjd_start=mjd_start,
            instrument_era=instrument_era,
        )
        dummy_paths[spec] = path

    if dry_run:
        for spec, path in dummy_paths.items():
            model = datamodels.open(path, meta_only=True)
            reports.append(
                SpectrographReport(
                    spectrograph=spec,
                    dummy_path=path,
                    instrument_era=instrument_era,
                    mjd_start=mjd_start,
                    associations=associate_calibrations(None, model, dry_run=True),
                )
            )
    else:
        with HISPECCalibrationStore(
            instrument_name=instrument_name,
            connect_remote=connect_remote,
            sync_on_init=False,
        ) as store:
            sync_caldb_from_scratch(
                store,
                connect_remote=connect_remote,
                download_all_missing=download_all_missing,
                redo_download=redo_download,
                dry_run=False,
            )
            for spec, path in dummy_paths.items():
                model = datamodels.open(path, meta_only=True)
                logger.info("Associating calibrations for %s (%s)...", spec, path)
                associations = associate_calibrations(store, model, dry_run=False)
                reports.append(
                    SpectrographReport(
                        spectrograph=spec,
                        dummy_path=path,
                        instrument_era=instrument_era,
                        mjd_start=mjd_start,
                        associations=associations,
                    )
                )

    report_path = os.path.join(cal_dir, "koa_caldb_test_report.json")
    _write_report(
        report_path,
        reports,
        extra={
            "output_dir": out_dir,
            "koa_calibration_cache": os.environ.get("KOA_CALIBRATION_CACHE"),
            "instrument_era": instrument_era,
            "mjd_start": mjd_start,
            "connect_remote": connect_remote,
            "download_all_missing": download_all_missing,
            "redo_download": redo_download,
            "dry_run": dry_run,
            "cal_roles": [k for k, _ in _CAL_ROLES],
            "note": (
                "Master fiberflat covers fibers 1-3; per-fiber FIBERFLAT rate maps "
                "are not caldb products. Roles with status 'missing' were not present "
                "in the KOA caldb for that spectrograph (e.g. ref_etalon / lfc_wavesol "
                "are often unavailable)."
            ),
        },
    )

    for r in reports:
        logger.info(
            "%s: %d/%d calibrations associated (%d missing/unavailable).",
            r.spectrograph,
            r.n_ok,
            len(r.associations),
            r.n_missing + sum(1 for a in r.associations if a.status == "error"),
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    setup_default_logging()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Create dummy BSPEC/RSPEC L0 files and associate Stage 0/1 calibrations "
            "from the KOA caldb (cache under test_data/ by default)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Directory for dummy L0 FITS, KOA cache (if unset), and JSON report.",
    )
    parser.add_argument(
        "--instrument-era",
        default=DEFAULT_INSTRUMENT_ERA,
        help="Instrument era metadata for selector matching.",
    )
    parser.add_argument(
        "--mjd-start",
        type=float,
        default=DEFAULT_MJD_START,
        help="Observation MJD for closest-in-time calibration selection.",
    )
    parser.add_argument(
        "--instrument-name",
        default=None,
        help="Instrument name for HISPECCalibrationStore (default: HISPEC / env).",
    )
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Do not connect to remote KOA; use only the local cache/DB.",
    )
    parser.add_argument(
        "--skip-bulk-download",
        action="store_true",
        help=(
            "Do not bulk-download all calibrations missing from the cache after "
            "sync (default is to download them). Association still downloads "
            "selected files on demand."
        ),
    )
    parser.add_argument(
        "--redo-download",
        action="store_true",
        help=(
            "Delete cached FITS for every local-DB calibration record and "
            "re-download them (implies bulk download)."
        ),
    )
    parser.add_argument(
        "--use-existing-cache",
        action="store_true",
        help=(
            "Do not override KOA_CALIBRATION_CACHE when it is already set "
            "(default: pin a fresh cache under --output-dir)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write dummy L0 files and report intended roles without KOA I/O.",
    )
    args = parser.parse_args(argv)

    reports = run(
        output_dir=args.output_dir,
        instrument_era=args.instrument_era,
        mjd_start=args.mjd_start,
        connect_remote=not args.no_remote,
        download_all_missing=not args.skip_bulk_download,
        redo_download=args.redo_download,
        dry_run=args.dry_run,
        instrument_name=args.instrument_name,
        force_cache_under_output=not args.use_existing_cache,
    )

    if args.dry_run:
        return 0

    # Fail if either spectrograph got zero successful associations.
    failed = [r.spectrograph for r in reports if r.n_ok == 0]
    if failed:
        logger.error(
            "No calibrations associated for: %s. "
            "Check KOA auth, KOA_CALIBRATION_CACHE, and instrument_era/mjd_start.",
            ", ".join(failed),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
