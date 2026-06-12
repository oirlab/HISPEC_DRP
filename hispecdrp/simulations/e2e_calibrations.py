"""
Resolve calibrations for end-to-end simulations: KOA store, user config, then simulated fallback.

Per calibration product, resolution order is:

1. Explicit path in ``calibration_config`` (YAML/JSON dict)
2. KOA calibration store (when enabled)
3. Simulated / HDF-derived products (see :mod:`hispecdrp.simulations.simulate_up_the_ramp`)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import datamodels
from ..calibrations import (
    BiasSelector,
    DarkSelector,
    DetectorFlatSelector,
    DQSelector,
    FiberFlatSelector,
    GainSelector,
    HISPECCalibrationStore,
    NonlinearCorrectionSelector,
    ReadNoiseSelector,
    SaturationSelector,
    TracePosSelector,
    TraceProfSelector,
    WavelengthSolutionSelector,
)
from ..utils.config import read_config_file
from . import e2e_support
from .simulate_up_the_ramp import create_simulated_calibration_models

logger = logging.getLogger(__name__)

# Stage 0 detector calibrations
STAGE0_CAL_KEYS: tuple[str, ...] = (
    "dq",
    "saturation",
    "gain",
    "detflat",
    "bias",
    "dark",
    "rn",
    "nonlin",
)

# Stage 1 / fiberflat pipeline products
STAGE1_CAL_KEYS: tuple[str, ...] = (
    "trace_pos",
    "trace_prof",
    "fiberflat",
    "wavesol",
)

KOA_SELECTOR_BY_KEY: dict[str, Any] = {
    "dq": DQSelector(),
    "saturation": SaturationSelector(),
    "gain": GainSelector(),
    "detflat": DetectorFlatSelector(),
    "bias": BiasSelector(),
    "dark": DarkSelector(),
    "rn": ReadNoiseSelector(),
    "nonlin": NonlinearCorrectionSelector(),
    "trace_pos": TracePosSelector(),
    "trace_prof": TraceProfSelector(),
    "fiberflat": FiberFlatSelector(),
    "wavesol": WavelengthSolutionSelector(),
}

STAGE1_TRACE_KEYS: tuple[str, ...] = ("trace_pos", "trace_prof", "fiberflat")

# Legacy KOA masters may still carry pre-schema object_fiber labels.
_LEGACY_OBJECT_FIBER: dict[str, str] = {
    "FIBERFLAT-B": "WHITELIGHT",
    "FIBERFLAT-R": "WHITELIGHT",
}
_OBJECT_FIBER_FITS_KEYS: tuple[str, ...] = ("OBJFIB1", "OBJFIB2", "OBJFIB3")


@dataclass
class E2ECalibrationBundle:
    """Resolved calibration products for an E2E run."""

    stage0: dict[str, datamodels.CalibrationModel] = field(default_factory=dict)
    trace_pos: datamodels.DataModel | None = None
    trace_prof: datamodels.DataModel | None = None
    fiberflat: datamodels.DataModel | None = None
    wavesol: datamodels.DataModel | None = None
    fiberflat_rate_paths: dict[int, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def has_master_trace_bundle(self) -> bool:
        """True when trace position, profile, and fiberflat are all available without the fiberflat pipeline."""
        return (
            self.trace_pos is not None
            and self.trace_prof is not None
            and self.fiberflat is not None
        )


def load_calibration_config(path_or_dict: str | Mapping[str, Any]) -> dict[str, Any]:
    """
    Load an E2E calibration config from a YAML file path or an in-memory dict.

    Expected structure::

        koa:  # optional overrides for store selection
          instrument_era: "0.0.1"
          mjd_start: 58941.35
          datetime_obs: null
          connect_remote: true
          sync_first: true
        calibrations:  # optional explicit FITS paths
          dq: /path/to/dq.fits
          trace_pos: ...
        fiberflat_rates:  # optional per-fiber rate FITS (skip dummy fiberflat generation)
          1: /path/to/HB....FIB1.fits
          2: ...
    """
    if isinstance(path_or_dict, Mapping):
        cfg = dict(path_or_dict)
    else:
        cfg = read_config_file(str(path_or_dict))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError("calibration config must be a mapping at the top level")
    return cfg


def _normalize_fiberflat_rates(raw: Any) -> dict[int, str]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("fiberflat_rates must be a mapping of fiber id -> file path")
    out: dict[int, str] = {}
    for k, v in raw.items():
        if v is None or str(v).strip() == "":
            continue
        out[int(k)] = os.path.abspath(str(v))
    return out


def _observation_meta(
    *,
    spectrograph: str,
    instrument_era: str,
    mjd_start: float | None,
    datetime_obs: str | None,
    instrument_name: str | None,
) -> tuple[dict[str, Any], datamodels.DataModel]:
    if mjd_start is None and datetime_obs is not None:
        import astropy.time as at

        mjd_start = float(at.Time(datetime_obs, format="isot", scale="utc").mjd)
    if mjd_start is None:
        import astropy.time as at

        mjd_start = float(at.Time.now().mjd)
    meta = {
        "instrument_era": instrument_era,
        "spectrograph": spectrograph.upper(),
        "mjd_start": mjd_start,
    }
    raw_name = instrument_name or os.getenv("HISPECDRP_INSTRUMENT_NAME", "HISPEC")
    instrument_name_meta = raw_name.upper() if isinstance(raw_name, str) else "HISPEC"
    input_model = datamodels.L0Model(
        meta={**meta, "instrument_name": instrument_name_meta},
        meta_only=True,
    )
    return meta, input_model


def _sync_koa_store(
    store: HISPECCalibrationStore,
    *,
    connect_remote: bool,
    sync_first: bool,
    download_missing: bool = True,
) -> None:
    """Sync KOA metadata into the local DB and download missing calibration files."""
    if not sync_first:
        return
    if connect_remote and store.remote_db is not None:
        try:
            remote_cals = store.remote_db.query() or []
            store.sync_records_from_remote(remote_cals, mode="id")
            logger.info("Synced %d calibration record(s) from remote KOA.", len(remote_cals))
        except Exception as e:
            logger.warning("KOA remote metadata sync failed: %s", e)
    try:
        added = store.sync_records_from_cached_files()
        if added:
            logger.info("Indexed %d calibration file(s) from local KOA cache.", len(added))
    except Exception as e:
        logger.warning("KOA local cache indexing failed: %s", e)
    if not download_missing:
        return
    try:
        missing = store.get_missing_local_files() or []
    except Exception as e:
        logger.warning("Could not list missing KOA calibration files: %s", e)
        return
    if not missing:
        return
    logger.info("Downloading %d missing KOA calibration file(s).", len(missing))
    for i, rec in enumerate(missing, 1):
        filename = rec.get("filename", "?")
        try:
            local_path, _ = store.get_calibration(cal=rec)
            logger.info("[%d/%d] Downloaded KOA cal: %s", i, len(missing), local_path or filename)
        except Exception as e:
            logger.warning("[%d/%d] Failed to download KOA cal %s: %s", i, len(missing), filename, e)


def _open_calibration_file(key: str, path: str) -> datamodels.CalibrationModel:
    """Open a calibration FITS, tolerating legacy ``FIBERFLAT-*`` object_fiber metadata."""
    import tempfile

    from astropy.io import fits

    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Calibration {key!r}: file not found: {p}")
    try:
        model = datamodels.open(p)
        logger.info("Opened calibration %s: %s", key, p)
        return model
    except ValueError as exc:
        if "object_fiber" not in str(exc):
            raise

    with fits.open(p) as hdul:
        header = hdul[0].header
        changed = False
        for kw in _OBJECT_FIBER_FITS_KEYS:
            val = header.get(kw)
            if isinstance(val, str) and val in _LEGACY_OBJECT_FIBER:
                header[kw] = _LEGACY_OBJECT_FIBER[val]
                changed = True
        if not changed:
            raise
        fd, tmp_path = tempfile.mkstemp(suffix=".fits", prefix="koa_cal_")
        os.close(fd)
        hdul.writeto(tmp_path, overwrite=True)

    try:
        model = datamodels.open(tmp_path)
        logger.info(
            "Opened calibration %s with legacy object_fiber remap: %s (temp %s)",
            key,
            p,
            tmp_path,
        )
        return model
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def fetch_koa_calibrations(
    keys: Sequence[str],
    meta: dict[str, Any],
    input_for_store: datamodels.DataModel,
    *,
    instrument_name: str | None = None,
    connect_remote: bool = True,
    sync_first: bool = True,
    download_missing: bool = True,
) -> dict[str, datamodels.CalibrationModel]:
    """
    Fetch calibration FITS from the KOA store for ``keys`` (best-effort per key).

    Missing keys are omitted from the returned dict (logged at warning level).
    """
    keys = [k for k in keys if k in KOA_SELECTOR_BY_KEY]
    if not keys:
        return {}

    result: dict[str, datamodels.CalibrationModel] = {}
    try:
        with HISPECCalibrationStore(
            instrument_name=instrument_name,
            connect_remote=connect_remote,
            sync_on_init=False,
        ) as store:
            _sync_koa_store(
                store,
                connect_remote=connect_remote,
                sync_first=sync_first,
                download_missing=download_missing,
            )

            store_meta = (
                input_for_store.meta if hasattr(input_for_store, "meta") else meta
            )
            for key in keys:
                selector = KOA_SELECTOR_BY_KEY[key]
                try:
                    filepath, _ = store.select_and_get_calibration(
                        store_meta, selector=selector
                    )
                    result[key] = _open_calibration_file(key, filepath)
                    logger.info("KOA calibration %s: %s", key, filepath)
                except Exception as e:
                    logger.warning(
                        "KOA calibration %r unavailable (instrument_era=%s, "
                        "spectrograph=%s, mjd_start=%s): %s",
                        key,
                        meta.get("instrument_era"),
                        meta.get("spectrograph"),
                        meta.get("mjd_start"),
                        e,
                    )
    except Exception as e:
        logger.warning(
            "KOA calibration store unavailable; missing keys will use simulated "
            "fallback after rate map is built: %s",
            e,
        )
    return result


def _open_calibration_path(key: str, path: str) -> datamodels.CalibrationModel:
    return _open_calibration_file(key, path)


def _local_master_search_dirs(output_dir: str) -> list[str]:
    dirs = [e2e_support.calibration_dir(output_dir)]
    cache_root = (os.environ.get("KOA_CALIBRATION_CACHE") or "").strip()
    if cache_root:
        dirs.append(os.path.join(cache_root, "calibrations", "hispec"))
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        d_abs = os.path.abspath(d)
        if os.path.isdir(d_abs) and d_abs not in seen:
            seen.add(d_abs)
            out.append(d_abs)
    return out


def _glob_local_master_path(key: str, spectrograph: str, search_dirs: Sequence[str]) -> str | None:
    prefix = "HB" if spectrograph.upper() == "BSPEC" else "HR"
    patterns = {
        "trace_pos": f"{prefix}.*mastercal-tracepos*.fits",
        "trace_prof": f"{prefix}.*mastercal-traceprof*.fits",
        "fiberflat": f"{prefix}.*mastercal-fiberflat*.fits",
    }
    pattern = patterns.get(key)
    if pattern is None:
        return None
    matches: list[str] = []
    for directory in search_dirs:
        matches.extend(str(p) for p in Path(directory).glob(pattern))
    if not matches:
        return None
    return sorted(matches)[-1]


def resolve_local_stage1_masters(
    bundle: E2ECalibrationBundle,
    *,
    output_dir: str,
    spectrograph: str,
) -> None:
    """Fill missing Stage 1 masters from local ``cals/`` or KOA cache directories."""
    search_dirs = _local_master_search_dirs(output_dir)
    for key in STAGE1_TRACE_KEYS:
        if getattr(bundle, key) is not None:
            continue
        local_path = _glob_local_master_path(key, spectrograph, search_dirs)
        if local_path is None:
            continue
        try:
            model = _open_calibration_file(key, local_path)
        except Exception as e:
            logger.warning("Local master %r at %s could not be opened: %s", key, local_path, e)
            continue
        setattr(bundle, key, model)
        bundle.sources[key] = f"local_cache:{local_path}"
        logger.info("Using local master calibration %s: %s", key, local_path)


_TESTING_DATA_STAGE1_REL_PATHS: dict[str, dict[str, str]] = {
    "BSPEC": {
        "trace_pos": "HISPEC/Cals/HB.20260606.00000.00.mastercal-tracepos.001.fits",
        "trace_prof": "HISPEC/Cals/HB.20260606.00000.00.mastercal-traceprof.001.fits",
        "fiberflat": "HISPEC/Cals/HB.20260606.00000.00.mastercal-fiberflat.001.fits",
    },
    "RSPEC": {
        "trace_pos": "HISPEC/Cals/HR.20260606.00000.00.mastercal-tracepos.001.fits",
        "trace_prof": "HISPEC/Cals/HR.20260606.00000.00.mastercal-traceprof.001.fits",
        "fiberflat": "HISPEC/Cals/HR.20260606.00000.00.mastercal-fiberflat.001.fits",
    },
}


def resolve_testing_data_stage1_masters(
    bundle: E2ECalibrationBundle,
    *,
    output_dir: str,
    spectrograph: str,
) -> None:
    """Download missing Stage 1 masters from bundled Google Drive test data (last resort)."""
    spec = spectrograph.upper()
    rel_paths = _TESTING_DATA_STAGE1_REL_PATHS.get(spec)
    if rel_paths is None:
        return

    from ..utils.gdrive import download_gdrive_file

    cal_dir = e2e_support.calibration_dir(output_dir)
    for key in STAGE1_TRACE_KEYS:
        if getattr(bundle, key) is not None:
            continue
        rel_path = rel_paths.get(key)
        if rel_path is None:
            continue
        try:
            local_path = download_gdrive_file(rel_path, output_dir=cal_dir, use_cached=True)
        except Exception as e:
            logger.warning("Could not download test-data master %r: %s", key, e)
            continue
        if local_path is None or not os.path.isfile(local_path):
            continue
        try:
            model = _open_calibration_file(key, local_path)
        except Exception as e:
            logger.warning("Downloaded test-data master %r at %s could not be opened: %s", key, local_path, e)
            continue
        setattr(bundle, key, model)
        bundle.sources[key] = f"testing_data:{local_path}"
        logger.info("Using test-data master calibration %s: %s", key, local_path)


def log_stage1_trace_bundle_status(bundle: E2ECalibrationBundle) -> None:
    """Log whether Stage 1 will use KOA/local masters or the fiberflat pipeline."""
    if bundle.has_master_trace_bundle:
        logger.info(
            "Stage 1 trace bundle complete (%s); ProcessFiberflatsPipeline will be skipped.",
            ", ".join(f"{k}={bundle.sources.get(k, '?')}" for k in STAGE1_TRACE_KEYS),
        )
        return
    missing = [k for k in STAGE1_TRACE_KEYS if getattr(bundle, k) is None]
    logger.warning(
        "Stage 1 trace bundle incomplete (missing %s); trace_pos and trace_prof will be "
        "generated on-the-fly by ProcessFiberflatsPipeline from dummy fiberflats.",
        missing,
    )


def log_missing_stage1_trace_masters(bundle: E2ECalibrationBundle) -> None:
    """Log which Stage 1 trace/fiberflat masters are still missing from KOA/config."""
    if bundle.has_master_trace_bundle:
        return
    missing = [k for k in STAGE1_TRACE_KEYS if getattr(bundle, k) is None]
    if missing:
        logger.warning(
            "KOA/config trace bundle incomplete (missing %s); dummy fiberflat "
            "simulation and ProcessFiberflatsPipeline will run unless paths are "
            "provided in calibration_config.",
            missing,
        )


def complete_stage0_calibrations(
    calibration_models: dict[str, datamodels.CalibrationModel],
    rate_map: np.ndarray,
    output_dir: str,
    sources: dict[str, str],
    *,
    calibration_kwargs: dict[str, Any] | None = None,
) -> None:
    """Fill any missing Stage 0 keys with simulated calibration models (in-place)."""
    rm = np.asarray(rate_map, dtype=np.float64)
    if rm.ndim != 2:
        raise ValueError(f"rate_map must be 2D; got shape {rm.shape}")
    eff_shape = (int(rm.shape[0]), int(rm.shape[1]))
    cal_kw = calibration_kwargs or {}

    for k in STAGE0_CAL_KEYS:
        if k in calibration_models:
            continue
        fp = e2e_support.dummy_cal_path(output_dir, k)
        if os.path.isfile(fp):
            calibration_models[k] = datamodels.open(fp)
            sources.setdefault(k, f"simulated_cached:{fp}")
            continue

    missing = [k for k in STAGE0_CAL_KEYS if k not in calibration_models]
    if not missing:
        return

    logger.info(
        "Simulating Stage 0 calibrations for missing keys %s (shape=%s).",
        missing,
        eff_shape,
    )
    new_models = create_simulated_calibration_models(eff_shape, **cal_kw)
    for k in missing:
        if k in new_models:
            calibration_models[k] = new_models[k]
            sources[k] = "simulated"
            try:
                new_models[k].save(filepath=e2e_support.dummy_cal_path(output_dir, k))
            except Exception as e:
                logger.warning("Could not save simulated calibration %s: %s", k, e)


def resolve_e2e_calibrations(
    *,
    output_dir: str,
    spectrograph: str,
    use_koa: bool = True,
    calibration_config: str | dict | None = None,
    instrument_name: str | None = None,
    instrument_era: str = "0.0.1",
    mjd_start: float | None = None,
    datetime_obs: str | None = None,
    connect_remote: bool = True,
    sync_caldb_first: bool = True,
    download_missing_koa: bool = True,
) -> E2ECalibrationBundle:
    """
    Resolve E2E calibrations: explicit config paths, then KOA, then simulated (Stage 0 completed later).

    Parameters
    ----------
    use_koa : bool
        When True, query the KOA calibration store for keys not set in ``calibration_config``.
        When False, only user config paths are used (remaining keys filled after rate map is built).
    """
    out_dir = e2e_support.ensure_output_directory(output_dir)
    e2e_support.ensure_koa_calibration_cache_env(out_dir)

    cfg: dict[str, Any] = {}
    if calibration_config is not None:
        cfg = load_calibration_config(calibration_config)

    koa_cfg = cfg.get("koa") if isinstance(cfg.get("koa"), dict) else {}
    instrument_era = str(koa_cfg.get("instrument_era", instrument_era))
    mjd_start = koa_cfg.get("mjd_start", mjd_start)
    datetime_obs = koa_cfg.get("datetime_obs", datetime_obs)
    connect_remote = bool(koa_cfg.get("connect_remote", connect_remote))
    sync_first = bool(koa_cfg.get("sync_first", sync_caldb_first))
    download_missing = bool(koa_cfg.get("download_missing", download_missing_koa))

    user_paths = cfg.get("calibrations") if isinstance(cfg.get("calibrations"), dict) else {}
    fiberflat_rates = _normalize_fiberflat_rates(cfg.get("fiberflat_rates"))

    meta, input_model = _observation_meta(
        spectrograph=spectrograph,
        instrument_era=instrument_era,
        mjd_start=mjd_start,
        datetime_obs=datetime_obs,
        instrument_name=instrument_name,
    )

    bundle = E2ECalibrationBundle(fiberflat_rate_paths=fiberflat_rates)

    all_keys = list(STAGE0_CAL_KEYS) + list(STAGE1_CAL_KEYS)
    for key in all_keys:
        if key in user_paths and user_paths[key]:
            try:
                model = _open_calibration_path(key, str(user_paths[key]))
                bundle.sources[key] = f"config:{user_paths[key]}"
                if key in STAGE0_CAL_KEYS:
                    bundle.stage0[key] = model
                elif key == "trace_pos":
                    bundle.trace_pos = model
                elif key == "trace_prof":
                    bundle.trace_prof = model
                elif key == "fiberflat":
                    bundle.fiberflat = model
                elif key == "wavesol":
                    bundle.wavesol = model
            except Exception as e:
                logger.warning("User calibration path for %r failed: %s", key, e)

    if use_koa:
        need_koa = [
            k
            for k in all_keys
            if k not in bundle.sources and k in KOA_SELECTOR_BY_KEY
        ]
        if need_koa:
            logger.info(
                "Fetching %d calibration product(s) from KOA (spectrograph=%s).",
                len(need_koa),
                spectrograph,
            )
            try:
                koa_models = fetch_koa_calibrations(
                    need_koa,
                    meta,
                    input_model,
                    instrument_name=instrument_name,
                    connect_remote=connect_remote,
                    sync_first=sync_first,
                    download_missing=download_missing,
                )
            except Exception as e:
                logger.warning(
                    "KOA calibration fetch failed; simulated fallback will be used "
                    "for missing products: %s",
                    e,
                )
                koa_models = {}
            for key, model in koa_models.items():
                bundle.sources[key] = "koa"
                if key in STAGE0_CAL_KEYS:
                    bundle.stage0[key] = model
                elif key == "trace_pos":
                    bundle.trace_pos = model
                elif key == "trace_prof":
                    bundle.trace_prof = model
                elif key == "fiberflat":
                    bundle.fiberflat = model
                elif key == "wavesol":
                    bundle.wavesol = model

    resolve_local_stage1_masters(
        bundle,
        output_dir=out_dir,
        spectrograph=spectrograph.upper(),
    )
    resolve_testing_data_stage1_masters(
        bundle,
        output_dir=out_dir,
        spectrograph=spectrograph.upper(),
    )
    log_stage1_trace_bundle_status(bundle)

    return bundle
