"""
Simulate up-the-ramp (UTR) readouts and Stage 0 / fiberflat helpers.

Build simulated calibration models, generate ramp data from a rate map, and run
the Stage 0 and fiberflat pipelines. Master calibrations come from
:func:`hispecdrp.calibrations.run_caldb.get_master_calibrations` when
``use_master_calibrations=True``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from .. import datamodels
from ..primitives.utils.spectral_utils import (
    get_spectrum_path,
    load_btsettl_spectrum,
)
from ..datamodels.dqflags import DQ_FLAGS
from ..pipelines import Stage0Pipeline, ProcessFiberflatsPipeline
from . import e2e_support
from .utr import create_ramp
from ..calibrations.run_caldb import get_master_calibrations
from .simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap, Simulator

# Directory containing bundled spectrum data (e.g. BT-Settl .h5 files)
# Default output filenames for simulated products
DEFAULT_SIMULATED_RAMP_FILENAME = "simulated_ramp.fits"

def _fiberflat_rate_filename(spectrograph: str, fiber: int, exposure: int = 1) -> str:
    """Default filename for a single-fiber fiberflat rate FITS (matches KOA-style naming)."""
    prefix = "HB" if spectrograph.upper() == "BSPEC" else "HR"
    base = f"{prefix}.20231006.00000.00_rate-FIBERFLAT-FIB{fiber}"
    if exposure <= 1:
        return f"{base}.fits"
    return f"{base}.{exposure:03d}.fits"


def get_expected_ramp_path(output_dir: str) -> str:
    """Return the path where the simulated ramp is saved (``simulated_ramp.fits`` under ``output_dir``)."""
    return os.path.join(os.path.abspath(output_dir), DEFAULT_SIMULATED_RAMP_FILENAME)


def get_expected_fiberflat_rate_paths(
    output_dir: str,
    spectrograph: str,
    fibers: list[int],
    fiberflat_filename: callable | None = None,
) -> list[str]:
    """Return paths where dummy fiber flat rate FITS would be saved (one per fiber in 1–3)."""
    if fiberflat_filename is None:
        fiberflat_filename = _fiberflat_rate_filename
    out_abs = e2e_support.calibration_dir(output_dir)
    return [
        os.path.join(out_abs, fiberflat_filename(spectrograph, f))
        for f in sorted(fibers)
        if f in (1, 2, 3)
    ]


def ramp_file_exists(output_dir: str) -> bool:
    """Return True if the simulated ramp file already exists in output_dir."""
    return os.path.isfile(get_expected_ramp_path(output_dir))


def fiberflat_rate_files_exist(
    output_dir: str,
    spectrograph: str,
    fibers: list[int],
    fiberflat_filename: callable | None = None,
) -> bool:
    """Return True if all expected dummy fiber flat rate files exist for the given fibers."""
    paths = get_expected_fiberflat_rate_paths(
        output_dir, spectrograph, fibers, fiberflat_filename=fiberflat_filename
    )
    return len(paths) > 0 and all(os.path.isfile(p) for p in paths)


def get_sim_stage1_master_paths(output_dir: str, spectrograph: str) -> dict[str, str]:
    """
    Stable paths under ``<output_dir>/cals`` for cached sim-HDF Stage-1 masters.

    These are written once by :func:`register_sim_stage1_calibrations` /
    :func:`~hispecdrp.simulations.e2e_pipeline.prepare_pipeline_example_stage1`
    and reused on later notebook / e2e runs.
    """
    prefix = "HB" if str(spectrograph).upper() == "BSPEC" else "HR"
    cal = e2e_support.calibration_dir(output_dir)
    return {
        "trace_pos": os.path.join(cal, f"{prefix}.sim_hdf_tracepos.fits"),
        "trace_prof": os.path.join(cal, f"{prefix}.sim_hdf_traceprof.fits"),
        "fiberflat": os.path.join(cal, f"{prefix}.sim_hdf_fiberflat.fits"),
        "meta": os.path.join(cal, f"{prefix}.sim_hdf_stage1.json"),
    }


def sim_stage1_masters_exist(
    output_dir: str,
    spectrograph: str,
    fibers: list[int],
    *,
    extraction_aperture: float = 10.0,
) -> bool:
    """True when cached Stage-1 masters exist and cover the requested fibers."""
    import json

    paths = get_sim_stage1_master_paths(output_dir, spectrograph)
    if not all(os.path.isfile(paths[k]) for k in ("trace_pos", "trace_prof", "fiberflat", "meta")):
        return False
    try:
        meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    except Exception:
        return False
    if str(meta.get("spectrograph", "")).upper() != str(spectrograph).upper():
        return False
    cached_fibers = {int(f) for f in (meta.get("fibers") or [])}
    if not set(int(f) for f in fibers).issubset(cached_fibers):
        return False
    try:
        if abs(float(meta.get("extraction_aperture", 10.0)) - float(extraction_aperture)) > 1e-6:
            return False
    except (TypeError, ValueError):
        return False
    return True


def load_cached_sim_stage1_masters(
    output_dir: str,
    spectrograph: str,
    *,
    meta: dict | None = None,
) -> dict[str, datamodels.DataModel]:
    """Open cached sim-HDF Stage-1 masters from ``<output_dir>/cals``."""
    from .sim_hdf_wavesol_packaging import load_packaged_sim_hdf_line_zp_wavesol

    paths = get_sim_stage1_master_paths(output_dir, spectrograph)
    wavesol_meta = {"spectrograph": str(spectrograph).upper()}
    if meta:
        wavesol_meta.update(meta)
    return {
        "trace_pos": datamodels.open(paths["trace_pos"]),
        "trace_prof": datamodels.open(paths["trace_prof"]),
        "fiberflat": datamodels.open(paths["fiberflat"]),
        "wavesol": load_packaged_sim_hdf_line_zp_wavesol(
            str(spectrograph).upper(), meta=wavesol_meta
        ),
        "cache_paths": paths,
    }


def save_sim_stage1_masters(
    output_dir: str,
    spectrograph: str,
    *,
    trace_pos: datamodels.DataModel,
    trace_prof: datamodels.DataModel,
    fiberflat: datamodels.DataModel,
    fibers: list[int],
    extraction_aperture: float = 10.0,
) -> dict[str, str]:
    """Persist Stage-1 masters under ``<output_dir>/cals`` for reuse."""
    import json

    paths = get_sim_stage1_master_paths(output_dir, spectrograph)
    for key, model in (
        ("trace_pos", trace_pos),
        ("trace_prof", trace_prof),
        ("fiberflat", fiberflat),
    ):
        model.save(filepath=paths[key])
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "spectrograph": str(spectrograph).upper(),
                "fibers": [int(f) for f in sorted(fibers)],
                "extraction_aperture": float(extraction_aperture),
            },
            fh,
            indent=2,
        )
        fh.write("\n")
    logger.info(
        "Cached sim-HDF Stage-1 masters under %s (fibers=%s).",
        e2e_support.calibration_dir(output_dir),
        sorted(int(f) for f in fibers),
    )
    return paths


__all__ = [
    "create_simulated_calibration_models",
    "simulate_ramp_from_rate_map",
    "create_ramp_and_calibrations",
    "create_ramp_from_spectrum",
    "create_dummy_fiberflat_rate_files",
    "get_expected_ramp_path",
    "get_expected_fiberflat_rate_paths",
    "get_sim_stage1_master_paths",
    "ramp_file_exists",
    "fiberflat_rate_files_exist",
    "sim_stage1_masters_exist",
    "load_cached_sim_stage1_masters",
    "save_sim_stage1_masters",
    "get_default_stage0_prim_args",
    "run_stage0_simulation",
    "run_fiberflat_pipeline",
    "create_wavelength_solution_model_from_sim",
    "register_sim_lfc_wavesol",
    "register_sim_stage1_calibrations",
    "prepare_fiberflat_models",
    "get_fiberflat_prim_args",
]


def create_wavelength_solution_model_from_sim(
    spectrograph: str,
    *,
    meta: dict | None = None,
):
    """
    Build an LFCWavesolModel from the SimulateRateMap HDF wavelength solution for the given spectrograph.

    Used so Stage 1 set_wavesol can run with a known wavelength solution when using simulated data.

    Parameters
    ----------
    spectrograph : str
        'BSPEC' or 'RSPEC'.
    meta : dict or None, optional
        Extra metadata fields (e.g. ``mjd_start``) merged into the model meta.

    Returns
    -------
    LFCWavesolModel
        Model with wave_fib# and spec_dq_fib# set from the HDF trace solution (micron; DQ zeros).
    """
    from ..datamodels import hispec_config
    from ..datamodels.lfcwavesol import LFCWavesolModel

    srm = SimulateRateMap()
    ws_dict = srm.get_wavelength_solution(spectrograph=spectrograph)
    spectrograph = spectrograph.upper()
    order_list = hispec_config.HISPECConfig.get_order_list(spectrograph)

    # Use the real HISPEC instrument config (provides metadata schema, num_fibers, etc.)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    wavesol_model = LFCWavesolModel(
        config=hispec_config.HISPECConfig,
        meta=model_meta,
    )
    for fiber in (1, 2, 3):
        if fiber not in ws_dict or not ws_dict[fiber]:
            continue
        wave_rows = []
        for o in order_list:
            if o not in ws_dict[fiber]:
                break
            wave_rows.append(ws_dict[fiber][o]["wavelength"])
        if wave_rows:
            wave_arr = np.stack(wave_rows, axis=0).astype(np.float64)
            setattr(wavesol_model, f"wave_fib{fiber}", wave_arr)
            setattr(
                wavesol_model,
                f"spec_dq_fib{fiber}",
                np.zeros(wave_arr.shape, dtype=np.uint32),
            )
    return wavesol_model


def register_sim_lfc_wavesol(
    spectrograph: str,
    *,
    meta: dict | None = None,
    connect_remote: bool = False,
    origin: str = "LOCAL",
    new_version: bool | None = None,
    prefer_packaged_line_zp: bool = True,
) -> tuple[str, dict]:
    """
    Build a simulated ``LFCWavesolModel`` and register it in the local caldb.

    Prefers the packaged sim-HDF **line-ZP** WAVE under ``hispecdrp/data/wavesol/``
    when present (``prefer_packaged_line_zp=True``); otherwise falls back to the
    geometric HDF deposit-inverse WAVE. The model uses ``cal_type='lfc_wavesol'``,
    so Stage 1 ``set_wavesol`` can select it without ``prim_args`` overrides.

    Packaged line-ZP registration defaults to ``new_version=True`` and tags
    ``meta.note`` so :class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`
    prefers it over a same-night geometric ``lfc_wavesol`` already in the cache.

    Parameters
    ----------
    spectrograph : str
        ``'BSPEC'`` or ``'RSPEC'``.
    meta : dict or None, optional
        Metadata merged into the wavesol (should include ``instrument_era`` and
        ``mjd_start`` matching the science ramp / L0 for selector matching).
    connect_remote : bool, optional
        Open a remote KOA connection when registering. Default False (local cache only).
    origin : str, optional
        Registration origin label. Default ``'LOCAL'``.
    new_version : bool or None, optional
        Passed to :meth:`~hispecdrp.calibrations.hispec_store.HISPECCalibrationStore.register_calibration`.
        ``None`` (default) uses ``True`` for packaged line-ZP and ``False`` for the
        geometric fallback (reuse an existing version family when present).
    prefer_packaged_line_zp : bool, optional
        Use packaged line-ZP WAVE when the FITS product exists. Default True.

    Returns
    -------
    tuple of (str, dict)
        Local filepath and caldb record from registration.
    """
    from ..calibrations import HISPECCalibrationStore
    from .sim_hdf_wavesol_packaging import (
        load_packaged_sim_hdf_line_zp_wavesol,
        packaged_sim_hdf_wavesol_path,
    )

    spectrograph = str(spectrograph).upper()
    packaged = packaged_sim_hdf_wavesol_path(spectrograph)
    if prefer_packaged_line_zp and packaged.is_file():
        wavesol_model = load_packaged_sim_hdf_line_zp_wavesol(spectrograph, meta=meta)
        source = f"packaged_line_zp:{packaged.name}"
        if new_version is None:
            new_version = True
    else:
        wavesol_model = create_wavelength_solution_model_from_sim(spectrograph, meta=meta)
        source = "hdf_deposit_inverse"
        if new_version is None:
            new_version = False
    with HISPECCalibrationStore(connect_remote=connect_remote, sync_on_init=False) as store:
        if prefer_packaged_line_zp and packaged.is_file():
            existing = _existing_packaged_line_zp_wavesol(store, wavesol_model)
            if existing is not None:
                local_path, cal_record = existing
                logger.info(
                    "Reusing packaged LFC wavesol in caldb (spectrograph=%s, source=%s): %s",
                    spectrograph,
                    source,
                    local_path,
                )
                return local_path, cal_record
        local_path, cal_record = store.register_calibration(
            wavesol_model, origin=origin, new_version=bool(new_version)
        )
        if prefer_packaged_line_zp and packaged.is_file() and cal_record:
            cal_record = _ensure_packaged_line_zp_note(store, cal_record)
    if local_path is None or not (cal_record or {}).get("filename"):
        raise RuntimeError(
            f"Failed to register simulated LFC wavesol for spectrograph={spectrograph!r}: "
            f"local_path={local_path!r}, filename={(cal_record or {}).get('filename')!r}. "
            "The caldb row must have a non-null filename so Stage 1 set_wavesol can resolve it."
        )
    logger.info(
        "Registered simulated LFC wavesol in caldb (spectrograph=%s, source=%s): %s",
        spectrograph,
        source,
        local_path,
    )
    return local_path, cal_record


def _ensure_packaged_line_zp_note(store, rec: dict) -> dict:
    """Backfill ``note=sim_hdf_wavesol.line_zp`` on a caldb row when missing."""
    from .sim_hdf_wavesol_packaging import (
        PACKAGED_SIM_HDF_LINE_ZP_NOTE,
        is_packaged_sim_hdf_line_zp_record,
    )

    if is_packaged_sim_hdf_line_zp_record(rec):
        return rec
    pk = rec.get("id")
    if not pk:
        return rec
    try:
        store.local_db.table.update(pk, {"note": PACKAGED_SIM_HDF_LINE_ZP_NOTE})
        rec = dict(rec)
        rec["note"] = PACKAGED_SIM_HDF_LINE_ZP_NOTE
        logger.info(
            "Tagged existing lfc_wavesol as packaged line-ZP (id=%s, filename=%s)",
            pk,
            rec.get("filename"),
        )
    except Exception as exc:
        logger.warning("Could not backfill packaged line-ZP note on id=%s: %s", pk, exc)
    return rec


def _refresh_packaged_line_zp_cache_file(local_path: str, wavesol_model) -> None:
    """Overwrite a stale KOA local WAVE FITS with current packaged bytes."""
    from .sim_hdf_wavesol_packaging import packaged_sim_hdf_wavesol_path

    spectrograph = str(getattr(getattr(wavesol_model, "meta", None), "spectrograph", "") or "").upper()
    if spectrograph not in ("BSPEC", "RSPEC"):
        # Infer from filename prefix (HB/HR) when meta is incomplete.
        name = Path(local_path).name.upper()
        spectrograph = "BSPEC" if name.startswith("HB.") else "RSPEC" if name.startswith("HR.") else ""
    if spectrograph not in ("BSPEC", "RSPEC"):
        return
    packaged = packaged_sim_hdf_wavesol_path(spectrograph)
    if not packaged.is_file():
        return
    dest = Path(local_path)
    if not dest.is_file():
        return
    try:
        src_bytes = packaged.read_bytes()
        if dest.read_bytes() == src_bytes:
            return
        dest.write_bytes(src_bytes)
        logger.info(
            "Refreshed KOA packaged line-ZP WAVE cache from %s -> %s",
            packaged.name,
            dest.name,
        )
    except Exception as exc:
        logger.warning("Could not refresh KOA packaged WAVE at %s: %s", dest, exc)


def _existing_packaged_line_zp_wavesol(store, wavesol_model) -> tuple[str, dict] | None:
    """Return an already-registered packaged line-ZP WAVE (by id, note, or family)."""
    from .sim_hdf_wavesol_packaging import is_packaged_sim_hdf_line_zp_record

    # Packaged FITS keeps a stable UUID; re-registration often hits mode='id' with
    # a row that predates the note tag — treat that as the packaged product.
    by_id = store.calibration_record_in_cache(wavesol_model, mode="id")
    if by_id:
        path, rec = store._existing_calibration_path_and_record(by_id)
        if path and rec and rec.get("filename"):
            _refresh_packaged_line_zp_cache_file(path, wavesol_model)
            return path, _ensure_packaged_line_zp_note(store, rec)

    existing = store.calibration_record_in_cache(wavesol_model, mode="version-family")
    if not existing:
        return None
    rows = existing if isinstance(existing, list) else [existing]
    for row in rows:
        if not is_packaged_sim_hdf_line_zp_record(row):
            continue
        path, rec = store._existing_calibration_path_and_record(row)
        if path and rec and rec.get("filename"):
            _refresh_packaged_line_zp_cache_file(path, wavesol_model)
            return path, rec
    return None


def _merge_ramp_meta_into_cal(model: datamodels.DataModel, ramp_meta, spectrograph: str) -> None:
    """Copy selector keys from the science ramp onto a simulated Stage-1 cal model."""
    meta = getattr(model, "meta", None)
    if meta is None:
        return
    meta.spectrograph = str(spectrograph).upper()
    if ramp_meta is None:
        return
    for key in ("instrument_name", "instrument_era", "mjd_start", "mjd_mid", "datetime_obs"):
        val = getattr(ramp_meta, key, None)
        if val is None and hasattr(ramp_meta, "get"):
            val = ramp_meta.get(key)
        if val is not None:
            setattr(meta, key, val)
    if getattr(meta, "instrument_era", None) is None:
        meta.instrument_era = "0.0.1"
    # Packaged line-ZP FITS ships with mjd_start=0 / 1858-11-17. Without syncing
    # datetime_obs to the science MJD, registration writes HB.18581117… and the
    # Stage-1 selector prefers a same-night geometric lfc_wavesol instead.
    mjd = getattr(meta, "mjd_start", None)
    if mjd is not None and getattr(meta, "datetime_obs", None) in (None, "", "1858-11-17T00:00:00.000"):
        import astropy.time

        meta.datetime_obs = astropy.time.Time(float(mjd), format="mjd", scale="utc").isot


def register_sim_stage1_calibrations(
    output_dir: str,
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fibers: list[int] | None = None,
    ramp_meta=None,
    connect_remote: bool = False,
    origin: str = "LOCAL",
    new_version: bool = False,
    extraction_aperture: float = 10.0,
    reuse_existing: bool = True,
) -> dict:
    """
    Build and register sim-HDF ``tracepos``, ``traceprof``, ``fiberflat``, and
    packaged line-ZP ``lfc_wavesol`` into the local KOA cache.

    Until these products are published to remote KOA, Stage 1/2 can resolve them
    via the default selectors (matching spectrograph / era / nearest MJD).

    When ``reuse_existing`` is True (default), products already present under
    ``<output_dir>/cals`` (Stage-1 masters and/or fiberflat rate frames) are
    reused instead of re-simulating.
    """
    from ..calibrations import HISPECCalibrationStore
    from .e2e_pipeline import prepare_pipeline_example_stage1

    spectrograph = str(spectrograph).upper()
    fibs = [int(f) for f in (fibers if fibers is not None else [2]) if int(f) in (1, 2, 3)]
    if not fibs:
        raise ValueError("fibers must include at least one of 1, 2, 3")
    bundle = prepare_pipeline_example_stage1(
        output_dir,
        rate_map,
        spectrograph=spectrograph,
        fibers=fibs,
        extraction_aperture=float(extraction_aperture),
        reuse_existing=bool(reuse_existing),
    )
    registered: dict[str, tuple[str, dict]] = {}
    with HISPECCalibrationStore(connect_remote=connect_remote, sync_on_init=False) as store:
        for key in ("trace_pos", "trace_prof", "fiberflat", "wavesol"):
            model = bundle[key]
            _merge_ramp_meta_into_cal(model, ramp_meta, spectrograph)
            # Packaged line-ZP WAVE must bump past a same-night geometric
            # lfc_wavesol (new_version=False would skip and keep the bad WAVE).
            # Reuse an already-tagged packaged row to avoid version churn.
            use_new_version = bool(new_version)
            if key == "wavesol":
                note = str(getattr(getattr(model, "meta", None), "note", "") or "")
                if "sim_hdf_wavesol.line_zp" in note:
                    existing = _existing_packaged_line_zp_wavesol(store, model)
                    if existing is not None:
                        registered[key] = existing
                        logger.info(
                            "Reusing simulated Stage-1 wavesol in caldb (spectrograph=%s): %s",
                            spectrograph,
                            existing[0],
                        )
                        continue
                    use_new_version = True
            local_path, cal_record = store.register_calibration(
                model, origin=origin, new_version=use_new_version
            )
            if local_path is None or not (cal_record or {}).get("filename"):
                raise RuntimeError(
                    f"Failed to register simulated Stage-1 {key} for "
                    f"spectrograph={spectrograph!r}: local_path={local_path!r}"
                )
            registered[key] = (local_path, cal_record)
            logger.info(
                "Registered simulated Stage-1 %s in caldb (spectrograph=%s): %s",
                key,
                spectrograph,
                local_path,
            )
    bundle["registered"] = registered
    return bundle


def create_simulated_calibration_models(
    shape: tuple[int, int] | None = None,
    *,
    gain: float = 1.0,
    bias_dn: float | None = None,
    dark_current: float = 10.0,
    read_noise: float = 0.0,
    saturation_dn: float | None = None,
    nonlin_coeffs: np.ndarray | None = None,
    dq_bad_rows: list[int] | None = None,
    detflat_scale: float = 0.1,
    seed: int | None = None,
) -> dict[str, datamodels.CalibrationModel]:
    """
    Create a set of calibration data models for use in up-the-ramp simulation.

    All arrays are 2D with shape (ny, nx). Defaults match a typical HISPEC
    detector simulation (e.g. gain 1 e-/ADU, dark 10 e-/s, zero read noise).

    Parameters
    ----------
    shape : tuple of int, optional
        (ny, nx) detector dimensions. Default is
        :data:`~hispecdrp.simulations.e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE`.
    gain : float, optional
        Detector gain in e-/ADU. Default is 1.0.
    bias_dn : float or None, optional
        Bias level in DN. If None, set to 1000 / gain so bias is 1000 e-.
        Default is None.
    dark_current : float, optional
        Dark current in e-/s/pixel. Default is 10.0.
    read_noise : float, optional
        Read noise in e-. Default is 0.0.
    saturation_dn : float or None, optional
        Saturation threshold in DN. If None, use max uint16. Default is None.
    nonlin_coeffs : np.ndarray or None, optional
        Polynomial coefficients for nonlinearity (e.g. [c0, c1, c2] for
        np.polyval). Default is None (no nonlinearity).
    dq_bad_rows : list of int or None, optional
        Row indices to mark as DO_NOT_USE in the DQ model. Default is None.
    detflat_scale : float, optional
        Scale (std) of random detector flat field (mean 1.0). Default is 0.1.
    seed : int or None, optional
        Random seed for detflat and dark maps. Default is None.

    Returns
    -------
    dict
        Dictionary with keys: ``dq``, ``saturation``, ``gain``, ``detflat``,
        ``bias``, ``dark``, ``rn``. Each value is the corresponding
        calibration data model instance.
    """
    ny, nx = shape if shape is not None else e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE
    rng = np.random.default_rng(seed)

    if bias_dn is None:
        bias_dn = 1000.0 / gain
    if saturation_dn is None:
        saturation_dn = float(np.iinfo(np.uint16).max)
    if nonlin_coeffs is None:
        nonlin_coeffs = np.array([-1e-6, 1.0, 0.0])

    # DQ
    dq_map = np.zeros((ny, nx), dtype=np.uint32)
    if dq_bad_rows is not None:
        for r in dq_bad_rows:
            dq_map[r, :] |= DQ_FLAGS["DO_NOT_USE"]
    dq_model = datamodels.DQModel(DQ=dq_map)

    # Saturation
    sat_map = np.full((ny, nx), saturation_dn, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(
        SAT_THRESH=sat_map,
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Gain
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(
        GAIN=gain_map,
        RN=np.zeros_like(gain_map),
        COV=np.zeros((2, 2, ny, nx), dtype=np.float32),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Detector flat
    detflat_map = rng.normal(loc=1.0, scale=detflat_scale, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(
        FLAT=detflat_map,
        FLAT_ERR=np.zeros_like(detflat_map),
        FLAT_VAR_RNOISE=np.zeros_like(detflat_map),
        FLAT_VAR_POISSON=np.zeros_like(detflat_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Bias (in DN)
    bias_map = np.full((ny, nx), bias_dn, dtype=np.float32)
    bias_model = datamodels.BiasModel(
        BIAS=bias_map,
        kTC=np.zeros_like(bias_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Dark (rate in e-/s, store as DN/s via gain)
    dark_map = rng.normal(loc=dark_current, scale=dark_current / 10.0, size=(ny, nx)).astype(np.float32)
    dark_model = datamodels.DarkModel(
        RATE=dark_map / gain_map,
        RATE_ERR=np.zeros_like(dark_map),
        RATE_VAR_RNOISE=np.zeros_like(dark_map),
        RATE_VAR_POISSON=np.zeros_like(dark_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Read noise
    rn_map = np.full((ny, nx), read_noise, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(
        RN=rn_map / gain_map,
        RN_ERR=np.zeros_like(rn_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    return {
        "dq": dq_model,
        "saturation": saturation_model,
        "gain": gain_model,
        "detflat": detflat_model,
        "bias": bias_model,
        "dark": dark_model,
        "rn": rn_model,
    }


def simulate_ramp_from_rate_map(
    rate_map: np.ndarray,
    calibration_models: dict[str, datamodels.CalibrationModel],
    *,
    readtime: float = 1.7,
    n_reads: int = 10,
    nonlin_coeffs: np.ndarray | None = None,
    max_cores: int = 1,
    spectrograph: str = "RSPEC",
    convert_to_uint16: bool = True,
    clip_ramps: bool = True,
    poisson_noise: bool = True,
    kTC_noise: float = 50.0,
    mjd_start: float | None = None,
    target_meta: dict | None = None,
    seed: int | None = None,
) -> datamodels.RampModel:
    """
    Simulate a ramp from a rate map (e-/s/pixel) using create_ramp and calibration models.

    Parameters
    ----------
    rate_map : np.ndarray
        2D rate map in e-/s/pixel (e.g. from SimulateRateMap).
    calibration_models : dict
        Output of :func:`create_simulated_calibration_models`. Must contain
        ``gain``, ``detflat``, ``bias``, ``dark``, ``rn``.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads in the ramp. Default is 10.
    nonlin_coeffs : np.ndarray or None, optional
        Nonlinearity polynomial coefficients. If None, use [-1e-6, 1, 0].
        Default is None.
    max_cores : int, optional
        Number of cores for create_ramp. Default is 1.
    spectrograph : str, optional
        Spectrograph name for ramp meta. Default is "RSPEC".
    convert_to_uint16 : bool, optional
        Whether to convert ramp data to uint16. Default is True.
    clip_ramps : bool, optional
        Whether to clip ramp to uint16 range. Default is True.
    poisson_noise : bool, optional
        Whether to add Poisson noise in create_ramp. Default is True.
    kTC_noise : float, optional
        kTC noise in e- for first read. Default is 50.0.

    Returns
    -------
    RampModel
        Ramp data model with ``data`` and ``dq_raw`` and meta set.
    """
    gain_model = calibration_models["gain"]
    detflat_model = calibration_models["detflat"]
    bias_model = calibration_models["bias"]
    dark_model = calibration_models["dark"]
    rn_model = calibration_models["rn"]

    rate_shape = tuple(np.asarray(rate_map, dtype=np.float64).shape)
    for key, model in (
        ("gain", gain_model),
        ("detflat", detflat_model),
        ("bias", bias_model),
        ("dark", dark_model),
        ("rn", rn_model),
    ):
        cal_shape = e2e_support.calibration_model_detector_shape(model)
        if cal_shape != rate_shape:
            raise ValueError(
                f"Calibration '{key}' shape {cal_shape} does not match rate_map shape "
                f"{rate_shape}. Regenerate simulated calibrations at the correct detector "
                f"size ({e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE}) or delete stale "
                f"sim_*.fits files under the output cals directory."
            )

    gain_map = gain_model.gain if hasattr(gain_model, "gain") else gain_model["GAIN"]
    detflat_map = detflat_model.flat if hasattr(detflat_model, "flat") else detflat_model["FLAT"]
    bias_map = bias_model.bias if hasattr(bias_model, "bias") else bias_model["BIAS"]
    dark_map = dark_model.rate if hasattr(dark_model, "rate") else dark_model["RATE"]
    rn_map = rn_model.rn if hasattr(rn_model, "rn") else rn_model["RN"]

    if nonlin_coeffs is None:
        nonlin_coeffs = np.array([-1e-6, 1.0, 0.0])

    # create_ramp expects bias in e-, dark in e-/s, read_noise in e-; calibration models store DN and DN/s
    gain_arr = np.asarray(gain_map, dtype=np.float64)
    bias_e = np.asarray(bias_map, dtype=np.float64) * gain_arr
    dark_e_per_s = np.asarray(dark_map, dtype=np.float64) * gain_arr
    rn_dn = np.asarray(rn_map, dtype=np.float64)
    read_noise_e = rn_dn * gain_arr

    eff_mjd_start = 60577.0 if mjd_start is None else float(mjd_start)
    ramp_dict = create_ramp(
        rate_map,
        readtime=readtime,
        n_reads=n_reads,
        nonlin_coeffs=nonlin_coeffs,
        gain=gain_map,
        flat=detflat_map,
        bias=bias_e,
        dark=dark_e_per_s,
        read_noise=read_noise_e,
        kTC_noise=kTC_noise,
        poisson_noise=poisson_noise,
        mjd_start=eff_mjd_start,
        convert_to_uint16=convert_to_uint16,
        clip_ramps=clip_ramps,
        max_cores=max_cores,
        seed=seed,
    )

    ramp_dict["meta"]["spectrograph"] = spectrograph
    exposure_mid_mjd = eff_mjd_start + 0.5 * max(int(n_reads) - 1, 0) * float(readtime) / 86400.0
    ramp_dict["meta"]["mjd_start"] = float(eff_mjd_start)
    ramp_dict["meta"]["mjd_mid"] = float(exposure_mid_mjd)
    if target_meta:
        ramp_dict["meta"].update(
            {k: v for k, v in target_meta.items() if v is not None}
        )
        ramp_dict["meta"]["mjd_start"] = float(eff_mjd_start)
        if target_meta.get("mjd_mid") is not None:
            ramp_dict["meta"]["mjd_mid"] = float(target_meta["mjd_mid"])
        else:
            ramp_dict["meta"]["mjd_mid"] = float(exposure_mid_mjd)
    ramp_model = datamodels.RampModel(
        data=ramp_dict["data"],
        dq_raw=ramp_dict["dq_raw"],
        meta=ramp_dict["meta"],
    )
    ramp_model.meta.filename = DEFAULT_SIMULATED_RAMP_FILENAME
    return ramp_model


def get_default_stage0_prim_args(
    *,
    max_cores: int | None = 1,
    ramp_fit_method: str = "jwst_likely",
    run_bias_sub: bool = False,
    run_nonlin_corr: bool = False,
    run_jump_detec: bool = False,
) -> dict:
    """
    Return default primitive arguments for the Stage 0 pipeline.

    Parameters
    ----------
    max_cores : int, optional
        Number of cores for parallel primitives. Default is 1.
    ramp_fit_method : str, optional
        Ramp fit method: 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'.
        Default is 'jwst_likely'.
    run_bias_sub : bool, optional
        Whether to run bias subtraction. Default is False.
    run_nonlin_corr : bool, optional
        Whether to run nonlinearity correction. Default is False.
    run_jump_detec : bool, optional
        Whether to run jump detection. Default is False.

    Returns
    -------
    dict
        Nested dict of primitive names to keyword arguments for
        Stage0Pipeline.apply(..., prim_args=...).
    """
    mc = max_cores if max_cores is not None else 1
    return {
        "dq_init": {"max_cores": mc},
        "saturation_check": {"max_cores": mc},
        "bias_sub": {"run": run_bias_sub, "max_cores": mc},
        "nonlin_corr": {"run": run_nonlin_corr, "max_cores": mc},
        "jump_detec": {
            "run": run_jump_detec,
            "method": "tpd",
            "max_cores": mc,
            "rejection_threshold": 4.0,
        },
        "ramp_fit": {
            "method": ramp_fit_method,
            "max_cores": mc,
            "jwst_jump_detection": True,
            "jwst_jump_rejection_threshold": 4.0,
        },
        "dark_sub": {"max_cores": mc},
        "gain_corr": {"max_cores": mc},
        "detflat_corr": {"max_cores": mc},
    }


def _build_fiberflat_rate_map(
    srm: SimulateRateMap,
    spectrograph: str,
    *,
    rate_map_method: str = "pyechelle",
    pyechelle_exposure_time_s: float = 300.0,
) -> np.ndarray:
    """Build a 2-D fiberflat rate map with pyechelle raytracing or the native PRF convolver."""
    method = str(rate_map_method).strip().lower()
    if method == "pyechelle":
        if Simulator is None:
            raise ImportError(
                "pyechelle is not installed but rate_map_method='pyechelle'. "
                "Install pyechelle (see docs/source/installation.rst) or pass rate_map_method='native'."
            )
        import astropy.units as u

        return srm.create_pyechelle_rate_map(
            spectrograph=spectrograph,
            exposure_time=pyechelle_exposure_time_s * u.s,
            save=False,
        )
    if method == "native":
        return srm.create_native_rate_map(
            spectrograph=spectrograph,
            method="prf_convolution",
            save=False,
        )
    raise ValueError(
        f"Unknown fiberflat rate_map_method {rate_map_method!r}; use 'pyechelle' or 'native'."
    )


def create_dummy_fiberflat_rate_files(
    fibers: list[int],
    spectrograph: str,
    output_dir: str,
    calibration_models: dict[str, datamodels.CalibrationModel],
    *,
    flat_scale: float = 2e-14,
    flat_temperature_k: float = 1000.0,
    readtime: float = 1.7,
    n_reads: int = 10,
    max_cores: int = 1,
    stage0_prim_args: dict | None = None,
    fiberflat_filename: callable | None = None,
    fiberflat_coadd_exposures: int = 1,
    fiberflat_duplicate_single_reduction: bool = False,
    rate_map_method: str = "pyechelle",
    pyechelle_exposure_time_s: float = 300.0,
    reuse_existing: bool = True,
) -> list[str]:
    """
    Create simulated fiber flat rate FITS files using the same Stage 0 path as the science ramp.

    For each requested fiber (1, 2, 3), a whitelight blackbody spectrum (0.8–3.0 µm, scaled by
    ``flat_scale``) is placed on that fiber only. A rate map is built with pyechelle raytracing
    (default) or the native PRF convolver, then one or more ramp + Stage 0 reductions are run.
    Paths are passed to :func:`run_fiberflat_pipeline`, which coadds frames sharing the same
    fiber illumination pattern and derives trace positions with :class:`TracePositionPrimitive`.

    By default one rate FITS is written per fiber (``fiberflat_coadd_exposures=1``). Optional
    multi-exposure coadd inputs are supported when ``fiberflat_coadd_exposures`` > 1.

    When ``reuse_existing`` is True (default), fibers whose expected rate FITS already
    exist under ``<output_dir>/cals`` are skipped (no re-raytrace / UTR / Stage 0).
    """
    import astropy.units as u
    from astropy.modeling.models import BlackBody

    if stage0_prim_args is None:
        stage0_prim_args = get_default_stage0_prim_args(max_cores=max_cores)
    if fiberflat_filename is None:
        fiberflat_filename = _fiberflat_rate_filename
    if fiberflat_coadd_exposures < 1:
        raise ValueError(f"fiberflat_coadd_exposures must be >= 1, got {fiberflat_coadd_exposures}")

    pipeline_root = e2e_support.ensure_output_directory(output_dir)
    out_dir_abs = e2e_support.calibration_dir(pipeline_root)
    paths: list[str] = []
    fibers_to_build: list[int] = []

    for f in sorted(fibers):
        if f not in (1, 2, 3):
            continue
        expected = [
            os.path.join(out_dir_abs, fiberflat_filename(spectrograph, f, exposure))
            for exposure in range(1, fiberflat_coadd_exposures + 1)
        ]
        if reuse_existing and all(os.path.isfile(p) for p in expected):
            logger.info(
                "Reusing existing fiberflat rate file(s) for fiber %s under %s.",
                f,
                out_dir_abs,
            )
            paths.extend(expected)
            continue
        fibers_to_build.append(f)

    if not fibers_to_build:
        return [
            os.path.join(out_dir_abs, fiberflat_filename(spectrograph, f, exposure))
            for f in sorted(fibers)
            if f in (1, 2, 3)
            for exposure in range(1, fiberflat_coadd_exposures + 1)
        ]

    # Coarse BB grid; upsampled to R~3e5 when continuous=True (required for smooth native PRF maps).
    n_wl_flat = 5_000 if str(rate_map_method).strip().lower() == "pyechelle" else 10_000
    wl_flat = np.linspace(0.8, 3.0, n_wl_flat) * u.micron
    wl_aa = wl_flat.to(u.AA)
    bb_scale = 1.0 * u.erg / (u.cm**2 * u.s * u.AA * u.sr)
    bb = BlackBody(temperature=flat_temperature_k * u.K, scale=bb_scale)
    bb_lam_aa = bb(wl_aa)
    # BlackBody returns surface brightness (per sr); use .value and attach flux-density unit (sr folded into flat_scale)
    # Convert per-Angstrom to per-micron: 1 micron = 1e4 AA => same flux density => multiply by 1e4
    flux_flat = (flat_scale * np.asarray(bb_lam_aa.value) * 1e4) * (u.erg / u.s / u.cm**2 / u.micron)

    for f in fibers_to_build:
        logger.info(
            "Creating simulated fiber flat rate file(s) for fiber %s "
            "(rate_map_method=%s, coadd_exposures=%s, duplicate_single_reduction=%s).",
            f,
            rate_map_method,
            fiberflat_coadd_exposures,
            fiberflat_duplicate_single_reduction,
        )
        srm = SimulateRateMap()
        srm.output_directory = out_dir_abs
        srm.pyechelle_outname = f"pyechelle_fiberflat_fib{f}.fits"
        # Continuum blackbody: upsample to R~3e5 so the native PRF convolver stamps overlap
        # along each order (continuous=False leaves ~15–20% of dispersion pixels empty).
        srm.set_fiber_spectrum(
            fiber_number=f,
            spectrum_wavelengths=wl_flat,
            spectrum_flux=flux_flat,
            continuous=True,
        )
        rate_map = _build_fiberflat_rate_map(
            srm,
            spectrograph,
            rate_map_method=rate_map_method,
            pyechelle_exposure_time_s=pyechelle_exposure_time_s,
        )

        def _finalize_and_save(rate_model: datamodels.DataModel, exposure: int) -> str:
            obj_flat = "WHITELIGHT"
            for fib in (1, 2, 3):
                setattr(
                    rate_model.meta,
                    f"object_fiber{fib}",
                    obj_flat if fib == f else "BLOCKED",
                )
            filename = fiberflat_filename(spectrograph, f, exposure)
            filepath = os.path.join(out_dir_abs, filename)
            rate_model.save(filepath=filepath)
            logger.debug("Saved dummy fiber flat rate file: %s", filepath)
            return filepath

        if fiberflat_coadd_exposures == 1:
            ramp_model = simulate_ramp_from_rate_map(
                rate_map,
                calibration_models,
                readtime=readtime,
                n_reads=n_reads,
                max_cores=max_cores,
                spectrograph=spectrograph,
                poisson_noise=True,
                convert_to_uint16=True,
            )
            rate_model = run_stage0_simulation(
                ramp_model,
                calibration_models,
                output_dir=out_dir_abs,
                prim_args=stage0_prim_args,
                max_cores=max_cores,
            )
            paths.append(_finalize_and_save(rate_model, 1))
            continue

        if fiberflat_duplicate_single_reduction:
            ramp_model = simulate_ramp_from_rate_map(
                rate_map,
                calibration_models,
                readtime=readtime,
                n_reads=n_reads,
                max_cores=max_cores,
                spectrograph=spectrograph,
                poisson_noise=True,
                convert_to_uint16=True,
            )
            rate_model = run_stage0_simulation(
                ramp_model,
                calibration_models,
                output_dir=out_dir_abs,
                prim_args=stage0_prim_args,
                max_cores=max_cores,
            )
            primary_path = _finalize_and_save(rate_model, 1)
            paths.append(primary_path)
            for exposure in range(2, fiberflat_coadd_exposures + 1):
                dup_path = os.path.join(
                    out_dir_abs, fiberflat_filename(spectrograph, f, exposure)
                )
                shutil.copy2(primary_path, dup_path)
                paths.append(dup_path)
            continue

        for exposure in range(1, fiberflat_coadd_exposures + 1):
            ramp_model = simulate_ramp_from_rate_map(
                rate_map,
                calibration_models,
                readtime=readtime,
                n_reads=n_reads,
                max_cores=max_cores,
                spectrograph=spectrograph,
                poisson_noise=True,
                convert_to_uint16=True,
            )
            rate_model = run_stage0_simulation(
                ramp_model,
                calibration_models,
                output_dir=out_dir_abs,
                prim_args=stage0_prim_args,
                max_cores=max_cores,
            )
            paths.append(_finalize_and_save(rate_model, exposure))

    # Stable fiber/exposure order for callers (reuse + newly built).
    return [
        os.path.join(out_dir_abs, fiberflat_filename(spectrograph, f, exposure))
        for f in sorted(fibers)
        if f in (1, 2, 3)
        for exposure in range(1, fiberflat_coadd_exposures + 1)
    ]


def run_stage0_simulation(
    ramp_model: datamodels.RampModel,
    calibration_models: dict[str, datamodels.CalibrationModel],
    output_dir: str = ".",
    prim_args: dict | None = None,
    *,
    max_cores: int | None = None,
) -> datamodels.DataModel:
    """
    Run the Stage 0 pipeline on a simulated ramp with the given calibrations.

    Parameters
    ----------
    ramp_model : RampModel
        Simulated ramp from :func:`simulate_ramp_from_rate_map`.
    calibration_models : dict
        Output of :func:`create_simulated_calibration_models`. Must contain
        ``dq``, ``saturation``, ``bias``, ``gain``, ``rn``, ``dark``, ``detflat``.
    output_dir : str, optional
        Directory for pipeline outputs. Default is ".".
    prim_args : dict or None, optional
        Primitive arguments for the pipeline. If None, use
        :func:`get_default_stage0_prim_args`(). Default is None.

    Returns
    -------
    DataModel
        Result of Stage0Pipeline.apply (e.g. rate or slope model).
    """
    if prim_args is None:
        prim_args = get_default_stage0_prim_args()
    if max_cores is not None:
        prim_args = e2e_support.apply_max_cores_to_stage0_prim_args(prim_args, max_cores)
    prim_args = e2e_support.apply_e2e_save_and_preview_flags(
        prim_args, Stage0Pipeline.PRIMITIVES
    )

    output_dir = e2e_support.ensure_output_directory(output_dir)
    logger.info("Running Stage 0 pipeline (output_dir=%s).", output_dir)
    ny, nx = int(ramp_model.data.shape[1]), int(ramp_model.data.shape[2])
    nonlin_model = calibration_models.get("nonlin") or e2e_support.identity_nonlin_calibration_model(
        ny, nx
    )

    pipe = Stage0Pipeline()
    pipe.dq_init.dq = calibration_models["dq"]
    pipe.saturation_check.saturation = calibration_models["saturation"]
    pipe.bias_sub.bias = calibration_models["bias"]
    pipe.nonlin_corr.nonlin = nonlin_model
    pipe.jump_detec.gain = calibration_models["gain"]
    pipe.jump_detec.rn = calibration_models["rn"]
    pipe.ramp_fit.gain = calibration_models["gain"]
    pipe.ramp_fit.rn = calibration_models["rn"]
    pipe.dark_sub.dark = calibration_models["dark"]
    pipe.gain_corr.gain = calibration_models["gain"]
    pipe.detflat_corr.detflat = calibration_models["detflat"]

    return pipe.apply(ramp_model, prim_args=prim_args, output_dir=output_dir)


def create_ramp_and_calibrations(
    rate_map: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
    *,
    fibers: list[int] | None = None,
    spectrum_wavelengths=None,
    spectrum_flux=None,
    radial_velocity_km_s: float = 0.0,
    output_dir: str = ".",
    calibration_models: dict[str, datamodels.CalibrationModel] | None = None,
    use_master_calibrations: bool = False,
    readtime: float = 1.7,
    n_reads: int = 10,
    spectrograph: str = "RSPEC",
    max_cores: int = 1,
    poisson_noise: bool = True,
    convert_to_uint16: bool = True,
    instrument_name: str | None = None,
    instrument_era: str = "0.0.1",
    mjd_start: float | None = None,
    datetime_obs: str | None = None,
    connect_remote: bool = True,
    sync_caldb_first: bool = True,
    template_mag_ab: float | None = None,
    template_2mass_band: str | None = None,
    template_filename: str | None = None,
    overwrite_ramp: bool = False,
    target_meta: dict | None = None,
    seed: int | None = None,
    **cal_kw,
) -> tuple[datamodels.RampModel, dict[str, datamodels.CalibrationModel]]:
    """
    Create calibration models and a simulated ramp (no Stage 0 pipeline).

    Use this to separate simulation setup from running Stage 0. If ``rate_map``
    is None, a default rate map is generated. Calibrations are either fetched
    from the store (when ``use_master_calibrations`` is True), passed in via
    ``calibration_models``, or created via
    :func:`create_simulated_calibration_models`(shape, **cal_kw).

    Parameters
    ----------
    rate_map : np.ndarray or None, optional
        2D rate map in e-/s/pixel. If None, a default is created via SimulateRateMap.
    shape : tuple of int or None, optional
        (ny, nx) for dummy calibrations when ``calibration_models`` is None and
        ``use_master_calibrations`` is False and ``rate_map`` is unavailable.
        Normally inferred from ``rate_map``; default is
        :data:`~hispecdrp.simulations.e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE`.
    fibers : list of int or None, optional
        Fiber numbers (1, 2, 3, and/or 4) to place spectra on when building the
        default rate map. Only 1, 2, 3 are supported by SimulateRateMap; others
        are ignored for rate map creation. If None, default is [1].
    spectrum_wavelengths : array-like with units or None, optional
        Wavelength array for the input spectrum when building the default rate
        map. If None (and spectrum_flux is None), a flat spectrum is used.
    spectrum_flux : array-like with units or None, optional
        Flux array for the input spectrum. Must be provided with
        spectrum_wavelengths to use an input spectrum.
    radial_velocity_km_s : float, optional
        Radial velocity in km/s to apply to the input spectrum before generating
        the rate map (Doppler shift). Must be in the range -500 to 500. The
        template is densified first, then shifted via
        ``SimulateRateMap.apply_radial_velocity`` (``λ × (1+z)``) before
        ``create_native_rate_map``. Default is 0.0.
    output_dir : str, optional
        Directory for saving the rate map when building the default rate map
        (when ``rate_map`` is None). Default is ".".
    calibration_models : dict or None, optional
        Calibration models to use. If None, master or dummy models are used.
    use_master_calibrations : bool, optional
        If True and ``calibration_models`` is None, fetch from calibration store.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads. Default is 10.
    spectrograph : str, optional
        Spectrograph name. Default is "RSPEC".
    max_cores : int, optional
        Cores for ramp creation. Default is 1.
    poisson_noise : bool, optional
        Whether to add Poisson noise in create_ramp. Default is True.
    convert_to_uint16 : bool, optional
        Whether to convert ramp data to uint16. Default is True.
    seed : int or None, optional
        RNG seed for UTR noise (Poisson / read / kTC). ``None`` is nondeterministic.
    instrument_name, instrument_era, mjd_start, datetime_obs : optional
        Used when ``use_master_calibrations`` is True.
    connect_remote : bool, optional
        Whether to connect to remote when fetching master calibrations.
    sync_caldb_first : bool, optional
        When using master calibrations, sync store first.
    template_mag_ab, template_2mass_band : optional
        When building the default rate map from the bundled BT-Settl template, normalize
        to this AB magnitude in the given 2MASS band (``'J'``, ``'H'``, or ``'K'``).
    **cal_kw
        Passed to :func:`create_simulated_calibration_models` when creating dummy calibrations.

    Returns
    -------
    tuple of (RampModel, dict, ndarray)
        ``(ramp_model, calibration_models, rate_map)``. Pass the first two to
        :func:`run_stage0_simulation` to run the Stage 0 pipeline. ``rate_map`` is the
        2D e-/s/pixel array used for ramp simulation (same data as ``rate_map.fits``).
    """
    if (template_mag_ab is not None) ^ (template_2mass_band is not None):
        raise ValueError(
            "template_mag_ab and template_2mass_band must both be set or both be None."
        )
    _band: str | None = None
    if template_2mass_band is not None:
        _band = str(template_2mass_band).strip().upper()
        if _band not in ("J", "H", "K"):
            raise ValueError("template_2mass_band must be one of 'J', 'H', 'K'")

    output_dir = e2e_support.ensure_output_directory(output_dir)
    e2e_support.ensure_koa_calibration_cache_env(output_dir)
    ramp_path = get_expected_ramp_path(output_dir)

    if rate_map is None:
        if fibers is None:
            fibers = list(e2e_support.DEFAULT_RATE_MAP_FIBERS)
        logger.info(
            "Building default rate map (fibers=%s, spectrograph=%s, output_dir=%s).",
            fibers,
            spectrograph,
            output_dir,
        )
        srm = SimulateRateMap()
        srm.output_directory = output_dir
        srm.rate_map_outname = DEFAULT_RATE_MAP_FILENAME
        import astropy.units as u

        if spectrum_wavelengths is not None and spectrum_flux is not None:
            wl = spectrum_wavelengths
            flux = spectrum_flux
        else:
            template_path = get_spectrum_path(
                template_filename or e2e_support.DEFAULT_TEMPLATE_FILENAME,
                base_dir=e2e_support.templates_data_dir(),
            )
            wl, flux = load_btsettl_spectrum(
                template_path,
                mag_ab=template_mag_ab,
                filter_band=_band,
            )
            wl_val = np.asarray(getattr(wl, "value", wl), dtype=np.float64)
            sort_idx = np.argsort(wl_val)
            wl = wl[sort_idx]
            flux = flux[sort_idx]
        # Densify the native template once, then Doppler-shift that densified
        # grid (λ × (1+z)). Keep densify-only PCHIP (no global linear rebuild),
        # which previously downsampled already-dense BSPEC BT-Settl lines.
        for f in fibers:
            if f in (1, 2, 3):
                srm.set_fiber_spectrum(
                    fiber_number=f,
                    spectrum_wavelengths=wl,
                    spectrum_flux=flux,
                    continuous=False,
                )
        from .simulate_rate_map import (
            _check_wavelength_sampling,
            _upsample_spectrum,
        )

        for f in fibers:
            if f not in (1, 2, 3) or f not in srm.fibers:
                continue
            fib = srm.fibers[f]
            w_f = fib["wavelength"]
            fl_f = fib["flux"]
            if not _check_wavelength_sampling(w_f, min_resolution=3e5):
                logger.info(
                    "Densifying fiber=%s spectrum to R>=3e5 before Doppler "
                    "(PCHIP, densify-only)",
                    f,
                )
                w_f, fl_f = _upsample_spectrum(
                    w_f, fl_f, desired_sampling_resolution=3e5
                )
                fib["wavelength"] = w_f
                fib["flux"] = fl_f
            fib["continuous"] = True
        if radial_velocity_km_s != 0.0:
            if not (-500.0 <= radial_velocity_km_s <= 500.0):
                raise ValueError(
                    f"radial_velocity_km_s must be in [-500, 500], got {radial_velocity_km_s}"
                )
            rv_quantity = radial_velocity_km_s * u.km / u.s
            logger.info(
                "Applying densified-template Doppler shift (topocentric) "
                "radial_velocity_km_s=%.6f",
                float(radial_velocity_km_s),
            )
            for f in fibers:
                if f in (1, 2, 3):
                    srm.apply_radial_velocity(fiber_number=f, rv=rv_quantity)
        rate_map = srm.create_native_rate_map(
            spectrograph=spectrograph, method="prf_convolution"
        )
        logger.debug("Rate map shape: %s", getattr(rate_map, "shape", None))

    if calibration_models is None:
        if use_master_calibrations:
            logger.info(
                "Fetching master calibrations from calibration store (spectrograph=%s).",
                spectrograph,
            )
            calibration_models = get_master_calibrations(
                meta=None,
                instrument_name=instrument_name,
                instrument_era=instrument_era,
                spectrograph=spectrograph,
                mjd_start=mjd_start,
                datetime_obs=datetime_obs,
                connect_remote=connect_remote,
                sync_first=sync_caldb_first,
            )
        else:
            rm = np.asarray(rate_map, dtype=np.float64)
            if rm.ndim != 2:
                raise ValueError(
                    f"rate_map must be 2D when creating dummy calibrations; got shape {rm.shape}"
                )
            eff_shape = (int(rm.shape[0]), int(rm.shape[1]))
            keys = ("dq", "saturation", "gain", "detflat", "bias", "dark", "rn")
            cal_models: dict[str, datamodels.CalibrationModel] = {}
            for k in keys:
                fp = e2e_support.dummy_cal_path(output_dir, k)
                cached = e2e_support.open_cached_stage0_calibration(
                    fp, eff_shape, key=k
                )
                if cached is not None:
                    cal_models[k] = cached
            missing = [k for k in keys if k not in cal_models]
            if missing:
                logger.debug(
                    "Creating simulated calibration models (shape=%s); missing=%s",
                    eff_shape,
                    missing,
                )
                new_models = create_simulated_calibration_models(eff_shape, **cal_kw)
                cal_models |= new_models
                for k in missing:
                    try:
                        cal_models[k].save(filepath=e2e_support.dummy_cal_path(output_dir, k))
                    except Exception as e:
                        logger.warning("Could not save dummy calibration %s: %s", k, e)
            calibration_models = cal_models

    logger.info("Simulating ramp (n_reads=%s, readtime=%s s).", n_reads, readtime)
    ramp_model = simulate_ramp_from_rate_map(
        rate_map,
        calibration_models,
        readtime=readtime,
        n_reads=n_reads,
        max_cores=max_cores,
        spectrograph=spectrograph,
        poisson_noise=poisson_noise,
        convert_to_uint16=convert_to_uint16,
        mjd_start=mjd_start,
        target_meta=target_meta,
        seed=seed,
    )
    if overwrite_ramp or not os.path.isfile(ramp_path):
        ramp_model.save(filepath=ramp_path)
    return ramp_model, calibration_models, np.asarray(rate_map, dtype=np.float64)


def create_ramp_from_spectrum(
    output_dir: str,
    *,
    spectrograph: str = "BSPEC",
    fibers: list[int] | None = None,
    template_filename: str | None = None,
    mag_ab: float | None = 10.0,
    filter_band: str | None = "J",
    spectrum_wavelengths=None,
    spectrum_flux=None,
    radial_velocity_km_s: float = 0.0,
    barycentric_velocity_km_s: float | None = 0.0,
    inject_rv_via: str = "spectrum",
    use_master_calibrations: bool = True,
    connect_remote: bool = False,
    sync_caldb_first: bool = False,
    n_reads: int = 10,
    overwrite_ramp: bool = True,
    register_wavesol: bool = True,
    register_stage1_sims: bool = True,
    reuse_stage1_sims: bool = True,
    seed: int | None = 0,
    **kwargs,
) -> tuple[datamodels.RampModel, dict[str, datamodels.CalibrationModel], np.ndarray]:
    """
    Build a native rate map from a 1D spectrum and return a simulated ``RampModel``.

    Convenience wrapper around :func:`create_ramp_and_calibrations` for the common
    notebook / user path: stamp fibers with a BT-Settl (or custom) spectrum via
    :class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`, then simulate
    an up-the-ramp readout using master calibrations from ``KOA_CALIBRATION_CACHE``
    (when ``use_master_calibrations`` is True).

    When ``register_stage1_sims`` is True (default), builds sim-HDF ``tracepos`` /
    ``traceprof`` / ``fiberflat`` plus the packaged line-ZP ``lfc_wavesol`` and
    registers them in the local caldb so Stage 1/2 default selectors resolve them
    (until those products are available from remote KOA). With
    ``reuse_stage1_sims=True`` (default), products already under
    ``<output_dir>/cals`` are reused instead of re-simulating fiberflat rates.
    When only ``register_wavesol`` is True, registers the packaged line-ZP WAVE alone.

    The returned ``calibration_models`` dict always includes ``wavesol``: the
    packaged sim-HDF **line-ZP** ``LFCWavesolModel`` (with baked CCF-order refine
    when present). That product is registered into the local caldb with a
    distinctive ``meta.note`` so Stage 1 ``set_wavesol`` selects it by default
    even when a same-night geometric ``lfc_wavesol`` is also cached. When Stage-1
    sims are registered, ``trace_pos``, ``trace_prof``, and ``fiberflat`` are
    also attached to ``calibration_models``.

    By default (``inject_rv_via='spectrum'``), stellar RV is applied physically:
    densify the 1D template, Doppler-shift that densified spectrum
    (``λ × (1+z)``), paint with ZEMAX, and attach the **unscaled** packaged WAVE
    for every injected RV. Pass ``inject_rv_via='wavesol'`` to keep the older
    closed-loop path (paint at RV=0, scale WAVE by ``(1+z)``) for comparison.

    Parameters
    ----------
    output_dir : str
        Directory for the rate map and raw ramp products.
    spectrograph : str, optional
        ``'BSPEC'`` or ``'RSPEC'``. Default ``'BSPEC'``.
    fibers : list of int or None, optional
        Fiber numbers to illuminate (1–3). Default ``[2]`` (fiber-2 RV path until
        packaged WAVE/masks are validated for fibers 1 and 3).
    template_filename : str or None, optional
        BT-Settl HDF5 filename under the package templates directory. Ignored when
        ``spectrum_wavelengths`` and ``spectrum_flux`` are provided. Default is the
        E2E default template when neither a custom spectrum nor this name is given.
    mag_ab, filter_band : optional
        Normalize the BT-Settl template to this AB magnitude in a 2MASS band
        (``'J'``, ``'H'``, or ``'K'``). Both must be set or both ``None``.
    spectrum_wavelengths, spectrum_flux : optional
        Custom 1D spectrum (with units). When both are set, the template is not loaded.
    radial_velocity_km_s : float, optional
        Injected stellar radial velocity (km/s). Interpretation depends on
        ``inject_rv_via`` (see below). Default 0.
    barycentric_velocity_km_s : float or None, optional
        Known barycentric Earth radial velocity for the simulated exposure (km/s).
        Default ``0.0`` stamps ``meta.force_berv_m_s`` so Stage-2 barycorr writes
        BERV=0 and is a no-op for the CCF. Pass ``None`` to leave BERV unset
        (barycorrpy will require target coordinates).
    inject_rv_via : {'spectrum', 'wavesol'}, optional
        How to realize ``radial_velocity_km_s``. ``'spectrum'`` (default): densify
        then Doppler-paint the 1D template; WAVE stays at the packaged line-ZP
        bake (RV=0 wavelengths). ``'wavesol'``: paint at RV=0 and scale the
        returned / Stage-1 WAVE by ``(1+z)`` (legacy closed-loop path).
    use_master_calibrations : bool, optional
        Fetch Stage-0 masters from the KOA local caldb. Default True.
    connect_remote, sync_caldb_first : bool, optional
        Remote KOA options for master-cal lookup. Defaults False (use local cache).
    n_reads : int, optional
        Number of UTR reads. Default 10.
    overwrite_ramp : bool, optional
        Overwrite an existing ramp FITS under ``output_dir``. Default True.
    register_wavesol : bool, optional
        Register a simulated ``lfc_wavesol`` (packaged line-ZP when available) when
        ``register_stage1_sims`` is False. Default True.
    register_stage1_sims : bool, optional
        Register sim-HDF Stage-1 products (traces, fiberflat, packaged WAVE).
        Default True. Implies WAVE registration.
    reuse_stage1_sims : bool, optional
        When True (default), reuse Stage-1 masters / fiberflat rate frames already
        present under ``<output_dir>/cals``. Set False to force re-simulation.
    seed : int or None, optional
        UTR noise seed (default ``0``). Keep fixed so the packaged WAVE CCF bake
        transfers across notebook re-runs; change only when intentionally drawing
        a new noise realization.
    **kwargs
        Forwarded to :func:`create_ramp_and_calibrations`.

    Returns
    -------
    tuple
        ``(ramp_model, calibration_models, rate_map)``. ``calibration_models``
        includes Stage-0 masters plus ``wavesol`` (packaged line-ZP) and, when
        Stage-1 sims are built, ``trace_pos`` / ``trace_prof`` / ``fiberflat``.
    """
    if fibers is None:
        fibers = [2]
    target_meta = dict(kwargs.pop("target_meta", None) or {})
    if barycentric_velocity_km_s is not None:
        target_meta["force_berv_m_s"] = float(barycentric_velocity_km_s) * 1000.0

    inject_mode = str(inject_rv_via or "spectrum").strip().lower()
    if inject_mode not in ("spectrum", "wavesol"):
        raise ValueError(
            f"inject_rv_via must be 'spectrum' or 'wavesol', got {inject_rv_via!r}"
        )
    inject_rv_km_s = float(radial_velocity_km_s)
    use_wave_inject = inject_mode == "wavesol"
    paint_rv_km_s = 0.0 if use_wave_inject else inject_rv_km_s
    if use_wave_inject and inject_rv_km_s != 0.0:
        logger.info(
            "inject_rv_via='wavesol': painting spectrum at RV=0 and "
            "injecting radial_velocity_km_s=%.6f via WAVE×(1+z)",
            inject_rv_km_s,
        )
    elif inject_rv_km_s != 0.0:
        logger.info(
            "inject_rv_via='spectrum': densify→Doppler-paint "
            "radial_velocity_km_s=%.6f; WAVE left at packaged line-ZP bake",
            inject_rv_km_s,
        )

    ramp_model, calibration_models, rate_map = create_ramp_and_calibrations(
        rate_map=None,
        output_dir=output_dir,
        spectrograph=spectrograph,
        fibers=fibers,
        spectrum_wavelengths=spectrum_wavelengths,
        spectrum_flux=spectrum_flux,
        radial_velocity_km_s=paint_rv_km_s,
        template_filename=template_filename,
        template_mag_ab=mag_ab,
        template_2mass_band=filter_band,
        use_master_calibrations=use_master_calibrations,
        connect_remote=connect_remote,
        sync_caldb_first=sync_caldb_first,
        n_reads=n_reads,
        overwrite_ramp=overwrite_ramp,
        target_meta=target_meta or None,
        seed=seed,
        **kwargs,
    )
    ramp_meta = getattr(ramp_model, "meta", None)
    wavesol_meta = _wavesol_meta_from_ramp(ramp_meta, spectrograph, kwargs.get("instrument_era", "0.0.1"))
    if register_stage1_sims:
        bundle = register_sim_stage1_calibrations(
            output_dir,
            rate_map,
            spectrograph=spectrograph,
            fibers=fibers,
            ramp_meta=ramp_meta,
            connect_remote=False,
            reuse_existing=bool(reuse_stage1_sims),
        )
        for key in ("trace_pos", "trace_prof", "fiberflat", "wavesol"):
            model = bundle.get(key)
            if model is not None:
                calibration_models[key] = model
    elif register_wavesol:
        register_sim_lfc_wavesol(
            spectrograph,
            meta=wavesol_meta,
            connect_remote=False,
            prefer_packaged_line_zp=True,
        )
        calibration_models["wavesol"] = _load_pipeline_example_wavesol(
            spectrograph, meta=wavesol_meta
        )
    else:
        calibration_models["wavesol"] = _load_pipeline_example_wavesol(
            spectrograph, meta=wavesol_meta
        )
    if use_wave_inject and inject_rv_km_s != 0.0 and calibration_models.get("wavesol") is not None:
        _scale_wavesol_for_injected_rv(calibration_models["wavesol"], inject_rv_km_s)
    elif calibration_models.get("wavesol") is not None:
        meta = getattr(calibration_models["wavesol"], "meta", None)
        if meta is not None:
            try:
                meta.injected_rv_km_s = inject_rv_km_s
                meta.injected_rv_via = "spectrum"
            except Exception:
                pass
    return ramp_model, calibration_models, rate_map


def _wavesol_meta_from_ramp(ramp_meta, spectrograph: str, instrument_era: str = "0.0.1") -> dict:
    """Selector-friendly meta for the packaged / registered sim WAVE."""
    wavesol_meta: dict = {"spectrograph": str(spectrograph).upper()}
    if ramp_meta is not None:
        for key in ("instrument_name", "instrument_era", "mjd_start", "mjd_mid", "datetime_obs"):
            val = getattr(ramp_meta, key, None)
            if val is None and hasattr(ramp_meta, "get"):
                val = ramp_meta.get(key)
            if val is not None:
                wavesol_meta[key] = val
    wavesol_meta.setdefault("instrument_era", instrument_era)
    mjd = wavesol_meta.get("mjd_start")
    if mjd is not None and not wavesol_meta.get("datetime_obs"):
        import astropy.time

        wavesol_meta["datetime_obs"] = astropy.time.Time(
            float(mjd), format="mjd", scale="utc"
        ).isot
    return wavesol_meta


def _load_pipeline_example_wavesol(
    spectrograph: str,
    *,
    meta: dict | None = None,
):
    """
    Load the packaged sim-HDF line-ZP WAVE (preferred) or geometric HDF WAVE.
    """
    from .sim_hdf_wavesol_packaging import (
        load_packaged_sim_hdf_line_zp_wavesol,
        packaged_sim_hdf_wavesol_path,
    )

    spectrograph = str(spectrograph).upper()
    packaged = packaged_sim_hdf_wavesol_path(spectrograph)
    if packaged.is_file():
        return load_packaged_sim_hdf_line_zp_wavesol(spectrograph, meta=meta)
    return create_wavelength_solution_model_from_sim(spectrograph, meta=meta)


def _scale_wavesol_for_injected_rv(wavesol_model, radial_velocity_km_s: float) -> None:
    """
    Doppler-scale ``WAVE_FIB*`` by ``(1 + v/c)`` for injected stellar RV.

    Packaged line-ZP WAVE is closed-loop calibrated at RV=0 (absorption ZP + CCF
    order refine). Spectrum Doppler paint does **not** transfer through that WAVE
    (effective recovered RV is systematically low). Injecting via WAVE matches the
    validated ``WAVE × (1+z)`` closed-loop path used in WAVE bake / RV tests.
    """
    import astropy.constants as const
    import astropy.units as u

    rv = float(radial_velocity_km_s)
    if rv == 0.0:
        return
    factor = float((1.0 + (rv * u.km / u.s) / const.c).decompose().value)
    for fiber in (1, 2, 3):
        attr = f"wave_fib{fiber}"
        wave = getattr(wavesol_model, attr, None)
        if wave is None:
            continue
        setattr(wavesol_model, attr, np.asarray(wave, dtype=np.float64) * factor)
    meta = getattr(wavesol_model, "meta", None)
    if meta is not None:
        try:
            meta.injected_rv_km_s = rv
            meta.injected_rv_via = "wavesol_scale"
        except Exception:
            pass
    logger.info(
        "Scaled packaged WAVE by (1+z) for injected radial_velocity_km_s=%.6f "
        "(factor=%.12f); science spectrum painted at RV=0",
        rv,
        factor,
    )



def _patch_fiberflat_model_object_fiber_meta(model: datamodels.DataModel, filepath: str) -> None:
    """
    Set object_fiber1/2/3 on model.meta from filename so group_by_illuminated_fiber finds fibers.

    Existing FITS (e.g. from disk or GDrive) may lack these; the fiber flat pipeline and
    trace primitives require them. Infers fiber from ...FIB1.../FIB2.../FIB3... and
    Uses WHITELIGHT for the illuminated fiber (per target.yaml object_fiber enum).
    """
    name = os.path.basename(filepath).upper()
    # Infer illuminated fiber from filename (e.g. rate-FIBERFLAT-FIB1.fits or ...FIB1...)
    match = re.search(r"FIB(?:ERFLAT-)?FIB([123])", name) or re.search(r"FIB([123])\b", name)
    fiber = int(match.group(1)) if match else 1
    obj_flat = "WHITELIGHT"
    for f in (1, 2, 3):
        setattr(model.meta, f"object_fiber{f}", obj_flat if f == fiber else "BLOCKED")


def prepare_fiberflat_models(fiberflat_filepaths: list[str]) -> list[datamodels.DataModel]:
    """
    Open fiberflat rate FITS files and patch object_fiber meta so pipelines can group by fiber.

    Parameters
    ----------
    fiberflat_filepaths : list of str
        Paths to fiberflat rate FITS files (one per fiber).

    Returns
    -------
    list of DataModel
        Opened models with object_fiber# meta set.
    """
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]
    for model, fp in zip(ff_models, fiberflat_filepaths):
        _patch_fiberflat_model_object_fiber_meta(model, fp)
    return ff_models


_DEFAULT_FIBERFLAT_DISP_RANGE = {"BSPEC": (300, 4000), "RSPEC": (400, 4000)}
# BSPEC order 110 sits near cross-disp pixel ~105; (135, …) clips it and yields 38/39 traces.
_DEFAULT_FIBERFLAT_XDISP_RANGE = {"BSPEC": (100, 4050), "RSPEC": (100, 4050)}
# Default peak-finding threshold (fraction of the brightest order) for simulated fiberflats.
# Robust order detection (NaN interpolation + order-spacing-based peak separation in
# ``find_trace_positions``) makes a single value reliable across the blaze dynamic range of
# both BSPEC and RSPEC, so a per-fiber tuned map is no longer required.
_DEFAULT_FIBERFLAT_PROMINENCE = 0.02


def get_fiberflat_prim_args(
    ff_models: list[datamodels.DataModel],
    *,
    disp_range: tuple[int, int] | None = None,
    xdisp_range: tuple[int, int] | None = None,
    deg_pos: int = 4,
    deg_width: int = 2,
    extraction_aperture: int = 10,
    prominence_factor: float | dict[str, float] | None = None,
) -> dict:
    """Build default primitive arguments for ProcessFiberflatsPipeline from opened fiberflat models."""
    do_orders = ff_models[0].make_order_vec() if hasattr(ff_models[0], "make_order_vec") else None
    # Multiple rate files per fiber (for coadd) share the same illumination pattern; prim_args
    # only need the distinct fiber ids, not one model per fiber.
    do_fibers: list[str] = []
    seen: set[str] = set()
    for model in ff_models:
        for fiber in range(1, model.num_fibers + 1):
            val = getattr(model.meta, f"object_fiber{fiber}", None)
            if val is None or val == "BLOCKED":
                continue
            key = str(fiber)
            if key not in seen:
                seen.add(key)
                do_fibers.append(key)
    do_fibers = sorted(do_fibers, key=int)
    spectrograph = str(getattr(ff_models[0].meta, "spectrograph", "RSPEC") or "RSPEC").upper()
    if disp_range is None:
        disp_range = _DEFAULT_FIBERFLAT_DISP_RANGE.get(spectrograph, (500, 4000))
    if xdisp_range is None:
        xdisp_range = _DEFAULT_FIBERFLAT_XDISP_RANGE.get(spectrograph, (100, 4050))
    if prominence_factor is None:
        prominence_factor = _DEFAULT_FIBERFLAT_PROMINENCE
    return {
        "trace_pos": {
            "deg_pos": deg_pos,
            "deg_width": deg_width,
            "disp_range": disp_range,
            "xdisp_range": xdisp_range,
            "prominence_factor": prominence_factor,
        },
        "spec_extract": {
            "extraction_aperture": extraction_aperture,
            "do_orders": do_orders,
            "do_fibers": do_fibers,
        },
        "set_wavesol": {"run": False},
        "register_tracepos_cal": {"run": False},
        "register_traceprof_cal": {"run": False},
        "register_fiberflat_cal": {"run": False},
    }


def run_fiberflat_pipeline(
    fiberflat_filepaths: list[str],
    prim_args: dict | None = None,
    *,
    output_dir: str | None = None,
    disp_range: tuple[int, int] | None = None,
    xdisp_range: tuple[int, int] | None = None,
    deg_pos: int = 4,
    deg_width: int = 2,
    extraction_aperture: int = 10,
    prominence_factor: float | dict[str, float] | None = None,
) -> dict[str, datamodels.DataModel]:
    """
    Run the ProcessFiberflatsPipeline on a list of fiberflat rate files.

    Parameters
    ----------
    fiberflat_filepaths : list of str
        Paths to fiberflat rate FITS files. Multiple paths with the same fiber
        illumination pattern are coadded before trace finding.
    prim_args : dict or None, optional
        Primitive arguments. If None, built from the other keyword arguments.
        Default is None.
    output_dir : str or None, optional
        If set, pipeline products are written under this directory. Default is None.
    disp_range : tuple of int or None, optional
        Fitting range in spectral direction (pixels). Default depends on spectrograph.
    xdisp_range : tuple of int or None, optional
        Cross-dispersion (order) range in pixels. Default depends on spectrograph.
    deg_pos : int, optional
        Polynomial degree for trace position. Default is 4.
    deg_width : int, optional
        Polynomial degree for trace width. Default is 2.
    extraction_aperture : int, optional
        Extraction aperture in pixels. Default is 10.

    Returns
    -------
    dict
        Pipeline results with keys such as ``trace_pos``, ``trace_prof``.
    """
    if output_dir is not None:
        e2e_support.ensure_koa_calibration_cache_env(output_dir)
    logger.info("Running fiberflat pipeline on %d file(s).", len(fiberflat_filepaths))
    ff_models = prepare_fiberflat_models(fiberflat_filepaths)
    if prim_args is None:
        prim_args = get_fiberflat_prim_args(
            ff_models,
            disp_range=disp_range,
            xdisp_range=xdisp_range,
            deg_pos=deg_pos,
            deg_width=deg_width,
            extraction_aperture=extraction_aperture,
            prominence_factor=prominence_factor,
        )
    prim_args = e2e_support.apply_e2e_save_and_preview_flags(
        prim_args, ProcessFiberflatsPipeline.PRIMITIVES
    )
    pipe = ProcessFiberflatsPipeline(prim_args=prim_args)
    apply_kw: dict = {}
    if output_dir is not None:
        apply_kw["output_dir"] = e2e_support.ensure_output_directory(output_dir)
    return pipe.apply(ff_models, **apply_kw)
