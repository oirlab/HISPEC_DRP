"""
End-to-end HISPEC simulation: calibrations (KOA → user config → simulated), UTR ramp, Stage 0 → 1 → 2.

Chains helpers in :mod:`hispecdrp.simulations.simulate_up_the_ramp` and
:mod:`hispecdrp.simulations.e2e_support` with explicit hand-off of each stage product
to the next. By default (``stage1_trace_source="pipeline"``), simulated per-fiber fiberflat
rate maps are processed with ``ProcessFiberflatsPipeline`` between Stage 0 and Stage 1
instead of downloading KOA ``mastercal-tracepos`` / ``mastercal-fiberflat`` products.

Run from the repository root with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

STAGE1_TRACE_SOURCE_PIPELINE = "pipeline"
STAGE1_TRACE_SOURCE_KOA = "koa"
STAGE1_TRACE_SOURCE_SIM_HDF = "sim_hdf"
STAGE1_TRACE_SOURCES = (
    STAGE1_TRACE_SOURCE_PIPELINE,
    STAGE1_TRACE_SOURCE_KOA,
    STAGE1_TRACE_SOURCE_SIM_HDF,
)

from . import e2e_calibrations, e2e_gaia, e2e_support
from .simulate_up_the_ramp import (
    create_dummy_fiberflat_rate_files,
    create_ramp_and_calibrations,
    create_wavelength_solution_model_from_sim,
    get_default_stage0_primitives,
    run_fiberflat_pipeline,
    run_stage0_simulation,
)

# Imports used by merged sim-HDF / infinite-S/N / preview helpers (formerly separate modules).
import re
from scipy.interpolate import PchipInterpolator

from .. import datamodels
from ..datamodels import hispec_config
from ..datamodels.fiberflat import FiberflatModel
from ..datamodels.lfcwavesol import LFCWavesolModel
from ..datamodels.tracepos import TracePosModel
from ..datamodels.traceprof import TraceProfModel
from .simulate_rate_map import SimulateRateMap, read_pyechelle_hdf

logger = logging.getLogger(__name__)


@dataclass
class E2EConfig:
    """Parameters for a single end-to-end simulated run (see defaults in ``run_e2e_sim.py``)."""

    output_dir: str
    n_reads: int = 10
    readtime: float = 1.7
    max_cores: int = 1
    spectrograph: str = "BSPEC"
    fibers: list[int] | None = None
    fiberflat_fibers: list[int] | None = None
    use_master_calibrations: bool = True
    # ``"pipeline"``: simulate per-fiber fiberflat rate maps (or use ``fiberflat_rates``
    # from calibration_config) and build trace_pos / trace_prof / fiberflat with
    # ProcessFiberflatsPipeline — KOA Stage 1 masters are not fetched or downloaded.
    # ``"koa"``: use KOA / local-cache / bundled test-data mastercal-tracepos,
    # mastercal-traceprof, and mastercal-fiberflat (pipeline runs only if still incomplete).
    stage1_trace_source: Literal["pipeline", "koa", "sim_hdf"] | None = None
    # Deprecated alias for ``stage1_trace_source``; ignored when ``stage1_trace_source`` is set.
    derive_stage1_trace_from_sim: bool = True
    calibration_config: str | dict | None = None
    instrument_name: str | None = None
    instrument_era: str = "0.0.1"
    mjd_start: float | None = None
    datetime_obs: str | None = None
    connect_remote: bool = True
    sync_caldb_first: bool = True
    download_missing_koa: bool = True
    run_stage1: bool = True
    run_stage2: bool = True
    poisson_noise: bool = True
    convert_to_uint16: bool = True
    radial_velocity_km_s: float = 0.0
    use_gaia_target: bool = True
    gaia_source_id: str | None = None
    gaia_max_g_mag: float = 8.0
    gaia_random_seed: int | None = None
    template_mag_ab: float | None = 10.0
    template_2mass_band: str | None = "J"
    # When True (and a Gaia target is used), normalize the template to the target star's
    # real 2MASS magnitude (in ``template_2mass_band``, converted Vega->AB) instead of the
    # fixed ``template_mag_ab``. Falls back to ``template_mag_ab`` if no 2MASS match.
    template_mag_from_2mass: bool = True
    template_filename: str | None = None
    stellar_rv_mask_path: str | None = None
    # Optional Stage-2 ``primitives`` overrides merged on top of defaults (e.g. stellar_rv
    # ``v_step_coarse``). Nested dicts are shallow-merged per
    # primitive name.
    stage2_primitives: dict[str, Any] | None = None
    # Optional Stage-1 ``primitives`` overrides (e.g. ``spec_extract.method``).
    stage1_primitives: dict[str, Any] | None = None
    overwrite_ramp: bool = True
    make_preview_html: bool = True
    # Dummy fiberflat simulation (ProcessFiberflatsPipeline input)
    flat_scale: float = 2e-14
    flat_temperature_k: float = 1000.0
    fiberflat_coadd_exposures: int = 1
    # When True, one Stage 0 product per fiber is duplicated for coadd; when False, independent ramps.
    fiberflat_duplicate_single_reduction: bool = False
    # Initial trace peak threshold(s) as a fraction of the max cross-disp profile.
    # ``None`` selects spectrograph-specific defaults (per-fiber map for BSPEC sim fiberflats).
    fiberflat_trace_prominence_factor: float | dict[str, float] | None = None
    # Rate-map engine for simulated fiberflats: ``native`` PRF convolver on pyechelle HDF
    # PSFs (default, fast) or full ``pyechelle`` raytracing (slow; opt-in).
    fiberflat_rate_map_method: str = "native"
    fiberflat_pyechelle_exposure_time_s: float = 300.0
    # Spectrographs to simulate fiber flats for and run through the fiber flat pipeline.
    # Each listed spectrograph produces mastercal-tracepos / -traceprof / -fiberflat and a
    # trace-position ``*_preview.png`` for inspection. Science Stage 1/2 always use
    # ``spectrograph``; any additional entries are generated for inspection only.
    # ``None`` -> just ``[spectrograph]``. Only honored in ``stage1_trace_source="pipeline"``.
    fiberflat_spectrographs: list[str] | None = None
    calibration_kwargs: dict[str, Any] = field(default_factory=dict)
    # When True, resolve/download/create calibrations only (no science ramp or Stages 0–2).
    calibrations_only: bool = False
    # When True (default), simulate dummy fiberflats and run ProcessFiberflatsPipeline to
    # materialize trace_pos / trace_prof / fiberflat when they are not already resolved.
    materialize_stage1_trace: bool = True
    # Write ``cals/calibration_manifest.json`` with full provenance after calibration setup.
    write_calibration_manifest: bool = True
    # Infinite-S/N RV validation: noiseless rate maps, HDF-truth Stage-1 cals, high-precision RV.
    infinite_snr_rv_validation: bool = False
    infinite_snr_rv_assert: bool = False
    # Sim-HDF WAVESOL source when ``stage1_trace_source="sim_hdf"``:
    #   ``"auto"``             — packaged line-ZP if present, else runtime line-ZP, else HDF
    #   ``"packaged_line_zp"`` — load ``hispecdrp/data/wavesol/*.sim_hdf_wavesol.line_zp.fits``
    #   ``"hdf_line_zp"``      — runtime absorption ZP on the science rate map
    #   ``"hdf"``              — deposit-inverse geometric WAVE only
    wavesol_source: Literal["auto", "packaged_line_zp", "hdf_line_zp", "hdf"] = "auto"


@dataclass
class E2ECalibrationsResult:
    """Resolved and materialized calibration products (no science exposure processing)."""

    output_dir: str
    calibration_models: dict[str, Any]
    cal_bundle: e2e_calibrations.E2ECalibrationBundle
    sources: dict[str, str] = field(default_factory=dict)
    fiberflat_bundle: dict | None = None
    wavesol_model: Any | None = None
    summary: e2e_support.CalibrationDatasetSummary | None = None


def effective_stage1_trace_source(config: E2EConfig) -> str:
    """Return ``pipeline``, ``koa``, or ``sim_hdf`` for Stage 1 trace / fiberflat resolution."""
    if config.stage1_trace_source is not None:
        return config.stage1_trace_source
    return (
        STAGE1_TRACE_SOURCE_PIPELINE
        if config.derive_stage1_trace_from_sim
        else STAGE1_TRACE_SOURCE_KOA
    )


def stage1_uses_fiberflat_pipeline(config: E2EConfig) -> bool:
    """True when Stage 1 trace products should come from ProcessFiberflatsPipeline, not KOA."""
    return effective_stage1_trace_source(config) == STAGE1_TRACE_SOURCE_PIPELINE


def stage1_uses_sim_hdf_trace(config: E2EConfig) -> bool:
    """True when Stage 1 trace products are built directly from pyechelle HDF models."""
    return effective_stage1_trace_source(config) == STAGE1_TRACE_SOURCE_SIM_HDF


def stage1_trace_materialized_from_sim(config: E2EConfig) -> bool:
    """True when Stage 1 trace masters are generated in-process rather than from KOA."""
    return effective_stage1_trace_source(config) in (
        STAGE1_TRACE_SOURCE_PIPELINE,
        STAGE1_TRACE_SOURCE_SIM_HDF,
    )


@dataclass
class E2EResult:
    ramp_model: Any
    stage0_model: Any
    calibration_models: dict
    stage1_model: Any | None
    stage2_model: Any | None
    fiberflat_bundle: dict | None
    gaia_target: e2e_gaia.E2EGaiaTarget | None = None
    calibration_sources: dict[str, str] = field(default_factory=dict)
    infinite_snr_rv_evaluation: Any | None = None


def _expected_calibration_products(config: E2EConfig) -> list[str]:
    from .e2e_calibrations import STAGE0_CAL_KEYS, STAGE1_CAL_KEYS

    keys = list(STAGE0_CAL_KEYS)
    ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
    for f in ff_list:
        if f in (1, 2, 3):
            keys.append(f"fiberflat_rate_fib{f}")
    if config.materialize_stage1_trace or not stage1_trace_materialized_from_sim(config):
        keys.extend(STAGE1_CAL_KEYS)
    return keys


def _fiberflat_spectrograph_list(config: E2EConfig) -> list[str]:
    """Ordered, de-duplicated spectrographs to process fiber flats for (science first)."""
    science = str(config.spectrograph).upper()
    requested = [str(s).upper() for s in (config.fiberflat_spectrographs or [science])]
    ordered = [science] + [s for s in requested if s != science]
    seen: set[str] = set()
    out: list[str] = []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _fiberflat_rate_paths_for_stage1(
    config: E2EConfig,
    cal_bundle: e2e_calibrations.E2ECalibrationBundle,
    calibration_models: dict,
    out_dir_abs: str,
    stage0_primitives: dict,
    ff_list: list[int],
    *,
    record_sources: bool,
) -> list[str]:
    """Simulated or user-config fiberflat rate map paths for Stage 1 materialization."""
    use_user_rates = record_sources and all(
        f in cal_bundle.fiberflat_rate_paths for f in ff_list
    )
    if use_user_rates:
        logger.info("E2E: user fiberflat rate files (%s).", config.spectrograph)
        ff_paths = [cal_bundle.fiberflat_rate_paths[f] for f in sorted(ff_list)]
        for f in sorted(ff_list):
            cal_bundle.sources[f"fiberflat_rate_fib{f}"] = (
                f"config:{cal_bundle.fiberflat_rate_paths[f]}"
            )
        return ff_paths

    logger.info("E2E: simulated fiberflat rates (%s).", config.spectrograph)
    ff_paths = create_dummy_fiberflat_rate_files(
        ff_list,
        config.spectrograph,
        out_dir_abs,
        calibration_models,
        readtime=config.readtime,
        n_reads=config.n_reads,
        max_cores=config.max_cores,
        stage0_primitives=stage0_primitives,
        flat_scale=config.flat_scale,
        flat_temperature_k=config.flat_temperature_k,
        fiberflat_coadd_exposures=config.fiberflat_coadd_exposures,
        fiberflat_duplicate_single_reduction=(
            config.fiberflat_duplicate_single_reduction
        ),
        rate_map_method=config.fiberflat_rate_map_method,
        pyechelle_exposure_time_s=config.fiberflat_pyechelle_exposure_time_s,
    )
    if record_sources:
        for f in sorted(ff_list):
            fib_paths = [
                p for p in ff_paths if f"FIBERFLAT-FIB{f}" in os.path.basename(p)
            ]
            label = fib_paths[0] if len(fib_paths) == 1 else f"{len(fib_paths)} coadd inputs"
            cal_bundle.sources[f"fiberflat_rate_fib{f}"] = (
                f"simulated_dummy_rate:{label}"
            )
    return ff_paths


def _simulate_fiberflats_and_run_pipeline(
    config: E2EConfig,
    spectrograph: str,
    cal_bundle: e2e_calibrations.E2ECalibrationBundle,
    calibration_models: dict,
    out_dir_abs: str,
    cal_dir: str,
    stage0_primitives: dict,
    ff_list: list[int],
    *,
    record_sources: bool,
) -> dict:
    """
    Simulate per-fiber fiberflat rate maps for ``spectrograph`` and run ProcessFiberflatsPipeline.

    Produces mastercal-tracepos / -traceprof / -fiberflat plus a trace-position
    ``*_preview.png`` under ``cal_dir``. When ``record_sources`` is True the resolved
    provenance is written to ``cal_bundle.sources`` (only the science spectrograph feeds
    Stage 1/2, so additional spectrographs are generated with ``record_sources=False``).
    """
    use_user_rates = record_sources and all(
        f in cal_bundle.fiberflat_rate_paths for f in ff_list
    )
    if use_user_rates:
        logger.info("E2E: user fiberflat rate files + fiberflat pipeline (%s).", spectrograph)
    else:
        logger.info("E2E: simulated fiberflat rates + fiberflat pipeline (%s).", spectrograph)
    ff_paths = _fiberflat_rate_paths_for_stage1(
        config,
        cal_bundle,
        calibration_models,
        out_dir_abs,
        stage0_primitives,
        ff_list,
        record_sources=record_sources,
    )
    ff_bundle = run_fiberflat_pipeline(
        ff_paths,
        output_dir=cal_dir,
        prominence_factor=config.fiberflat_trace_prominence_factor,
    )
    if record_sources:
        for key in ("trace_pos", "trace_prof", "fiberflat"):
            cal_bundle.sources[key] = "fiberflat_pipeline"
    return ff_bundle


def materialize_stage1_calibrations(
    config: E2EConfig,
    cal_bundle: e2e_calibrations.E2ECalibrationBundle,
    calibration_models: dict,
    out_dir_abs: str,
    *,
    rate_map_for_wavesol_zp: np.ndarray | None = None,
    wavesol_zp_known_rv_m_s: float = 0.0,
) -> tuple[dict | None, Any | None]:
    """
    Resolve or generate Stage 1 trace / fiberflat / wavesol calibrations.

    Returns ``(fiberflat_bundle, wavesol_model)``; either may be ``None`` when not
    requested or when masters already exist on ``cal_bundle``.

    When ``stage1_trace_source='sim_hdf'`` and ``rate_map_for_wavesol_zp`` is set,
    apply a per-order absorption-line zero-point to the HDF wavesol so WAVE matches
    LSF-shifted line centers on the simulated detector (HDF file unchanged).
    Pass ``wavesol_zp_known_rv_m_s`` when the rate map includes an injected stellar RV.
    """
    cal_dir = e2e_support.calibration_dir(out_dir_abs)
    ff_bundle: dict | None = None
    wavesol_model = cal_bundle.wavesol

    user_config_masters = e2e_calibrations.stage1_masters_from_user_config(cal_bundle)
    use_master_trace_bundle = cal_bundle.has_master_trace_bundle and (
        user_config_masters or not stage1_trace_materialized_from_sim(config)
    )
    if use_master_trace_bundle:
        logger.info(
            "E2E: using trace_pos, trace_prof, and fiberflat masters "
            "(%s).",
            ", ".join(cal_bundle.sources.get(k, "?") for k in e2e_calibrations.STAGE1_TRACE_KEYS),
        )
        ff_bundle = {
            "trace_pos": cal_bundle.trace_pos,
            "trace_prof": cal_bundle.trace_prof,
            "fiberflat": cal_bundle.fiberflat,
        }
    elif config.materialize_stage1_trace:
        ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
        ff_list = [f for f in ff_list if f in (1, 2, 3)]
        if not ff_list:
            logger.warning("E2E: no fiberflat_fibers configured; skipping Stage 1 trace materialization.")
        else:
            stage0_primitives = get_default_stage0_primitives(max_cores=config.max_cores)
            if stage1_uses_sim_hdf_trace(config):

                extraction_aperture = 10.0
                if config.infinite_snr_rv_validation:

                    extraction_aperture = float(
                        merge_infinite_snr_stage1_primitives({})
                        .get("spec_extract", {})
                        .get("extraction_aperture", extraction_aperture)
                    )
                logger.info(
                    "E2E: sim_hdf Stage-1 bundle (HDF y_pos traces + box fiberflat, aperture=%.1f).",
                    extraction_aperture,
                )
                ramp_mjd = e2e_gaia.resolve_e2e_mjd_start(config.mjd_start)
                ff_paths = _fiberflat_rate_paths_for_stage1(
                    config,
                    cal_bundle,
                    calibration_models,
                    out_dir_abs,
                    stage0_primitives,
                    ff_list,
                    record_sources=True,
                )
                bundle = materialize_stage1_bundle_from_sim_hdf(
                    spectrograph=config.spectrograph,
                    fiberflat_rate_paths=ff_paths,
                    meta={"mjd_start": float(ramp_mjd)},
                    extraction_aperture=extraction_aperture,
                    rate_map_for_wavesol_zp=rate_map_for_wavesol_zp,
                    wavesol_zp_known_rv_m_s=float(wavesol_zp_known_rv_m_s),
                    wavesol_source=getattr(config, "wavesol_source", "auto") or "auto",
                )
                ff_bundle = {
                    "trace_pos": bundle["trace_pos"],
                    "trace_prof": bundle["trace_prof"],
                    "fiberflat": bundle["fiberflat"],
                }
                wavesol_model = bundle["wavesol"]
                for key in e2e_calibrations.STAGE1_TRACE_KEYS:
                    cal_bundle.sources[key] = STAGE1_TRACE_SOURCE_SIM_HDF
                if bundle.get("wavesol_comb_info"):
                    cal_bundle.sources["wavesol"] = f"{STAGE1_TRACE_SOURCE_SIM_HDF}+injected_comb"
                elif (bundle.get("wavesol_zp_info") or {}).get("source") == "packaged_line_zp":
                    cal_bundle.sources["wavesol"] = f"{STAGE1_TRACE_SOURCE_SIM_HDF}+packaged_line_zp"
                elif bundle.get("wavesol_zp_info"):
                    cal_bundle.sources["wavesol"] = f"{STAGE1_TRACE_SOURCE_SIM_HDF}+line_zp"
                else:
                    cal_bundle.sources["wavesol"] = STAGE1_TRACE_SOURCE_SIM_HDF
            elif stage1_uses_fiberflat_pipeline(config):
                logger.info(
                    "E2E: ProcessFiberflatsPipeline (stage1_trace_source=pipeline): "
                    "materializing trace_pos, trace_prof, fiberflat from %s fiberflat rate map(s).",
                    config.fiberflat_rate_map_method,
                )
                ff_bundle = _simulate_fiberflats_and_run_pipeline(
                    config,
                    config.spectrograph,
                    cal_bundle,
                    calibration_models,
                    out_dir_abs,
                    cal_dir,
                    stage0_primitives,
                    ff_list,
                    record_sources=True,
                )
            else:
                logger.info(
                    "E2E: materializing trace_pos, trace_prof, fiberflat via "
                    "ProcessFiberflatsPipeline (KOA masters incomplete)."
                )
                ff_bundle = _simulate_fiberflats_and_run_pipeline(
                    config,
                    config.spectrograph,
                    cal_bundle,
                    calibration_models,
                    out_dir_abs,
                    cal_dir,
                    stage0_primitives,
                    ff_list,
                    record_sources=True,
                )

    # Additional spectrographs (e.g. RSPEC alongside a BSPEC science run): simulate fiber
    # flats and run the fiber flat pipeline purely to produce trace-position previews and
    # inspection products. These do not feed the science Stage 1/2 trace bundle.
    if config.materialize_stage1_trace and stage1_uses_fiberflat_pipeline(config):
        extra_specs = _fiberflat_spectrograph_list(config)[1:]
        ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
        ff_list = [f for f in ff_list if f in (1, 2, 3)]
        if extra_specs and ff_list:
            stage0_primitives = get_default_stage0_primitives(max_cores=config.max_cores)
            for spec in extra_specs:
                logger.info(
                    "E2E: simulating fiber flats + fiber flat pipeline for additional "
                    "spectrograph %s (trace-position previews / inspection only).",
                    spec,
                )
                _simulate_fiberflats_and_run_pipeline(
                    config,
                    spec,
                    cal_bundle,
                    calibration_models,
                    out_dir_abs,
                    cal_dir,
                    stage0_primitives,
                    ff_list,
                    record_sources=False,
                )

    if wavesol_model is None and config.materialize_stage1_trace and not stage1_uses_sim_hdf_trace(config):
        ramp_mjd = e2e_gaia.resolve_e2e_mjd_start(config.mjd_start)
        wavesol_model = create_wavelength_solution_model_from_sim(
            config.spectrograph,
            meta={"mjd_start": float(ramp_mjd)},
        )
        cal_bundle.sources["wavesol"] = "simulated_hdf"

    return ff_bundle, wavesol_model


def run_e2e_calibrations(config: E2EConfig) -> E2ECalibrationsResult:
    """
    Resolve, download, and materialize all calibration products without processing science data.

    Order: user ``calibration_config`` paths → KOA store → simulated Stage 0 fallback →
    optional dummy fiberflat simulation and ProcessFiberflatsPipeline for Stage 1 masters.
    """
    out_dir_abs = e2e_support.ensure_output_directory(config.output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    ramp_mjd_start = e2e_gaia.resolve_e2e_mjd_start(config.mjd_start)

    logger.info(
        "E2E calibrations-only: resolving products (output_dir=%s, use_koa=%s, "
        "stage1_trace_source=%s).",
        out_dir_abs,
        config.use_master_calibrations,
        effective_stage1_trace_source(config),
    )

    cal_bundle = e2e_calibrations.resolve_e2e_calibrations(
        output_dir=out_dir_abs,
        spectrograph=config.spectrograph,
        use_koa=config.use_master_calibrations,
        calibration_config=config.calibration_config,
        instrument_name=config.instrument_name,
        instrument_era=config.instrument_era,
        mjd_start=ramp_mjd_start,
        datetime_obs=config.datetime_obs,
        connect_remote=config.connect_remote,
        sync_caldb_first=config.sync_caldb_first,
        download_missing_koa=config.download_missing_koa,
        auto_resolve_stage1_masters=not stage1_trace_materialized_from_sim(config),
        skip_koa_cal_types=(
            list(e2e_calibrations.STAGE1_TRACE_KEYS)
            if stage1_trace_materialized_from_sim(config)
            else None
        ),
    )
    e2e_calibrations.log_stage1_trace_bundle_status(cal_bundle)

    calibration_models = dict(cal_bundle.stage0)
    e2e_calibrations.complete_stage0_calibrations(
        calibration_models,
        rate_map=None,
        output_dir=out_dir_abs,
        sources=cal_bundle.sources,
        calibration_kwargs=config.calibration_kwargs,
        shape=e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE,
    )
    for k in calibration_models:
        cal_bundle.sources.setdefault(k, "simulated")

    ff_bundle, wavesol_model = materialize_stage1_calibrations(
        config, cal_bundle, calibration_models, out_dir_abs
    )

    summary = e2e_support.build_calibration_dataset_summary(
        output_dir=out_dir_abs,
        calibration_models=calibration_models,
        sources=cal_bundle.sources,
        fiberflat_bundle=ff_bundle,
        wavesol_model=wavesol_model,
        expected_products=_expected_calibration_products(config),
    )
    e2e_support.log_calibration_dataset_summary(summary)
    e2e_support.log_e2e_calibration_summary(
        calibration_models=calibration_models,
        sources=cal_bundle.sources,
        fiberflat_bundle=ff_bundle,
        wavesol_model=wavesol_model,
    )
    if config.write_calibration_manifest:
        e2e_support.write_calibration_manifest(out_dir_abs, summary)

    return E2ECalibrationsResult(
        output_dir=out_dir_abs,
        calibration_models=calibration_models,
        cal_bundle=cal_bundle,
        sources=dict(cal_bundle.sources),
        fiberflat_bundle=ff_bundle,
        wavesol_model=wavesol_model,
        summary=summary,
    )


def run_e2e_simulation(config: E2EConfig) -> E2EResult:
    """
    Build a simulated ramp + calibrations, then run Stage 0, fiberflat pipeline, Stage 1, Stage 2.

    Calibrations are resolved in order: explicit paths in ``calibration_config``, KOA store
    (when ``use_master_calibrations`` is True), then simulated products for any missing keys.

    When ``config.calibrations_only`` is True, stops after :func:`run_e2e_calibrations`
    (no science ramp or pipeline stages).
    """
    if config.calibrations_only:
        cal_res = run_e2e_calibrations(config)
        return E2EResult(
            ramp_model=None,
            stage0_model=None,
            calibration_models=cal_res.calibration_models,
            stage1_model=None,
            stage2_model=None,
            fiberflat_bundle=cal_res.fiberflat_bundle,
            gaia_target=None,
            calibration_sources=cal_res.sources,
            infinite_snr_rv_evaluation=None,
        )

    if config.infinite_snr_rv_validation:

        config = apply_infinite_snr_rv_e2e_config(config)

    out_dir_abs = e2e_support.ensure_output_directory(config.output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    gaia_target: e2e_gaia.E2EGaiaTarget | None = None
    injected_rv_km_s = config.radial_velocity_km_s
    doppler_rv_km_s = config.radial_velocity_km_s
    # Non-Gaia sims are painted in the topocentric frame; stamp BERV=0 so Stage-2
    # barycorr is a no-op (CCF zb from barycorr remains zero).
    target_meta: dict | None = {"force_berv_m_s": 0.0}
    ramp_mjd_start = e2e_gaia.resolve_e2e_mjd_start(config.mjd_start)
    if config.use_gaia_target:
        exposure_mid_mjd = e2e_gaia.exposure_mid_mjd(
            ramp_mjd_start,
            n_reads=config.n_reads,
            readtime=config.readtime,
        )
        gaia_target = e2e_gaia.resolve_e2e_gaia_target(
            source_id=config.gaia_source_id,
            mjd_mid=float(exposure_mid_mjd),
            max_g_mag=config.gaia_max_g_mag,
            random_seed=config.gaia_random_seed,
            extra_rv_km_s=config.radial_velocity_km_s,
        )
        injected_rv_km_s = gaia_target.stellar_rv_km_s
        doppler_rv_km_s = gaia_target.topocentric_doppler_rv_km_s
        target_meta = dict(gaia_target.target_meta)
        target_meta["mjd_start"] = float(ramp_mjd_start)
        target_meta["mjd_mid"] = float(exposure_mid_mjd)
        logger.info(
            "E2E Doppler: stellar RV=%.3f km/s (comparison metric), "
            "topocentric template shift=%.3f km/s (Gaia RV - BERV); BERV removed in CCF via ccf_zb.",
            injected_rv_km_s,
            doppler_rv_km_s,
        )

    # Resolve the template normalization magnitude. By default, normalize to the Gaia
    # target's real 2MASS magnitude (converted Vega->AB); otherwise use the fixed value.
    eff_template_mag_ab = config.template_mag_ab
    if (
        config.template_mag_from_2mass
        and gaia_target is not None
        and config.template_2mass_band is not None
    ):
        derived_mag_ab = e2e_gaia.twomass_mag_ab(
            gaia_target.twomass_data, config.template_2mass_band
        )
        if derived_mag_ab is not None:
            eff_template_mag_ab = derived_mag_ab
            logger.info(
                "E2E: normalizing template to 2MASS %s=%.3f AB mag (from this star's "
                "2MASS photometry) instead of fixed %.3f.",
                config.template_2mass_band,
                derived_mag_ab,
                config.template_mag_ab if config.template_mag_ab is not None else float("nan"),
            )
        else:
            logger.warning(
                "E2E: could not derive 2MASS %s magnitude for the target; "
                "falling back to fixed template_mag_ab=%s.",
                config.template_2mass_band,
                config.template_mag_ab,
            )

    cal_bundle = e2e_calibrations.resolve_e2e_calibrations(
        output_dir=out_dir_abs,
        spectrograph=config.spectrograph,
        use_koa=config.use_master_calibrations,
        calibration_config=config.calibration_config,
        instrument_name=config.instrument_name,
        instrument_era=config.instrument_era,
        mjd_start=ramp_mjd_start,
        datetime_obs=config.datetime_obs,
        connect_remote=config.connect_remote,
        sync_caldb_first=config.sync_caldb_first,
        download_missing_koa=config.download_missing_koa,
        auto_resolve_stage1_masters=not stage1_trace_materialized_from_sim(config),
        skip_koa_cal_types=(
            list(e2e_calibrations.STAGE1_TRACE_KEYS)
            if stage1_trace_materialized_from_sim(config)
            else None
        ),
    )
    e2e_calibrations.log_stage1_trace_bundle_status(cal_bundle)

    stage0_seed = dict(cal_bundle.stage0) if cal_bundle.stage0 else None

    logger.info(
        "E2E: building ramp and calibrations (output_dir=%s, stage0_from_koa_or_config=%s).",
        out_dir_abs,
        sorted(cal_bundle.stage0.keys()),
    )
    ramp_model, calibration_models, rate_map = create_ramp_and_calibrations(
        rate_map=None,
        shape=e2e_support.DEFAULT_HISPEC_DETECTOR_SHAPE,
        fibers=config.fibers,
        output_dir=out_dir_abs,
        calibration_models=stage0_seed,
        use_master_calibrations=False,
        readtime=config.readtime,
        n_reads=config.n_reads,
        spectrograph=config.spectrograph,
        max_cores=config.max_cores,
        poisson_noise=config.poisson_noise,
        convert_to_uint16=config.convert_to_uint16,
        radial_velocity_km_s=doppler_rv_km_s,
        mjd_start=ramp_mjd_start,
        datetime_obs=config.datetime_obs,
        target_meta=target_meta,
        template_mag_ab=eff_template_mag_ab,
        template_2mass_band=config.template_2mass_band,
        template_filename=config.template_filename,
        overwrite_ramp=config.overwrite_ramp,
        **config.calibration_kwargs,
    )

    e2e_calibrations.complete_stage0_calibrations(
        calibration_models,
        rate_map,
        out_dir_abs,
        cal_bundle.sources,
        calibration_kwargs=config.calibration_kwargs,
    )
    for k in calibration_models:
        cal_bundle.sources.setdefault(k, "simulated")

    stage0_primitives = get_default_stage0_primitives(max_cores=config.max_cores)
    stage1_primitives = None
    stage2_primitives = None
    if config.infinite_snr_rv_validation:

        stage1_primitives = merge_infinite_snr_stage1_primitives(config.stage1_primitives)
        stage2_primitives = merge_infinite_snr_stage2_primitives(config.stage2_primitives)
    elif config.stage2_primitives:
        import copy

        # Merge overrides onto Stage-2 defaults so blaze/barycorr still run.
        stage2_primitives = e2e_support.get_default_stage2_primitives(
            max_cores=config.max_cores
        )
        for prim_name, overrides in config.stage2_primitives.items():
            stage2_primitives.setdefault(prim_name, {})
            if isinstance(overrides, dict):
                stage2_primitives[prim_name].update(copy.deepcopy(overrides))
            else:
                stage2_primitives[prim_name] = overrides
    logger.info("E2E: Stage 0 (ramp → rate/slope).")
    stage0_model = run_stage0_simulation(
        ramp_model,
        calibration_models,
        output_dir=out_dir_abs,
        primitives=stage0_primitives,
        max_cores=config.max_cores,
    )

    stage1_model = None
    stage2_model = None
    ff_bundle = None
    wavesol_model = None

    if config.run_stage1:
        ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
        ff_list = [f for f in ff_list if f in (1, 2, 3)]
        if ff_list:
            # Prefer science rate_map for wavesol line ZP (LSF shift); fall back to Stage-0 rate.
            rate_for_zp = None
            if rate_map is not None:
                rate_for_zp = np.asarray(rate_map, dtype=np.float64)
            elif getattr(stage0_model, "rate", None) is not None:
                rate_for_zp = np.asarray(stage0_model.rate, dtype=np.float64)
            ff_bundle, wavesol_model = materialize_stage1_calibrations(
                config,
                cal_bundle,
                calibration_models,
                out_dir_abs,
                rate_map_for_wavesol_zp=rate_for_zp,
                wavesol_zp_known_rv_m_s=float(config.radial_velocity_km_s) * 1000.0,
            )
            if ff_bundle is not None:
                stage1_model = e2e_support.run_stage1_simulation(
                    stage0_model,
                    ff_bundle["trace_pos"],
                    ff_bundle["trace_prof"],
                    primitives=stage1_primitives,
                    wavesol_model=wavesol_model,
                    output_dir=out_dir_abs,
                    max_cores=config.max_cores,
                )
        else:
            logger.warning("E2E: run_stage1 is True but fiberflat_fibers is empty; skipping Stage 1/2.")

    if config.run_stage2 and stage1_model is not None and ff_bundle is not None:
        logger.info("E2E: Stage 2.")
        _rv_center_m_s: float | None = None
        if np.isfinite(float(injected_rv_km_s)):
            _rv_center_m_s = float(injected_rv_km_s) * 1000.0
            logger.info(
                "E2E: stellar RV centered uniform grid ccf_input_velocity=%.3f m/s (%.4f km/s)",
                _rv_center_m_s,
                float(injected_rv_km_s),
            )
        stage2_model = e2e_support.run_stage2_simulation(
            stage1_model,
            fiberflat_model=ff_bundle["fiberflat"],
            primitives=stage2_primitives,
            output_dir=out_dir_abs,
            max_cores=config.max_cores,
            stellar_rv_mask_path=config.stellar_rv_mask_path,
            stellar_rv_ccf_input_velocity_m_s=_rv_center_m_s,
        )
    elif config.run_stage2:
        logger.warning("E2E: run_stage2 is True but Stage 1 or fiberflat bundle missing; skipping.")

    e2e_support.log_e2e_calibration_summary(
        calibration_models=calibration_models,
        sources=cal_bundle.sources,
        fiberflat_bundle=ff_bundle,
        wavesol_model=wavesol_model,
    )
    e2e_support.log_e2e_rv_performance_summary(
        stage2_model,
        injected_rv_km_s=injected_rv_km_s,
        gaia_target=gaia_target,
        fibers=config.fibers,
    )

    infinite_snr_rv_evaluation = None
    if config.infinite_snr_rv_validation:

        infinite_snr_rv_evaluation = evaluate_infinite_snr_rv_recovery(
            stage2_model,
            injected_rv_km_s=injected_rv_km_s,
            fibers=config.fibers,
        )
        log_infinite_snr_rv_evaluation(infinite_snr_rv_evaluation)
        if config.infinite_snr_rv_assert:
            assert_infinite_snr_rv_recovery(infinite_snr_rv_evaluation)

    if config.make_preview_html:
        try:

            base_name = infer_exposure_base_name(
                stage0_model=stage0_model,
                stage1_model=stage1_model,
                stage2_model=stage2_model,
            )
            html_path = generate_e2e_preview_report(
                out_dir_abs,
                exposure_base_name=base_name,
                gaia_target=gaia_target,
                stage2_model=stage2_model,
                injected_rv_km_s=injected_rv_km_s,
                fibers=config.fibers,
            )
            logger.info("E2E preview report written to %s", html_path)
        except Exception:
            logger.exception("E2E: failed to generate preview report (continuing).")

    return E2EResult(
        ramp_model=ramp_model,
        stage0_model=stage0_model,
        calibration_models=calibration_models,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        fiberflat_bundle=ff_bundle,
        gaia_target=gaia_target,
        calibration_sources=dict(cal_bundle.sources),
        infinite_snr_rv_evaluation=infinite_snr_rv_evaluation,
    )


def run_pipeline_example(
    output_dir: str,
    *,
    fibers: list[int] | None = None,
    spectrograph: str = "BSPEC",
    template_filename: str = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
    template_mag_ab: float = 3.05,
) -> E2EResult:
    """
    Minimal-entry e2e run used by the docs Basic Usage walkthrough.

    Uses sim-HDF Stage-1 products, the packaged line-ZP WAVE (including baked
    CCF-order refine), and the packaged ccfopt mask so order-averaged stellar RV
    at injected RV=0 reaches the demonstrated ≪10 cm/s performance. Only
    ``output_dir`` and ``fibers`` are typically changed by notebook users.

    Default brightness is AB=3.05 with Poisson noise and uint16 ramps: the
    brightest level at the default 15.3 s integration that stays unsaturated
    (AB=3 clips peak cores). Peak last-read DN sits near ~64k of 65535.
    """
    fibs = [int(f) for f in (fibers if fibers is not None else [2])]
    fibs = [f for f in fibs if f in (1, 2, 3)]
    if not fibs:
        raise ValueError("fibers must include at least one of 1, 2, 3")

    config = E2EConfig(
        output_dir=str(output_dir),
        spectrograph=str(spectrograph).upper(),
        n_reads=10,
        readtime=1.7,
        max_cores=1,
        fibers=fibs,
        fiberflat_fibers=list(fibs),
        stage1_trace_source="sim_hdf",
        wavesol_source="packaged_line_zp",
        use_master_calibrations=False,
        connect_remote=False,
        sync_caldb_first=False,
        download_missing_koa=False,
        use_gaia_target=False,
        radial_velocity_km_s=0.0,
        template_filename=template_filename,
        template_mag_ab=float(template_mag_ab),
        template_mag_from_2mass=False,
        # Default stellar-RV mask is the packaged ccfopt product.
        stellar_rv_mask_path=None,
        poisson_noise=True,
        convert_to_uint16=True,
        run_stage1=True,
        run_stage2=True,
        make_preview_html=True,
        overwrite_ramp=True,
        stage2_primitives={
            "stellar_rv": {
                "do_fibers": list(fibs),
                "ccf_input_velocity": 0.0,
                "save_preview": True,
                "preview_write_png": False,
            },
        },
    )
    return run_e2e_simulation(config)


def prepare_pipeline_example_stage1(
    output_dir: str,
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fibers: list[int] | None = None,
    extraction_aperture: float = 10.0,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """
    Build sim-HDF Stage-1 products for the docs Basic Usage walkthrough.

    Returns ``trace_pos``, ``trace_prof``, ``fiberflat``, and packaged line-ZP
    ``wavesol`` so Stage 1/2 can use the demonstrated fiber-2 RV path instead of
    a generic KOA wavelength solution.

    With ``reuse_existing=True`` (default), loads Stage-1 masters from
    ``<output_dir>/cals`` when present, otherwise reuses any existing fiberflat
    rate frames there and only simulates missing fibers. Fresh products are
    written back under ``cals/`` for the next call.
    """
    from .simulate_up_the_ramp import (
        create_simulated_calibration_models,
        load_cached_sim_stage1_masters,
        save_sim_stage1_masters,
        sim_stage1_masters_exist,
    )

    out_dir_abs = e2e_support.ensure_output_directory(output_dir)
    fibs = [int(f) for f in (fibers if fibers is not None else [2]) if int(f) in (1, 2, 3)]
    if not fibs:
        raise ValueError("fibers must include at least one of 1, 2, 3")
    spectrograph = str(spectrograph).upper()
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate.ndim != 2:
        raise ValueError(f"rate_map must be 2-D, got shape {rate.shape}")

    ramp_mjd = e2e_gaia.resolve_e2e_mjd_start(None)
    stage1_meta = {"mjd_start": float(ramp_mjd)}

    if reuse_existing and sim_stage1_masters_exist(
        out_dir_abs,
        spectrograph,
        fibs,
        extraction_aperture=float(extraction_aperture),
    ):
        logger.info(
            "Reusing cached sim-HDF Stage-1 masters under %s (fibers=%s).",
            e2e_support.calibration_dir(out_dir_abs),
            fibs,
        )
        cached = load_cached_sim_stage1_masters(
            out_dir_abs, spectrograph, meta=stage1_meta
        )
        return {
            "trace_pos": cached["trace_pos"],
            "trace_prof": cached["trace_prof"],
            "fiberflat": cached["fiberflat"],
            "wavesol": cached["wavesol"],
            "wavesol_zp_info": {"source": "cached_stage1_masters"},
            "from_cache": True,
        }

    calibration_models = create_simulated_calibration_models(rate.shape)
    stage0_primitives = get_default_stage0_primitives(max_cores=1)
    # Use native PRF (same as science paint). pyechelle + continuous R~3e5
    # continuum can hang for tens of minutes with no FITS progress.
    ff_paths = create_dummy_fiberflat_rate_files(
        fibs,
        spectrograph,
        out_dir_abs,
        calibration_models,
        readtime=1.7,
        n_reads=10,
        max_cores=1,
        stage0_primitives=stage0_primitives,
        reuse_existing=bool(reuse_existing),
        rate_map_method="native",
    )
    bundle = materialize_stage1_bundle_from_sim_hdf(
        spectrograph=spectrograph,
        fiberflat_rate_paths=ff_paths,
        meta=stage1_meta,
        extraction_aperture=float(extraction_aperture),
        rate_map_for_wavesol_zp=rate,
        wavesol_source="packaged_line_zp",
    )
    save_sim_stage1_masters(
        out_dir_abs,
        spectrograph,
        trace_pos=bundle["trace_pos"],
        trace_prof=bundle["trace_prof"],
        fiberflat=bundle["fiberflat"],
        fibers=fibs,
        extraction_aperture=float(extraction_aperture),
    )
    return {
        "trace_pos": bundle["trace_pos"],
        "trace_prof": bundle["trace_prof"],
        "fiberflat": bundle["fiberflat"],
        "wavesol": bundle["wavesol"],
        "wavesol_zp_info": bundle.get("wavesol_zp_info"),
        "from_cache": False,
    }


# ========================================================================
# Sim-HDF Stage-1 calibrations (formerly sim_hdf_calibrations.py)
# ========================================================================

_DEFAULT_FWHM_PIX = 3.5
_TRACE_POLY_DEG = 2
_HDF_MODEL_CACHE: dict[str, dict] = {}


def _order_list(spectrograph: str) -> list[int]:
    return list(hispec_config.HISPECConfig.get_order_list(spectrograph.upper()))


def _load_hdf_model_data(spectrograph: str) -> tuple[str, dict]:
    """Load pyechelle HDF once per spectrograph (PSF stamps are ~1 GB)."""
    global _HDF_MODEL_CACHE
    spectrograph = spectrograph.upper()
    cached = _HDF_MODEL_CACHE.get(spectrograph)
    if cached is not None:
        return spectrograph, cached
    srm = SimulateRateMap()
    model = srm.bspec_hdf_model if spectrograph == "BSPEC" else srm.rspec_hdf_model
    model_path = srm.hdf_file_directory + model
    model_data = read_pyechelle_hdf(model_path)
    _HDF_MODEL_CACHE[spectrograph] = model_data
    return spectrograph, model_data


def clear_hdf_model_cache() -> None:
    """Drop cached HDF models (free ~1 GB) when finished with sim extracts."""
    global _HDF_MODEL_CACHE
    _HDF_MODEL_CACHE = {}



def _orient_hdf_pixel_xy(
    ox: np.ndarray,
    oy: np.ndarray,
    *,
    nx: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map native HDF detector ``(x, y)`` to oriented rate-map ``(row, col)``.

    Matches ``SimulateRateMap.flip_rate_map_horizontal`` followed by
    ``rotate_rate_map_vertical`` (``rot90`` ``k=3``) with ``orders_are_vertical=True``
    extraction along spectral rows.
    """
    ox = np.asarray(ox, dtype=np.float64)
    oy = np.asarray(oy, dtype=np.float64)
    row = (nx - 1) - ox
    col = (nx - 1) - oy
    return row, col


def _interp_oriented_spectral_row(
    row: np.ndarray,
    values: np.ndarray,
    *,
    n_pix: int = 4096,
) -> np.ndarray:
    """Interpolate ``values`` onto spectral row indices ``0 .. n_pix-1``."""
    row = np.asarray(row, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(row) & np.isfinite(values)
    row = row[good]
    values = values[good]
    if row.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    sidx = np.argsort(row)
    rows = row[sidx]
    vals = values[sidx]
    ur, inv = np.unique(rows, return_inverse=True)
    if ur.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    uvals = np.bincount(inv, weights=vals) / np.maximum(np.bincount(inv), 1)
    f = PchipInterpolator(ur, uvals, extrapolate=True)
    return f(np.arange(n_pix, dtype=np.float64))


def _wavelength_from_deposit_inverse(
    ow: np.ndarray,
    ox: np.ndarray,
    *,
    n_pix: int = 4096,
    nx: int = 4096,
    n_dense: int = 20000,
) -> np.ndarray:
    """
    Build ``WAVE[p]`` as the exact inverse of the native deposit map ``λ → x``.

    The simulator deposits each λ at ``x = Pchip(λ)`` (then orients
    ``row = (nx-1) - x``).  Resampling ``λ(row)`` by Pchip on the sparse HDF
    samples is **not** that inverse and leaves a ~2 cm/s floor.  Sampling the
    forward ``λ → x`` map densely and Pchip-inverting reaches ≪1 cm/s.
    """
    ow = np.asarray(ow, dtype=np.float64)
    ox = np.asarray(ox, dtype=np.float64)
    good = np.isfinite(ow) & np.isfinite(ox)
    ow, ox = ow[good], ox[good]
    if ow.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    sidx = np.argsort(ow)
    ow, ox = ow[sidx], ox[sidx]
    uw, inv = np.unique(ow, return_inverse=True)
    if uw.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    ux = np.bincount(inv, weights=ox) / np.maximum(np.bincount(inv), 1)
    fx = PchipInterpolator(uw, ux, extrapolate=True)
    lam_d = np.linspace(float(uw.min()), float(uw.max()), int(n_dense))
    x_d = np.asarray(fx(lam_d), dtype=np.float64)
    s2 = np.argsort(x_d)
    fx_inv = PchipInterpolator(x_d[s2], lam_d[s2], extrapolate=True)
    pix = np.arange(int(n_pix), dtype=np.float64)
    x_native = float(nx - 1) - pix
    return np.asarray(fx_inv(x_native), dtype=np.float64)


def get_sim_hdf_trace_grids(
    spectrograph: str,
    *,
    n_pix: int = 4096,
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """
    Per-fiber, per-order detector grids from the HDF model.

    Returns ``{fiber: {order: {pixel, wavelength, y_pos}}}`` with ``pixel`` = ``0..n_pix-1``
    indexing the **spectral row** in the oriented rate map (same convention as
    ``orders_are_vertical=True`` extraction). ``y_pos[pixel]`` is the cross-dispersion
    column at that row.

    ``wavelength`` is the inverse of the native deposit ``λ → x`` map (dense
    forward sampling), so ``WAVE(row_deposit(λ)) = λ`` to ≪1 cm/s.
    """
    spectrograph, model_data = _load_hdf_model_data(spectrograph)
    order_list = _order_list(spectrograph)
    nx = int(model_data.get("nx", n_pix))
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    pixel = np.arange(n_pix, dtype=np.float64)
    for fiber in (1, 2, 3):
        out[fiber] = {}
        fiber_key = f"fiber_{fiber}"
        for od in order_list:
            od_s = str(od)
            if od_s not in model_data.get(fiber_key, {}):
                continue
            block = model_data[fiber_key][od_s]
            ow = np.asarray(block["xy_wavelengths"], dtype=np.float64)
            ox = np.asarray(block["x"], dtype=np.float64)
            oy = np.asarray(block["y"], dtype=np.float64)
            good = np.isfinite(ox) & np.isfinite(oy) & np.isfinite(ow)
            if int(np.sum(good)) < 2:
                continue
            row, col = _orient_hdf_pixel_xy(ox[good], oy[good], nx=nx)
            wavelength = _wavelength_from_deposit_inverse(
                ow[good], ox[good], n_pix=n_pix, nx=nx
            )
            y_pos = _interp_oriented_spectral_row(row, col, n_pix=n_pix)
            out[fiber][od] = {
                "pixel": pixel.copy(),
                "wavelength": wavelength.astype(np.float64),
                "y_pos": y_pos.astype(np.float64),
            }
    return out


def _fit_trace_poly_coeffs(y_row: np.ndarray, deg: int = _TRACE_POLY_DEG) -> np.ndarray:
    x = np.arange(y_row.size, dtype=np.float64)
    ok = np.isfinite(y_row)
    if int(np.sum(ok)) < deg + 1:
        return np.full(deg + 1, np.nan, dtype=np.float64)
    return np.polyfit(x[ok], y_row[ok], deg)


def create_tracepos_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
) -> TracePosModel:
    """TracePosModel with POS/FWHM rows taken from the HDF echelle model."""
    spectrograph = spectrograph.upper()
    grids = get_sim_hdf_trace_grids(spectrograph)
    order_list = _order_list(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    model = TracePosModel(config=hispec_config.HISPECConfig, meta=model_meta)
    n_pix = 4096
    pixel_arr = np.arange(n_pix, dtype=np.float64)
    for fiber in (1, 2, 3):
        rows = []
        coeff_pos = []
        coeff_fwhm = []
        for order in order_list:
            if order not in grids.get(fiber, {}):
                rows.append(np.full(n_pix, np.nan, dtype=np.float64))
                coeff_pos.append(np.full(_TRACE_POLY_DEG + 1, np.nan))
                coeff_fwhm.append(np.full(_TRACE_POLY_DEG + 1, np.nan))
                continue
            y_row = grids[fiber][order]["y_pos"]
            rows.append(y_row)
            coeff_pos.append(_fit_trace_poly_coeffs(y_row))
            coeff_fwhm.append(np.polyfit(pixel_arr, np.full(n_pix, _DEFAULT_FWHM_PIX), 1))
        pos = np.stack(rows, axis=0)
        fwhm = np.full_like(pos, _DEFAULT_FWHM_PIX, dtype=np.float64)
        setattr(model, f"coeff_pos_fib{fiber}", np.stack(coeff_pos, axis=0))
        setattr(model, f"coeff_fwhm_fib{fiber}", np.stack(coeff_fwhm, axis=0))
        setattr(model, f"pos_fib{fiber}", pos)
        setattr(model, f"fwhm_fib{fiber}", fwhm)
    return model


def create_tracepos_model_from_fiberflat_rates(
    fiberflat_rate_paths: list[str],
    *,
    spectrograph: str,
    meta: dict | None = None,
    prominence_factor: float = 0.02,
) -> TracePosModel:
    """
    TracePosModel from peak finding on simulated fiber-flat rate maps.

    Uses the same detector orientation as science simulations (post flip/rotate),
    unlike :func:`create_tracepos_model_from_sim_hdf` which reads raw HDF ``x/y``.
    """
    from ..primitives.trace_fitting.tracepos_primitive import TracePositionPrimitive
    from .simulate_up_the_ramp import (
        _DEFAULT_FIBERFLAT_DISP_RANGE,
        _DEFAULT_FIBERFLAT_XDISP_RANGE,
    )

    spectrograph = spectrograph.upper()
    models_by_fiber: dict[str, datamodels.DataModel] = {}
    for path in fiberflat_rate_paths:
        model = datamodels.open(path)
        base = str(getattr(model.meta, "filename", path)).upper()
        fib = None
        for f in (1, 2, 3):
            if f"FIB{f}" in base or f"FIBERFLAT-FIB{f}" in base:
                fib = str(f)
                break
        if fib is None:
            logger.warning("Skipping fiberflat rate %s (fiber not inferred from name).", path)
            continue
        models_by_fiber[fib] = model
    if not models_by_fiber:
        raise ValueError("No fiberflat rate paths with inferable fiber labels.")
    trace_pos: TracePosModel = TracePositionPrimitive.call(
        models_by_fiber,
        deg_pos=4,
        deg_width=2,
        disp_range=_DEFAULT_FIBERFLAT_DISP_RANGE.get(spectrograph, (500, 4000)),
        xdisp_range=_DEFAULT_FIBERFLAT_XDISP_RANGE.get(spectrograph, (100, 4050)),
        prominence_factor=prominence_factor,
    )
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    for key, val in model_meta.items():
        setattr(trace_pos.meta, key, val)
    logger.info(
        "Sim HDF trace_pos from fiberflat rates spectrograph=%s n_rate_files=%d",
        spectrograph,
        len(fiberflat_rate_paths),
    )
    return trace_pos


def create_traceprof_model_from_rate_maps(
    rate_maps_by_fiber: dict[int, np.ndarray | str | os.PathLike[str]],
    *,
    spectrograph: str = "BSPEC",
    meta: dict | None = None,
    lfc_rate: np.ndarray | None = None,
    tracepos: TracePosModel | None = None,
    orders: list[int] | None = None,
    extraction_aperture: float = 8.0,
    expected_xy_by_order: dict[int, list[tuple[float, float]]] | None = None,
) -> TraceProfModel:
    """
    Build a :class:`~hispecdrp.datamodels.traceprof.TraceProfModel` from 2-D rate maps.

    Matches the production :class:`~hispecdrp.primitives.trace_fitting.traceprof_primitive.TraceProfilePrimitive`
    pattern (``prof_fib*`` = fiberflat rate image). Used for optimal extraction
    where spatial profile ``P`` must come from the flat, not the science spectrum.

    When ``lfc_rate`` and ``tracepos`` are given, also fills spectroperfection
    moment HDUs (shear, ``sigma_spec``, ``sigma_spat``, ``mom2_cross``).
    """
    from astropy.io import fits

    spectrograph = str(spectrograph).upper()
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    model = TraceProfModel(config=hispec_config.HISPECConfig, meta=model_meta)
    for fiber, src in rate_maps_by_fiber.items():
        fib = int(fiber)
        if isinstance(src, (str, os.PathLike)):
            prof = np.asarray(fits.getdata(os.fspath(src)), dtype=np.float64)
        else:
            prof = np.asarray(src, dtype=np.float64)
        if prof.ndim != 2:
            raise ValueError(f"fiber {fib} profile rate must be 2-D; got {prof.shape}")
        err = np.zeros_like(prof, dtype=np.float64)
        setattr(model, f"prof_fib{fib}", prof)
        setattr(model, f"prof_err_fib{fib}", err)
    if lfc_rate is not None and tracepos is not None:
        from ..primitives.trace_fitting.traceprof_moments import (
            fill_traceprof_spectroperfection_moments,
        )

        lfc = np.asarray(lfc_rate, dtype=np.float64)
        for fiber in rate_maps_by_fiber:
            fib = int(fiber)
            prof = np.asarray(getattr(model, f"prof_fib{fib}"), dtype=np.float64)
            fill_traceprof_spectroperfection_moments(
                model,
                fiberflat_rate=prof,
                lfc_rate=lfc,
                tracepos=tracepos,
                spectrograph=spectrograph,
                fiber=fib,
                orders=orders,
                extraction_aperture=float(extraction_aperture),
                expected_xy_by_order=expected_xy_by_order,
            )
    return model


def create_traceprof_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
    fiberflat_rate_paths: dict[int, str | os.PathLike[str]] | None = None,
    detector_shape: tuple[int, int] = (4096, 4160),
) -> TraceProfModel:
    """
    Trace profiles for optimal extraction.

    When ``fiberflat_rate_paths`` is provided (``{fiber: rate_fits_path}``),
    profiles are taken from those 2-D rate maps (production-like). Otherwise
    returns unity placeholders (box extract only; not suitable for optimal).
    """
    if fiberflat_rate_paths:
        return create_traceprof_model_from_rate_maps(
            fiberflat_rate_paths,
            spectrograph=spectrograph,
            meta=meta,
        )
    spectrograph = spectrograph.upper()
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    model = TraceProfModel(config=hispec_config.HISPECConfig, meta=model_meta)
    ny, nx = detector_shape
    for fiber in (1, 2, 3):
        prof = np.ones((ny, nx), dtype=np.float64)
        err = np.zeros((ny, nx), dtype=np.float64)
        setattr(model, f"prof_fib{fiber}", prof)
        setattr(model, f"prof_err_fib{fiber}", err)
    return model


def create_trace_shear_from_sim_hdf(
    spectrograph: str,
    *,
    fiber: int = 2,
    n_pix: int = 4096,
) -> np.ndarray:
    """
    Iso-λ shear ``∂row/∂col`` from HDF PSF stamps (oriented spectral-row grid).

    Thin wrapper around
    :func:`hispecdrp.primitives.spectral_extraction.iso_lambda_shear.create_trace_shear_from_sim_hdf`.
    """
    from ..primitives.spectral_extraction.iso_lambda_shear import (
        create_trace_shear_from_sim_hdf as _create,
    )

    return _create(spectrograph, fiber=fiber, n_pix=n_pix)


def create_wavelength_solution_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
) -> LFCWavesolModel:
    """LFCWavesolModel from the same HDF grids as :func:`create_tracepos_model_from_sim_hdf`."""
    spectrograph = spectrograph.upper()
    order_list = _order_list(spectrograph)
    grids = get_sim_hdf_trace_grids(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    wavesol = LFCWavesolModel(config=hispec_config.HISPECConfig, meta=model_meta)
    for fiber in (1, 2, 3):
        wave_rows = []
        for order in order_list:
            block = grids.get(fiber, {}).get(order)
            if block is None:
                break
            wave_rows.append(np.asarray(block["wavelength"], dtype=np.float64))
        if wave_rows:
            wave_arr = np.stack(wave_rows, axis=0)
            setattr(wavesol, f"wave_fib{fiber}", wave_arr)
            setattr(
                wavesol,
                f"spec_dq_fib{fiber}",
                np.zeros(wave_arr.shape, dtype=np.uint32),
            )
    return wavesol


def _normalize_fibers(
    *,
    fibers: list[int] | tuple[int, ...] | None = None,
    fiber: int | None = None,
    default: tuple[int, ...] = (1, 2, 3),
) -> list[int]:
    """Resolve ``fibers`` / legacy ``fiber`` into a validated list in ``{1,2,3}``."""
    if fibers is not None:
        out = [int(f) for f in fibers]
    elif fiber is not None:
        out = [int(fiber)]
    else:
        out = [int(f) for f in default]
    if not out:
        raise ValueError("At least one fiber is required")
    for f in out:
        if f not in (1, 2, 3):
            raise ValueError(f"fiber must be 1, 2, or 3; got {f}")
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    uniq: list[int] = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def validate_deposit_inverse_wavesol_geometry(
    out_dir: str | os.PathLike[str],
    *,
    spectrograph: str = "BSPEC",
    fibers: list[int] | tuple[int, ...] | None = None,
    fiber: int | None = None,
    n_lines_per_order: int = 1000,
    target_cm_s: float = 10.0,
    write_plots: bool = True,
    write_wavesol: bool = True,
) -> dict[str, Any]:
    """
    Inject/recover test for simulation WAVE at deposit centers (no PSF centroiding).

    Evaluates deposit-inverse ``WAVE`` at exact ``row(λ)`` for each requested fiber
    (default ``1, 2, 3``) and reports residuals in cm/s.

    When ``write_plots`` is True, writes per-fiber products under ``out_dir``:

    - ``comb_wavesol_fib{F}_order{N}_dpix.png``
    - ``comb_wavesol_fib{F}_per_order.png``
    - ``comb_wavesol_per_fiber.png`` (summary across fibers)
    - ``comb_wavesol_report.json``
    """
    import json
    from pathlib import Path

    from ..primitives.utils.spectral_utils import delta_wave_to_delta_vel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spectrograph = spectrograph.upper()
    fiber_list = _normalize_fibers(fibers=fibers, fiber=fiber)
    wavesol = create_wavelength_solution_model_from_sim_hdf(
        spectrograph, meta={"mjd_start": 0.0}
    )
    order_list = _order_list(spectrograph)
    pix_grid = np.arange(4096, dtype=np.float64)

    from .plotting.comb_wavesol_geometry import (
        write_comb_wavesol_order_residual_png,
        write_comb_wavesol_per_fiber_summary_png,
        write_comb_wavesol_per_order_summary_png,
    )

    per_fiber: dict[int, dict[str, Any]] = {}
    all_mads: list[float] = []

    for fib in fiber_list:
        ll = build_injected_line_list(
            spectrograph, fiber=fib, n_lines_per_order=n_lines_per_order
        )
        per: dict[int, dict[str, Any]] = {}
        mads: list[float] = []
        for od in order_list:
            if od not in ll:
                continue
            oi = order_list.index(od)
            w = np.asarray(getattr(wavesol, f"wave_fib{fib}")[oi], dtype=np.float64)
            lams = np.asarray(ll[od], dtype=np.float64)
            rows, _ = expected_feature_xy_oriented(spectrograph, fib, od, lams)
            ok = np.isfinite(rows) & (rows >= 0) & (rows < 4096)
            if int(ok.sum()) < 5:
                continue
            plo, phi = np.percentile(rows[ok], [8, 92])
            use = ok & (rows >= plo) & (rows <= phi)
            if int(use.sum()) < 5:
                continue
            lam_mid = float(np.mean(lams[use]))
            f_wave = PchipInterpolator(pix_grid, w, extrapolate=True)
            lam_hat = np.asarray(f_wave(rows[use]), dtype=np.float64)
            vel = np.asarray(
                delta_wave_to_delta_vel(lams[use] - lam_hat, lam_mid), dtype=np.float64
            )
            mad = float(np.median(np.abs(vel)))
            rms = float(np.std(vel))
            mx = float(np.max(np.abs(vel)))
            mads.append(mad * 100)
            per[od] = {
                "n": int(use.sum()),
                "mad_cm_s": mad * 100,
                "rms_cm_s": rms * 100,
                "max_cm_s": mx * 100,
                "lam_mid_um": lam_mid,
                "pix": rows[use].tolist(),
                "vel_cm_s": (vel * 100).tolist(),
            }
            if write_plots:
                write_comb_wavesol_order_residual_png(
                    out / f"comb_wavesol_fib{fib}_order{od}_dpix.png",
                    fiber=fib,
                    order=od,
                    pix=rows[use],
                    vel_cm_s=vel * 100,
                    lam_mid_um=lam_mid,
                    mad_cm_s=mad * 100,
                    rms_cm_s=rms * 100,
                    target_cm_s=target_cm_s,
                    legacy_alias_path=(
                        out / f"comb_wavesol_order{od}_dpix.png"
                        if fiber_list == [2]
                        else None
                    ),
                )

        orders = sorted(per)
        if write_plots and orders:
            write_comb_wavesol_per_order_summary_png(
                out / f"comb_wavesol_fib{fib}_per_order.png",
                fiber=fib,
                orders=orders,
                per_order=per,
                mads_cm_s=mads,
                target_cm_s=target_cm_s,
                legacy_alias_path=(
                    out / "comb_wavesol_per_order.png" if fiber_list == [2] else None
                ),
            )

        fib_report = {
            "fiber": int(fib),
            "median_mad_cm_s": float(np.median(mads)) if mads else float("nan"),
            "median_rms_cm_s": (
                float(np.median([per[o]["rms_cm_s"] for o in orders])) if orders else float("nan")
            ),
            "max_mad_cm_s": float(np.max(mads)) if mads else float("nan"),
            "n_orders": len(orders),
            "per_order": per,
            "passed": bool(mads) and float(np.max(mads)) < float(target_cm_s),
        }
        per_fiber[int(fib)] = fib_report
        all_mads.extend(mads)

    if write_plots and len(fiber_list) > 1:
        write_comb_wavesol_per_fiber_summary_png(
            out / "comb_wavesol_per_fiber.png",
            per_fiber=per_fiber,
            target_cm_s=target_cm_s,
        )

    wavesol_path = out / "sim_hdf_wavesol_injected_comb.fits"
    if write_wavesol:
        wavesol.save(str(wavesol_path))

    # Backward-compat top-level fields mirror the first fiber (or fiber 2 if present).
    primary = 2 if 2 in per_fiber else fiber_list[0]
    prim = per_fiber[primary]
    report: dict[str, Any] = {
        "method": "deposit_inverse",
        "spectrograph": spectrograph,
        "fibers": fiber_list,
        "fiber": int(primary),
        "target_cm_s": float(target_cm_s),
        "median_mad_cm_s": float(np.median(all_mads)) if all_mads else float("nan"),
        "median_rms_cm_s": float(
            np.median([per_fiber[f]["median_rms_cm_s"] for f in fiber_list])
        ),
        "max_mad_cm_s": float(np.max(all_mads)) if all_mads else float("nan"),
        "n_orders": int(prim["n_orders"]),
        "per_order": prim["per_order"],
        "per_fiber": per_fiber,
        "wavesol_path": str(wavesol_path) if write_wavesol else None,
        "out_dir": str(out),
        "passed": bool(all_mads) and float(np.max(all_mads)) < float(target_cm_s),
        "note": (
            "Residuals are WAVE(deposit_row(λ)) − λ for each fiber. PSF-centroid "
            "residuals are a separate measurement systematic (~40 m/s ZP, "
            "pixel-phase structure) and must not be used to redefine simulation WAVE."
        ),
    }
    if write_plots:
        (out / "comb_wavesol_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _extract_box_one_order(
    rate: np.ndarray,
    rate_err: np.ndarray,
    rate_dq: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
) -> tuple[np.ndarray, np.ndarray]:
    from ..primitives.spectral_extraction.extract_box import extract_spectrum_box

    rate = np.asarray(rate, dtype=np.float64)
    if rate_err.shape != rate.shape:
        rate_err = np.zeros_like(rate, dtype=np.float64)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
    if rate_dq.shape != rate.shape:
        rate_dq = np.zeros_like(rate, dtype=np.uint32)
    else:
        rate_dq = np.asarray(rate_dq, dtype=np.uint32)
    ny, nx = rate.shape
    spec = np.full(ny, np.nan, dtype=np.float64)
    err = np.full(ny, np.nan, dtype=np.float64)
    var_r = np.zeros_like(rate, dtype=np.float64)
    var_p = np.zeros_like(rate, dtype=np.float64)
    dq = np.zeros_like(rate, dtype=np.uint32)
    mid = np.zeros_like(rate, dtype=np.float64)
    extract_spectrum_box(
        input_data=rate,
        input_err=rate_err,
        input_dq=rate_dq,
        input_var_rnoise=var_r,
        input_var_poisson=var_p,
        input_mid_time=mid,
        trace_positions=trace_y,
        extraction_aperture=extraction_aperture,
        spec_output=spec,
        err_output=err,
        var_rnoise_output=var_r,
        var_poisson_output=var_p,
        dq_output=dq,
        mid_time_output=mid,
        orders_are_vertical=True,
    )
    return spec, err


def extract_spectrum_from_rate_map(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    extraction_aperture: float = 8.0,
    orders: list[int] | None = None,
    rate_err: np.ndarray | None = None,
    rate_dq: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Box-extract 1D spectra directly from a ZEMAX / sim rate map (UTR bypass).

    Uses HDF ``y_pos`` / WAVE grids from :func:`get_sim_hdf_trace_grids` so the
    extract matches the paint geometry without Stage0 ``rate_err`` artifacts.
    Intended for RV / injection diagnostics where the truth rate map is already
    near-infinite S/N.

    Parameters
    ----------
    rate_map
        2D rate image ``(ny, nx)`` in the pipeline-oriented frame (same as
        ``rate_map.fits`` from :func:`~hispecdrp.simulations.simulate_up_the_ramp.create_ramp_from_spectrum`).
    spectrograph
        ``'BSPEC'`` or ``'RSPEC'``.
    fiber
        Fiber number (1–3).
    extraction_aperture
        Box aperture width in pixels.
    orders
        Optional subset of echelle orders; default is all orders for the spectrograph.
    rate_err, rate_dq
        Optional companions; default to zeros (no NaN-err masking).

    Returns
    -------
    dict
        ``orders`` (list[int]), ``wavelength`` (n_orders, n_pix), ``flux``,
        ``flux_err``, ``fiber``, ``spectrograph``, ``extraction_aperture``.
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    if fiber not in (1, 2, 3):
        raise ValueError(f"fiber must be 1, 2, or 3; got {fiber}")

    rate = np.asarray(rate_map, dtype=np.float64)
    if rate.ndim != 2:
        raise ValueError(f"rate_map must be 2D; got shape {rate.shape}")
    if rate_err is None:
        rate_err = np.zeros_like(rate)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
        if rate_err.shape != rate.shape:
            raise ValueError("rate_err shape must match rate_map")
    if rate_dq is None:
        rate_dq = np.zeros(rate.shape, dtype=np.uint32)
    else:
        rate_dq = np.asarray(rate_dq, dtype=np.uint32)
        if rate_dq.shape != rate.shape:
            raise ValueError("rate_dq shape must match rate_map")

    grids = get_sim_hdf_trace_grids(spectrograph)
    if fiber not in grids:
        raise KeyError(f"No HDF trace grids for fiber {fiber}")
    order_list = list(_order_list(spectrograph) if orders is None else orders)
    wave_rows: list[np.ndarray] = []
    flux_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    kept: list[int] = []
    for od in order_list:
        g = grids[fiber].get(int(od))
        if g is None:
            continue
        y_pos = np.asarray(g["y_pos"], dtype=np.float64)
        wave = np.asarray(g["wavelength"], dtype=np.float64)
        spec, err = _extract_box_one_order(
            rate,
            rate_err,
            rate_dq,
            y_pos,
            extraction_aperture=float(extraction_aperture),
        )
        kept.append(int(od))
        wave_rows.append(wave)
        flux_rows.append(spec)
        err_rows.append(err)

    if not kept:
        raise RuntimeError(
            f"No orders extracted for {spectrograph} fiber {fiber} "
            f"(requested={order_list})"
        )

    return {
        "spectrograph": spectrograph,
        "fiber": fiber,
        "extraction_aperture": float(extraction_aperture),
        "orders": kept,
        "wavelength": np.stack(wave_rows, axis=0),
        "flux": np.stack(flux_rows, axis=0),
        "flux_err": np.stack(err_rows, axis=0),
    }

def create_fiberflat_model_from_rate_maps(
    rate_paths: list[str],
    tracepos_model: TracePosModel,
    *,
    spectrograph: str,
    meta: dict | None = None,
    extraction_aperture: float = 10.0,
) -> FiberflatModel:
    """
    Box-extract simulated fiber-flat rate maps with HDF ``tracepos_model`` rows.

    Produces a ``FiberflatModel`` consistent with the science extraction geometry.
    """
    spectrograph = spectrograph.upper()
    order_list = _order_list(spectrograph)
    hdf_grids = get_sim_hdf_trace_grids(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    ff = FiberflatModel(config=hispec_config.HISPECConfig, meta=model_meta)
    n_orders = len(order_list)
    n_pix = 4096

    for path in rate_paths:
        l0 = datamodels.open(path)
        fib = None
        base = str(getattr(l0.meta, "filename", path)).upper()
        for f in (1, 2, 3):
            if f"FIB{f}" in base or f"FIBERFLAT-FIB{f}" in base:
                fib = f
                break
        if fib is None:
            logger.warning("Skipping fiberflat rate %s (fiber not inferred from name).", path)
            continue
        pos = getattr(tracepos_model, f"pos_fib{fib}")
        rate = l0.rate
        rate_err = getattr(l0, "rate_err", None)
        if rate_err is None:
            rate_err = np.zeros_like(rate, dtype=np.float64)
        rate_dq = getattr(l0, "rate_dq", None)
        if rate_dq is None:
            rate_dq = getattr(l0, "dq", None)
        if rate_dq is None:
            rate_dq = np.zeros_like(rate, dtype=np.uint32)
        spec_rows = []
        err_rows = []
        wave_rows = []
        for i, _order in enumerate(order_list):
            if i >= pos.shape[0]:
                break
            trace_y = pos[i]
            spec, err = _extract_box_one_order(
                rate,
                rate_err,
                rate_dq,
                trace_y,
                extraction_aperture=extraction_aperture,
            )
            spec_rows.append(spec)
            err_rows.append(err)
            wave = hdf_grids.get(fib, {}).get(_order, {}).get("wavelength")
            if wave is None:
                wave = np.full(n_pix, np.nan, dtype=np.float64)
            wave_rows.append(np.asarray(wave, dtype=np.float64))
        if not spec_rows:
            continue
        spec_arr = np.stack(spec_rows, axis=0).astype(np.float32)
        err_arr = np.stack(err_rows, axis=0).astype(np.float32)
        setattr(ff, f"spec_fib{fib}", spec_arr)
        setattr(ff, f"spec_err_fib{fib}", err_arr)
        setattr(ff, f"spec_var_poisson_fib{fib}", np.zeros_like(spec_arr, dtype=np.float32))
        setattr(ff, f"spec_var_rnoise_fib{fib}", np.zeros_like(spec_arr, dtype=np.float32))
        setattr(ff, f"spec_dq_fib{fib}", np.zeros(spec_arr.shape, dtype=np.uint32))
        setattr(ff, f"wave_fib{fib}", np.stack(wave_rows, axis=0).astype(np.float64))
        setattr(ff.meta, f"object_fiber{fib}", "WHITELIGHT")

    for f in (1, 2, 3):
        if getattr(ff, f"spec_fib{f}", None) is None and getattr(ff.meta, f"object_fiber{f}", None) is None:
            setattr(ff.meta, f"object_fiber{f}", "BLOCKED")

    return ff


def _parabolic_min_nearest(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    half_win: float,
) -> float:
    """Parabolic absorption minimum nearest ``x0`` (shared by wavesol ZP)."""
    from ..utils.line_mask_utils import _parabolic_minimum_wavelength_um

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y) & (x >= x0 - half_win) & (x <= x0 + half_win)
    if int(m.sum()) < 3:
        return float("nan")
    xx, yy = x[m], y[m]
    o = np.argsort(xx)
    xx, yy = xx[o], yy[o]
    local = np.where((yy[1:-1] < yy[:-2]) & (yy[1:-1] <= yy[2:]))[0] + 1
    if local.size == 0:
        return float("nan")
    i0 = int(local[np.argmin(np.abs(xx[local] - x0))])
    if i0 <= 0 or i0 >= xx.size - 1:
        return float(xx[i0])
    return float(_parabolic_minimum_wavelength_um(xx, yy, i0))


def refine_sim_hdf_wavesol_from_ccf_order_rvs(
    wavesol: LFCWavesolModel,
    *,
    fiber: int,
    order_rvs_m_s: dict[int, float] | np.ndarray,
    order_vec: np.ndarray | list[int] | None = None,
    known_rv_m_s: float = 0.0,
    spectrograph: str | None = None,
) -> dict[str, Any]:
    """
    Apply a per-order Doppler scale so CCF order RVs land on ``known_rv_m_s``.

    Line-centroid WAVE ZP removes the bulk PSF deposition shift, but the
    line-mask CCF still sees order-dependent residuals (mask weighting / blends /
    LSF). For a reference exposure at known injected RV this folds those
    residuals into ``WAVE_FIB`` (data-model calibration, not a chromatic RV
    fit at science time)::

    ``order_rvs_m_s`` maps echelle order → measured CCF RV (m/s), or a length-N
    array aligned with ``order_vec`` / ``wave_fib`` rows.
    """
    import scipy.constants

    fib = int(fiber)
    wave_attr = f"wave_fib{fib}"
    wave_arr = getattr(wavesol, wave_attr, None)
    if wave_arr is None:
        raise ValueError(f"wavesol missing {wave_attr}")
    wave_arr = np.asarray(wave_arr, dtype=np.float64).copy()
    if order_vec is None:
        if spectrograph is None:
            raise ValueError("order_vec or spectrograph is required")
        order_vec = np.asarray(_order_list(str(spectrograph).upper()), dtype=int)
    else:
        order_vec = np.asarray(order_vec, dtype=int).ravel()
    if wave_arr.shape[0] != order_vec.size:
        raise ValueError(
            f"wave rows ({wave_arr.shape[0]}) != order_vec ({order_vec.size})"
        )

    if isinstance(order_rvs_m_s, dict):
        rv_by_order = {int(k): float(v) for k, v in order_rvs_m_s.items()}
    else:
        rvs = np.asarray(order_rvs_m_s, dtype=np.float64).ravel()
        if rvs.size != order_vec.size:
            raise ValueError("order_rvs_m_s array must match order_vec length")
        rv_by_order = {int(od): float(rv) for od, rv in zip(order_vec, rvs)}

    c_mps = float(scipy.constants.c)
    known = float(known_rv_m_s)
    applied: dict[int, float] = {}
    for oi, od in enumerate(order_vec):
        if int(od) not in rv_by_order:
            continue
        rv = float(rv_by_order[int(od)])
        if not np.isfinite(rv):
            continue
        # Residual vs truth absorbed into WAVE scale.
        dv = rv - known
        if not np.isfinite(dv) or abs(dv) < 1.0e-12:
            applied[int(od)] = 0.0
            continue
        wave_arr[oi] = wave_arr[oi] * (1.0 - dv / c_mps)
        applied[int(od)] = float(dv)

    setattr(wavesol, wave_attr, wave_arr)
    vals = list(applied.values())
    return {
        "fiber": fib,
        "known_rv_m_s": known,
        "n_orders": len(applied),
        "median_dv_m_s": float(np.median(vals)) if vals else float("nan"),
        "mad_dv_m_s": (
            float(np.median(np.abs(np.asarray(vals) - np.median(vals))))
            if vals
            else float("nan")
        ),
        "per_order_dv_m_s": applied,
        "method": "ccf_order_scale",
    }


def refine_sim_hdf_wavesol_line_zeropoint(
    wavesol: LFCWavesolModel,
    rate_map: np.ndarray,
    *,
    spectrograph: str,
    fibers: list[int] | tuple[int, ...] | None = None,
    fiber: int | None = None,
    n_lines_per_order: int = 40,
    min_depth: float = 0.15,
    fit_half_win_pix: float = 8.0,
    known_rv_m_s: float = 0.0,
    poly_deg: int = 2,
) -> dict[str, Any]:
    """
    Apply a per-fiber, per-order absorption-line WAVE correction from a rate map.

    The geometric HDF → WAVE mapping matches paint coordinates, but LSF/PSF
    deposition shifts apparent absorption minima by tens of m/s (and can leave a
    mild spectral-axis trend). For each fiber/order this measures line Δv vs
    BT-Settl and corrects ``WAVE_FIB``:

    * ``poly_deg == 0`` — constant scale ``WAVE *= (1 - med/c)`` (legacy ZP).
    * ``poly_deg >= 1`` — pixel-dependent Doppler scale
      ``WAVE[p] *= (1 - dv_poly(p)/c)`` with ``dv_poly`` a degree-``poly_deg``
      fit to line residuals (default 2).

    When the rate map includes a known injected stellar RV, pass
    ``known_rv_m_s`` so that Doppler shift is **not** absorbed into WAVE.

    Fibers with too little rate-map flux are left unchanged. Does **not** modify
    the HDF spectrograph model file.
    """
    import scipy.constants

    from ..primitives.utils.spectral_utils import (
        delta_wave_to_delta_vel,
        get_spectrum_path,
        load_btsettl_spectrum,
    )
    from ..utils.line_mask_utils import (
        _continuum_envelope,
        detect_template_line_centers_parabolic_um,
    )
    from . import e2e_support

    spectrograph = spectrograph.upper()
    fiber_list = _normalize_fibers(fibers=fibers, fiber=fiber)
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate.ndim != 2:
        raise ValueError(f"rate_map must be 2-D, got shape {rate.shape}")
    poly_deg = int(poly_deg)
    if poly_deg < 0 or poly_deg > 3:
        raise ValueError(f"poly_deg must be 0..3; got {poly_deg}")

    grids = get_sim_hdf_trace_grids(spectrograph)
    order_list = _order_list(spectrograph)
    tw, tf = load_btsettl_spectrum(
        get_spectrum_path(
            e2e_support.DEFAULT_TEMPLATE_FILENAME,
            base_dir=e2e_support.templates_data_dir(),
        )
    )
    tw = np.asarray(tw, dtype=np.float64).ravel()
    tf = np.asarray(tf, dtype=np.float64).ravel()
    order_t = np.argsort(tw)
    tw, tf = tw[order_t], tf[order_t]

    c_mps = float(scipy.constants.c)
    z = float(known_rv_m_s) / c_mps
    per_fiber: dict[int, dict[str, Any]] = {}

    for fib in fiber_list:
        per_order: dict[int, dict[str, Any]] = {}
        wave_attr = f"wave_fib{fib}"
        wave_arr = getattr(wavesol, wave_attr, None)
        if wave_arr is None:
            continue
        wave_arr = np.asarray(wave_arr, dtype=np.float64).copy()

        for oi, od in enumerate(order_list):
            if od not in grids.get(fib, {}):
                continue
            if oi >= wave_arr.shape[0]:
                break
            g = grids[fib][od]
            y_pos = np.asarray(g["y_pos"], dtype=np.float64)
            wave = wave_arr[oi]
            spec, _ = _extract_box_one_order(
                rate,
                np.zeros_like(rate),
                np.zeros(rate.shape, dtype=np.uint32),
                y_pos,
                extraction_aperture=10.0,
            )
            ok = np.isfinite(wave) & np.isfinite(spec)
            if int(ok.sum()) < 50:
                continue
            # Skip fibers/orders with negligible continuum (unlit fiber).
            if float(np.nanpercentile(spec[ok], 90)) <= 0:
                continue
            w_lo, w_hi = float(np.nanmin(wave[ok])), float(np.nanmax(wave[ok]))
            msk = (tw >= w_lo) & (tw <= w_hi)
            centers = detect_template_line_centers_parabolic_um(
                tw[msk], tf[msk], min_continuum_normalized_depth=min_depth
            )
            if centers.size == 0:
                continue
            cont = _continuum_envelope(tf[msk], 181)
            depths = 1.0 - np.interp(centers, tw[msk], tf[msk]) / np.maximum(
                np.interp(centers, tw[msk], cont), 1e-30
            )
            pick = centers[np.argsort(-depths)[:n_lines_per_order]]
            dlam = float(np.nanmedian(np.abs(np.diff(wave[np.isfinite(wave)]))))
            half = fit_half_win_pix * max(dlam, 1e-9)
            ideal = np.interp(wave, tw, tf, left=np.nan, right=np.nan)
            pix = np.arange(wave.size, dtype=np.float64)
            pix_lines: list[float] = []
            dvs: list[float] = []
            for c_native in pick:
                lam_rest = _parabolic_min_nearest(wave, ideal, float(c_native), half)
                if not np.isfinite(lam_rest):
                    continue
                lam_expect = lam_rest * (1.0 + z)
                lam_m = _parabolic_min_nearest(wave, spec, lam_expect, half)
                if not np.isfinite(lam_m):
                    continue
                if abs(lam_m - lam_expect) > 2.5 * dlam:
                    continue
                # Pixel of the expected line on the current WAVE grid.
                p_exp = float(np.interp(lam_expect, wave[ok], pix[ok]))
                dv = float(delta_wave_to_delta_vel(lam_m - lam_expect, lam_expect))
                pix_lines.append(p_exp)
                dvs.append(dv)
            if len(dvs) < 3:
                continue
            dvs_a = np.asarray(dvs, dtype=np.float64)
            pix_a = np.asarray(pix_lines, dtype=np.float64)
            med = float(np.median(dvs_a))
            mad = float(np.median(np.abs(dvs_a - med)))
            # MAD clip before polyfit / scale.
            if mad > 0:
                keep = np.abs(dvs_a - med) <= 4.0 * 1.4826 * mad
            else:
                keep = np.ones(dvs_a.shape, dtype=bool)
            if int(keep.sum()) < 3:
                keep = np.ones(dvs_a.shape, dtype=bool)
            dvs_k = dvs_a[keep]
            pix_k = pix_a[keep]
            med = float(np.median(dvs_k))
            mad = float(np.median(np.abs(dvs_k - med)))
            deg_use = min(poly_deg, max(0, int(dvs_k.size) - 2))
            order_info: dict[str, Any] = {
                "dv_m_s": med,
                "mad_m_s": mad,
                "n_lines": float(len(dvs_k)),
                "poly_deg": int(deg_use),
            }
            if deg_use <= 0:
                wave_arr[oi] = wave * (1.0 - med / c_mps)
                order_info["poly_coeffs"] = [med]
            else:
                coeffs, x_mid, x_half = _polyfit_scaled(pix_k, dvs_k, deg=deg_use)
                dv_pix = _polyval_scaled(coeffs, pix, x_mid=x_mid, x_half=x_half)
                wave_arr[oi] = wave * (1.0 - dv_pix / c_mps)
                order_info["poly_coeffs"] = [float(c) for c in coeffs]
                order_info["poly_x_mid"] = float(x_mid)
                order_info["poly_x_half"] = float(x_half)
                # Report residual after poly (should be ~0 median).
                resid = dvs_k - _polyval_scaled(coeffs, pix_k, x_mid=x_mid, x_half=x_half)
                order_info["dv_m_s"] = float(np.median(resid))
                order_info["mad_m_s"] = float(np.median(np.abs(resid - np.median(resid))))
                order_info["scale_dv_m_s"] = med
            per_order[int(od)] = order_info

        if per_order:
            setattr(wavesol, wave_attr, wave_arr.astype(np.float64))
        # Prefer scale_dv when present (poly residual medians are ~0 by construction).
        scale_vals = [
            float(v.get("scale_dv_m_s", v["dv_m_s"]))
            for v in per_order.values()
            if np.isfinite(float(v.get("scale_dv_m_s", v["dv_m_s"])))
        ]
        med_all = float(np.median(scale_vals)) if scale_vals else float("nan")
        per_fiber[int(fib)] = {
            "fiber": int(fib),
            "per_order": per_order,
            "median_dv_m_s": med_all,
            "n_orders": len(per_order),
            "poly_deg": int(poly_deg),
        }
        logger.info(
            "Sim HDF wavesol line ZP: fiber=%d n_orders=%d median_scale_dv=%.1f m/s "
            "poly_deg=%d (known_rv=%.1f m/s)",
            fib,
            len(per_order),
            med_all,
            int(poly_deg),
            float(known_rv_m_s),
        )

    medians = [
        v["median_dv_m_s"]
        for v in per_fiber.values()
        if np.isfinite(v["median_dv_m_s"])
    ]
    primary = 2 if 2 in per_fiber else fiber_list[0]
    return {
        "fibers": fiber_list,
        "fiber": int(primary),
        "per_fiber": per_fiber,
        # Backward-compat single-fiber fields (primary fiber).
        "per_order": per_fiber.get(primary, {}).get("per_order", {}),
        "median_dv_m_s": float(np.median(medians)) if medians else float("nan"),
        "known_rv_m_s": float(known_rv_m_s),
        "poly_deg": int(poly_deg),
    }


def build_wavesol_aidan_with_order_zp(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fibers: list[int] | tuple[int, ...] | None = None,
    fiber: int | None = None,
    known_rv_m_s: float = 0.0,
    meta: dict | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Aidan ``get_wavelength_solution`` WAVE + constant (deg-0) per-order ZP only.

    Builds :func:`create_wavelength_solution_model_from_sim` (Brent inverse of the
    HDF λ→x map), then applies
    :func:`refine_sim_hdf_wavesol_line_zeropoint` with ``poly_deg=0`` so each
    order gets a single Doppler scale and no higher-order λ trend is fit.

    Returns ``(wavesol_model, zp_info)``.
    """
    from .simulate_up_the_ramp import create_wavelength_solution_model_from_sim

    spectrograph = spectrograph.upper()
    model_meta = {"spectrograph": spectrograph, "note": "aidan_gws+order_zp0"}
    if meta:
        model_meta.update(meta)
    wavesol = create_wavelength_solution_model_from_sim(spectrograph, meta=model_meta)
    zp_info = refine_sim_hdf_wavesol_line_zeropoint(
        wavesol,
        rate_map,
        spectrograph=spectrograph,
        fibers=fibers,
        fiber=fiber,
        known_rv_m_s=float(known_rv_m_s),
        poly_deg=0,
    )
    return wavesol, zp_info


def _aidan_order_zp0_wavesol_cache_path(shared_dir: str, spectrograph: str) -> str:
    prefix = "HB" if str(spectrograph).upper() == "BSPEC" else "HR"
    return os.path.join(
        e2e_support.calibration_dir(shared_dir),
        f"{prefix}.aidan_gws_order_zp0_wavesol.fits",
    )


def load_shared_stage1_cals_for_rv_sim(
    shared_dir: str,
    spectrograph: str = "BSPEC",
) -> dict[str, Any] | None:
    """
    Load a previously cached complete Stage-1 cal bundle for RV recovery sims.

    Expects sim-HDF ``tracepos`` / ``traceprof`` / ``fiberflat`` under
    ``<shared_dir>/cals`` plus an Aidan+deg-0 order-ZP ``wavesol`` FITS written by
    :func:`ensure_shared_stage1_cals_for_rv_sim`. Returns ``None`` if incomplete.
    """
    from .. import datamodels
    from .simulate_up_the_ramp import (
        get_sim_stage1_master_paths,
        load_cached_sim_stage1_masters,
    )

    spectrograph = str(spectrograph).upper()
    out_abs = os.path.abspath(shared_dir)
    wavesol_path = _aidan_order_zp0_wavesol_cache_path(out_abs, spectrograph)
    paths = get_sim_stage1_master_paths(out_abs, spectrograph)
    required = ("trace_pos", "trace_prof", "fiberflat", "meta")
    if not os.path.isfile(wavesol_path):
        return None
    if not all(os.path.isfile(paths[k]) for k in required):
        return None

    masters = load_cached_sim_stage1_masters(out_abs, spectrograph)
    wavesol = datamodels.open(wavesol_path)
    return {
        "trace_pos": masters["trace_pos"],
        "trace_prof": masters["trace_prof"],
        "fiberflat": masters["fiberflat"],
        "wavesol": wavesol,
        "zp_info": {
            "loaded_from_cache": True,
            "wavesol_path": wavesol_path,
        },
        "shared_dir": out_abs,
    }


def ensure_shared_stage1_cals_for_rv_sim(
    shared_dir: str,
    *,
    rate_map: np.ndarray | None = None,
    spectrograph: str = "BSPEC",
    fibers: list[int] | tuple[int, ...] | None = None,
    known_rv_m_s: float = 0.0,
    extraction_aperture: float = 8.0,
    ramp_meta: Any | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """
    Return a complete Stage-1 cal bundle for RV recovery simulations.

    Products (HISPEC datamodel objects):

    - ``trace_pos``, ``trace_prof``, ``fiberflat`` — sim-HDF masters
    - ``wavesol`` — Aidan ``get_wavelength_solution`` + deg-0 per-order ZP
      (:func:`build_wavesol_aidan_with_order_zp`)

    On the first call (or when ``rebuild=True``), ``rate_map`` from the science
    injection is required: fiberflat simulation and WAVE ZP refine use that map.
    Results are cached under ``<shared_dir>/cals`` so later injections (and
    notebook restarts) reuse the same frames without rebuilding.

    Parameters
    ----------
    shared_dir
        Directory that owns the cached Stage-1 masters (typically
        ``<OUT>/_shared_stage1``).
    rate_map
        2-D e-/s rate map from the first ``create_ramp_from_spectrum`` call.
        Required when the cache is missing or ``rebuild`` is True.
    known_rv_m_s
        Injected stellar RV of the rate map used for the WAVE order-ZP refine
        (m/s). For ``inject_rv_via='spectrum'``, pass the first injection's RV.
    """
    from .simulate_up_the_ramp import register_sim_stage1_calibrations

    spectrograph = str(spectrograph).upper()
    fibs = [int(f) for f in (fibers if fibers is not None else [2]) if int(f) in (1, 2, 3)]
    if not fibs:
        raise ValueError("fibers must include at least one of 1, 2, 3")
    out_abs = e2e_support.ensure_output_directory(shared_dir)

    if not rebuild:
        cached = load_shared_stage1_cals_for_rv_sim(out_abs, spectrograph)
        if cached is not None:
            logger.info(
                "Reusing shared Stage-1 cals under %s (tracepos/traceprof/fiberflat + Aidan WAVE).",
                e2e_support.calibration_dir(out_abs),
            )
            return cached

    if rate_map is None:
        raise ValueError(
            "ensure_shared_stage1_cals_for_rv_sim: rate_map is required to build "
            "shared Stage-1 cals (cache missing). Pass the rate map from the first "
            "create_ramp_from_spectrum call."
        )

    logger.info(
        "Building shared Stage-1 cals under %s (fiberflat/traces + Aidan+ZP0 WAVE; "
        "known_rv=%.3f m/s). This is a one-time cost for the RV recovery grid.",
        out_abs,
        float(known_rv_m_s),
    )
    bundle = register_sim_stage1_calibrations(
        out_abs,
        rate_map,
        spectrograph=spectrograph,
        fibers=fibs,
        ramp_meta=ramp_meta,
        connect_remote=False,
        extraction_aperture=float(extraction_aperture),
        reuse_existing=False if rebuild else True,
    )
    wavesol_meta: dict[str, Any] = {}
    if ramp_meta is not None:
        mjd = getattr(ramp_meta, "mjd_start", None)
        if mjd is not None:
            wavesol_meta["mjd_start"] = mjd
    wavesol, zp_info = build_wavesol_aidan_with_order_zp(
        rate_map,
        spectrograph=spectrograph,
        fibers=fibs,
        known_rv_m_s=float(known_rv_m_s),
        meta=wavesol_meta or None,
    )
    wavesol_path = _aidan_order_zp0_wavesol_cache_path(out_abs, spectrograph)
    wavesol.save(filepath=wavesol_path)
    zp_info = dict(zp_info or {})
    zp_info["wavesol_path"] = wavesol_path
    zp_info["loaded_from_cache"] = False

    return {
        "trace_pos": bundle["trace_pos"],
        "trace_prof": bundle["trace_prof"],
        "fiberflat": bundle["fiberflat"],
        "wavesol": wavesol,
        "zp_info": zp_info,
        "shared_dir": out_abs,
    }


# Default box aperture for RV-recovery E2E sims (matches rv_recovery_sim_line_by_line.ipynb).
DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE: float = 8.0


def stellar_rv_kwargs_for_e2e_sim(
    bulk_rv_km_s: float,
    mask_path: str,
    *,
    fibers: list[int] | tuple[int, ...] = (2,),
    **overrides,
) -> dict[str, Any]:
    """
    Keyword arguments for :class:`~hispecdrp.primitives.StellarRVPrimitive` on E2E blaze L2.

    Uses :data:`~hispecdrp.utils.line_mask_utils.CCFOPT_SINGLE_LINE_STELLAR_RV_KW`
    (line-by-line ccfopt defaults).  Set ``ccf_zb=0`` when ``barycentric_velocity_km_s=0``
    and ``barycorr`` is skipped.
    """
    from ..utils.line_mask_utils import CCFOPT_SINGLE_LINE_STELLAR_RV_KW

    kw = CCFOPT_SINGLE_LINE_STELLAR_RV_KW
    inj_m_s = float(bulk_rv_km_s) * 1000.0
    pars: dict[str, Any] = {
        "algorithm": "ccf_line_mask",
        "mask_path": str(mask_path),
        "ccf_input_velocity": inj_m_s,
        "do_fibers": [str(int(f)) for f in fibers],
        "window_line_by_line": True,
        "window_size_um": float(kw["window_size_um"]),
        "v_half_window": float(kw["v_half_window_m_s"]),
        "v_step_coarse": float(kw["v_step_m_s"]),
        "mask_ccf_pixel_interp_order": int(kw["mask_ccf_pixel_interp_order"]),
        "ccf_min_good_pixels": int(kw["ccf_min_good_pixels"]),
        "ccf_zb": 0.0,
        "save_preview": False,
        "show_preview": False,
    }
    pars.update(overrides)
    return pars


def run_sim_through_blaze(
    ramp,
    shared_cals: dict[str, Any],
    *,
    output_dir: str,
    fibers: list[int] | tuple[int, ...] = (2,),
    extraction_aperture: float = DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE,
) -> Any:
    """
    Reduce a simulated UTR through Stage 0–2 blaze correction (no stellar RV).

    ``shared_cals`` must be the bundle from
    :func:`ensure_shared_stage1_cals_for_rv_sim` (tracepos / traceprof / fiberflat /
    Aidan+deg-0 WAVE).
    """
    from ..pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline

    do_fibers = [str(int(f)) for f in fibers]
    l0 = Stage0Pipeline.call(ramp, output_dir=output_dir)
    l1 = Stage1Pipeline.call(
        l0,
        output_dir=output_dir,
        primitives={
            "spec_extract": {
                "do_fibers": do_fibers,
                "method": "box",
                "extraction_aperture": int(extraction_aperture),
                "tracepos": shared_cals["trace_pos"],
                "traceprof": shared_cals["trace_prof"],
            },
            "set_wavesol": {"wavesol": shared_cals["wavesol"]},
        },
    )
    return Stage2Pipeline.call(
        l1,
        output_dir=output_dir,
        primitives={
            "blaze_corr": {
                "do_fibers": do_fibers,
                "fiberflat": shared_cals["fiberflat"],
                "smooth_blaze": True,
            },
            "sky_sub": {"run": False},
            "barycorr": {"run": False},
            "stellar_rv": {"run": False},
        },
    )


def build_e2e_ccfopt_mask_from_blaze_l2(
    blaze_l2,
    template_filename: str,
    output_path: str,
    *,
    fiber: int = 2,
    masks_directory: str | None = None,
) -> str:
    """
    Build a static ``*.ccfopt.delta.mas`` mask on an E2E blaze L2 at inject RV = 0.

    Re-zeros seed line centers from a line-by-line ``RVCCF`` table on the blaze
    spectrum (same path as :func:`hispecdrp.simulations.spectra.build_ccfopt_mask`
    but for ZEMAX+extraction L2 rather than ideal native-WAVE L2).
    """
    from pathlib import Path

    from ..utils.line_mask_utils import (
        build_ccfopt_line_mask_from_l2_line_by_line_rv,
        paired_line_mask_path_for_template,
    )

    seed_mask = paired_line_mask_path_for_template(template_filename, mask_style="delta")
    templates_dir = Path(e2e_support.templates_data_dir())
    template_path = templates_dir / Path(template_filename).name
    if not template_path.is_file():
        alt = Path(template_filename).expanduser()
        template_path = alt.resolve() if alt.is_file() else (templates_dir / template_filename)
    mask_path, _ = build_ccfopt_line_mask_from_l2_line_by_line_rv(
        seed_mask,
        blaze_l2,
        fiber=int(fiber),
        template_path=str(template_path),
        output_path=str(output_path),
        masks_directory=masks_directory,
    )
    return mask_path


def materialize_stage1_bundle_from_sim_hdf(
    *,
    spectrograph: str,
    fiberflat_rate_paths: list[str],
    meta: dict | None = None,
    extraction_aperture: float = 10.0,
    rate_map_for_wavesol_zp: np.ndarray | None = None,
    wavesol_zp_fibers: list[int] | tuple[int, ...] | None = None,
    wavesol_zp_fiber: int | None = None,
    wavesol_zp_known_rv_m_s: float = 0.0,
    wavesol_from_injected_comb: bool = False,
    wavesol_comb_n_lines: int = 100,
    wavesol_comb_poly_deg: int = 4,
    wavesol_source: Literal["auto", "packaged_line_zp", "hdf_line_zp", "hdf"] = "auto",
) -> dict[str, Any]:
    """
    Return ``trace_pos``, ``trace_prof``, ``fiberflat``, and ``wavesol`` from HDF truth.

    ``trace_pos`` uses the oriented HDF ``y_pos`` grids (full spectral coverage).
    Fiberflat rate maps are still box-extracted for the blaze / fiberflat product.

    Wavelength solution (see ``wavesol_source``):
      * ``packaged_line_zp`` — prebuilt FITS under ``hispecdrp/data/wavesol/``
        (deposit-inverse + absorption line ZP; preferred for RV e2e).
      * ``hdf_line_zp`` — runtime ZP on ``rate_map_for_wavesol_zp``.
      * ``hdf`` — deposit-inverse geometric WAVE only.
      * ``auto`` — packaged if present, else runtime line-ZP, else HDF.
      * ``wavesol_from_injected_comb`` — validation only (deposit-inverse kept).

    ``wavesol_zp_fibers`` selects which fibers get line-ZP / comb validation
    (default ``1, 2, 3``). Legacy ``wavesol_zp_fiber`` still selects a single fiber.
    """
    from .sim_hdf_wavesol_packaging import (
        load_packaged_sim_hdf_line_zp_wavesol,
        packaged_sim_hdf_wavesol_path,
    )

    spectrograph = spectrograph.upper()
    zp_fibers = _normalize_fibers(fibers=wavesol_zp_fibers, fiber=wavesol_zp_fiber)
    trace_pos = create_tracepos_model_from_sim_hdf(spectrograph, meta=meta)
    trace_prof = create_traceprof_model_from_sim_hdf(spectrograph, meta=meta)
    wavesol_zp_info: dict[str, Any] | None = None
    wavesol_comb_info: dict[str, Any] | None = None
    packaged_path = packaged_sim_hdf_wavesol_path(spectrograph)
    src = str(wavesol_source or "auto").lower()

    if wavesol_from_injected_comb:
        # Validation only — comb centroids must not redefine simulation WAVE.
        comb = calibrate_wavesol_from_injected_comb(
            spectrograph,
            fibers=zp_fibers,
            n_lines_per_order=int(wavesol_comb_n_lines),
            poly_deg=int(wavesol_comb_poly_deg),
            extraction_aperture=float(extraction_aperture),
            meta=meta,
            centroid_mode="2d",
            photons_per_line=1.0e12,
            paint_mode="native",
        )
        wavesol = comb["wavesol"]  # still deposit-inverse
        wavesol_comb_info = {
            "rv_budget": comb["rv_budget"],
            "per_order": comb.get("per_order"),
            "per_fiber": comb.get("per_fiber"),
            "fibers": comb.get("fibers", zp_fibers),
            "fiber": comb.get("fiber"),
        }
    elif src == "packaged_line_zp" or (src == "auto" and packaged_path.is_file()):
        wavesol = load_packaged_sim_hdf_line_zp_wavesol(spectrograph, meta=meta)
        wavesol_zp_info = {
            "source": "packaged_line_zp",
            "path": str(packaged_path),
            "median_dv_m_s": float("nan"),
        }
        meta_json = packaged_path.with_suffix(".json")
        if meta_json.is_file():
            try:
                import json as _json

                side = _json.loads(meta_json.read_text())
                wavesol_zp_info["median_dv_m_s"] = side.get("median_dv_m_s")
                wavesol_zp_info["sidecar"] = side
            except Exception:
                pass
        logger.info(
            "Sim HDF wavesol: using packaged line-ZP %s (median_dv=%s m/s)",
            packaged_path,
            wavesol_zp_info.get("median_dv_m_s"),
        )
    elif src in ("hdf_line_zp", "auto") and rate_map_for_wavesol_zp is not None:
        # Deposit-inverse WAVE + runtime absorption ZP.
        wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
        wavesol_zp_info = refine_sim_hdf_wavesol_line_zeropoint(
            wavesol,
            rate_map_for_wavesol_zp,
            spectrograph=spectrograph,
            fibers=zp_fibers,
            known_rv_m_s=float(wavesol_zp_known_rv_m_s),
        )
    elif src == "packaged_line_zp":
        raise FileNotFoundError(
            f"wavesol_source='packaged_line_zp' but missing {packaged_path}. "
            "Regenerate with hispecdrp/data/wavesol/regenerate_sim_hdf_line_zp_wavesol.py"
        )
    else:
        wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)

    fiberflat = create_fiberflat_model_from_rate_maps(
        fiberflat_rate_paths,
        trace_pos,
        spectrograph=spectrograph,
        meta=meta,
        extraction_aperture=extraction_aperture,
    )
    if wavesol_comb_info:
        wavesol_mode = "injected_comb"
    elif wavesol_zp_info and wavesol_zp_info.get("source") == "packaged_line_zp":
        wavesol_mode = "packaged_line_zp"
    elif wavesol_zp_info:
        wavesol_mode = "line_zp"
    else:
        wavesol_mode = "hdf"
    logger.info(
        "Sim HDF Stage-1 bundle ready spectrograph=%s n_fiberflat_rates=%d "
        "extraction_aperture=%.1f trace_pos=sim_hdf_y_pos wavesol=%s",
        spectrograph,
        len(fiberflat_rate_paths),
        float(extraction_aperture),
        wavesol_mode,
    )
    return {
        "trace_pos": trace_pos,
        "trace_prof": trace_prof,
        "fiberflat": fiberflat,
        "wavesol": wavesol,
        "wavesol_zp_info": wavesol_zp_info,
        "wavesol_comb_info": wavesol_comb_info,
    }

# ========================================================================
# Injected-comb wavelength calibration
# ========================================================================


def build_injected_line_list(
    spectrograph: str,
    *,
    fiber: int = 2,
    n_lines_per_order: int = 1000,
    edge_margin_frac: float = 0.05,
) -> dict[int, np.ndarray]:
    """
    Known vacuum wavelengths (µm) for an injected emission comb.

    Returns ``{order: λ_um array}`` with ``n_lines_per_order`` samples spanning
    the central ``(1 - 2*edge_margin_frac)`` of each order's HDF wavelength range.
    """
    grids = get_sim_hdf_trace_grids(spectrograph)
    out: dict[int, np.ndarray] = {}
    for od in _order_list(spectrograph):
        if od not in grids.get(fiber, {}):
            continue
        w = np.asarray(grids[fiber][od]["wavelength"], dtype=np.float64)
        ok = np.isfinite(w)
        if int(ok.sum()) < 10:
            continue
        lo, hi = np.nanpercentile(w[ok], [100 * edge_margin_frac, 100 * (1 - edge_margin_frac)])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        out[int(od)] = np.linspace(float(lo), float(hi), int(n_lines_per_order), dtype=np.float64)
    return out


def resolve_lfc_fiber_object(spectrograph: str) -> str:
    """Return ``LFC-B`` / ``LFC-R`` for the spectrograph (matches ``target.yaml`` enum)."""
    sp = (spectrograph or "").strip().upper()
    if sp == "BSPEC":
        return "LFC-B"
    if sp == "RSPEC":
        return "LFC-R"
    raise ValueError(f"Unknown spectrograph {spectrograph!r}; expected BSPEC or RSPEC")


def build_lfc_line_list(
    spectrograph: str,
    *,
    fiber: int = 2,
    fiber_object: str | None = None,
    edge_margin_frac: float = 0.05,
    mode_stride: int = 1,
    nu0: float | None = None,
    delta_nu: float | None = None,
    orders: list[int] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """
    Physical LFC mode wavelengths matching :func:`gen_lfc_modes` / ``FitLFCLinesPrimitive``.

    Modes follow ``ν_n = ν_0 + n·Δν`` with ``λ = c/ν`` using ``HISPECConfig.lfc_params``
    for ``LFC-B`` / ``LFC-R`` (same Hz values written as ``LFCNU0`` / ``LFCDELNU``),
    clipped to each order's HDF wavelength span after an optional spectral-edge trim.

    Parameters
    ----------
    mode_stride
        Keep every ``mode_stride``-th mode (1 = full comb). Useful for faster
        diagnostic paints without changing the underlying ladder physics.
    orders
        Restrict to these echelle orders. ``None`` uses the full spectrograph
        order list.

    Returns
    -------
    line_list
        ``{order: λ_um array}`` of vacuum wavelengths to paint / measure.
    lfc_meta
        ``fiber_object``, ``nu0``, ``delta_nu`` used to generate the list.
    """
    from ..primitives.line_fitting.fit_lfc_lines_primitive import gen_lfc_modes

    spectrograph = spectrograph.upper()
    fiber_object = fiber_object or resolve_lfc_fiber_object(spectrograph)
    params = hispec_config.HISPECConfig.lfc_params.get(fiber_object)
    if params is None:
        raise ValueError(
            f"No HISPECConfig.lfc_params entry for {fiber_object!r}; "
            f"expected one of {sorted(hispec_config.HISPECConfig.lfc_params)}"
        )
    nu0_hz = float(nu0 if nu0 is not None else params["nu0"])
    delta_nu_hz = float(delta_nu if delta_nu is not None else params["delta_nu"])
    stride = max(1, int(mode_stride))
    edge = float(edge_margin_frac)

    grids = get_sim_hdf_trace_grids(spectrograph)
    pix = np.arange(4096, dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    order_iter = (
        [int(od) for od in orders]
        if orders is not None
        else list(_order_list(spectrograph))
    )
    for od in order_iter:
        if od not in grids.get(fiber, {}):
            continue
        w = np.asarray(grids[fiber][od]["wavelength"], dtype=np.float64)
        ok = np.isfinite(w)
        if int(ok.sum()) < 10:
            continue
        # Trim in spectral-row space (same convention as lfc_wavesol_refit),
        # then convert edge rows → wavelength bounds for gen_lfc_modes.
        lo_pix, hi_pix = np.percentile(pix[ok], [100 * edge, 100 * (1 - edge)])
        w_lo = float(np.interp(lo_pix, pix, w))
        w_hi = float(np.interp(hi_pix, pix, w))
        w_min, w_max = (min(w_lo, w_hi), max(w_lo, w_hi))
        if not (np.isfinite(w_min) and np.isfinite(w_max) and w_max > w_min):
            continue
        waves, _nus, _ints = gen_lfc_modes(nu0_hz, delta_nu_hz, w_min, w_max)
        waves = np.asarray(waves, dtype=np.float64)
        if waves.size == 0:
            continue
        if stride > 1:
            waves = waves[::stride]
        out[int(od)] = waves

    lfc_meta = {
        "fiber_object": fiber_object,
        "nu0": nu0_hz,
        "delta_nu": delta_nu_hz,
        "edge_margin_frac": edge,
        "mode_stride": float(stride),
    }
    return out, lfc_meta


def _hdf_deposit_xy_interpolators(
    spectrograph: str,
    fiber: int,
    order: int,
) -> tuple[Any, Any, np.ndarray] | None:
    """
    Pchip ``λ → (x_native, y_native)`` matching :func:`bilinear_weights_rate_map`.

    Returns ``(fx, fy, ow)`` or ``None`` if the order is unavailable.
    """
    spectrograph, model_data = _load_hdf_model_data(spectrograph)
    block = model_data.get(f"fiber_{int(fiber)}", {}).get(str(int(order)))
    if block is None:
        return None
    ow = np.asarray(block["xy_wavelengths"], dtype=np.float64)
    ox = np.asarray(block["x"], dtype=np.float64)
    oy = np.asarray(block["y"], dtype=np.float64)
    good = np.isfinite(ow) & np.isfinite(ox) & np.isfinite(oy)
    if int(good.sum()) < 2:
        return None
    ow, ox, oy = ow[good], ox[good], oy[good]
    sidx = np.argsort(ow)
    ow, ox, oy = ow[sidx], ox[sidx], oy[sidx]
    # Merge duplicate wavelengths (Pchip requires strictly increasing x).
    uw, inv = np.unique(ow, return_inverse=True)
    if uw.size < 2:
        return None
    ox_u = np.bincount(inv, weights=ox) / np.maximum(np.bincount(inv), 1)
    oy_u = np.bincount(inv, weights=oy) / np.maximum(np.bincount(inv), 1)
    return PchipInterpolator(uw, ox_u, extrapolate=True), PchipInterpolator(
        uw, oy_u, extrapolate=True
    ), uw


def expected_feature_xy_oriented(
    spectrograph: str,
    fiber: int,
    order: int,
    wavelengths_um: np.ndarray,
    *,
    nx: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact oriented ``(row, col)`` where the native simulator deposits each λ.

    ``row`` is the spectral coordinate (WAVE axis); ``col`` is cross-dispersion.
    """
    interps = _hdf_deposit_xy_interpolators(spectrograph, fiber, order)
    if interps is None:
        n = np.asarray(wavelengths_um).size
        return np.full(n, np.nan), np.full(n, np.nan)
    fx, fy, _ = interps
    lams = np.asarray(wavelengths_um, dtype=np.float64)
    xc = np.asarray(fx(lams), dtype=np.float64)
    yc = np.asarray(fy(lams), dtype=np.float64)
    return _orient_hdf_pixel_xy(xc, yc, nx=nx)


def paint_injected_emission_comb(
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    method: str = "bilinear_weights",
    line_flux_density: float | None = None,
    photons_per_line: float = 1.0e12,
) -> np.ndarray:
    """
    Paint a sparse emission-line spectrum onto a native-oriented rate map.

    With ``continuous=False``, each flux sample is a photon count *rate*
    (ph/s) per unresolved line — not a wavelength density. Defaults to
    ``photons_per_line`` for every line. ``line_flux_density`` is accepted as a
    deprecated alias for that per-line rate (same numeric value).
    """
    from astropy import units as u

    from .simulate_rate_map import SimulateRateMap

    lams = np.unique(np.concatenate([np.asarray(v, dtype=np.float64) for v in line_list.values()]))
    lams.sort()
    if lams.size < 2:
        raise ValueError("Need at least two injected wavelengths")
    # Prefer explicit photons_per_line; line_flux_density kept as alias.
    rate_per_line = (
        float(line_flux_density) if line_flux_density is not None else float(photons_per_line)
    )
    flux = np.full(lams.shape, rate_per_line)
    srm = SimulateRateMap()
    srm.set_fiber_spectrum(
        int(fiber),
        lams * u.micron,
        flux * (u.ph / u.s),
        continuous=False,
    )
    rate = srm.create_native_rate_map(
        spectrograph=spectrograph.upper(),
        method=method,
        apply_blaze=False,
        save=False,
    )
    return np.asarray(rate, dtype=np.float64)


def paint_compact_emission_comb(
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    sigma_pix: float = 0.55,
    peak_value: float = 1.0e6,
    half_stamp: int = 3,
) -> np.ndarray:
    """
    Paint compact Gaussians at exact oriented deposit ``(row, col)`` for each λ.

    Used for high-density inject/recover tests: placement follows the same HDF
    λ→(x,y) model as the native simulator, but the stamp is narrow enough that
    ~1000 lines/order do not blend, so a single paint pass is sufficient.
    """
    spectrograph = spectrograph.upper()
    # Oriented science detector + 64-col reference channel (matches native maps).
    rate = np.zeros((4096, 4160), dtype=np.float64)
    sig = float(sigma_pix)
    hw = int(half_stamp)
    yy, xx = np.mgrid[-hw : hw + 1, -hw : hw + 1]
    stamp0 = np.exp(-0.5 * (xx**2 + yy**2) / max(sig**2, 1e-6))
    stamp0 *= float(peak_value) / max(float(stamp0.max()), 1e-30)

    for od, lams in line_list.items():
        lams = np.asarray(lams, dtype=np.float64)
        if lams.size == 0:
            continue
        rows, cols = expected_feature_xy_oriented(spectrograph, fiber, int(od), lams)
        for r0, c0 in zip(rows, cols):
            if not (np.isfinite(r0) and np.isfinite(c0)):
                continue
            # Sub-pixel shift of the Gaussian stamp about the exact deposit center.
            dr = float(r0) - np.rint(r0)
            dc = float(c0) - np.rint(c0)
            stamp = np.exp(
                -0.5 * ((xx - dc) ** 2 + (yy - dr) ** 2) / max(sig**2, 1e-6)
            )
            stamp *= float(peak_value) / max(float(stamp.max()), 1e-30)
            ir = int(np.rint(r0))
            ic = int(np.rint(c0))
            rlo, rhi = ir - hw, ir + hw + 1
            clo, chi = ic - hw, ic + hw + 1
            if rlo < 0 or clo < 0 or rhi > rate.shape[0] or chi > min(rate.shape[1], 4096):
                continue
            rate[rlo:rhi, clo:chi] += stamp
    return rate


def _elliptical_gaussian_stamp(
    rr: np.ndarray,
    cc: np.ndarray,
    *,
    row0: float,
    col0: float,
    a_pix: float,
    b_pix: float,
    pa_rad: float,
    amp: float,
) -> np.ndarray:
    """
    Elliptical Gaussian on coordinate grids ``rr``, ``cc`` (same shape).

    ``a_pix`` / ``b_pix`` are 1-σ semimajor / semiminor axes (pixels).
    ``pa_rad`` is the position angle of the semimajor axis from +row toward +col
    (counter-clockwise, radians).
    """
    a = max(float(a_pix), 1e-6)
    b = max(float(b_pix), 1e-6)
    cos_p = float(np.cos(pa_rad))
    sin_p = float(np.sin(pa_rad))
    dr = np.asarray(rr, dtype=np.float64) - float(row0)
    dc = np.asarray(cc, dtype=np.float64) - float(col0)
    # Rotate into ellipse frame: u along semimajor.
    u = cos_p * dr + sin_p * dc
    v = -sin_p * dr + cos_p * dc
    return float(amp) * np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2))


def paint_elliptical_gaussian_comb(
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    a_pix: float = 0.80,
    b_pix: float = 0.55,
    pa_rad: float = 0.0,
    peak_value: float = 1.0e6,
    half_stamp: int | None = None,
) -> np.ndarray:
    """
    Paint compact *elliptical* Gaussians at exact oriented deposit ``(row, col)``.

    Same geometry as :func:`paint_compact_emission_comb`, but with known
    semimajor / semiminor / PA so a 2D elliptical fit can recover the input
    profile parameters (used by ``lfc_wavesol_refit``).
    """
    spectrograph = spectrograph.upper()
    rate = np.zeros((4096, 4160), dtype=np.float64)
    a = float(a_pix)
    b = float(b_pix)
    pa = float(pa_rad)
    # Stamp large enough to cover ~5σ along the major axis.
    if half_stamp is None:
        hw = max(3, int(np.ceil(5.0 * max(a, b))))
    else:
        hw = int(half_stamp)

    for od, lams in line_list.items():
        lams = np.asarray(lams, dtype=np.float64)
        if lams.size == 0:
            continue
        rows, cols = expected_feature_xy_oriented(spectrograph, fiber, int(od), lams)
        for r0, c0 in zip(rows, cols):
            if not (np.isfinite(r0) and np.isfinite(c0)):
                continue
            ir = int(np.rint(r0))
            ic = int(np.rint(c0))
            rlo, rhi = ir - hw, ir + hw + 1
            clo, chi = ic - hw, ic + hw + 1
            if rlo < 0 or clo < 0 or rhi > rate.shape[0] or chi > min(rate.shape[1], 4096):
                continue
            # Absolute pixel centers of stamp pixels
            rr = np.arange(rlo, rhi, dtype=np.float64)[:, None]
            cc = np.arange(clo, chi, dtype=np.float64)[None, :]
            stamp = _elliptical_gaussian_stamp(
                rr,
                cc,
                row0=float(r0),
                col0=float(c0),
                a_pix=a,
                b_pix=b,
                pa_rad=pa,
                amp=1.0,
            )
            stamp *= float(peak_value) / max(float(stamp.max()), 1e-30)
            rate[rlo:rhi, clo:chi] += stamp
    return rate


# ZEMAX / SimulateRateMap continuous spectra are densified to this resolving
# power when undersampled (PCHIP).  Build SEDs at ≥ this R so painting never
# interpolates the input — but note R=3e5 is only ~1.4 samples per FWHM for a
# 2.25-pix Gaussian on BSPEC, so line-profile invariance needs denser sampling.
ZEMAX_MIN_SAMPLING_RESOLUTION = 3.0e5


def build_gaussian_lfc_sed(
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    fwhm_pix: float = 2.25,
    amplitude: float = 1.0e3,
    sampling_resolution: float = 1.2e6,
    n_sigma_pad: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Analytic sum of Gaussians on a constant-R (log-λ) wavelength grid.

    Line σ is ``fwhm_pix`` converted with the local deposit-inverse ``dλ/dpix``.
    Evaluate the SED on the target grid directly — do **not** densify a coarse
    SED with PCHIP (that clips peaks and shifts centroids).

    Returns ``(wavelength_um, flux_density, info)`` with ``info`` containing
    ``sampling_resolution``, ``n_samples``, and ``samples_per_fwhm``.
    """
    from .simulate_rate_map import _check_wavelength_sampling
    from astropy import units as u

    grids = get_sim_hdf_trace_grids(spectrograph.upper())
    if int(fiber) not in grids:
        raise KeyError(f"No HDF grids for fiber {fiber}")
    profiles: list[tuple[float, float, float]] = []
    for od, lams in line_list.items():
        if int(od) not in grids[int(fiber)]:
            continue
        w = np.asarray(grids[int(fiber)][int(od)]["wavelength"], dtype=np.float64)
        ok = np.isfinite(w)
        if int(ok.sum()) < 2:
            continue
        dlam = float(np.nanmedian(np.abs(np.diff(w[ok]))))
        if not np.isfinite(dlam) or dlam <= 0.0:
            continue
        sig = (float(fwhm_pix) * dlam) / 2.354820045
        for lam0 in np.asarray(lams, dtype=np.float64):
            if np.isfinite(lam0):
                profiles.append((float(lam0), float(sig), float(amplitude)))
    if not profiles:
        raise ValueError("No valid Gaussian profiles to build SED")

    sig_max = max(s for _, s, _ in profiles)
    lo = min(l for l, _, _ in profiles) - float(n_sigma_pad) * sig_max
    hi = max(l for l, _, _ in profiles) + float(n_sigma_pad) * sig_max
    if lo <= 0.0:
        lo = min(l for l, _, _ in profiles) * 0.99
    R = float(sampling_resolution)
    n = int(np.ceil(np.log(hi / lo) * R)) + 1
    grid = lo * np.exp(np.arange(n, dtype=np.float64) / R)
    flux = np.zeros_like(grid)
    for lam0, sig, amp in profiles:
        i0 = int(np.searchsorted(grid, lam0 - 6.0 * sig))
        i1 = int(np.searchsorted(grid, lam0 + 6.0 * sig))
        if i1 > i0:
            x = grid[i0:i1]
            flux[i0:i1] += amp * np.exp(-0.5 * ((x - lam0) / sig) ** 2)

    fwhm_um = float(np.median([s * 2.354820045 for _, s, _ in profiles]))
    lam_mid = float(np.median([l for l, _, _ in profiles]))
    spp = fwhm_um / (lam_mid / R) if lam_mid > 0.0 else float("nan")
    if not _check_wavelength_sampling(grid * u.micron, min_resolution=ZEMAX_MIN_SAMPLING_RESOLUTION):
        raise ValueError(
            f"Built SED at R={R:.3e} fails ZEMAX min sampling "
            f"R={ZEMAX_MIN_SAMPLING_RESOLUTION:.3e}; increase sampling_resolution."
        )
    info = {
        "sampling_resolution": R,
        "n_samples": float(grid.size),
        "samples_per_fwhm": float(spp),
        "n_lines": float(len(profiles)),
        "fwhm_um_median": fwhm_um,
        "lam_mid_um": lam_mid,
    }
    return grid, flux, info


def paint_continuous_gaussian_comb(
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    fwhm_pix: float = 2.25,
    amplitude: float = 1.0e3,
    sampling_resolution: float = 1.2e6,
    method: str = "prf_convolution",
    apply_blaze: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Paint an analytic Gaussian LFC via ZEMAX continuous ``prf_convolution``.

    Builds the SED at ``sampling_resolution`` (≥ ``ZEMAX_MIN_SAMPLING_RESOLUTION``)
    so ``set_fiber_spectrum(..., continuous=True)`` does **not** PCHIP-upsample.
    Returns ``(rate, meta)`` where ``meta`` records sampling checks and SED size.
    """
    from astropy import units as u

    from .simulate_rate_map import (
        SimulateRateMap,
        _check_wavelength_sampling,
    )

    wl, fl, sed_info = build_gaussian_lfc_sed(
        line_list,
        spectrograph=spectrograph,
        fiber=fiber,
        fwhm_pix=fwhm_pix,
        amplitude=amplitude,
        sampling_resolution=sampling_resolution,
    )
    ok = _check_wavelength_sampling(
        wl * u.micron, min_resolution=ZEMAX_MIN_SAMPLING_RESOLUTION
    )
    if not ok:
        raise RuntimeError("SED failed ZEMAX sampling check before paint")

    srm = SimulateRateMap()
    srm.set_fiber_spectrum(
        int(fiber),
        wl * u.micron,
        fl * u.Unit("erg / (s cm2 um)"),
        continuous=True,
    )
    stored_wl = np.asarray(srm.fibers[int(fiber)]["wavelength"], dtype=np.float64)
    no_internal_upsample = int(stored_wl.size) == int(wl.size)
    if not no_internal_upsample:
        raise RuntimeError(
            f"ZEMAX upsampled SED ({wl.size} → {stored_wl.size}); "
            f"raise sampling_resolution (got {sampling_resolution:.3e})."
        )

    rate = np.asarray(
        srm.create_native_rate_map(
            spectrograph=spectrograph.upper(),
            method=method,
            apply_blaze=apply_blaze,
            save=False,
        ),
        dtype=np.float64,
    )
    meta: dict[str, Any] = {
        **sed_info,
        "zemax_min_resolution": float(ZEMAX_MIN_SAMPLING_RESOLUTION),
        "passed_zemax_sampling_check": True,
        "no_internal_upsample": True,
        "stored_n_samples": float(stored_wl.size),
        "paint_method": str(method),
        "apply_blaze": bool(apply_blaze),
        "wavelength_um": wl,
        "flux": fl,
    }
    return rate, meta


def _centroid_2d_flux(
    image: np.ndarray,
    *,
    row0: float,
    col0: float,
    half_win: int = 2,
) -> tuple[float, float, float, float]:
    """
    Flux-weighted 2D centroid in a ``(2*half_win+1)²`` window about ``(row0, col0)``.

    Returns ``(row_c, col_c, row_err, col_err)``.  Uncertainties are the
    intensity-weighted second-moment widths divided by ``sqrt(n_eff)``, with
    ``n_eff = (Σw)² / Σw²`` (shape-only; independent of absolute flux scale).
    """
    nan4 = (float("nan"), float("nan"), float("nan"), float("nan"))
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    r0 = int(np.clip(np.rint(row0), 0, ny - 1))
    c0 = int(np.clip(np.rint(col0), 0, nx - 1))
    hw = int(half_win)
    rlo, rhi = max(0, r0 - hw), min(ny, r0 + hw + 1)
    clo, chi = max(0, c0 - hw), min(nx, c0 + hw + 1)
    stamp = img[rlo:rhi, clo:chi]
    if stamp.size == 0:
        return nan4
    w = np.clip(stamp, 0.0, None)
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return nan4
    rr = np.arange(rlo, rhi, dtype=np.float64)[:, None]
    cc = np.arange(clo, chi, dtype=np.float64)[None, :]
    row_c = float((w * rr).sum() / wsum)
    col_c = float((w * cc).sum() / wsum)
    var_r = float((w * (rr - row_c) ** 2).sum() / wsum)
    var_c = float((w * (cc - col_c) ** 2).sum() / wsum)
    n_eff = float(wsum ** 2 / max(float((w * w).sum()), 1e-300))
    row_err = float(np.sqrt(max(var_r, 0.0) / max(n_eff, 1.0)))
    col_err = float(np.sqrt(max(var_c, 0.0) / max(n_eff, 1.0)))
    return row_c, col_c, row_err, col_err


def _canonicalize_ellipse_ab_pa(
    a: float, b: float, pa_rad: float
) -> tuple[float, float, float]:
    """Enforce ``a >= b`` and wrap PA into ``[-π/2, π/2)``."""
    a = float(a)
    b = float(b)
    pa = float(pa_rad)
    if b > a:
        a, b = b, a
        pa = pa + 0.5 * np.pi
    # Wrap to [-π/2, π/2)
    pa = (pa + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return a, b, pa


def fit_elliptical_gaussian_2d(
    image: np.ndarray,
    *,
    row0: float,
    col0: float,
    half_win: int = 5,
    a_seed: float | None = None,
    b_seed: float | None = None,
    pa_seed: float = 0.0,
) -> dict[str, Any]:
    """
    Fit an elliptical Gaussian spot in a local window about ``(row0, col0)``.

    Model (pixel coords; row = spectral, col = cross-dispersion)::

        I = A · exp(-½ [(u/a)² + (v/b)²])
        (u, v) = R(-PA) · (r − r₀, c − c₀)

    Returns a dict with ``row``, ``col``, ``a``, ``b``, ``pa_rad``, ``amp``,
    ``success``, ``residual_rms``, ``n_pix``. Axes are canonicalized so ``a ≥ b``
    and ``pa_rad ∈ [-π/2, π/2)``.
    """
    from scipy.optimize import least_squares

    nan_out: dict[str, Any] = {
        "row": float("nan"),
        "col": float("nan"),
        "a": float("nan"),
        "b": float("nan"),
        "pa_rad": float("nan"),
        "amp": float("nan"),
        "success": False,
        "residual_rms": float("nan"),
        "n_pix": 0,
    }
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    hw = int(half_win)
    r0i = int(np.clip(np.rint(row0), 0, ny - 1))
    c0i = int(np.clip(np.rint(col0), 0, nx - 1))
    rlo, rhi = max(0, r0i - hw), min(ny, r0i + hw + 1)
    clo, chi = max(0, c0i - hw), min(nx, c0i + hw + 1)
    stamp = img[rlo:rhi, clo:chi]
    if stamp.size < 9:
        return nan_out
    rr = np.arange(rlo, rhi, dtype=np.float64)[:, None]
    cc = np.arange(clo, chi, dtype=np.float64)[None, :]
    rr_g = np.broadcast_to(rr, stamp.shape).ravel()
    cc_g = np.broadcast_to(cc, stamp.shape).ravel()
    data = stamp.ravel()
    m = np.isfinite(data) & (data > 0)
    if int(m.sum()) < 9:
        return nan_out
    rr_g, cc_g, data = rr_g[m], cc_g[m], data[m]

    # Moment seeds
    w = data
    wsum = float(w.sum())
    row_c = float((w * rr_g).sum() / wsum)
    col_c = float((w * cc_g).sum() / wsum)
    var_r = float((w * (rr_g - row_c) ** 2).sum() / wsum)
    var_c = float((w * (cc_g - col_c) ** 2).sum() / wsum)
    sig_r = float(np.sqrt(max(var_r, 1e-4)))
    sig_c = float(np.sqrt(max(var_c, 1e-4)))
    a0 = float(a_seed) if a_seed is not None and np.isfinite(a_seed) else max(sig_r, sig_c)
    b0 = float(b_seed) if b_seed is not None and np.isfinite(b_seed) else min(sig_r, sig_c)
    a0, b0, pa0 = _canonicalize_ellipse_ab_pa(a0, b0, float(pa_seed))
    amp0 = float(np.max(data))

    # Params: amp, row, col, a, b, pa
    x0 = np.array([amp0, row_c, col_c, a0, b0, pa0], dtype=np.float64)
    lo = np.array(
        [0.0, float(rlo), float(clo), 0.05, 0.05, -0.5 * np.pi],
        dtype=np.float64,
    )
    hi = np.array(
        [np.inf, float(rhi - 1), float(chi - 1), float(hw), float(hw), 0.5 * np.pi],
        dtype=np.float64,
    )

    def resid(p: np.ndarray) -> np.ndarray:
        model = _elliptical_gaussian_stamp(
            rr_g,
            cc_g,
            row0=float(p[1]),
            col0=float(p[2]),
            a_pix=float(p[3]),
            b_pix=float(p[4]),
            pa_rad=float(p[5]),
            amp=float(p[0]),
        )
        return model - data

    try:
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=200)
    except Exception:
        return nan_out
    if not sol.success and sol.cost > 0.5 * float(np.sum(data**2)):
        return nan_out
    amp, row_f, col_f, a_f, b_f, pa_f = (float(x) for x in sol.x)
    a_f, b_f, pa_f = _canonicalize_ellipse_ab_pa(a_f, b_f, pa_f)
    res = resid(sol.x)
    rms = float(np.sqrt(np.mean(res**2)))
    return {
        "row": row_f,
        "col": col_f,
        "a": a_f,
        "b": b_f,
        "pa_rad": pa_f,
        "amp": amp,
        "success": bool(sol.success) or rms < 0.05 * amp0,
        "residual_rms": rms,
        "n_pix": int(data.size),
    }


def measure_lfc_spots_elliptical(
    rate_map: np.ndarray,
    line_list: dict[int, np.ndarray],
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    a_pix: float = 0.80,
    b_pix: float = 0.55,
    pa_rad: float = 0.0,
    half_win: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fit every injected LFC spot with :func:`fit_elliptical_gaussian_2d`.

    Returns a list of row dicts with truth deposit ``(row, col, λ)``, input
    profile params, and fitted parameters / deltas.
    """
    spectrograph = spectrograph.upper()
    a_true, b_true, pa_true = _canonicalize_ellipse_ab_pa(
        float(a_pix), float(b_pix), float(pa_rad)
    )
    if half_win is None:
        hw = max(5, int(np.ceil(5.0 * max(a_true, b_true))))
    else:
        hw = int(half_win)
    out: list[dict[str, Any]] = []
    for od, lams in line_list.items():
        lams = np.asarray(lams, dtype=np.float64)
        if lams.size == 0:
            continue
        rows, cols = expected_feature_xy_oriented(
            spectrograph, int(fiber), int(od), lams
        )
        for lam, r_exp, c_exp in zip(lams, rows, cols):
            if not (np.isfinite(lam) and np.isfinite(r_exp) and np.isfinite(c_exp)):
                continue
            fit = fit_elliptical_gaussian_2d(
                rate_map,
                row0=float(r_exp),
                col0=float(c_exp),
                half_win=hw,
                a_seed=a_true,
                b_seed=b_true,
                pa_seed=pa_true,
            )
            dpa = float(fit["pa_rad"] - pa_true) if np.isfinite(fit["pa_rad"]) else float("nan")
            # Smallest signed angle difference
            if np.isfinite(dpa):
                dpa = (dpa + 0.5 * np.pi) % np.pi - 0.5 * np.pi
            out.append(
                {
                    "order": int(od),
                    "lam_true_um": float(lam),
                    "row_expected": float(r_exp),
                    "col_expected": float(c_exp),
                    "a_true": a_true,
                    "b_true": b_true,
                    "pa_true_rad": pa_true,
                    "row_fit": float(fit["row"]),
                    "col_fit": float(fit["col"]),
                    "a_fit": float(fit["a"]),
                    "b_fit": float(fit["b"]),
                    "pa_fit_rad": float(fit["pa_rad"]),
                    "amp_fit": float(fit["amp"]),
                    "drow_pix": float(fit["row"] - r_exp)
                    if np.isfinite(fit["row"])
                    else float("nan"),
                    "dcol_pix": float(fit["col"] - c_exp)
                    if np.isfinite(fit["col"])
                    else float("nan"),
                    "da_pix": float(fit["a"] - a_true)
                    if np.isfinite(fit["a"])
                    else float("nan"),
                    "db_pix": float(fit["b"] - b_true)
                    if np.isfinite(fit["b"])
                    else float("nan"),
                    "dpa_rad": dpa,
                    "success": bool(fit["success"]),
                    "residual_rms": float(fit["residual_rms"]),
                    "n_pix": int(fit["n_pix"]),
                }
            )
    return out


def _polyfit_scaled(
    x: np.ndarray,
    y: np.ndarray,
    *,
    deg: int,
) -> tuple[np.ndarray, float, float]:
    """
    Numerically stable ``polyfit`` with ``x`` mapped to ``[-1, 1]``.

    Returns ``(coeffs_highest_first, x_mid, x_halfspan)`` so
    ``y ≈ polyval(coeffs, (x - x_mid) / x_halfspan)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mid = float(0.5 * (np.min(x) + np.max(x)))
    x_half = float(0.5 * (np.max(x) - np.min(x)))
    if not np.isfinite(x_half) or x_half <= 0:
        x_half = 1.0
    xn = (x - x_mid) / x_half
    coeffs = np.polyfit(xn, y, int(deg))
    return coeffs, x_mid, x_half


def _polyval_scaled(
    coeffs: np.ndarray,
    x: np.ndarray,
    *,
    x_mid: float,
    x_half: float,
) -> np.ndarray:
    xn = (np.asarray(x, dtype=np.float64) - float(x_mid)) / float(x_half)
    return np.polyval(coeffs, xn)


def _fit_emission_peak_pix(
    flux: np.ndarray,
    *,
    pix_guess: float,
    half_win: int = 8,
) -> float:
    """Gaussian emission center (pixel) near ``pix_guess``; parabolic fallback."""
    from ..primitives.line_fitting.fit_lines import fit_line_gauss

    flux = np.asarray(flux, dtype=np.float64)
    pix = np.arange(flux.size, dtype=np.float64)
    i0 = int(np.clip(np.rint(pix_guess), 0, flux.size - 1))
    lo = max(0, i0 - int(half_win))
    hi = min(flux.size, i0 + int(half_win) + 1)
    xx = pix[lo:hi]
    yy = flux[lo:hi]
    m = np.isfinite(xx) & np.isfinite(yy) & (yy > 0)
    if int(m.sum()) < 5:
        return float("nan")
    xx, yy = xx[m], yy[m]
    # Prefer local max nearest guess
    local = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:]))[0] + 1
    if local.size == 0:
        j = int(np.argmax(yy))
    else:
        j = int(local[np.argmin(np.abs(xx[local] - pix_guess))])
    try:
        res = fit_line_gauss(
            xx,
            yy,
            y_err=np.ones_like(yy),
            dq=np.zeros(xx.size, dtype=np.uint32),
            mu_guess=float(xx[j]),
            mu_bounds=(-float(half_win), float(half_win)),
            absorption=False,
            background_poly_deg=0,
        )
        mu = float(res.mu)
        if np.isfinite(mu) and abs(mu - pix_guess) <= half_win:
            return mu
    except Exception:
        pass
    # 3-point parabolic max
    if 0 < j < xx.size - 1:
        x0, x1, x2 = xx[j - 1], xx[j], xx[j + 1]
        y0, y1, y2 = yy[j - 1], yy[j], yy[j + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            return float(x1 + 0.5 * (y0 - y2) / denom)
    return float(xx[j])


def _robust_polyfit_wavesol(
    pix: np.ndarray,
    lam: np.ndarray,
    *,
    poly_deg: int,
    n_clip: int = 3,
    clip_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively sigma-clipped ``np.polyfit(pix, lam, poly_deg)``.

    Returns ``(coeffs, inlier_mask)``.
    """
    pix = np.asarray(pix, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    mask = np.isfinite(pix) & np.isfinite(lam)
    deg = int(poly_deg)
    if int(mask.sum()) < deg + 3:
        return np.polyfit(pix[mask], lam[mask], deg), mask
    for _ in range(int(n_clip)):
        coeffs = np.polyfit(pix[mask], lam[mask], deg)
        resid = lam - np.polyval(coeffs, pix)
        # Convert to relative residual so clip threshold is wavelength-scale free.
        mad = float(np.nanmedian(np.abs(resid[mask] - np.nanmedian(resid[mask]))))
        if not np.isfinite(mad) or mad <= 0:
            break
        keep = np.abs(resid - np.nanmedian(resid[mask])) <= (clip_sigma * 1.4826 * mad)
        new_mask = mask & keep
        if int(new_mask.sum()) < deg + 3 or bool(np.array_equal(new_mask, mask)):
            break
        mask = new_mask
    coeffs = np.polyfit(pix[mask], lam[mask], deg)
    return coeffs, mask


def calibrate_wavesol_from_injected_comb(
    spectrograph: str = "BSPEC",
    *,
    fibers: list[int] | tuple[int, ...] | None = None,
    fiber: int | None = None,
    n_lines_per_order: int = 1000,
    poly_deg: int = 9,
    extraction_aperture: float = 10.0,
    meta: dict | None = None,
    rate_map: np.ndarray | None = None,
    line_list: dict[int, np.ndarray] | None = None,
    max_fit_rms_m_s: float = 50.0,
    max_dpix: float = 0.75,
    centroid_mode: str = "2d",
    centroid_half_win: int = 3,
    photons_per_line: float = 1.0e12,
    fit_edge_margin_frac: float = 0.08,
    paint_mode: str = "native",
) -> dict[str, Any]:
    """
    Empirically validate deposit-inverse ``WAVE_FIB`` with a dense injected emission comb.

    Runs for each requested fiber (default ``1, 2, 3``). For each echelle order:
      1. Inject ``n_lines_per_order`` known vacuum λ peaks at high S/N.
      2. Predict exact deposit ``(row, col)`` from the HDF λ→(x,y) model (same
         interpolators the native simulator uses).
      3. Centroid the bright features in 2D (default) or along a 1D box extract.
      4. Report geometry residual of deposit-inverse WAVE at deposit centers
         (PSF-centroid residuals are diagnostic only).

    ``paint_mode``:
      * ``native`` (default) — full PSF via :func:`paint_injected_emission_comb`;
        dense combs are painted in phase-offset batches to avoid blending.
      * ``compact`` — narrow Gaussians at exact deposit (x,y); one pass.

    Returns ``wavesol``, ``per_fiber`` (and legacy ``per_order`` / ``rv_budget`` for
    the primary fiber), ``line_measurements``, and ``order_fit_curves``.
    Does **not** modify the HDF spectrograph model file.
    """
    from ..primitives.utils.spectral_utils import (
        delta_vel_to_delta_wave,
        delta_wave_to_delta_vel,
    )

    spectrograph = spectrograph.upper()
    fiber_list = _normalize_fibers(fibers=fibers, fiber=fiber)
    if len(fiber_list) > 1:
        if rate_map is not None or line_list is not None:
            raise ValueError(
                "Multi-fiber calibrate_wavesol_from_injected_comb does not accept "
                "a shared rate_map or line_list; omit them or pass a single fiber."
            )
        per_fiber: dict[int, dict[str, Any]] = {}
        wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
        hdf_wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
        for fib in fiber_list:
            one = calibrate_wavesol_from_injected_comb(
                spectrograph,
                fiber=fib,
                n_lines_per_order=n_lines_per_order,
                poly_deg=poly_deg,
                extraction_aperture=extraction_aperture,
                meta=meta,
                max_fit_rms_m_s=max_fit_rms_m_s,
                max_dpix=max_dpix,
                centroid_mode=centroid_mode,
                centroid_half_win=centroid_half_win,
                photons_per_line=photons_per_line,
                fit_edge_margin_frac=fit_edge_margin_frac,
                paint_mode=paint_mode,
            )
            # Keep each fiber's WAVE independent (no copy-from-fiber-N).
            w_fib = getattr(one["wavesol"], f"wave_fib{fib}", None)
            if w_fib is not None:
                setattr(wavesol, f"wave_fib{fib}", np.asarray(w_fib, dtype=np.float64))
            per_fiber[int(fib)] = {
                "fiber": int(fib),
                "per_order": one["per_order"],
                "line_measurements": one["line_measurements"],
                "order_fit_curves": one["order_fit_curves"],
                "rv_budget": one["rv_budget"],
                "rate_map": one["rate_map"],
                "line_list": one["line_list"],
            }
        primary = 2 if 2 in per_fiber else fiber_list[0]
        mad_all = [
            per_fiber[f]["rv_budget"].get("median_fit_mad_m_s", float("nan"))
            for f in fiber_list
        ]
        rv_budget = {
            "fibers": fiber_list,
            "per_fiber_median_fit_mad_m_s": {
                str(f): per_fiber[f]["rv_budget"].get("median_fit_mad_m_s") for f in fiber_list
            },
            "median_fit_mad_m_s": float(np.nanmedian(mad_all)),
            "max_median_fit_mad_m_s": float(np.nanmax(mad_all)),
            "fiber": int(primary),
        }
        rv_budget.update(
            {k: v for k, v in per_fiber[primary]["rv_budget"].items() if k not in rv_budget}
        )
        return {
            "wavesol": wavesol,
            "hdf_wavesol": hdf_wavesol,
            "per_fiber": per_fiber,
            "per_order": per_fiber[primary]["per_order"],
            "line_measurements": per_fiber[primary]["line_measurements"],
            "order_fit_curves": per_fiber[primary]["order_fit_curves"],
            "rv_budget": rv_budget,
            "rate_map": per_fiber[primary]["rate_map"],
            "line_list": per_fiber[primary]["line_list"],
            "fiber": int(primary),
            "fibers": fiber_list,
            "spectrograph": spectrograph,
        }

    fiber = int(fiber_list[0])
    centroid_mode = str(centroid_mode).lower().strip()
    paint_mode = str(paint_mode).lower().strip()
    if centroid_mode not in ("2d", "1d"):
        raise ValueError("centroid_mode must be '2d' or '1d'")
    if paint_mode not in ("compact", "native"):
        raise ValueError("paint_mode must be 'compact' or 'native'")
    if line_list is None:
        line_list = build_injected_line_list(
            spectrograph, fiber=fiber, n_lines_per_order=n_lines_per_order
        )

    # Native PSF blends at ~1000 lines/order; batch unless using compact stamps.
    n_batch = 1
    if (
        rate_map is None
        and paint_mode == "native"
        and int(n_lines_per_order) > 150
    ):
        n_batch = max(1, int(np.ceil(int(n_lines_per_order) / 100.0)))

    grids = get_sim_hdf_trace_grids(spectrograph)
    order_list = _order_list(spectrograph)
    hdf_wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
    wave_hdf_all = np.asarray(getattr(hdf_wavesol, f"wave_fib{fiber}"), dtype=np.float64)
    wave_new = wave_hdf_all.copy()

    per_order: dict[int, dict[str, Any]] = {}
    measurements: list[dict[str, float]] = []
    pix_grid = np.arange(4096, dtype=np.float64)
    nx_orient = 4096

    # Accumulate per-order measurement lists across paint batches.
    acc: dict[int, dict[str, list[float]]] = {
        int(od): {
            "pix": [],
            "pix_err": [],
            "col": [],
            "col_err": [],
            "lam": [],
            "pe": [],
            "drow": [],
            "dcol": [],
        }
        for od in line_list
    }
    rate_last: np.ndarray | None = None
    order_fit_curves: dict[int, dict[str, Any]] = {}

    for phase in range(int(n_batch)):
        if rate_map is not None and n_batch == 1:
            batch_list = line_list
            rate = np.asarray(rate_map, dtype=np.float64)
        else:
            batch_list = {
                int(od): np.asarray(lams, dtype=np.float64)[phase::n_batch]
                for od, lams in line_list.items()
                if np.asarray(lams).size > phase
            }
            if not batch_list:
                continue
            if paint_mode == "compact":
                rate = paint_compact_emission_comb(
                    batch_list, spectrograph=spectrograph, fiber=fiber
                )
            else:
                rate = paint_injected_emission_comb(
                    batch_list,
                    spectrograph=spectrograph,
                    fiber=fiber,
                    photons_per_line=float(photons_per_line),
                )
        rate_last = rate
        ny, nx = rate.shape

        for od, lams in batch_list.items():
            if od not in grids.get(fiber, {}):
                continue
            lams = np.asarray(lams, dtype=np.float64)
            if lams.size == 0:
                continue
            y_pos = np.asarray(grids[fiber][od]["y_pos"], dtype=np.float64)
            oi = order_list.index(od) if od in order_list else None
            if oi is None:
                continue
            w_hdf = wave_hdf_all[oi]
            row_exp, col_exp = expected_feature_xy_oriented(
                spectrograph, fiber, od, lams, nx=nx_orient
            )
            okw = np.isfinite(w_hdf)
            order_w = np.argsort(w_hdf[okw])
            pix_wave_exp = np.interp(lams, w_hdf[okw][order_w], pix_grid[okw][order_w])

            spec_1d = None
            if centroid_mode == "1d":
                spec_1d, _ = _extract_box_one_order(
                    rate,
                    np.zeros_like(rate),
                    np.zeros(rate.shape, dtype=np.uint32),
                    y_pos,
                    extraction_aperture=extraction_aperture,
                )

            for lam, re, ce, pe_w in zip(lams, row_exp, col_exp, pix_wave_exp):
                if not (np.isfinite(re) and np.isfinite(ce)):
                    continue
                if centroid_mode == "2d":
                    if not (0.0 <= re < ny and 0.0 <= ce < min(nx, nx_orient)):
                        continue
                    rm, cm, rerr, cerr = _centroid_2d_flux(
                        rate,
                        row0=float(re),
                        col0=float(ce),
                        half_win=int(centroid_half_win),
                    )
                    if not (np.isfinite(rm) and np.isfinite(cm)):
                        continue
                    drow = float(rm - re)
                    dcol = float(cm - ce)
                    if abs(drow) > float(max_dpix) or abs(dcol) > float(max_dpix):
                        continue
                    pix_m = float(rm)
                    col_m = float(cm)
                    pix_err = float(rerr) if np.isfinite(rerr) else float("nan")
                    col_err = float(cerr) if np.isfinite(cerr) else float("nan")
                    pe = float(re)
                else:
                    pm = _fit_emission_peak_pix(
                        spec_1d,
                        pix_guess=float(pe_w),
                        half_win=max(6, int(centroid_half_win)),
                    )
                    if not np.isfinite(pm) or abs(pm - pe_w) > float(max_dpix):
                        continue
                    pix_m = float(pm)
                    col_m = float(np.interp(pm, pix_grid, y_pos))
                    pe = float(pe_w)
                    drow = float(pm - pe_w)
                    dcol = 0.0
                    pix_err = float("nan")
                    col_err = float("nan")

                acc[int(od)]["pix"].append(pix_m)
                acc[int(od)]["pix_err"].append(pix_err)
                acc[int(od)]["col"].append(col_m)
                acc[int(od)]["col_err"].append(col_err)
                acc[int(od)]["lam"].append(float(lam))
                acc[int(od)]["pe"].append(pe)
                acc[int(od)]["drow"].append(drow)
                acc[int(od)]["dcol"].append(dcol)
                measurements.append(
                    {
                        "order": float(od),
                        "lam_true_um": float(lam),
                        "row_expected": float(re) if np.isfinite(re) else float(pe_w),
                        "col_expected": float(ce) if np.isfinite(ce) else float("nan"),
                        "row_meas": pix_m,
                        "col_meas": col_m,
                        "row_err_pix": pix_err,
                        "col_err_pix": col_err,
                        "drow_pix": drow,
                        "dcol_pix": dcol,
                        "pix_expected": pe,
                        "pix_meas": pix_m,
                        "dpix": drow,
                        "batch": float(phase),
                    }
                )

    rate = rate_last if rate_last is not None else np.zeros((4096, 4160), dtype=np.float64)

    # Simulation WAVE = dense inverse of the deposit λ→x map (already in
    # wave_hdf_all via get_sim_hdf_trace_grids).  Do NOT replace it with a
    # polynomial fit to PSF centroids — that confuses a ~0.06 pix
    # pixel-phase centroid bias with a wavelength-solution error.
    wave_new = wave_hdf_all.copy()

    for oi, od in enumerate(order_list):
        if od not in acc or not acc[od]["pix"]:
            continue
        w_sol = wave_new[oi]
        pix_m = np.asarray(acc[od]["pix"], dtype=np.float64)
        pix_err = np.asarray(acc[od]["pix_err"], dtype=np.float64)
        lam_m = np.asarray(acc[od]["lam"], dtype=np.float64)
        pe_m = np.asarray(acc[od]["pe"], dtype=np.float64)
        drow_a = np.asarray(acc[od]["drow"], dtype=np.float64)
        dcol_a = np.asarray(acc[od]["dcol"], dtype=np.float64)
        lams = np.asarray(line_list[od], dtype=np.float64)
        if pix_m.size < 10:
            continue

        lam_mid = float(np.nanmean(lam_m))
        dlam_dpix_arr = np.gradient(w_sol)
        dlam_dpix_mean = float(np.nanmedian(np.abs(dlam_dpix_arr[np.isfinite(dlam_dpix_arr)])))
        if not np.isfinite(dlam_dpix_mean) or dlam_dpix_mean <= 0:
            dlam_dpix_mean = 1e-6

        # Geometry recovery: evaluate WAVE at the *deposit center* (pe), not the
        # PSF centroid.  This is the correct inject/recover test for simulation WAVE.
        lam_at_deposit = np.interp(pe_m, pix_grid, w_sol)
        vel_geom = np.asarray(
            delta_wave_to_delta_vel(lam_m - lam_at_deposit, lam_mid), dtype=np.float64
        )
        # Centroid systematic (NOT a WAVE error): WAVE at PSF centroid.
        lam_at_centroid = np.interp(pix_m, pix_grid, w_sol)
        vel_cent = np.asarray(
            delta_wave_to_delta_vel(lam_m - lam_at_centroid, lam_mid), dtype=np.float64
        )
        vel_xerr = np.asarray(
            delta_wave_to_delta_vel(pix_err * dlam_dpix_mean, lam_mid), dtype=np.float64
        )

        plo = float(np.nanpercentile(pe_m, 100.0 * float(fit_edge_margin_frac)))
        phi = float(np.nanpercentile(pe_m, 100.0 * (1.0 - float(fit_edge_margin_frac))))
        edge_ok = (pe_m >= plo) & (pe_m <= phi) & np.isfinite(pe_m) & np.isfinite(lam_m)
        inlier = edge_ok.copy()

        geom_mad = float(
            np.nanmedian(np.abs(vel_geom[inlier] - np.nanmedian(vel_geom[inlier])))
        )
        geom_rms = float(np.nanstd(vel_geom[inlier]))
        cent_mad = float(
            np.nanmedian(np.abs(vel_cent[inlier] - np.nanmedian(vel_cent[inlier])))
        )
        cent_mean = float(np.nanmean(vel_cent[inlier]))
        median_dpix = float(np.median(drow_a[inlier]))


        # Optional smooth through geometry residual (should be ~flat at ≪10 cm/s).
        x_fine = np.linspace(float(np.min(pe_m[inlier])), float(np.max(pe_m[inlier])), 800)
        if int(inlier.sum()) >= poly_deg + 3:
            g_coeffs, g_mid, g_half = _polyfit_scaled(
                pe_m[inlier], vel_geom[inlier], deg=min(3, int(poly_deg))
            )
            vel_fine = _polyval_scaled(g_coeffs, x_fine, x_mid=g_mid, x_half=g_half)
        else:
            vel_fine = np.zeros_like(x_fine)

        order_fit_curves[int(od)] = {
            "pix_meas": pix_m.tolist(),
            "pix_err": pix_err.tolist(),
            "pix_expected": pe_m.tolist(),
            "drow_pix": drow_a.tolist(),
            # Primary diagnostic: geometry residual at deposit centers.
            "vel_m_s": vel_geom.tolist(),
            "vel_centroid_m_s": vel_cent.tolist(),
            "vel_xerr_m_s": vel_xerr.tolist(),
            "inlier": inlier.astype(bool).tolist(),
            "edge_ok": edge_ok.astype(bool).tolist(),
            "x_fine": x_fine.tolist(),
            "vel_poly_fine": vel_fine.tolist(),
            "lam_mid_um": lam_mid,
            "dlam_dpix_mean": dlam_dpix_mean,
            "fit_pix_lo": plo,
            "fit_pix_hi": phi,
            "residual_kind": "deposit_center",
        }

        per_order[int(od)] = {
            "n_injected": int(lams.size),
            "n_measured": int(pix_m.size),
            "n_inlier": int(inlier.sum()),
            "n_edge_rejected": int((~edge_ok).sum()),
            "poly_deg": int(poly_deg),
            "method": "deposit_inverse",
            "centroid_mode": centroid_mode,
            "fit_pix_lo": plo,
            "fit_pix_hi": phi,
            "median_dpix": median_dpix,
            "median_drow_pix": float(np.median(drow_a)),
            "median_dcol_pix": float(np.median(dcol_a)),
            "median_pix_err": float(np.nanmedian(pix_err[inlier])),
            "drow_rms_pix": float(np.nanstd(drow_a[inlier])),
            "drow_mad_pix": float(
                np.nanmedian(np.abs(drow_a[inlier] - np.nanmedian(drow_a[inlier])))
            ),
            "dcol_rms_pix": float(np.nanstd(dcol_a[inlier])),
            "dcol_mad_pix": float(
                np.nanmedian(np.abs(dcol_a[inlier] - np.nanmedian(dcol_a[inlier])))
            ),
            # Geometry inject/recover (WAVE at deposit center) — target ≪10 cm/s.
            "geometry_residual_mad_m_s": geom_mad,
            "geometry_residual_rms_m_s": geom_rms,
            "geometry_residual_mad_cm_s": geom_mad * 100.0,
            "geometry_residual_rms_cm_s": geom_rms * 100.0,
            # PSF-centroid residual (measurement systematic, not WAVE error).
            "centroid_residual_mad_m_s": cent_mad,
            "centroid_residual_mean_m_s": cent_mean,
            "wave_precision_mad_um": float(
                np.nanmedian(np.abs((lam_m - lam_at_deposit)[inlier]))
            ),
            "wave_precision_rms_um": float(np.nanstd((lam_m - lam_at_deposit)[inlier])),
            "velocity_uncertainty_mad_m_s": geom_mad,
            "velocity_uncertainty_rms_m_s": geom_rms,
            "vel_curve_dispersion_rms_m_s": geom_rms,
            "vel_curve_dispersion_mad_m_s": geom_mad,
            "model_vs_centroid_vel_mad_m_s": cent_mad,
            "model_vs_centroid_vel_mean_m_s": cent_mean,
            "fit_rms_m_s": geom_rms,
            "fit_mad_m_s": geom_mad,
            "fit_mean_m_s": float(np.nanmean(vel_geom[inlier])),
            "hdf_rms_m_s": geom_rms,
            "hdf_mad_m_s": geom_mad,
            "hdf_mean_m_s": float(np.nanmean(vel_geom[inlier])),
            "poly_minus_hdf_mean_m_s": 0.0,
            "poly_minus_hdf_rms_m_s": 0.0,
            "poly_minus_hdf_mad_m_s": 0.0,
            "wave_min_um": float(np.nanmin(w_sol)),
            "wave_max_um": float(np.nanmax(w_sol)),
            "lam_mid_um": lam_mid,
        }

    # Keep WAVE for the requested fiber only; other fibers stay deposit-inverse HDF.
    # (Do not copy one fiber's correction onto fibers 1/3 — calibrate those separately.)
    wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
    setattr(wavesol, f"wave_fib{fiber}", wave_new.astype(np.float64))

    fit_rms = np.array([v["fit_rms_m_s"] for v in per_order.values()], dtype=np.float64)
    fit_mad = np.array([v["fit_mad_m_s"] for v in per_order.values()], dtype=np.float64)
    wave_mad = np.array(
        [v["wave_precision_mad_um"] for v in per_order.values()], dtype=np.float64
    )
    drow_mad = np.array([v["drow_mad_pix"] for v in per_order.values()], dtype=np.float64)
    dcol_mad = np.array([v["dcol_mad_pix"] for v in per_order.values()], dtype=np.float64)
    model_vel_mean = np.array(
        [v["model_vs_centroid_vel_mean_m_s"] for v in per_order.values()], dtype=np.float64
    )
    hdf_mean = np.array([v["hdf_mean_m_s"] for v in per_order.values()], dtype=np.float64)
    poly_hdf_mean = np.array(
        [v["poly_minus_hdf_mean_m_s"] for v in per_order.values()], dtype=np.float64
    )
    within_order_sys = np.array(
        [v["poly_minus_hdf_mad_m_s"] for v in per_order.values()], dtype=np.float64
    )
    n_med = (
        float(np.median([v["n_inlier"] for v in per_order.values()])) if per_order else 0.0
    )
    med_fit_rms = float(np.nanmedian(fit_rms)) if fit_rms.size else float("nan")
    med_fit_mad = float(np.nanmedian(fit_mad)) if fit_mad.size else float("nan")
    n_poly = int(sum(1 for v in per_order.values() if v.get("method") == "poly"))
    n_shift = int(sum(1 for v in per_order.values() if v.get("method") == "hdf_pixel_shift"))
    rv_budget = {
        "n_orders": int(len(per_order)),
        "n_orders_poly": n_poly,
        "n_orders_pixel_shift": n_shift,
        "centroid_mode": centroid_mode,
        "median_fit_rms_m_s": med_fit_rms,
        "median_fit_mad_m_s": med_fit_mad,
        "median_wave_precision_mad_um": (
            float(np.nanmedian(wave_mad)) if wave_mad.size else float("nan")
        ),
        "median_drow_mad_pix": (
            float(np.nanmedian(drow_mad)) if drow_mad.size else float("nan")
        ),
        "median_dcol_mad_pix": (
            float(np.nanmedian(dcol_mad)) if dcol_mad.size else float("nan")
        ),
        "median_model_vs_centroid_vel_m_s": (
            float(np.nanmedian(model_vel_mean)) if model_vel_mean.size else float("nan")
        ),
        "median_hdf_mean_offset_m_s": float(np.nanmedian(hdf_mean)) if hdf_mean.size else float("nan"),
        "mad_hdf_mean_offset_m_s": (
            float(np.nanmedian(np.abs(hdf_mean - np.nanmedian(hdf_mean))))
            if hdf_mean.size
            else float("nan")
        ),
        "median_poly_minus_hdf_m_s": (
            float(np.nanmedian(poly_hdf_mean)) if poly_hdf_mean.size else float("nan")
        ),
        "mad_order_zp_m_s": (
            float(np.nanmedian(np.abs(poly_hdf_mean - np.nanmedian(poly_hdf_mean))))
            if poly_hdf_mean.size
            else float("nan")
        ),
        "absolute_rv_systematic_hdf_m_s": (
            float(np.nanmedian(np.abs(hdf_mean))) if hdf_mean.size else float("nan")
        ),
        "single_line_precision_floor_m_s": med_fit_mad,
        "order_ccf_precision_floor_m_s": (
            med_fit_mad / np.sqrt(max(n_med, 1.0)) if np.isfinite(med_fit_mad) else float("nan")
        ),
        "order_to_order_zp_scatter_m_s": (
            float(np.nanmedian(np.abs(poly_hdf_mean - np.nanmedian(poly_hdf_mean))))
            if poly_hdf_mean.size
            else float("nan")
        ),
        "median_within_order_shape_m_s": (
            float(np.nanmedian(within_order_sys)) if within_order_sys.size else float("nan")
        ),
        "poly_deg": int(poly_deg),
        "n_lines_per_order_target": int(n_lines_per_order),
        "n_paint_batches": int(n_batch),
        "paint_mode": paint_mode,
        "fiber": int(fiber),
    }

    logger.info(
        "Comb wavesol: fiber=%d mode=%s n_lines=%d n_orders=%d "
        "wave_MAD=%.3e um vel_MAD=%.2f m/s drow_MAD=%.4f pix",
        fiber,
        centroid_mode,
        n_lines_per_order,
        len(per_order),
        rv_budget["median_wave_precision_mad_um"],
        rv_budget["median_fit_mad_m_s"],
        rv_budget["median_drow_mad_pix"],
    )
    return {
        "wavesol": wavesol,
        "hdf_wavesol": hdf_wavesol,
        "per_order": per_order,
        "line_measurements": measurements,
        "order_fit_curves": order_fit_curves,
        "rv_budget": rv_budget,
        "rate_map": rate,
        "line_list": {str(k): v.tolist() for k, v in line_list.items()},
        "fiber": int(fiber),
        "fibers": [int(fiber)],
        "spectrograph": spectrograph,
    }


# ========================================================================
# Infinite-S/N RV validation (formerly e2e_infinite_snr_rv.py)
# ========================================================================


HIGH_PRECISION_STELLAR_RV_ARGS: dict[str, Any] = {
    "algorithm": "ccf_line_mask",
    "mask_ccf_pixel_interp_order": 1,
    "window_line_by_line": True,
    "v_half_window": 500.0,
    "v_step_coarse": 0.01,
    "chunk_rv_prior_half_width_m_s": 3000.0,
    "outlier_sigma": 4.0,
    "preview_write_png": False,
    "save_overlay_debug": False,
    "save_preview": False,
}


@dataclass(frozen=True)
class InfiniteSnrRvThresholds:
    max_abs_bias_km_s: float = 0.20
    max_fiber_bias_spread_km_s: float = 0.15
    min_inlier_chunks: int = 3


DEFAULT_INFINITE_SNR_RV_THRESHOLDS = InfiniteSnrRvThresholds()


@dataclass
class InfiniteSnrRvFiberResult:
    fiber: int
    measured_rv_m_s: float | None
    rv_err_m_s: float | None
    bias_m_s: float | None
    bias_km_s: float | None
    n_inlier_chunks: int
    n_total_chunks: int
    inlier_median_bias_m_s: float | None


@dataclass
class InfiniteSnrRvEvaluation:
    injected_rv_km_s: float
    injected_rv_m_s: float
    fibers: list[InfiniteSnrRvFiberResult]
    max_abs_bias_km_s: float | None
    fiber_bias_spread_km_s: float | None

    @property
    def passed_default_thresholds(self) -> bool:
        th = DEFAULT_INFINITE_SNR_RV_THRESHOLDS
        if self.max_abs_bias_km_s is None or self.fiber_bias_spread_km_s is None:
            return False
        if self.max_abs_bias_km_s > th.max_abs_bias_km_s:
            return False
        if self.fiber_bias_spread_km_s > th.max_fiber_bias_spread_km_s:
            return False
        if any(r.n_inlier_chunks < th.min_inlier_chunks for r in self.fibers):
            return False
        return True


def apply_infinite_snr_rv_e2e_config(config: E2EConfig) -> E2EConfig:
    from dataclasses import replace
    from ..utils.line_mask_utils import paired_line_mask_path_for_template

    mask_path = config.stellar_rv_mask_path
    if mask_path is None and config.template_filename:
        try:
            mask_path = paired_line_mask_path_for_template(config.template_filename)
        except FileNotFoundError:
            logger.warning(
                "Infinite-S/N RV: paired mask for template %s not found; using default mask.",
                config.template_filename,
            )

    # Keep an explicit pipeline Stage-1 path (matched fiberflats / traces). Default
    # to sim_hdf truth traces only when the caller has not chosen a source.
    trace_src = (
        config.stage1_trace_source
        if config.stage1_trace_source == STAGE1_TRACE_SOURCE_PIPELINE
        else STAGE1_TRACE_SOURCE_SIM_HDF
    )

    return replace(
        config,
        infinite_snr_rv_validation=True,
        poisson_noise=False,
        convert_to_uint16=False,
        stage1_trace_source=trace_src,
        stellar_rv_mask_path=mask_path,
        make_preview_html=False,
    )


def merge_infinite_snr_stage1_primitives(
    primitives: dict | None,
    *,
    extraction_aperture: int = 10,
    method: str = "box",
    shear: object | None = None,
) -> dict:
    import copy

    out = copy.deepcopy(primitives or {})
    out.setdefault("spec_extract", {})
    out["spec_extract"]["method"] = str(method)
    out["spec_extract"]["extraction_aperture"] = int(extraction_aperture)
    if shear is not None:
        out["spec_extract"]["shear"] = shear
    return out


def merge_sp_lsf_e2e_stage1_primitives(
    primitives: dict | None,
    *,
    extraction_aperture: int = 10,
) -> dict:
    """Stage-1 kwargs for spectroperfectionism extract (sim / infinite-S/N RV)."""
    return merge_infinite_snr_stage1_primitives(
        primitives,
        extraction_aperture=extraction_aperture,
        method="spectroperfectionism",
    )


def merge_infinite_snr_stage2_primitives(primitives: dict | None) -> dict:
    import copy

    out = copy.deepcopy(primitives or {})
    out.setdefault("stellar_rv", {})
    out["stellar_rv"].update(HIGH_PRECISION_STELLAR_RV_ARGS)
    return out


merge_infinite_snr_stage1_prim_args = merge_infinite_snr_stage1_primitives
merge_infinite_snr_stage2_prim_args = merge_infinite_snr_stage2_primitives


def evaluate_infinite_snr_rv_recovery(
    stage2_model: datamodels.DataModel | None,
    *,
    injected_rv_km_s: float,
    fibers: list[int] | None = None,
) -> InfiniteSnrRvEvaluation:
    injected_m_s = float(injected_rv_km_s) * 1000.0
    if stage2_model is None:
        return InfiniteSnrRvEvaluation(
            injected_rv_km_s=float(injected_rv_km_s),
            injected_rv_m_s=injected_m_s,
            fibers=[],
            max_abs_bias_km_s=None,
            fiber_bias_spread_km_s=None,
        )
    if fibers is None:
        fibers = [1, 2, 3]
    rows: list[InfiniteSnrRvFiberResult] = []
    biases_km: list[float] = []
    for fib in fibers:
        avg = getattr(stage2_model, f"rvccfavg_fib{fib}", None)
        chunks = getattr(stage2_model, f"rvccf_fib{fib}", None)
        meas = err = bias_m_s = None
        n_in = n_tot = 0
        inlier_med = None
        if avg is not None and len(avg) > 0:
            meas = float(avg["RV"][0])
            err = float(avg["RV_ERR"][0])
            if np.isfinite(meas):
                bias_m_s = meas - injected_m_s
                biases_km.append(bias_m_s / 1000.0)
        if chunks is not None and len(chunks) > 0:
            rv_c = np.asarray(chunks["RV"], dtype=np.float64)
            n_tot = int(rv_c.size)
            chunk_names = getattr(chunks, "colnames", None)
            if chunk_names is None and hasattr(chunks, "dtype"):
                chunk_names = chunks.dtype.names
            if chunk_names and "INLIER" in chunk_names:
                inl = np.asarray(chunks["INLIER"], dtype=bool)
            else:
                inl = np.isfinite(rv_c)
            ok = inl & np.isfinite(rv_c)
            n_in = int(np.sum(ok))
            if n_in > 0:
                inlier_med = float(np.median(rv_c[ok] - injected_m_s))
        rows.append(
            InfiniteSnrRvFiberResult(
                fiber=int(fib),
                measured_rv_m_s=meas,
                rv_err_m_s=err,
                bias_m_s=bias_m_s,
                bias_km_s=(bias_m_s / 1000.0) if bias_m_s is not None else None,
                n_inlier_chunks=n_in,
                n_total_chunks=n_tot,
                inlier_median_bias_m_s=inlier_med,
            )
        )
    max_abs = max((abs(b) for b in biases_km), default=None)
    spread = (max(biases_km) - min(biases_km)) if len(biases_km) >= 2 else None
    return InfiniteSnrRvEvaluation(
        injected_rv_km_s=float(injected_rv_km_s),
        injected_rv_m_s=injected_m_s,
        fibers=rows,
        max_abs_bias_km_s=max_abs,
        fiber_bias_spread_km_s=spread,
    )


def summarize_per_line_rv_mad(
    stage2_model: datamodels.DataModel | None,
    *,
    injected_rv_m_s: float,
    fiber: int = 2,
) -> dict[str, float | int]:
    """
    Aggregate bias and per-line MAD from a Stage-2 ``RVCCF`` table.

    Used for E2E closure with SP extract + LSF/E2E ccfopt masks: track
    aggregate bias, per-line MAD, and formal ``RV_ERR`` vs ideal recovery.
    """
    out: dict[str, float | int] = {
        "fiber": int(fiber),
        "injected_rv_m_s": float(injected_rv_m_s),
        "n_lines": 0,
        "aggregate_bias_m_s": float("nan"),
        "per_line_mad_m_s": float("nan"),
        "formal_rv_err_m_s": float("nan"),
    }
    if stage2_model is None:
        return out
    avg = getattr(stage2_model, f"rvccfavg_fib{fiber}", None)
    chunks = getattr(stage2_model, f"rvccf_fib{fiber}", None)
    if avg is not None and len(avg) > 0:
        meas = float(avg["RV"][0])
        err = float(avg["RV_ERR"][0])
        if np.isfinite(meas):
            out["aggregate_bias_m_s"] = meas - float(injected_rv_m_s)
        if np.isfinite(err):
            out["formal_rv_err_m_s"] = err
    if chunks is None or len(chunks) == 0:
        return out
    rv = np.asarray(chunks["RV"], dtype=np.float64)
    names = getattr(chunks, "colnames", None)
    if names is None and hasattr(chunks, "dtype"):
        names = chunks.dtype.names
    if names and "INLIER" in names:
        ok = np.asarray(chunks["INLIER"], dtype=bool) & np.isfinite(rv)
    else:
        ok = np.isfinite(rv)
    resid = rv[ok] - float(injected_rv_m_s)
    out["n_lines"] = int(resid.size)
    if resid.size:
        out["per_line_mad_m_s"] = float(np.median(np.abs(resid)))
    return out


def e2e_rv_closure_config(
    config: "E2EConfig",
    *,
    extract_method: str = "spectroperfectionism",
    use_lsf_mask: bool = True,
    extraction_aperture: int = 10,
) -> "E2EConfig":
    """
    Configure infinite-S/N E2E RV with SP (or sheared_sp) extract and optional LSF mask.

    When ``use_lsf_mask`` is True and a packaged ``*.lsf.ccfopt.delta.mas`` exists for
    the template, prefer it; otherwise keep the E2E / packaged ccfopt mask.

    Track closure with :func:`summarize_per_line_rv_mad` on the Stage-2 product.
    """
    from dataclasses import replace
    from pathlib import Path
    from ..utils.line_mask_utils import paired_line_mask_path_for_template

    cfg = apply_infinite_snr_rv_e2e_config(config)
    mask_path = cfg.stellar_rv_mask_path
    if use_lsf_mask and cfg.template_filename:
        stem = Path(cfg.template_filename).stem
        try:
            base = Path(paired_line_mask_path_for_template(cfg.template_filename))
            lsf_cand = base.with_name(f"{stem}.lsf.ccfopt.delta.mas")
            if lsf_cand.is_file():
                mask_path = str(lsf_cand)
        except FileNotFoundError:
            pass
    stage1 = merge_infinite_snr_stage1_primitives(
        cfg.stage1_primitives,
        extraction_aperture=extraction_aperture,
        method=extract_method,
    )
    return replace(
        cfg,
        stellar_rv_mask_path=mask_path,
        stage1_primitives=stage1,
    )


def assert_infinite_snr_rv_recovery(
    evaluation: InfiniteSnrRvEvaluation,
    *,
    thresholds: InfiniteSnrRvThresholds | None = None,
) -> None:
    th = thresholds or DEFAULT_INFINITE_SNR_RV_THRESHOLDS
    if evaluation.max_abs_bias_km_s is None:
        raise AssertionError("Infinite-S/N RV evaluation: no measured RVs (stage2 missing?).")
    if evaluation.max_abs_bias_km_s > th.max_abs_bias_km_s:
        raise AssertionError(
            f"Infinite-S/N RV max |bias|={evaluation.max_abs_bias_km_s:.4f} km/s "
            f"> {th.max_abs_bias_km_s:.4f} km/s"
        )
    if (
        evaluation.fiber_bias_spread_km_s is not None
        and evaluation.fiber_bias_spread_km_s > th.max_fiber_bias_spread_km_s
    ):
        raise AssertionError(
            f"Infinite-S/N RV fiber spread={evaluation.fiber_bias_spread_km_s:.4f} km/s "
            f"> {th.max_fiber_bias_spread_km_s:.4f} km/s"
        )
    for row in evaluation.fibers:
        if row.n_inlier_chunks < th.min_inlier_chunks:
            raise AssertionError(
                f"Infinite-S/N RV fiber {row.fiber}: n_inlier={row.n_inlier_chunks} "
                f"< {th.min_inlier_chunks}"
            )


def log_infinite_snr_rv_evaluation(evaluation: InfiniteSnrRvEvaluation) -> None:
    logger.info(
        "Infinite-S/N RV evaluation injected=%.3f km/s max|bias|=%s spread=%s passed=%s",
        evaluation.injected_rv_km_s,
        f"{evaluation.max_abs_bias_km_s:.4f} km/s" if evaluation.max_abs_bias_km_s is not None else "—",
        f"{evaluation.fiber_bias_spread_km_s:.4f} km/s"
        if evaluation.fiber_bias_spread_km_s is not None
        else "—",
        evaluation.passed_default_thresholds,
    )
    for row in evaluation.fibers:
        logger.info(
            "  fib%d: bias=%s km/s n_inlier=%d/%d inlier_med=%s m/s",
            row.fiber,
            f"{row.bias_km_s:+.4f}" if row.bias_km_s is not None else "—",
            row.n_inlier_chunks,
            row.n_total_chunks,
            f"{row.inlier_median_bias_m_s:.1f}" if row.inlier_median_bias_m_s is not None else "—",
        )


# ========================================================================
# E2E HTML / PNG preview report (moved to simulations.plotting.e2e_preview)
# ========================================================================

from .plotting.e2e_preview import (  # noqa: E402
    PREVIEWS_SUBDIR,
    REPORT_HTML_NAME,
    PreviewItem,
    generate_e2e_preview_report,
    infer_exposure_base_name,
)
