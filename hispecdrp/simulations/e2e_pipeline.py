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
    get_default_stage0_prim_args,
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
    stage0_prim_args: dict,
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
        stage0_prim_args=stage0_prim_args,
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
    stage0_prim_args: dict,
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
        stage0_prim_args,
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
) -> tuple[dict | None, Any | None]:
    """
    Resolve or generate Stage 1 trace / fiberflat / wavesol calibrations.

    Returns ``(fiberflat_bundle, wavesol_model)``; either may be ``None`` when not
    requested or when masters already exist on ``cal_bundle``.
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
            stage0_prim_args = get_default_stage0_prim_args(max_cores=config.max_cores)
            if stage1_uses_sim_hdf_trace(config):

                extraction_aperture = 10.0
                if config.infinite_snr_rv_validation:

                    extraction_aperture = float(
                        merge_infinite_snr_stage1_prim_args({})
                        .get("spec_extract", {})
                        .get("extraction_aperture", extraction_aperture)
                    )
                logger.info(
                    "E2E: sim_hdf Stage-1 bundle (HDF trace + box fiberflat, aperture=%.1f).",
                    extraction_aperture,
                )
                ramp_mjd = e2e_gaia.resolve_e2e_mjd_start(config.mjd_start)
                ff_paths = _fiberflat_rate_paths_for_stage1(
                    config,
                    cal_bundle,
                    calibration_models,
                    out_dir_abs,
                    stage0_prim_args,
                    ff_list,
                    record_sources=True,
                )
                bundle = materialize_stage1_bundle_from_sim_hdf(
                    spectrograph=config.spectrograph,
                    fiberflat_rate_paths=ff_paths,
                    meta={"mjd_start": float(ramp_mjd)},
                    extraction_aperture=extraction_aperture,
                )
                ff_bundle = {
                    "trace_pos": bundle["trace_pos"],
                    "trace_prof": bundle["trace_prof"],
                    "fiberflat": bundle["fiberflat"],
                }
                wavesol_model = bundle["wavesol"]
                for key in e2e_calibrations.STAGE1_TRACE_KEYS:
                    cal_bundle.sources[key] = STAGE1_TRACE_SOURCE_SIM_HDF
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
                    stage0_prim_args,
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
                    stage0_prim_args,
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
            stage0_prim_args = get_default_stage0_prim_args(max_cores=config.max_cores)
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
                    stage0_prim_args,
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
    target_meta: dict | None = None
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

    stage0_prim_args = get_default_stage0_prim_args(max_cores=config.max_cores)
    stage1_prim_args = None
    stage2_prim_args = None
    if config.infinite_snr_rv_validation:

        stage1_prim_args = merge_infinite_snr_stage1_prim_args(None)
        stage2_prim_args = merge_infinite_snr_stage2_prim_args(None)
    logger.info("E2E: Stage 0 (ramp → rate/slope).")
    stage0_model = run_stage0_simulation(
        ramp_model,
        calibration_models,
        output_dir=out_dir_abs,
        prim_args=stage0_prim_args,
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
            ff_bundle, wavesol_model = materialize_stage1_calibrations(
                config, cal_bundle, calibration_models, out_dir_abs
            )
            if ff_bundle is not None:
                stage1_model = e2e_support.run_stage1_simulation(
                    stage0_model,
                    ff_bundle["trace_pos"],
                    ff_bundle["trace_prof"],
                    prim_args=stage1_prim_args,
                    wavesol_model=wavesol_model,
                    output_dir=out_dir_abs,
                    max_cores=config.max_cores,
                )
        else:
            logger.warning("E2E: run_stage1 is True but fiberflat_fibers is empty; skipping Stage 1/2.")

    if config.run_stage2 and stage1_model is not None and ff_bundle is not None:
        logger.info("E2E: Stage 2.")
        _rv_center_m_s: float | None = None
        if gaia_target is not None and np.isfinite(float(injected_rv_km_s)):
            _rv_center_m_s = float(injected_rv_km_s) * 1000.0
            logger.info(
                "E2E: stellar RV centered uniform grid ccf_input_velocity=%.3f m/s (%.4f km/s)",
                _rv_center_m_s,
                float(injected_rv_km_s),
            )
        stage2_model = e2e_support.run_stage2_simulation(
            stage1_model,
            fiberflat_model=ff_bundle["fiberflat"],
            prim_args=stage2_prim_args,
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

# ========================================================================
# Sim-HDF Stage-1 calibrations (formerly sim_hdf_calibrations.py)
# ========================================================================

_DEFAULT_FWHM_PIX = 3.5
_TRACE_POLY_DEG = 2


def _order_list(spectrograph: str) -> list[int]:
    return list(hispec_config.HISPECConfig.get_order_list(spectrograph.upper()))


def _load_hdf_model_data(spectrograph: str) -> tuple[str, dict]:
    srm = SimulateRateMap()
    spectrograph = spectrograph.upper()
    model = srm.bspec_hdf_model if spectrograph == "BSPEC" else srm.rspec_hdf_model
    model_path = srm.hdf_file_directory + model
    return spectrograph, read_pyechelle_hdf(model_path)


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
            wavelength = _interp_oriented_spectral_row(row, ow[good], n_pix=n_pix)
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
    prim = TracePositionPrimitive()
    prim.init_args(
        deg_pos=4,
        deg_width=2,
        disp_range=_DEFAULT_FIBERFLAT_DISP_RANGE.get(spectrograph, (500, 4000)),
        xdisp_range=_DEFAULT_FIBERFLAT_XDISP_RANGE.get(spectrograph, (100, 4050)),
        prominence_factor=prominence_factor,
        output_model=TracePosModel,
    )
    trace_pos: TracePosModel = prim.apply(models_by_fiber)
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


def create_traceprof_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
    detector_shape: tuple[int, int] = (4096, 4160),
) -> TraceProfModel:
    """Placeholder trace profiles (box extraction does not use them)."""
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

    return ff


def materialize_stage1_bundle_from_sim_hdf(
    *,
    spectrograph: str,
    fiberflat_rate_paths: list[str],
    meta: dict | None = None,
    extraction_aperture: float = 10.0,
) -> dict[str, Any]:
    """Return ``trace_pos``, ``trace_prof``, ``fiberflat``, and ``wavesol`` from HDF truth."""
    spectrograph = spectrograph.upper()
    trace_pos = create_tracepos_model_from_fiberflat_rates(
        fiberflat_rate_paths,
        spectrograph=spectrograph,
        meta=meta,
    )
    trace_prof = create_traceprof_model_from_sim_hdf(spectrograph, meta=meta)
    wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
    fiberflat = create_fiberflat_model_from_rate_maps(
        fiberflat_rate_paths,
        trace_pos,
        spectrograph=spectrograph,
        meta=meta,
        extraction_aperture=extraction_aperture,
    )
    logger.info(
        "Sim HDF Stage-1 bundle ready spectrograph=%s n_fiberflat_rates=%d extraction_aperture=%.1f",
        spectrograph,
        len(fiberflat_rate_paths),
        float(extraction_aperture),
    )
    return {
        "trace_pos": trace_pos,
        "trace_prof": trace_prof,
        "fiberflat": fiberflat,
        "wavesol": wavesol,
    }

# ========================================================================
# Infinite-S/N RV validation (formerly e2e_infinite_snr_rv.py)
# ========================================================================


HIGH_PRECISION_STELLAR_RV_ARGS: dict[str, Any] = {
    "algorithm": "ccf_line_mask",
    "mask_ccf_pixel_interp_order": 1,
    "mask_auto_center_scout": False,
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

    return replace(
        config,
        infinite_snr_rv_validation=True,
        poisson_noise=False,
        convert_to_uint16=False,
        stage1_trace_source=STAGE1_TRACE_SOURCE_SIM_HDF,
        stellar_rv_mask_path=mask_path,
        make_preview_html=False,
    )


def merge_infinite_snr_stage1_prim_args(
    prim_args: dict | None,
    *,
    extraction_aperture: int = 10,
) -> dict:
    import copy

    out = copy.deepcopy(prim_args or {})
    out.setdefault("spec_extract", {})
    out["spec_extract"]["method"] = "box"
    out["spec_extract"]["extraction_aperture"] = int(extraction_aperture)
    return out


def merge_infinite_snr_stage2_prim_args(prim_args: dict | None) -> dict:
    import copy

    out = copy.deepcopy(prim_args or {})
    out.setdefault("stellar_rv", {})
    out["stellar_rv"].update(HIGH_PRECISION_STELLAR_RV_ARGS)
    return out


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
# E2E HTML / PNG preview report (formerly e2e_previews.py)
# ========================================================================

PREVIEWS_SUBDIR = "previews"
REPORT_HTML_NAME = "index.html"

# Longest image dimension (pixels) for rendered detector/rate previews.
_MAX_IMAGE_DIM = 1100

# Canonical ordering and display labels for Stage 0 / Stage 1 step tokens. Tokens are the text
# after the suffix dash in the saved filename (e.g. ``HB...._raw-dqinit.fits`` -> ``dqinit``).
_STAGE0_TOKEN_ORDER: dict[str, str] = {
    "dqinit": "DQ Init",
    "satcheck": "Saturation Check",
    "biassub": "Bias Subtraction",
    "refpixcorr": "Reference Pixel Correction",
    "nonlincorr": "Nonlinearity Correction",
    "jumpdetec": "Jump Detection",
    "rampfit": "Ramp Fitting",
    "darksub": "Dark Subtraction",
    "gaincorr": "Gain Correction",
    "detflatcorr": "Detector Flat Correction",
}
_STAGE1_TOKEN_ORDER: dict[str, str] = {
    "specextract": "Spectral Extraction",
    "wavesolinit": "Initial Wavelength Solution",
}
_STAGE2_TOKEN_ORDER: dict[str, str] = {
    "blaze_correct": "Blaze Correction",
    "barycorr": "Barycentric Correction",
    "stellar_rv": "Stellar RV (CCF)",
}

# Filename suffix tokens: <base>_<group>-<token>.fits
_PRODUCT_RE = re.compile(r"^(?P<base>.+?)_(?P<group>raw|rate|specuncal|spec)-(?P<token>[A-Za-z0-9_]+)\.fits$")


@dataclass
class PreviewItem:
    """One rendered preview belonging to a (stage, primitive) section."""

    primitive_label: str
    primitive_order: int
    images: list[str] = field(default_factory=list)  # paths relative to the report dir
    links: list[tuple[str, str]] = field(default_factory=list)  # (label, href)
    caption: str = ""


def infer_exposure_base_name(
    *,
    stage0_model=None,
    stage1_model=None,
    stage2_model=None,
) -> str | None:
    """Best-effort science exposure base name (e.g. ``HB.20260612.57502.95``) from a model."""
    for model in (stage2_model, stage1_model, stage0_model):
        if model is None:
            continue
        meta = getattr(model, "meta", None)
        fname = getattr(meta, "filename", None) if meta is not None else None
        if not fname:
            fname = getattr(model, "_filepath", None)
        if fname:
            base = _strip_product_suffix(os.path.basename(str(fname)))
            if base:
                return base
    return None


def _strip_product_suffix(filename: str) -> str | None:
    m = _PRODUCT_RE.match(filename)
    if m:
        return m.group("base")
    # Fall back to a bare ``<base>_rate.fits`` / ``<base>_specuncal.fits`` style name.
    m2 = re.match(r"^(?P<base>.+?)_(raw|rate|specuncal|spec)\.fits$", filename)
    if m2:
        return m2.group("base")
    return None


def _infer_base_from_dir(output_dir: str) -> str | None:
    """Pick the exposure base with the most products directly under ``output_dir``."""
    counts: dict[str, int] = {}
    for name in os.listdir(output_dir):
        if not name.endswith(".fits"):
            continue
        m = _PRODUCT_RE.match(name)
        if not m:
            continue
        counts[m.group("base")] = counts.get(m.group("base"), 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


# --------------------------------------------------------------------------------------
# Image rendering helpers
# --------------------------------------------------------------------------------------


def _downsample_2d(arr: np.ndarray, max_dim: int = _MAX_IMAGE_DIM) -> np.ndarray:
    """Stride-downsample a 2-D array so its longest dimension is <= ``max_dim``."""
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    ny, nx = a.shape
    step = max(1, int(np.ceil(max(ny, nx) / float(max_dim))))
    if step <= 1:
        return a
    return a[::step, ::step]


def _finite_mean_std(arr: np.ndarray) -> tuple[float, float]:
    """Mean and sample std of finite pixels in a 2-D (or collapsible) array."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _format_2d_frame_title(base_title: str, mean: float, std: float) -> str:
    """Append pixel mean and standard deviation to a matplotlib frame title."""
    return f"{base_title}\nMean={mean:.2f}, Standard Deviation={std:.2f}"


def _save_detector_image_png(arr2d: np.ndarray, out_path: str, title: str) -> None:
    """Render a 2-D detector/rate image with robust percentile scaling to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = _downsample_2d(np.asarray(arr2d, dtype=np.float64))
    finite = np.isfinite(a)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(a[finite], [1.0, 99.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(a[finite]))
            vmax = float(np.nanmax(a[finite]))
            if vmin == vmax:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    im = ax.imshow(a, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("detector x (downsampled)")
    ax.set_ylabel("detector y (downsampled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def _save_orders_spectra_png(
    flux: np.ndarray,
    dq: np.ndarray | None,
    out_path: str,
    title: str,
) -> None:
    """Render 1-D extracted spectra (one panel per order) for a single fiber."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.asarray(flux, dtype=np.float64)
    if f.ndim != 2:
        f = np.atleast_2d(f)
    n_orders, n_pix = f.shape
    dq_arr = np.asarray(dq) if dq is not None else None

    ncols = 4
    nrows = int(np.ceil(n_orders / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.4, nrows * 1.7), squeeze=False
    )
    x = np.arange(n_pix)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n_orders:
            ax.axis("off")
            continue
        y = f[k].copy()
        if dq_arr is not None and dq_arr.shape == f.shape:
            y = np.where(dq_arr[k] != 0, np.nan, y)
        ax.plot(x, y, lw=0.5, color="#1f77b4")
        ax.set_title(f"order {k}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.margins(x=0.0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Per-stage preview builders
# --------------------------------------------------------------------------------------


def _read_hdu_data(path: str, names: list[str]):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        available = {hdu.name.upper(): i for i, hdu in enumerate(hdul)}
        for name in names:
            idx = available.get(name.upper())
            if idx is not None and hdul[idx].data is not None:
                return np.asarray(hdul[idx].data)
    return None


def _build_stage0_item(
    path: str,
    token: str,
    *,
    kind: str,
    previews_dir: str,
    report_dir: str,
) -> PreviewItem | None:
    label = _STAGE0_TOKEN_ORDER.get(token, token)
    order = list(_STAGE0_TOKEN_ORDER).index(token) if token in _STAGE0_TOKEN_ORDER else 999
    out_png = os.path.join(previews_dir, f"stage0_{token}.png")
    if kind == "ramp_mean":
        cube = _read_hdu_data(path, ["DATA", "SCI"])
        if cube is None or cube.ndim < 3:
            logger.warning("Stage 0 preview: no ramp cube in %s; skipping.", path)
            return None
        # Mean across the read axis (axis 0 for (nreads, ny, nx); collapse leading axes if 4-D).
        img = np.asarray(cube, dtype=np.float64)
        while img.ndim > 2:
            img = np.nanmean(img, axis=0)
        base_title = f"{label}: mean of detector across all reads"
        mean, std = _finite_mean_std(img)
        title = _format_2d_frame_title(base_title, mean, std)
        caption = (
            "Average of the detector counts across all UTR reads. "
            f"Mean={mean:.2f}, Standard Deviation={std:.2f}."
        )
    else:  # rate_map
        img = _read_hdu_data(path, ["RATE", "SCI", "DATA"])
        if img is None:
            logger.warning("Stage 0 preview: no rate image in %s; skipping.", path)
            return None
        img = np.asarray(img, dtype=np.float64)
        while img.ndim > 2:
            img = img[0]
        base_title = f"{label}: rate map"
        mean, std = _finite_mean_std(img)
        title = _format_2d_frame_title(base_title, mean, std)
        caption = (
            "Rate map (e-/s per pixel) after this step. "
            f"Mean={mean:.2f}, Standard Deviation={std:.2f}."
        )
    _save_detector_image_png(img, out_png, title)
    rel = os.path.relpath(out_png, report_dir)
    return PreviewItem(
        primitive_label=label,
        primitive_order=order,
        images=[rel],
        caption=caption,
    )


def _build_stage1_item(
    path: str,
    token: str,
    *,
    previews_dir: str,
    report_dir: str,
) -> PreviewItem | None:
    from astropy.io import fits

    label = _STAGE1_TOKEN_ORDER.get(token, token)
    order = list(_STAGE1_TOKEN_ORDER).index(token) if token in _STAGE1_TOKEN_ORDER else 999
    images: list[str] = []
    with fits.open(path, memmap=False) as hdul:
        hdu_names = [h.name.upper() for h in hdul]
        spec_names = sorted(
            (hdu.name for hdu in hdul if re.match(r"^SPEC_UNCAL_FIB\d+$", hdu.name.upper())),
            key=lambda s: int(re.search(r"\d+", s).group()),
        )
        for spec_name in spec_names:
            fib = int(re.search(r"\d+", spec_name).group())
            flux = hdul[spec_name].data
            if flux is None:
                continue
            dq_name = f"SPEC_DQ_FIB{fib}"
            dq = hdul[dq_name].data if dq_name.upper() in hdu_names else None
            out_png = os.path.join(previews_dir, f"stage1_{token}_fib{fib}.png")
            _save_orders_spectra_png(
                np.asarray(flux),
                np.asarray(dq) if dq is not None else None,
                out_png,
                f"{label}: extracted 1-D spectra, fiber {fib}",
            )
            images.append(os.path.relpath(out_png, report_dir))
    if not images:
        return None
    return PreviewItem(
        primitive_label=label,
        primitive_order=order,
        images=images,
        caption="Extracted 1-D spectrum per echelle order (flux vs. detector column).",
    )


def _collect_stage2_items(
    output_dir: str,
    base: str,
    report_dir: str,
) -> list[PreviewItem]:
    """Collect the pipeline's default Stage 2 previews already written next to the L2 products."""
    items: dict[str, PreviewItem] = {}
    for name in sorted(os.listdir(output_dir)):
        if not name.startswith(base + "_spec"):
            continue
        if not name.lower().endswith((".png", ".html")):
            continue  # default previews only (skip .fits products)
        path = os.path.join(output_dir, name)
        token = None
        for tok in _STAGE2_TOKEN_ORDER:
            if f"_spec-{tok}" in name or f"-{tok}_" in name or f"-{tok}." in name:
                token = tok
                break
        if "stellar_rv" in name:
            token = "stellar_rv"
        if token is None:
            continue
        label = _STAGE2_TOKEN_ORDER.get(token, token)
        order = list(_STAGE2_TOKEN_ORDER).index(token) if token in _STAGE2_TOKEN_ORDER else 999
        item = items.setdefault(
            token, PreviewItem(primitive_label=label, primitive_order=order, caption="")
        )
        rel = os.path.relpath(path, report_dir)
        if name.lower().endswith(".png"):
            item.images.append(rel)
        elif name.lower().endswith(".html"):
            item.links.append((name.replace(base + "_", ""), rel))
    return [it for it in items.values() if it.images or it.links]


def _collect_fiberflat_items(output_dir: str, report_dir: str) -> list[PreviewItem]:
    """Collect trace-position previews from the fiberflat pipeline under ``output_dir/cals/``."""
    cals_dir = os.path.join(output_dir, "cals")
    if not os.path.isdir(cals_dir):
        return []

    previews: list[tuple[float, str]] = []
    for name in os.listdir(cals_dir):
        lower = name.lower()
        if "mastercal-tracepos" not in lower or not lower.endswith("_preview.png"):
            continue
        path = os.path.join(cals_dir, name)
        if os.path.isfile(path):
            previews.append((os.path.getmtime(path), path))

    if not previews:
        return []

    # Use the most recently written trace-pos preview (current E2E run).
    _, best_path = max(previews, key=lambda t: t[0])
    rel = os.path.relpath(best_path, report_dir)
    return [
        PreviewItem(
            primitive_label="Trace Positions",
            primitive_order=0,
            images=[rel],
            caption=(
                "Fitted echelle trace centroids on simulated fiberflat rate maps "
                "(ProcessFiberflatsPipeline / TracePositionPrimitive)."
            ),
        )
    ]


# --------------------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------------------

_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 1.5rem; line-height: 1.4; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem 0; }
.subtitle { color: #888; margin-bottom: 1.25rem; font-size: 0.9rem; }
details.stage { border: 1px solid #8884; border-radius: 8px; margin: 0 0 1rem 0; padding: 0.25rem 0.75rem; }
details.stage > summary { font-size: 1.15rem; font-weight: 600; cursor: pointer; padding: 0.5rem 0; }
details.prim { border-top: 1px dashed #8884; margin: 0.25rem 0; padding: 0.25rem 0.25rem 0.5rem 0.75rem; }
details.prim > summary { font-size: 1rem; font-weight: 500; cursor: pointer; padding: 0.4rem 0; }
.caption { color: #888; font-size: 0.85rem; margin: 0.1rem 0 0.5rem 0; }
.gallery { display: flex; flex-wrap: wrap; gap: 0.75rem; max-height: 80vh; overflow: auto;
           padding: 0.25rem; border: 1px solid #8882; border-radius: 6px; }
.gallery img { max-width: 100%; height: auto; border: 1px solid #8884; border-radius: 4px; }
.gallery figure { margin: 0; }
.links a { display: inline-block; margin: 0.2rem 0.6rem 0.2rem 0; }
.rv-summary { margin: 1.25rem 0 0.5rem 0; }
.rv-summary h2 { font-size: 1.15rem; margin: 0 0 0.5rem 0; }
.rv-summary table { border-collapse: collapse; width: 100%; max-width: 960px; font-size: 0.9rem; }
.rv-summary th, .rv-summary td { border: 1px solid #8884; padding: 0.45rem 0.55rem; text-align: right; }
.rv-summary th:first-child, .rv-summary td:first-child { text-align: center; }
.rv-summary th { background: #8882; font-weight: 600; }
.rv-summary caption { caption-side: top; text-align: left; color: #888; font-size: 0.85rem; margin-bottom: 0.35rem; }
"""


def _render_item_html(item: PreviewItem) -> str:
    parts = [f'<details class="prim" open><summary>{_esc(item.primitive_label)}</summary>']
    if item.caption:
        parts.append(f'<div class="caption">{_esc(item.caption)}</div>')
    if item.images:
        parts.append('<div class="gallery">')
        for img in item.images:
            parts.append(
                f'<figure><a href="{_esc(img)}" target="_blank">'
                f'<img src="{_esc(img)}" loading="lazy"></a></figure>'
            )
        parts.append("</div>")
    if item.links:
        parts.append('<div class="links">')
        for label, href in item.links:
            parts.append(f'<a href="{_esc(href)}" target="_blank">{_esc(label)} &#8599;</a>')
        parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


def _render_stage_html(stage_title: str, items: list[PreviewItem]) -> str:
    items = sorted(items, key=lambda it: (it.primitive_order, it.primitive_label))
    body = "\n".join(_render_item_html(it) for it in items)
    return (
        f'<details class="stage" open><summary>{_esc(stage_title)}</summary>\n{body}\n</details>'
    )


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_num(value: float | None, *, decimals: int = 3, unit: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}{unit}"


def _render_rv_performance_table_html(
    perf_rows: list,
    *,
    injected_rv_km_s: float,
    injected_berv_km_s: float | None = None,
) -> str:
    """HTML table of E2E RV recovery metrics (from :func:`collect_e2e_rv_performance_rows`)."""
    from . import e2e_support

    if not perf_rows:
        return ""

    header = (
        f"Injected stellar RV = {_fmt_num(injected_rv_km_s, decimals=3)} km/s "
        f"({_fmt_num(injected_rv_km_s * 1000.0, decimals=1)} m/s); "
        "excludes BERV (simulation Doppler uses topocentric ``Gaia RV - BERV``; "
        "Stage 2 CCF converts measured RV to barycentric via ``ccf_zb``)"
    )
    if injected_berv_km_s is not None and np.isfinite(injected_berv_km_s):
        header += (
            f". Reference BERV at ``mjd_mid`` = {_fmt_num(injected_berv_km_s, decimals=3)} km/s "
            f"({_fmt_num(injected_berv_km_s * 1000.0, decimals=1)} m/s)"
        )
    header += (
        ". Measured RV and σ(⟨RV⟩) from RVCCFAVG; BERV column from Stage 2 "
        "``BARYCORR_FIB*``; inlier scatter is RMS of "
        "inlier chunk RVs about ⟨RV⟩ (same metric as the Stage 2 preview sidebar)."
    )
    body_rows: list[str] = []
    for row in perf_rows:
        if not isinstance(row, e2e_support.E2ERVPerformanceRow):
            continue
        body_rows.append(
            "<tr>"
            f"<td>{row.fiber}</td>"
            f"<td>{_fmt_num(row.injected_rv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.berv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.measured_rv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.bias_m_s, decimals=1)}</td>"
            f"<td>{_fmt_num(row.rv_err_m_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.inlier_scatter_m_s, decimals=3)}</td>"
            f"<td>{row.n_inlier_chunks} / {row.n_total_chunks}</td>"
            "</tr>"
        )
    if not body_rows:
        return ""

    return f"""
<section class="rv-summary">
<h2>RV performance summary</h2>
<table>
<caption>{_esc(header)}</caption>
<thead>
<tr>
<th>Fiber</th>
<th>Injected stellar RV (km/s)</th>
<th>BERV (km/s)</th>
<th>Measured RV (km/s)</th>
<th>Bias (m/s)</th>
<th>σ(⟨RV⟩) (m/s)</th>
<th>Inlier scatter (m/s)</th>
<th>Inlier chunks</th>
</tr>
</thead>
<tbody>
{os.linesep.join(body_rows)}
</tbody>
</table>
</section>
"""


def generate_e2e_preview_report(
    output_dir: str,
    *,
    exposure_base_name: str | None = None,
    gaia_target=None,
    stage2_model=None,
    injected_rv_km_s: float | None = None,
    fibers: list[int] | None = None,
) -> str:
    """
    Render PNG previews and write ``index.html`` summarizing the E2E run.

    Parameters
    ----------
    output_dir : str
        Directory containing the saved per-step FITS products (the E2E ``output_dir``).
    exposure_base_name : str or None
        Science exposure base name (e.g. ``HB.20260612.57502.95``). If ``None``, inferred
        from the products in ``output_dir``.
    gaia_target : optional
        :class:`~hispecdrp.simulations.e2e_gaia.E2EGaiaTarget` for a short header summary.
    stage2_model : optional
        Stage 2 L2 model for the RV performance table. When omitted, the table is skipped.
    injected_rv_km_s : float or None
        Template Doppler RV injected before rate-map generation. Defaults to
        ``gaia_target.injected_rv_km_s`` when available.
    fibers : list of int or None
        Fibers to include in the RV table (default ``[1, 2, 3]``).

    Returns
    -------
    str
        Absolute path to the written ``index.html``.
    """
    output_dir = os.path.abspath(output_dir)
    base = exposure_base_name or _infer_base_from_dir(output_dir)
    if not base:
        raise RuntimeError(f"Could not determine exposure base name in {output_dir!r}")

    previews_dir = os.path.join(output_dir, PREVIEWS_SUBDIR)
    os.makedirs(previews_dir, exist_ok=True)
    report_dir = output_dir
    report_path = os.path.join(output_dir, REPORT_HTML_NAME)

    stage0_items: list[PreviewItem] = []
    stage1_items: list[PreviewItem] = []

    for name in sorted(os.listdir(output_dir)):
        if not name.startswith(base + "_") or not name.endswith(".fits"):
            continue
        m = _PRODUCT_RE.match(name)
        if not m:
            continue
        group = m.group("group")
        token = m.group("token")
        path = os.path.join(output_dir, name)
        try:
            if group == "raw":
                item = _build_stage0_item(
                    path, token, kind="ramp_mean",
                    previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage0_items.append(item)
            elif group == "rate":
                item = _build_stage0_item(
                    path, token, kind="rate_map",
                    previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage0_items.append(item)
            elif group == "specuncal" and token == "specextract":
                item = _build_stage1_item(
                    path, token, previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage1_items.append(item)
        except Exception:
            logger.exception("Failed to build preview for %s", name)

    stage2_items = _collect_stage2_items(output_dir, base, report_dir)
    fiberflat_items = _collect_fiberflat_items(output_dir, report_dir)

    # Header summary
    summary_bits = [f"Exposure: <code>{_esc(base)}</code>"]
    if gaia_target is not None:
        meta = getattr(gaia_target, "target_meta", {}) or {}
        summary_bits.append(f"Target: <code>{_esc(meta.get('target_name', '?'))}</code>")
        summary_bits.append(f"Gaia DR3 <code>{_esc(getattr(gaia_target, 'source_id', '?'))}</code>")
        summary_bits.append(f"injected RV = {getattr(gaia_target, 'injected_rv_km_s', float('nan')):.3f} km/s")

    stages_html = []
    if stage0_items:
        stages_html.append(_render_stage_html("Stage 0 — detector reduction", stage0_items))
    if fiberflat_items:
        stages_html.append(
            _render_stage_html("Fiberflat calibration — trace positions", fiberflat_items)
        )
    if stage1_items:
        stages_html.append(_render_stage_html("Stage 1 — spectral extraction", stage1_items))
    if stage2_items:
        stages_html.append(_render_stage_html("Stage 2 — spectral calibration & RV", stage2_items))

    eff_injected_rv = injected_rv_km_s
    if eff_injected_rv is None and gaia_target is not None:
        eff_injected_rv = getattr(gaia_target, "injected_rv_km_s", None)
    eff_injected_berv = None
    if gaia_target is not None:
        eff_injected_berv = getattr(gaia_target, "berv_km_s", None)
    rv_table_html = ""
    if stage2_model is not None and eff_injected_rv is not None:
        from . import e2e_support

        perf_rows = e2e_support.collect_e2e_rv_performance_rows(
            stage2_model,
            injected_rv_km_s=float(eff_injected_rv),
            injected_berv_km_s=(
                float(eff_injected_berv)
                if eff_injected_berv is not None and np.isfinite(eff_injected_berv)
                else None
            ),
            fibers=fibers,
        )
        rv_table_html = _render_rv_performance_table_html(
            perf_rows,
            injected_rv_km_s=float(eff_injected_rv),
            injected_berv_km_s=(
                float(eff_injected_berv)
                if eff_injected_berv is not None and np.isfinite(eff_injected_berv)
                else None
            ),
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HISPEC E2E previews — {_esc(base)}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>HISPEC end-to-end simulation previews</h1>
<div class="subtitle">{' &middot; '.join(summary_bits)}</div>
{os.linesep.join(stages_html)}
{rv_table_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(
        "E2E previews: wrote %d stage-0, %d fiberflat, %d stage-1, %d stage-2 sections to %s",
        len(stage0_items), len(fiberflat_items), len(stage1_items), len(stage2_items), report_path,
    )
    return report_path
