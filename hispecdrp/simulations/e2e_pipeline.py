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
                from .sim_hdf_calibrations import materialize_stage1_bundle_from_sim_hdf

                extraction_aperture = 10.0
                if config.infinite_snr_rv_validation:
                    from .e2e_infinite_snr_rv import merge_infinite_snr_stage1_prim_args

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
        from .e2e_infinite_snr_rv import apply_infinite_snr_rv_e2e_config

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
        from .e2e_infinite_snr_rv import (
            merge_infinite_snr_stage1_prim_args,
            merge_infinite_snr_stage2_prim_args,
        )

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
                "E2E: stellar RV hybrid grid center ccf_input_velocity=%.3f m/s (%.4f km/s)",
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
        from .e2e_infinite_snr_rv import (
            assert_infinite_snr_rv_recovery,
            evaluate_infinite_snr_rv_recovery,
            log_infinite_snr_rv_evaluation,
        )

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
            from . import e2e_previews

            base_name = e2e_previews.infer_exposure_base_name(
                stage0_model=stage0_model,
                stage1_model=stage1_model,
                stage2_model=stage2_model,
            )
            html_path = e2e_previews.generate_e2e_preview_report(
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
