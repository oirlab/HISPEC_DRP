"""
End-to-end HISPEC simulation: calibrations (KOA → user config → simulated), UTR ramp, Stage 0 → 1 → 2.

Chains helpers in :mod:`hispecdrp.simulations.simulate_up_the_ramp` and
:mod:`hispecdrp.simulations.e2e_support` with explicit hand-off of each stage product
to the next. When trace/fiberflat masters are not available from KOA or config, dummy
fiberflat rates and the fiberflat pipeline run between Stage 0 and Stage 1.

Run from the repository root with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

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
    # When True (default), trace_pos/trace_prof/fiberflat are derived from simulated
    # dummy fiberflats + ProcessFiberflatsPipeline using the same instrument model as
    # the science ramp. Stale KOA/local master trace products are not auto-loaded.
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
    calibration_kwargs: dict[str, Any] = field(default_factory=dict)


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


def run_e2e_simulation(config: E2EConfig) -> E2EResult:
    """
    Build a simulated ramp + calibrations, then run Stage 0, fiberflat pipeline, Stage 1, Stage 2.

    Calibrations are resolved in order: explicit paths in ``calibration_config``, KOA store
    (when ``use_master_calibrations`` is True), then simulated products for any missing keys.
    """
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
        auto_resolve_stage1_masters=not config.derive_stage1_trace_from_sim,
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
        shape=(4096, 4096),
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
    cal_dir = e2e_support.calibration_dir(out_dir_abs)

    if config.run_stage1:
        ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
        ff_list = [f for f in ff_list if f in (1, 2, 3)]
        if ff_list:
            use_master_trace_bundle = (
                cal_bundle.has_master_trace_bundle
                and not config.derive_stage1_trace_from_sim
            )
            if use_master_trace_bundle:
                logger.info(
                    "E2E: using trace_pos, trace_prof, and fiberflat masters "
                    "(explicit config and/or KOA); skipping dummy fiberflat simulation "
                    "and ProcessFiberflatsPipeline."
                )
                ff_bundle = {
                    "trace_pos": cal_bundle.trace_pos,
                    "trace_prof": cal_bundle.trace_prof,
                    "fiberflat": cal_bundle.fiberflat,
                }
            else:
                if config.derive_stage1_trace_from_sim:
                    logger.info(
                        "E2E: deriving trace_pos, trace_prof, and fiberflat from simulated "
                        "fiberflats (%s rate maps + ProcessFiberflatsPipeline).",
                        config.fiberflat_rate_map_method,
                    )
                if all(f in cal_bundle.fiberflat_rate_paths for f in ff_list):
                    logger.info("E2E: user fiberflat rate files + fiberflat pipeline + Stage 1.")
                    ff_paths = [
                        cal_bundle.fiberflat_rate_paths[f] for f in sorted(ff_list)
                    ]
                    for f in sorted(ff_list):
                        path = cal_bundle.fiberflat_rate_paths[f]
                        cal_bundle.sources[f"fiberflat_rate_fib{f}"] = f"config:{path}"
                else:
                    logger.info("E2E: simulated fiberflats + fiberflat pipeline + Stage 1.")
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
                    for f in sorted(ff_list):
                        fib_paths = [
                            p for p in ff_paths if f"FIBERFLAT-FIB{f}" in os.path.basename(p)
                        ]
                        label = fib_paths[0] if len(fib_paths) == 1 else f"{len(fib_paths)} coadd inputs"
                        cal_bundle.sources[f"fiberflat_rate_fib{f}"] = (
                            f"simulated_dummy_rate:{label}"
                        )
                ff_bundle = run_fiberflat_pipeline(
                    ff_paths,
                    output_dir=cal_dir,
                    prominence_factor=config.fiberflat_trace_prominence_factor,
                )
                for key in ("trace_pos", "trace_prof", "fiberflat"):
                    cal_bundle.sources[key] = "fiberflat_pipeline"

            if cal_bundle.wavesol is not None:
                wavesol_model = cal_bundle.wavesol
            else:
                wavesol_model = create_wavelength_solution_model_from_sim(
                    config.spectrograph,
                    meta={"mjd_start": float(ramp_mjd_start)},
                )
                cal_bundle.sources["wavesol"] = "simulated_hdf"

            stage1_model = e2e_support.run_stage1_simulation(
                stage0_model,
                ff_bundle["trace_pos"],
                ff_bundle["trace_prof"],
                wavesol_model=wavesol_model,
                output_dir=out_dir_abs,
                max_cores=config.max_cores,
            )
        else:
            logger.warning("E2E: run_stage1 is True but fiberflat_fibers is empty; skipping Stage 1/2.")

    if config.run_stage2 and stage1_model is not None and ff_bundle is not None:
        logger.info("E2E: Stage 2.")
        stage2_model = e2e_support.run_stage2_simulation(
            stage1_model,
            fiberflat_model=ff_bundle["fiberflat"],
            output_dir=out_dir_abs,
            max_cores=config.max_cores,
            stellar_rv_mask_path=config.stellar_rv_mask_path,
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
    )
