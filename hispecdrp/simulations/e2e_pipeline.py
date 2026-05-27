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

from . import e2e_calibrations, e2e_support
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
    calibration_config: str | dict | None = None
    instrument_name: str | None = None
    instrument_era: str = "0.0.1"
    mjd_start: float | None = None
    datetime_obs: str | None = None
    connect_remote: bool = True
    sync_caldb_first: bool = True
    run_stage1: bool = True
    run_stage2: bool = True
    poisson_noise: bool = True
    convert_to_uint16: bool = True
    radial_velocity_km_s: float = 0.0
    template_mag_ab: float | None = None
    template_2mass_band: str | None = None
    calibration_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class E2EResult:
    ramp_model: Any
    stage0_model: Any
    calibration_models: dict
    stage1_model: Any | None
    stage2_model: Any | None
    fiberflat_bundle: dict | None
    calibration_sources: dict[str, str] = field(default_factory=dict)


def run_e2e_simulation(config: E2EConfig) -> E2EResult:
    """
    Build a simulated ramp + calibrations, then run Stage 0, fiberflat pipeline, Stage 1, Stage 2.

    Calibrations are resolved in order: explicit paths in ``calibration_config``, KOA store
    (when ``use_master_calibrations`` is True), then simulated products for any missing keys.
    """
    out_dir_abs = e2e_support.ensure_output_directory(config.output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    cal_bundle = e2e_calibrations.resolve_e2e_calibrations(
        output_dir=out_dir_abs,
        spectrograph=config.spectrograph,
        use_koa=config.use_master_calibrations,
        calibration_config=config.calibration_config,
        instrument_name=config.instrument_name,
        instrument_era=config.instrument_era,
        mjd_start=config.mjd_start,
        datetime_obs=config.datetime_obs,
        connect_remote=config.connect_remote,
        sync_caldb_first=config.sync_caldb_first,
    )

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
        radial_velocity_km_s=config.radial_velocity_km_s,
        template_mag_ab=config.template_mag_ab,
        template_2mass_band=config.template_2mass_band,
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
            if cal_bundle.has_master_trace_bundle:
                logger.info("E2E: using trace/fiberflat masters from config or KOA; Stage 1.")
                ff_bundle = {
                    "trace_pos": cal_bundle.trace_pos,
                    "trace_prof": cal_bundle.trace_prof,
                    "fiberflat": cal_bundle.fiberflat,
                }
            else:
                if all(f in cal_bundle.fiberflat_rate_paths for f in ff_list):
                    logger.info("E2E: user fiberflat rate files + fiberflat pipeline + Stage 1.")
                    ff_paths = [
                        cal_bundle.fiberflat_rate_paths[f] for f in sorted(ff_list)
                    ]
                    for f in sorted(ff_list):
                        path = cal_bundle.fiberflat_rate_paths[f]
                        cal_bundle.sources[f"fiberflat_rate_fib{f}"] = f"config:{path}"
                else:
                    logger.info("E2E: dummy fiberflats + fiberflat pipeline + Stage 1.")
                    ff_paths = create_dummy_fiberflat_rate_files(
                        ff_list,
                        config.spectrograph,
                        out_dir_abs,
                        calibration_models,
                        readtime=config.readtime,
                        n_reads=config.n_reads,
                        max_cores=config.max_cores,
                        stage0_prim_args=stage0_prim_args,
                    )
                    for f, path in zip(sorted(ff_list), ff_paths):
                        cal_bundle.sources[f"fiberflat_rate_fib{f}"] = (
                            f"simulated_dummy_rate:{path}"
                        )
                ff_bundle = run_fiberflat_pipeline(ff_paths, output_dir=cal_dir)
                for key in ("trace_pos", "trace_prof", "fiberflat"):
                    cal_bundle.sources[key] = "fiberflat_pipeline"

            if cal_bundle.wavesol is not None:
                wavesol_model = cal_bundle.wavesol
            else:
                wavesol_model = create_wavelength_solution_model_from_sim(
                    config.spectrograph
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
        injected_rv_km_s=config.radial_velocity_km_s,
        fibers=config.fibers,
    )

    return E2EResult(
        ramp_model=ramp_model,
        stage0_model=stage0_model,
        calibration_models=calibration_models,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        fiberflat_bundle=ff_bundle,
        calibration_sources=dict(cal_bundle.sources),
    )
