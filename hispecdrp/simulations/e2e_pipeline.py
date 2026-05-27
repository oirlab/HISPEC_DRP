"""
End-to-end HISPEC simulation: dummy calibrations, simulated UTR ramp, Stage 0 → 1 → 2.

This module chains the same building blocks as ``benchmarks/end_to_end_simulation.py``,
with **explicit** hand-off of each stage’s product to the next (Stage 0 rate model →
Stage 1 L2 input; Stage 1 output → Stage 2). Dummy fiberflats and the fiberflat
pipeline run between Stage 0 and Stage 1 when ``run_stage1`` is true.

**Repository layout:** importing :mod:`benchmarks.end_to_end_simulation` requires the
repository root (parent of the ``hispecdrp`` package) on ``sys.path``. The helper
:func:`ensure_repo_on_path` is called automatically by :func:`run_e2e_simulation`.
Running ``python -m hispecdrp.simulations.run_e2e_sim`` from the clone root is enough.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_repo_on_path() -> Path:
    """
    Insert the HISPEC_DRP repository root on ``sys.path`` so ``benchmarks`` resolves.

    ``hispecdrp/simulations/e2e_pipeline.py`` → parents[2] is the repo root.
    """
    repo_root = Path(__file__).resolve().parents[2]
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo_root


def import_benchmarks_end_to_end():
    """Load ``benchmarks.end_to_end_simulation`` (must exist in a source checkout)."""
    ensure_repo_on_path()
    return importlib.import_module("benchmarks.end_to_end_simulation")


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
    use_master_calibrations: bool = False
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


def run_e2e_simulation(config: E2EConfig) -> E2EResult:
    """
    Build a simulated ramp + calibrations, then run Stage 0, fiberflat pipeline, Stage 1, Stage 2.

    Stage flow
    ----------
    1. **Ramp + calibrations** — :func:`benchmarks.end_to_end_simulation.create_ramp_and_calibrations`
       (BT-Settl template or user spectrum; dummy or master cals).
    2. **Stage 0** — ramp and cal dict → rate/slope product.
    3. **Fiberflats** (if ``run_stage1``) — dummy per-fiber rates under ``<output_dir>/cals/``,
       then :class:`~hispecdrp.pipelines.calibration.process_fiberflats_pipeline.ProcessFiberflatsPipeline`.
    4. **Stage 1** — Stage 0 product + trace models + sim wavelength solution → L2-style product.
    5. **Stage 2** (if ``run_stage2`` and Stage 1 ran) — Stage 1 product + fiberflat bundle.
    """
    m = import_benchmarks_end_to_end()
    import os

    out_dir_abs = m._ensure_output_directory(config.output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    m._ensure_koa_calibration_cache_env(out_dir_abs)

    logger.info("E2E: building ramp and calibrations (output_dir=%s).", out_dir_abs)
    ramp_model, calibration_models = m.create_ramp_and_calibrations(
        rate_map=None,
        shape=(4096, 4096),
        fibers=config.fibers,
        output_dir=out_dir_abs,
        calibration_models=None,
        use_master_calibrations=config.use_master_calibrations,
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

    stage0_prim_args = m.get_default_stage0_prim_args()
    logger.info("E2E: Stage 0 (ramp → rate/slope).")
    stage0_model = m.run_stage0_simulation(
        ramp_model,
        calibration_models,
        output_dir=out_dir_abs,
        prim_args=stage0_prim_args,
        max_cores=config.max_cores,
        profile_records=None,
    )

    stage1_model = None
    stage2_model = None
    ff_bundle = None
    cal_dir = m._calibration_dir(out_dir_abs)

    if config.run_stage1:
        ff_list = config.fiberflat_fibers if config.fiberflat_fibers is not None else [1, 2, 3]
        ff_list = [f for f in ff_list if f in (1, 2, 3)]
        if ff_list:
            logger.info("E2E: dummy fiberflats + fiberflat pipeline + Stage 1.")
            ff_paths = m.create_dummy_fiberflat_rate_files(
                ff_list,
                config.spectrograph,
                out_dir_abs,
                calibration_models,
                readtime=config.readtime,
                n_reads=config.n_reads,
                max_cores=config.max_cores,
                stage0_prim_args=stage0_prim_args,
            )
            ff_bundle = m.run_fiberflat_pipeline(ff_paths, output_dir=cal_dir)
            wavesol_model = m.create_wavelength_solution_model_from_sim(config.spectrograph)
            stage1_prim_args = m.get_default_stage1_prim_args(max_cores=config.max_cores)
            stage1_model = m.run_stage1_simulation(
                stage0_model,
                ff_bundle["trace_pos"],
                ff_bundle["trace_prof"],
                prim_args=stage1_prim_args,
                wavesol_model=wavesol_model,
                output_dir=out_dir_abs,
                max_cores=config.max_cores,
                profile_records=None,
            )
        else:
            logger.warning("E2E: run_stage1 is True but fiberflat_fibers is empty; skipping Stage 1/2.")

    if config.run_stage2 and stage1_model is not None and ff_bundle is not None:
        logger.info("E2E: Stage 2.")
        stage2_prim_args = m.get_default_stage2_prim_args(max_cores=config.max_cores)
        stage2_model = m.run_stage2_simulation(
            stage1_model,
            fiberflat_model=ff_bundle["fiberflat"],
            prim_args=stage2_prim_args,
            output_dir=out_dir_abs,
            max_cores=config.max_cores,
            profile_records=None,
        )
    elif config.run_stage2:
        logger.warning("E2E: run_stage2 is True but Stage 1 or fiberflat bundle missing; skipping.")

    return E2EResult(
        ramp_model=ramp_model,
        stage0_model=stage0_model,
        calibration_models=calibration_models,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        fiberflat_bundle=ff_bundle,
    )
