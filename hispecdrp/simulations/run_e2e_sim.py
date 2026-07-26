#!/usr/bin/env python
"""
Hard-coded entry point for :mod:`hispecdrp.simulations.e2e_pipeline`.

Run from the **repository root** with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim
    python -m hispecdrp.simulations.run_e2e_sim --calibrations-only
    python -m hispecdrp.simulations.run_e2e_sim --infinite-snr-rv

Adjust the literals in ``_base_config()`` below for output directory, reads, cores, etc.
Set ``calibration_config`` to a YAML path (see
``docs/source/simulations/e2e_calibrations.example.yaml``) to pin specific FITS files;
set ``use_master_calibrations=False`` to skip KOA.

Stage 1 trace / fiberflat products (``mastercal-tracepos``, ``mastercal-traceprof``,
``mastercal-fiberflat``) are controlled by ``stage1_trace_source``:

* ``"pipeline"`` (default): simulate per-fiber fiberflat rate maps and run
  ``ProcessFiberflatsPipeline`` — KOA Stage 1 masters are not downloaded.
* ``"koa"``: fetch/use KOA or local-cache master cals instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START
from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_calibrations, run_e2e_simulation
from hispecdrp.simulations.e2e_support import repository_root
from hispecdrp.utils.line_mask_utils import packaged_default_line_mask_path

# Default simulation target: 4500 K K-type star.
# Template: bundled BT-Settl lte049.0-2.5 (K-type giant) and its delta-width CCF mask.
DEFAULT_TEMPLATE = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"
# Template brightness: by default the template is normalized to the Gaia target star's
# real 2MASS magnitude (in DEFAULT_TEMPLATE_2MASS_BAND, converted Vega->AB) so the
# simulated flux matches the actual star. DEFAULT_TEMPLATE_MAG_AB is only used as a
# fallback when no 2MASS counterpart is found (or template_mag_from_2mass=False).
DEFAULT_TEMPLATE_MAG_AB = 10.0
DEFAULT_TEMPLATE_2MASS_BAND = "J"
# Gaia DR3 source matching the K-type giant used for coordinates/parallax/PM/RV.
DEFAULT_GAIA_SOURCE_ID = "3438436918943830144"


def _base_config(*, output_dir: str | None = None) -> E2EConfig:
    """Shared defaults for the full E2E run and CLI mode variants."""
    out = output_dir or str(repository_root() / "test_data")
    return E2EConfig(
        output_dir=out,
        n_reads=10,
        readtime=1.7,
        max_cores=1,
        spectrograph="BSPEC",
        mjd_start=DEFAULT_E2E_MJD_START,
        fibers=[1, 2, 3],
        fiberflat_fibers=[1, 2, 3],
        # Simulate fiber flats and run the fiber flat pipeline for both spectrographs so
        # trace-position *_preview.png plots are produced for BSPEC and RSPEC. Science
        # Stage 1/2 still run on ``spectrograph`` above (BSPEC); RSPEC is inspection-only.
        fiberflat_spectrographs=["BSPEC", "RSPEC"],
        use_master_calibrations=True,
        # Build trace_pos / trace_prof / fiberflat with ProcessFiberflatsPipeline
        # (simulated dummy fiberflat rates). Set to "koa" to use KOA master cals instead.
        stage1_trace_source="pipeline",
        calibration_config=None,
        run_stage1=True,
        run_stage2=True,
        poisson_noise=True,
        convert_to_uint16=True,
        radial_velocity_km_s=0.0,
        use_gaia_target=True,
        gaia_source_id=DEFAULT_GAIA_SOURCE_ID,
        gaia_max_g_mag=8.0,
        gaia_random_seed=None,
        template_mag_ab=DEFAULT_TEMPLATE_MAG_AB,
        template_2mass_band=DEFAULT_TEMPLATE_2MASS_BAND,
        template_mag_from_2mass=True,
        template_filename=DEFAULT_TEMPLATE,
        stellar_rv_mask_path=packaged_default_line_mask_path(),
        overwrite_ramp=True,
        make_preview_html=True,
        fiberflat_rate_map_method="native",
        fiberflat_pyechelle_exposure_time_s=300.0,
        flat_scale=2e-14,
        flat_temperature_k=1000.0,
        fiberflat_coadd_exposures=1,
        fiberflat_duplicate_single_reduction=False,
        fiberflat_trace_prominence_factor=None,
        calibration_kwargs={},
    )


def _run_full(config: E2EConfig) -> int:
    res = run_e2e_simulation(config)
    log = logging.getLogger(__name__)
    log.info(
        "Done: ramp=%s stage0=%s stage1=%s stage2=%s cal_keys=%s",
        type(res.ramp_model).__name__,
        type(res.stage0_model).__name__,
        type(res.stage1_model).__name__ if res.stage1_model is not None else None,
        type(res.stage2_model).__name__ if res.stage2_model is not None else None,
        sorted(res.calibration_models.keys()),
    )
    return 0


def _run_calibrations_only(config: E2EConfig) -> int:
    """Resolve/materialize calibrations only (no science ramp or Stages 0–2)."""
    config.calibrations_only = True
    config.write_calibration_manifest = True
    config.materialize_stage1_trace = True
    config.connect_remote = True
    config.sync_caldb_first = True
    config.download_missing_koa = True
    config.run_stage1 = False
    config.run_stage2 = False
    config.make_preview_html = False

    res = run_e2e_calibrations(config)
    log = logging.getLogger(__name__)
    n_warn = len(res.summary.validation_warnings) if res.summary else 0
    log.info(
        "Calibrations-only run complete: %d product(s), %d validation warning(s), "
        "manifest in %s/cals/calibration_manifest.json",
        len(res.summary.rows) if res.summary else 0,
        n_warn,
        res.output_dir,
    )
    return 1 if n_warn else 0


def _run_infinite_snr_rv(config: E2EConfig) -> int:
    """Noiseless / HDF-truth Stage-1 / high-precision RV validation run."""
    from hispecdrp.simulations.e2e_pipeline import (
        apply_infinite_snr_rv_e2e_config,
        assert_infinite_snr_rv_recovery,
        log_infinite_snr_rv_evaluation,
    )

    # Dedicated subdirectory so full-E2E products are not overwritten.
    if Path(config.output_dir).name != "infinite_snr_rv":
        config.output_dir = str(Path(config.output_dir) / "infinite_snr_rv")
    config.overwrite_ramp = False
    config.infinite_snr_rv_validation = True
    config.infinite_snr_rv_assert = False
    config = apply_infinite_snr_rv_e2e_config(config)

    res = run_e2e_simulation(config)
    log = logging.getLogger(__name__)
    if res.infinite_snr_rv_evaluation is not None:
        log_infinite_snr_rv_evaluation(res.infinite_snr_rv_evaluation)
        if config.infinite_snr_rv_assert:
            assert_infinite_snr_rv_recovery(res.infinite_snr_rv_evaluation)

    log.info(
        "Infinite-S/N RV validation done: stage2=%s passed=%s",
        type(res.stage2_model).__name__ if res.stage2_model is not None else None,
        res.infinite_snr_rv_evaluation.passed_default_thresholds
        if res.infinite_snr_rv_evaluation is not None
        else None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HISPEC DRP end-to-end simulation entry point.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--calibrations-only",
        action="store_true",
        help="Resolve and materialize calibrations only (no science ramp / Stages 0–2).",
    )
    mode.add_argument(
        "--infinite-snr-rv",
        action="store_true",
        help=(
            "Noiseless rate map + sim-HDF Stage-1 cals + high-precision stellar RV; "
            "evaluate recovery vs injected Gaia barycentric RV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <repo>/test_data, or …/infinite_snr_rv for that mode).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = _base_config(output_dir=args.output_dir)

    if args.calibrations_only:
        return _run_calibrations_only(config)
    if args.infinite_snr_rv:
        return _run_infinite_snr_rv(config)
    return _run_full(config)


if __name__ == "__main__":
    raise SystemExit(main())
