#!/usr/bin/env python
"""
Infinite signal-to-noise RV validation entry point.

Runs the full E2E simulation with noiseless rate maps, HDF-truth Stage-1
calibrations (no fiber-flat pipeline trace walker), and high-precision
``StellarRVPrimitive`` settings, then evaluates recovery against the injected
Gaia barycentric RV.

Run from the repository root with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_infinite_snr_rv

Optional: set ``infinite_snr_rv_assert=True`` in ``main()`` to raise when
default thresholds are not met (max |bias| 0.2 km/s, fiber spread 0.15 km/s).
By default the run logs the evaluation only (assertions off until RV tuning converges).
"""

from __future__ import annotations

import logging

from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START
from hispecdrp.simulations.e2e_infinite_snr_rv import (
    apply_infinite_snr_rv_e2e_config,
    assert_infinite_snr_rv_recovery,
    log_infinite_snr_rv_evaluation,
)
from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation
from hispecdrp.simulations.e2e_support import repository_root
from hispecdrp.simulations.run_e2e_sim import (
    DEFAULT_GAIA_SOURCE_ID,
    DEFAULT_TEMPLATE,
    DEFAULT_TEMPLATE_2MASS_BAND,
    DEFAULT_TEMPLATE_MAG_AB,
)

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    output_dir = str(repository_root() / "test_data" / "infinite_snr_rv")
    config = apply_infinite_snr_rv_e2e_config(
        E2EConfig(
            output_dir=output_dir,
            n_reads=10,
            readtime=1.7,
            max_cores=1,
            spectrograph="BSPEC",
            mjd_start=DEFAULT_E2E_MJD_START,
            fibers=[1, 2, 3],
            fiberflat_fibers=[1, 2, 3],
            use_master_calibrations=True,
            calibration_config=None,
            run_stage1=True,
            run_stage2=True,
            use_gaia_target=True,
            gaia_source_id=DEFAULT_GAIA_SOURCE_ID,
            gaia_max_g_mag=8.0,
            gaia_random_seed=None,
            template_mag_ab=DEFAULT_TEMPLATE_MAG_AB,
            template_2mass_band=DEFAULT_TEMPLATE_2MASS_BAND,
            template_mag_from_2mass=True,
            template_filename=DEFAULT_TEMPLATE,
            overwrite_ramp=False,
            fiberflat_rate_map_method="native",
            fiberflat_pyechelle_exposure_time_s=300.0,
            flat_scale=2e-14,
            flat_temperature_k=1000.0,
            fiberflat_coadd_exposures=1,
            fiberflat_duplicate_single_reduction=False,
            fiberflat_trace_prominence_factor=None,
            calibration_kwargs={},
            infinite_snr_rv_validation=True,
            infinite_snr_rv_assert=False,
        )
    )

    res = run_e2e_simulation(config)
    if res.infinite_snr_rv_evaluation is not None:
        log_infinite_snr_rv_evaluation(res.infinite_snr_rv_evaluation)
        if config.infinite_snr_rv_assert:
            assert_infinite_snr_rv_recovery(res.infinite_snr_rv_evaluation)

    logger.info(
        "Infinite-S/N RV validation done: stage2=%s passed=%s",
        type(res.stage2_model).__name__ if res.stage2_model is not None else None,
        res.infinite_snr_rv_evaluation.passed_default_thresholds
        if res.infinite_snr_rv_evaluation is not None
        else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
