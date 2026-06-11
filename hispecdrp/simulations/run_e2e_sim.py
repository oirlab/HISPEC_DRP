#!/usr/bin/env python
"""
Hard-coded entry point for :mod:`hispecdrp.simulations.e2e_pipeline`.

Run from the **repository root** with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim

Adjust the literals in ``main()`` below for output directory, reads, cores, etc.
Set ``calibration_config`` to a YAML path (see ``e2e_calibrations.example.yaml``)
to pin specific FITS files; set ``use_master_calibrations=False`` to skip KOA.
"""

from __future__ import annotations

import logging

from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation
from hispecdrp.simulations.e2e_support import repository_root


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    output_dir = str(repository_root() / "test_data")
    n_reads = 10
    max_cores = 1
    spectrograph = "BSPEC"

    config = E2EConfig(
        output_dir=output_dir,
        n_reads=n_reads,
        readtime=1.7,
        max_cores=max_cores,
        spectrograph=spectrograph,
        fibers=[1, 2, 3],
        fiberflat_fibers=[1, 2, 3],
        use_master_calibrations=True,
        calibration_config=None,
        run_stage1=True,
        run_stage2=True,
        poisson_noise=True,
        convert_to_uint16=True,
        radial_velocity_km_s=0.0,
        template_mag_ab=None,
        template_2mass_band=None,
        overwrite_ramp=True,
        calibration_kwargs={},
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
