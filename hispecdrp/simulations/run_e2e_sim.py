#!/usr/bin/env python
"""
Hard-coded entry point for :mod:`hispecdrp.simulations.e2e_pipeline`.

Run from the **repository root** with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim

Adjust the literals in ``main()`` below for output directory, reads, cores, etc.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Repo root on path (needed for ``benchmarks`` when hispecdrp is not editable-only).
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    output_dir = str(_repo / "test_data")
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
        use_master_calibrations=False,
        run_stage1=True,
        run_stage2=True,
        poisson_noise=True,
        convert_to_uint16=True,
        radial_velocity_km_s=0.0,
        template_mag_ab=None,
        template_2mass_band=None,
        calibration_kwargs={},
    )

    res = run_e2e_simulation(config)
    logging.getLogger(__name__).info(
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
