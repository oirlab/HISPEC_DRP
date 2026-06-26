#!/usr/bin/env python
"""
Resolve and materialize E2E calibrations only (no science ramp or Stages 0–2).

Run from the **repository root** with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_cals

Downloads or creates calibration FITS under ``output_dir/cals/``, validates provenance
(KOA, user config, simulated, etc.), logs a full summary, and writes
``cals/calibration_manifest.json``.
"""

from __future__ import annotations

import logging

from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START
from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_calibrations
from hispecdrp.simulations.e2e_support import repository_root

# Same defaults as ``run_e2e_sim.py`` for spectrograph / KOA / fiberflat behaviour.
DEFAULT_SPECTROGRAPH = "BSPEC"


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    output_dir = str(repository_root() / "test_data")

    config = E2EConfig(
        output_dir=output_dir,
        spectrograph=DEFAULT_SPECTROGRAPH,
        mjd_start=DEFAULT_E2E_MJD_START,
        fiberflat_fibers=[1, 2, 3],
        use_master_calibrations=True,
        stage1_trace_source="pipeline",
        materialize_stage1_trace=True,
        calibration_config=None,
        connect_remote=True,
        sync_caldb_first=True,
        download_missing_koa=True,
        calibrations_only=True,
        write_calibration_manifest=True,
        run_stage1=False,
        run_stage2=False,
        calibration_kwargs={},
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
