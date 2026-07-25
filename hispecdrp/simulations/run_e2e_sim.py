#!/usr/bin/env python
"""
Hard-coded entry point for :mod:`hispecdrp.simulations.e2e_pipeline`.

Run from the **repository root** with the ``hispec`` conda env::

    python -m hispecdrp.simulations.run_e2e_sim

Adjust the literals in ``main()`` below for output directory, reads, cores, etc.
Set ``calibration_config`` to a YAML path (see ``e2e_calibrations.example.yaml``)
to pin specific FITS files; set ``use_master_calibrations=False`` to skip KOA.

Stage 1 trace / fiberflat products (``mastercal-tracepos``, ``mastercal-traceprof``,
``mastercal-fiberflat``) are controlled by ``stage1_trace_source``:

* ``"pipeline"`` (default): simulate per-fiber fiberflat rate maps and run
  ``ProcessFiberflatsPipeline`` — KOA Stage 1 masters are not downloaded.
* ``"koa"``: fetch/use KOA or local-cache master cals instead.
"""

from __future__ import annotations

import logging

from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START
from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation
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
