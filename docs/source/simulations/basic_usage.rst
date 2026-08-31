Basic Usage
===========

Short worked example for running HISPEC DRP end-to-end from a 1D stellar
template: build a simulated BSPEC ramp, reduce it through Stage 0–2, and
inspect the stellar-RV product using line-by-line ``ccf_line_mask`` with the
packaged **BT-Settl lte040** template.

The packaged ``*.ccfopt.delta.mas`` mask paired with lte040 is calibrated on
**ideal** noiseless L2 (see :doc:`examples/ideal_spectrum_rv_recovery`).  For
full ZEMAX→ramp→extraction E2E, build a mask on a blaze L2 at inject RV = 0
and reuse a shared Stage-1 calibration bundle (same pattern as
``rv_recovery_sim_line_by_line.ipynb``).

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../_static/notebooks/basic_usage.ipynb"
        data-notebook-download="basic_usage.ipynb"
        download="basic_usage.ipynb">Download as notebook</a>
   </p>

.. warning::
   **Setup.** Set these in your shell / notebook kernel before running (or let
   the first cell create a local cache under ``OUT``):

.. list-table::
   :header-rows: 1
   :widths: 28, 28, 44

   * - Variable
     - Required when
     - Purpose
   * - ``KOA_CALIBRATION_CACHE``
     - Always
     - Local KOA calibration cache directory (and SQLite DB)
   * - ``KECK_OBSERVER_EMAIL``
     - Optional remote sync
     - Keck Observer account email
   * - ``KECK_OBSERVER_PASSWORD``
     - Optional remote sync
     - Keck Observer account password

Imports and output directory
----------------------------

.. code-block:: python

   import os
   import hispecdrp

   OUT = "/path/to/hispec_pipeline_example"
   os.makedirs(OUT, exist_ok=True)
   os.environ.setdefault(
       "KOA_CALIBRATION_CACHE",
       os.path.join(OUT, "cals", "KOA_CALIBRATION_CACHE"),
   )
   print(f"OUT={OUT}")

1. KOA caldb
------------

The pipeline resolves KOA calibrations by default when reducing real data.
This step is still needed here so ``create_ramp_from_spectrum`` can populate
and use a local cache while building the simulated UTR.

.. code-block:: python

   from hispecdrp.calibrations import verify_koa_calibrations

   SPECTROGRAPH = "BSPEC"
   summary = verify_koa_calibrations(
       SPECTROGRAPH, ignore_missing_cal_types=["lfc_wavesol"]
   )
   print(f"{summary['spectrograph']}: {summary['n_records']} records")

2. Shared Stage-1 cals and E2E line mask (inject RV = 0)
--------------------------------------------------------

Build a **one-time** Stage-1 bundle (sim traces / fiberflat / Aidan WAVE with
deg-0 order ZP) from a calibration injection at inject RV = 0, reduce it to
blaze L2, and write an E2E ``ccfopt`` mask under ``OUT/masks/``.  Later science
injections reuse the cached bundle under ``OUT/_shared_stage1``.

.. code-block:: python

   from pathlib import Path

   from hispecdrp.simulations.e2e_pipeline import (
       DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE,
       build_e2e_ccfopt_mask_from_blaze_l2,
       ensure_shared_stage1_cals_for_rv_sim,
       run_sim_through_blaze,
   )
   from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum

   TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
   FIBERS = [2]
   SHARED_DIR = os.path.join(OUT, "_shared_stage1")
   MASK_DIR = os.path.join(OUT, "masks")
   os.makedirs(MASK_DIR, exist_ok=True)
   MASK_PATH = os.path.join(MASK_DIR, "lte040.e2e.ccfopt.delta.mas")

   CAL_RV_KM_S = 0.0
   cal_dir = os.path.join(OUT, "calibration_inj0")
   cal_ramp, _, cal_rate_map = create_ramp_from_spectrum(
       cal_dir,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       template_filename=TEMPLATE,
       mag_ab=3.05,
       filter_band="J",
       seed=0,
       radial_velocity_km_s=CAL_RV_KM_S,
       barycentric_velocity_km_s=0.0,
       inject_rv_via="spectrum",
   )
   shared_cals = ensure_shared_stage1_cals_for_rv_sim(
       SHARED_DIR,
       rate_map=cal_rate_map,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       known_rv_m_s=0.0,
       extraction_aperture=DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE,
       ramp_meta=cal_ramp.meta,
   )
   cal_l2 = run_sim_through_blaze(
       cal_ramp,
       shared_cals,
       output_dir=cal_dir,
       fibers=FIBERS,
   )
   if not Path(MASK_PATH).is_file():
       build_e2e_ccfopt_mask_from_blaze_l2(
           cal_l2,
           TEMPLATE,
           MASK_PATH,
           fiber=FIBERS[0],
           masks_directory=MASK_DIR,
       )
   print(f"E2E mask: {MASK_PATH}")

3. Science UTR
--------------

Doppler-shift by ``INJECTED_RV``, paint with ZEMAX, and build the science UTR.
Stage-0 KOA masters are registered into the local caldb as before.

.. code-block:: python

   INJECTED_RV = 1.0  # km/s

   ramp, _, rate_map = create_ramp_from_spectrum(
       OUT,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       template_filename=TEMPLATE,
       mag_ab=3.05,
       filter_band="J",
       seed=0,
       radial_velocity_km_s=INJECTED_RV,
       barycentric_velocity_km_s=0.0,
       inject_rv_via="spectrum",
   )
   # Reuse the cached shared Stage-1 bundle (no rebuild).
   shared_cals = ensure_shared_stage1_cals_for_rv_sim(
       SHARED_DIR,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
   )

4. Stage 0–2
------------

Same pipeline classes as real data.  Pass the shared trace / fiberflat / WAVE
frames explicitly, skip ``barycorr`` when BERV = 0, and run line-by-line
``ccf_line_mask`` with the E2E mask from step 2.

.. code-block:: python

   from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline
   from hispecdrp.simulations.e2e_pipeline import stellar_rv_kwargs_for_e2e_sim

   l0 = Stage0Pipeline.call(ramp, output_dir=OUT)

   do_fibers = [str(f) for f in FIBERS]
   l1 = Stage1Pipeline.call(
       l0,
       output_dir=OUT,
       primitives={
           "spec_extract": {
               "do_fibers": do_fibers,
               "method": "box",
               "extraction_aperture": int(DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE),
               "tracepos": shared_cals["trace_pos"],
               "traceprof": shared_cals["trace_prof"],
           },
           "set_wavesol": {"wavesol": shared_cals["wavesol"]},
       },
   )

   injected_m_s = float(INJECTED_RV) * 1000.0
   l2 = Stage2Pipeline.call(
       l1,
       output_dir=OUT,
       primitives={
           "blaze_corr": {
               "do_fibers": do_fibers,
               "fiberflat": shared_cals["fiberflat"],
               "smooth_blaze": True,
           },
           "sky_sub": {"run": False},
           "barycorr": {"run": False},
           "stellar_rv": stellar_rv_kwargs_for_e2e_sim(
               INJECTED_RV,
               MASK_PATH,
               fibers=FIBERS,
               save_preview=True,
           ),
       },
   )

   print(f"L2: {l2.meta.model_type}  {l2.meta.filename}")
   for fib in FIBERS:
       avg = getattr(l2, f"rvccfavg_fib{fib}", None)
       if avg is not None and len(avg):
           rv = float(avg["RV"][0])
           err = float(avg["RV_ERR"][0])
           print(f"fiber {fib} <RV> = {rv:.4f} +/- {err:.4f} m/s")
           print(f"fiber {fib} bias = {rv - injected_m_s:.4f} m/s")

5. Consistency check: stellar RV preview
----------------------------------------

The Stage-2 stellar-RV HTML preview is written next to the FITS product:

.. code-block:: python

   preview = l2.get_output_path(output_dir=OUT, suffix="stellar_rv").replace(
       ".fits", "_preview.html"
   )
   print(preview)

Alternative: all-in-one E2E simulation
--------------------------------------

To run the full end-to-end simulation from the command line:

.. code-block:: bash

   python -m hispecdrp.simulations.run_e2e_sim

.. code-block:: python

   from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation

   config = E2EConfig(
       output_dir=OUT,
       spectrograph="BSPEC",
       fibers=[2],
       n_reads=10,
       derive_stage1_trace_from_sim=True,
   )
   # result = run_e2e_simulation(config)
