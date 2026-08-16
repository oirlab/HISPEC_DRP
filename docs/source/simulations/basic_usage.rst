Basic Usage
===========

This page is a short worked example for running HISPEC DRP end-to-end from a
1D stellar template: build a simulated BSPEC ramp, reduce it through Stage 0–2,
and inspect the stellar-RV product.

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../_static/notebooks/basic_usage.ipynb"
        data-notebook-download="basic_usage.ipynb"
        download="basic_usage.ipynb">Download as notebook</a>
   </p>

**Environment variables** (set these in your shell / notebook kernel before running):

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

If ``KOA_CALIBRATION_CACHE`` is unset, the setup cell sets it under ``OUT``.

Imports and output directory
----------------------------

.. code-block:: python

   import os
   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   # OUTPUT DIRECTORY - update this to your desired output directory
   OUT = "/path/to/hispec_pipeline_example"
   os.makedirs(OUT, exist_ok=True)

   os.environ.setdefault(
       "KOA_CALIBRATION_CACHE",
       os.path.join(OUT, "cals", "KOA_CALIBRATION_CACHE"),
   )
   print("output:", OUT)
   print("KOA_CALIBRATION_CACHE:", os.environ["KOA_CALIBRATION_CACHE"])

1. Sync / verify KOA caldb (BSPEC)
----------------------------------

Pull BSPEC calibration metadata into the local caldb and download any missing
files.

.. code-block:: python

   from hispecdrp.calibrations import verify_koa_calibrations

   SPECTROGRAPH = "BSPEC"

   try:
       summary = verify_koa_calibrations(
           SPECTROGRAPH,
           ignore_missing_cal_types=["lfc_wavesol"],
       )
   except RuntimeError as exc:
       msg = str(exc)
       if "Failed to query metadata" not in msg and "404" not in msg:
           raise
       print(
           f"Remote KOA sync failed ({exc.__class__.__name__}); "
           "using local cache only."
       )
       summary = verify_koa_calibrations(
           SPECTROGRAPH,
           connect_remote=False,
           sync_remote=False,
           ignore_missing_cal_types=["lfc_wavesol"],
       )

   print(
       f"{summary['spectrograph']} caldb: {summary['n_records']} record(s); "
       f"downloaded {summary['n_downloaded']}/{summary['n_missing']} "
       f"missing file(s)"
   )
   print("types:", summary["cal_types"])

2. Simulated rate map from a 1D spectrum to ``RampModel``
---------------------------------------------------------

Load a BT-Settl template, densify it, Doppler-shift the densified 1D spectrum by
``radial_velocity_km_s`` (``inject_rv_via='spectrum'``), paint with ZEMAX, then
build a raw UTR ``RampModel``. Stage-0 KOA masters plus simulation Stage-1
products (packaged line-ZP WAVE, traces, fiberflat) are registered into the
local caldb.

.. code-block:: python

   from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum

   FIBERS = [2]

   # Injected stellar RV (km/s).
   INJECTED_RV = 1.0

   ramp, cals, rate_map = create_ramp_from_spectrum(
       OUT,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       template_filename="lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
       mag_ab=3.05,
       filter_band="J",  # BSPEC (yJ)
       seed=0,
       radial_velocity_km_s=INJECTED_RV,
       barycentric_velocity_km_s=0.0,
       inject_rv_via="spectrum",
   )

   print(f"ramp type: {ramp.meta.model_type}")
   print(f"ramp filename: {ramp.meta.filename}")
   print(f"ramp data shape: {ramp.data.shape}")

3. Stage 0 to 2 via the default pipelines
-----------------------------------------

Reduce the simulated ramp with the same pipeline classes used on real data.

.. code-block:: python

   from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline

   l0 = Stage0Pipeline(output_dir=OUT).apply(ramp)

   do_fibers = [str(f) for f in FIBERS]
   stage1_args = {
       "spec_extract": {"do_fibers": do_fibers},
       "set_wavesol": {"wavesol": cals["wavesol"]},
   }
   l1 = Stage1Pipeline(output_dir=OUT, prim_args=stage1_args).apply(l0)

   injected_m_s = float(INJECTED_RV) * 1000.0
   l2 = Stage2Pipeline(
       output_dir=OUT,
       prim_args={
           "blaze_corr": {"do_fibers": do_fibers},
           "stellar_rv": {
               "ccf_input_velocity": injected_m_s,
               "do_fibers": list(FIBERS),
               "save_preview": True,
               "preview_write_png": False,
           },
       },
   ).apply(l1)

   print(f"L2: {l2.meta.model_type}  {l2.meta.filename}")
   for fib in FIBERS:
       avg = getattr(l2, f"rvccfavg_fib{fib}", None)
       if avg is not None and len(avg):
           rv = float(avg["RV"][0])
           err = float(avg["RV_ERR"][0])
           print(f"fiber {fib} <RV> = {rv:.4f} +/- {err:.4f} m/s")
           print(
               f"fiber {fib} bias = {rv - injected_m_s:.4f} m/s "
               "(recovered - injected)"
           )

4. Stellar RV preview path
--------------------------

The Stage-2 stellar-RV HTML preview is written next to the FITS product:

.. code-block:: python

   preview = l2.get_output_path(output_dir=OUT, suffix="stellar_rv").replace(
       ".fits", "_preview.html"
   )
   print(preview)

5. Alternative: an all-in-one E2E simulation
--------------------------------------------

To run the full end-to-end simulation from the command line interface, run:

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
