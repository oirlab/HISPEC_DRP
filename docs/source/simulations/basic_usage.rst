Basic Usage
===========

Short worked example for running HISPEC DRP end-to-end from a 1D stellar
template: build a simulated BSPEC ramp, reduce it through Stage 0–2, and
inspect the stellar-RV product using line-by-line ``ccf_line_mask`` (packaged
ccfopt settings).

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">github</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../_static/notebooks/basic_usage.ipynb"
        data-notebook-download="basic_usage.ipynb"
        download="basic_usage.ipynb">Download as notebook</a>
   </p>

.. warning::
   **Setup.** This example uses a local KOA calibration cache. If
   ``KOA_CALIBRATION_CACHE`` is unset, the first cell sets it under ``OUT``.
   Optional remote sync uses ``KECK_OBSERVER_EMAIL`` and
   ``KECK_OBSERVER_PASSWORD``.

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

2. Simulate UTR
---------------

Load a BT-Settl template, Doppler-shift by ``radial_velocity_km_s``, paint with
ZEMAX, and build a raw UTR. Stage-0 KOA masters plus simulation Stage-1
products are registered into the local caldb.

.. code-block:: python

   from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum

   FIBERS = [2]
   INJECTED_RV = 1.0  # km/s

   ramp, cals, rate_map = create_ramp_from_spectrum(
       OUT,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       template_filename="lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
       mag_ab=3.05,
       filter_band="J",
       seed=0,
       radial_velocity_km_s=INJECTED_RV,
       barycentric_velocity_km_s=0.0,
       inject_rv_via="spectrum",
   )
   print(f"Stage-0 cals: {sorted(cals)}")

3. Stage 0–2
------------

Same pipeline classes as real data. Stage-2 stellar RV uses line-by-line
``ccf_line_mask`` with the packaged ``ccfopt`` mask and
:data:`~hispecdrp.utils.line_mask_utils.CCFOPT_SINGLE_LINE_STELLAR_RV_KW`.

.. code-block:: python

   from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline
   from hispecdrp.utils.line_mask_utils import (
       CCFOPT_SINGLE_LINE_STELLAR_RV_KW,
       packaged_default_line_mask_path,
   )

   l0 = Stage0Pipeline.call(ramp, output_dir=OUT)

   do_fibers = [str(f) for f in FIBERS]
   l1 = Stage1Pipeline.call(
       l0,
       output_dir=OUT,
       primitives={
           "spec_extract": {"do_fibers": do_fibers},
           "set_wavesol": {"wavesol": cals["wavesol"]},
       },
   )

   injected_m_s = float(INJECTED_RV) * 1000.0
   kw = CCFOPT_SINGLE_LINE_STELLAR_RV_KW
   l2 = Stage2Pipeline.call(
       l1,
       output_dir=OUT,
       primitives={
           "blaze_corr": {"do_fibers": do_fibers},
           # No sky fiber in this single-science-fiber sim; skip sky subtraction.
           "sky_sub": {"run": False},
           "stellar_rv": {
               "algorithm": "ccf_line_mask",
               "mask_path": packaged_default_line_mask_path(),
               "ccf_input_velocity": injected_m_s,
               "do_fibers": list(FIBERS),
               "window_line_by_line": True,
               "window_size_um": float(kw["window_size_um"]),
               "v_half_window": float(kw["v_half_window_m_s"]),
               "v_step_coarse": float(kw["v_step_m_s"]),
               "mask_auto_center_scout": bool(kw["mask_auto_center_scout"]),
               "mask_ccf_pixel_interp_order": int(kw["mask_ccf_pixel_interp_order"]),
               "ccf_min_good_pixels": int(kw["ccf_min_good_pixels"]),
               "ccf_zb": 0.0,
               "ccf_zb_from_barycorr": False,
               "save_preview": True,
               "preview_write_png": False,
           },
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

4. Consistency check: stellar RV preview
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
