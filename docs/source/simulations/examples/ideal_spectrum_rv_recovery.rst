Ideal-Spectrum RV Recovery
==========================

From the rest-frame **BT-Settl lte040** (4000 K) template, we simulate a BSPEC L2
data model then recover its injected RV with the matched delta line mask and
``StellarRVPrimitive``.  In this example, this is done for injected RVs from
0–50 km/s in steps of 10 km/s and the injected versus recovered RVs are plotted
at the end, including the recovered RV uncertainty and residual.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/ideal_spectrum_rv_recovery.ipynb"
        data-notebook-download="ideal_spectrum_rv_recovery.ipynb"
        download="ideal_spectrum_rv_recovery.ipynb">Download as notebook</a>
   </p>

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   OUT = Path("/path/to/ideal_spectrum_rv_recovery")
   OUT.mkdir(parents=True, exist_ok=True)
   print("output:", OUT.resolve())

1. Configure template and mask
------------------------------

Packaged lte040 template and the paired ``*.ccfopt.delta.mas`` line mask. Recovery
uses the ccfopt line-by-line defaults on the full BSPEC echelle order range.

.. code-block:: python

   from hispecdrp.utils.line_mask_utils import paired_line_mask_path_for_template

   FIBER = 1
   BULK_RV_KM_S = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
   TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
   MASK_PATH = paired_line_mask_path_for_template(TEMPLATE, mask_style="ccfopt.delta")

   print(f"mask: {MASK_PATH}")

2. Inject, recover, and collate
-------------------------------

For each bulk shift, :func:`~hispecdrp.simulations.spectra.build_l2`
injects the template into a BSPEC L2 data model, then
:class:`~hispecdrp.primitives.StellarRVPrimitive` recovers RV with the matched
delta mask and line-by-line ``ccf_line_mask``.

.. code-block:: python

   import numpy as np

   from hispecdrp.primitives import StellarRVPrimitive
   from hispecdrp.simulations.spectra import build_l2, quiet_pipeline_logging

   hdr = (
       f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
       f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
   )
   print(hdr)
   print("-" * len(hdr))

   inj_m_s = []
   rec_m_s = []
   err_m_s = []

   # This suppresses logging from the pipeline so the output is cleaner
   with quiet_pipeline_logging():
       for bulk in BULK_RV_KM_S:
           l2 = build_l2(bulk, template=TEMPLATE, fiber=FIBER)
           inj = float(bulk) * 1000.0
           out = StellarRVPrimitive.call(
               l2,
               mask_path=str(MASK_PATH),
               ccf_input_velocity=inj,
           )
           avg = getattr(out, f"rvccfavg_fib{FIBER}")
           rec = float(avg["RV"][0])
           err = float(avg["RV_ERR"][0])
           bias = rec - inj
           inj_m_s.append(inj)
           rec_m_s.append(rec)
           err_m_s.append(err)
           print(
               f"{bulk:10.1f}  {inj:14.4f}  {rec:16.6f}  {err:12.6f}  "
               f"{bias:12.6e}  {bias * 1e3:10.4f}"
           )

   inj_m_s = np.asarray(inj_m_s, dtype=float)
   rec_m_s = np.asarray(rec_m_s, dtype=float)
   err_m_s = np.asarray(err_m_s, dtype=float)
   bias_m_s = rec_m_s - inj_m_s
   max_abs = float(np.max(np.abs(bias_m_s)))
   rms = float(np.sqrt(np.mean(bias_m_s**2)))
   d_bias = bias_m_s - bias_m_s[0]
   max_d = float(np.max(np.abs(d_bias[1:])))
   print()
   print(f"max|bias| = {max_abs:.6e} m/s ({max_abs * 1e3:.4f} mm/s)")
   print(f"rms bias  = {rms:.6e} m/s ({rms * 1e3:.4f} mm/s)")
   print(f"max|differential bias| (vs RV=0) = {max_d:.6e} m/s ({max_d * 1e3:.4f} mm/s)")

3. Plot
-------

The lower panel shows residuals from the 1:1 line (recovered minus injected) on
a zoomed scale so bias can be compared with the reported ``RV_ERR`` bars.

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_ideal_spectrum_rv_recovery

   summary = plot_ideal_spectrum_rv_recovery(
       inj_m_s,
       rec_m_s,
       err_m_s,
       out_path=OUT / "lte040_simulated_vs_recovered_rv.png",
   )
   if summary.get("out_path"):
       print("wrote", summary["out_path"])
