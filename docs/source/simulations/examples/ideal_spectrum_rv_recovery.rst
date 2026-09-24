Ideal-Spectrum RV Recovery
==========================

Recover an injected RV from the rest-frame **BT-Settl lte040** (4000 K)
template using the packaged full-band ``*.ccfopt.delta.mas`` mask (~887 lines)
and :func:`~hispecdrp.simulations.spectra.recover_stellar_rv`. The same
-50...+50 km/s grid (1 km/s, 101 samples) is run on three independent paths.
Run the preamble and section 1 first, then any of sections 2–4. Each path
prints the same recovery table and residual plot (cm/s; mean and ``np.std``
with ``ddof=0``).

1. **Native L2** (section 2):
   :func:`~hispecdrp.simulations.spectra.build_l2` onto the echelle WAVE
   grid. Fast; no KOA cache.
2. **Rate map → Stage 1** (section 3): native rate map, wrap as L0, extract
   and blaze.
3. **Rate map → UTR → Stages 0–2** (section 4): same as (2) through a
   simulated ramp and Stage 0. Slowest.

Pipeline logs are suppressed during each sweep. For an LSF-matched closed loop,
set ``APPLY_LSF = True`` and pass a full-band ``*.lsf.ccfopt.delta.mas`` from
:func:`~hispecdrp.simulations.spectra.ensure_lsf_ccfopt_mask`. The packaged
``lte040...lsf.ccfopt.delta.mas`` is a two-order smoke file and is not used
here.

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

Packaged lte040 template, native ``*.ccfopt.delta.mas`` mask, and the -50...+50
km/s grid. ``run_injected_rv_grid`` prints the recovery table and residual plot
used by sections 2–4.

.. code-block:: python

   import numpy as np

   from hispecdrp.simulations.plotting import plot_ideal_spectrum_rv_recovery
   from hispecdrp.simulations.spectra import quiet_pipeline_logging
   from hispecdrp.utils.line_mask_utils import (
       paired_line_mask_path_for_template,
       read_mask_generation_meta,
   )

   FIBER = 1
   BULK_RV_KM_S = np.arange(-50.0, 51.0, 1.0)
   TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
   APPLY_LSF = False
   MASK_PATH = paired_line_mask_path_for_template(TEMPLATE, mask_style="ccfopt.delta")
   HDR = (
       f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
       f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
   )
   meta = read_mask_generation_meta(MASK_PATH) or {}
   print(f"mask: {MASK_PATH}")
   print(
       f"n_lines={meta.get('n_lines')}  n_seed={meta.get('n_seed_lines')}  "
       f"n_drop={meta.get('n_dropped_lines')}"
   )
   print(
       f"n_bulk={len(BULK_RV_KM_S)}  "
       f"grid={BULK_RV_KM_S[0]:.0f}..{BULK_RV_KM_S[-1]:.0f} km/s step 1"
   )


   def run_injected_rv_grid(recover_one, *, title, out_path):
       inj_m_s, rec_m_s, err_m_s = [], [], []
       print(HDR)
       print("-" * len(HDR))
       for bulk in BULK_RV_KM_S:
           with quiet_pipeline_logging():
               inj, rec, err = recover_one(bulk)
           bias = rec - inj
           inj_m_s.append(float(inj))
           rec_m_s.append(float(rec))
           err_m_s.append(float(err))
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
       max_d = float(np.max(np.abs(d_bias[1:]))) if bias_m_s.size > 1 else 0.0
       print()
       print(f"max|bias| = {max_abs:.6e} m/s ({max_abs * 1e3:.4f} mm/s)")
       print(f"rms bias  = {rms:.6e} m/s ({rms * 1e3:.4f} mm/s)")
       print(
           f"max|differential bias| (vs RV=0) = {max_d:.6e} m/s "
           f"({max_d * 1e3:.4f} mm/s)"
       )
       summary = plot_ideal_spectrum_rv_recovery(
           inj_m_s,
           rec_m_s,
           err_m_s,
           title=title,
           out_path=out_path,
       )
       if summary.get("out_path"):
           print("wrote", summary["out_path"])
           print(
               f"mean bias = {summary['mean_bias_cm_s']:.3f} cm/s  "
               f"std = {summary['std_bias_cm_s']:.3f} cm/s"
           )
       return summary

2. Native L2 injection
----------------------

:func:`~hispecdrp.simulations.spectra.build_l2` paints the Doppler-shifted
template onto a native BSPEC L2, then
:func:`~hispecdrp.simulations.spectra.recover_stellar_rv` recovers the RV.
Requires the preamble and section 1 only.

.. code-block:: python

   from hispecdrp.simulations.spectra import build_l2, recover_stellar_rv


   def recover_native(bulk):
       l2 = build_l2(
           bulk,
           template=TEMPLATE,
           fiber=FIBER,
           apply_lsf=APPLY_LSF,
       )
       return recover_stellar_rv(
           l2,
           bulk,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           spectrograph="BSPEC",
           fiber=FIBER,
       )


   summary_native = run_injected_rv_grid(
       recover_native,
       title="BT-Settl lte040 (native L2): injected vs recovered RV",
       out_path=OUT / "lte040_native_l2_simulated_vs_recovered_rv.png",
   )

3. Rate map → Stage 1
---------------------

Paint a native rate map, wrap it as L0, box-extract and blaze with simulated
Stage-1 calibrations, then recover RV. The first point builds masters under
``OUT``; later points reuse them. Requires the preamble and section 1 only.

.. code-block:: python

   from hispecdrp.simulations.ideal_spectrum_rv import recover_rv_from_rate_map


   def recover_stage1(bulk):
       return recover_rv_from_rate_map(
           bulk,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           output_dir=OUT / "rate_map_stage1",
           fiber=FIBER,
       )


   summary_s1 = run_injected_rv_grid(
       recover_stage1,
       title="BT-Settl lte040 (rate map -> Stage 1): injected vs recovered RV",
       out_path=OUT / "lte040_rate_map_stage1_simulated_vs_recovered_rv.png",
   )

4. Rate map → UTR → Stages 0–2
------------------------------

Same grid through a simulated up-the-ramp exposure and Stage 0, then the
section 3 extract / blaze / CCF path. Poisson noise is off. Each iteration
deletes its working directory after the RV is recorded. Requires the preamble
and section 1 only.

.. code-block:: python

   from hispecdrp.simulations.ideal_spectrum_rv import recover_rv_from_ramp


   def recover_stage0(bulk):
       return recover_rv_from_ramp(
           bulk,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           output_dir=OUT / "ramp_stage0",
           fiber=FIBER,
       )


   summary_s0 = run_injected_rv_grid(
       recover_stage0,
       title="BT-Settl lte040 (rate map -> UTR -> Stages 0-2): injected vs recovered RV",
       out_path=OUT / "lte040_rate_map_stage0_simulated_vs_recovered_rv.png",
   )
