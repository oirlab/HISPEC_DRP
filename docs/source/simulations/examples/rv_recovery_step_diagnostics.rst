RV Recovery Step Diagnostics
============================

Track injected-RV recovery through the same densified **BT-Settl lte040**
spectrum at every pipeline stage. The default grid is 0, 10, 20, 30, 40,
50 km/s. Every :class:`~hispecdrp.primitives.StellarRVPrimitive` call keeps
**all** finite lines (``outlier_sigma=1e30``). Run the preamble and section 1
first, then sections 2–4 in order. Section 5 compares whatever steps have
finished. Section 6 re-extracts the saved rate maps with simulated wavecals
to separate WAVE association from the extract.

1. **Truth 1D** (section 1): densify the rest-frame template, Doppler-shift it,
   and verify recovered RV ≈ 0 at injected RV = 0.
2. **Native L2** (section 2): paint that shifted 1D onto the echelle WAVE grid.
3. **Rate map → Stage 1** (section 3): ZEMAX/native 2D paint, ZEMAX-matched
   fiberflat, box extract, blaze, recover.
4. **UTR → Stages 0–2** (section 4): simulated ramp and Stage-0 detector cals,
   then the same extract / blaze / CCF path.
5. **Compare** (section 5): which step adds the most bias and ``RV_ERR``.
6. **WAVE vs extract** (section 6): same rate maps, new WAVE (line-ZP and
   extracted comb).

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/rv_recovery_step_diagnostics.ipynb"
        data-notebook-download="rv_recovery_step_diagnostics.ipynb"
        download="rv_recovery_step_diagnostics.ipynb">Download as notebook</a>
   </p>

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   OUT = Path("/path/to/rv_recovery_step_diagnostics")
   OUT.mkdir(parents=True, exist_ok=True)
   print("output:", OUT.resolve())

1. Truth 1D spectrum and RV=0 check
-----------------------------------

Densify the packaged lte040 template to R=3e5. At injected RV = 0, paint that
rest-frame 1D onto L2 and recover with every line kept. Then build the
Doppler-shifted truth spectrum for each bulk RV used in sections 2–4.

.. code-block:: python

   from hispecdrp.simulations.rv_step_diagnostics import (
       STEP_BULK_RV_KM_S,
       STELLAR_RV_NO_REJECT_KW,
       doppler_shift_spectrum,
       load_densified_template,
       run_step_grid,
       verify_zero_rv_recovery,
   )
   from hispecdrp.utils.line_mask_utils import (
       paired_line_mask_path_for_template,
       read_mask_generation_meta,
   )

   FIBER = 1
   TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
   MASK_PATH = paired_line_mask_path_for_template(TEMPLATE, mask_style="ccfopt.delta")
   BULK_RV_KM_S = STEP_BULK_RV_KM_S.copy()
   meta = read_mask_generation_meta(MASK_PATH) or {}
   print(f"mask: {MASK_PATH}")
   print(
       f"n_lines={meta.get('n_lines')}  outlier_sigma={STELLAR_RV_NO_REJECT_KW['outlier_sigma']:.0e}"
   )
   print(f"grid={list(BULK_RV_KM_S)} km/s")

   wl_rest, flux_rest = load_densified_template(TEMPLATE)
   print(f"truth 1D: n={wl_rest.size}  λ={wl_rest[0]:.4f}–{wl_rest[-1]:.4f}")

   verify_zero_rv_recovery(
       template=TEMPLATE,
       mask_path=str(MASK_PATH),
       fiber=FIBER,
   )

   TRUTH = {
       float(rv): doppler_shift_spectrum(wl_rest, flux_rest, float(rv))
       for rv in BULK_RV_KM_S
   }
   summaries = {}

2. Paint truth onto L2
----------------------

Resample each Doppler-shifted 1D spectrum onto the native BSPEC WAVE grid and
recover RV. This is the interpolation-only reference for later steps.

.. code-block:: python

   from hispecdrp.simulations.rv_step_diagnostics import recover_rv_from_truth_l2


   def recover_l2(bulk):
       wl, flux = TRUTH[float(bulk)]
       return recover_rv_from_truth_l2(
           bulk,
           wavelengths=wl,
           flux=flux,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           fiber=FIBER,
       )


   summaries["L2"] = run_step_grid(
       recover_l2,
       name="L2",
       title="BT-Settl lte040 (truth 1D -> L2)",
       out_path=OUT / "lte040_step_l2.png",
       bulk_rv_km_s=BULK_RV_KM_S,
   )

3. Rate map → extract → blaze
-----------------------------

Paint the same truth 1D onto a native/ZEMAX rate map. Stage-1 traces, WAVE, and
fiberflat come from that ZEMAX model
(:func:`~hispecdrp.simulations.ideal_spectrum_rv.prepare_stage1_cals`). Then
box-extract, blaze-correct, and recover RV. The first point builds masters
under ``OUT``.

.. code-block:: python

   from hispecdrp.simulations.rv_step_diagnostics import recover_rv_from_truth_rate_map


   def recover_rate_map(bulk):
       wl, flux = TRUTH[float(bulk)]
       return recover_rv_from_truth_rate_map(
           bulk,
           wavelengths=wl,
           flux=flux,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           output_dir=OUT / "rate_map_stage1",
           fiber=FIBER,
       )


   summaries["rate map"] = run_step_grid(
       recover_rate_map,
       name="rate map",
       title="BT-Settl lte040 (truth 1D -> rate map -> Stage 1)",
       out_path=OUT / "lte040_step_rate_map.png",
       bulk_rv_km_s=BULK_RV_KM_S,
   )

4. UTR → Stages 0–2
-------------------

Realize each truth 1D as a simulated up-the-ramp exposure with matching 2D
Stage-0 calibration frames, run Stage 0, then the same Stage 1 / blaze / CCF
path as section 3. Poisson noise is off so residual scatter isolates the ramp
fit. Each iteration deletes its working directory after the RV is recorded.

.. code-block:: python

   from hispecdrp.simulations.rv_step_diagnostics import recover_rv_from_truth_ramp


   def recover_ramp(bulk):
       wl, flux = TRUTH[float(bulk)]
       return recover_rv_from_truth_ramp(
           bulk,
           wavelengths=wl,
           flux=flux,
           template=TEMPLATE,
           mask_path=str(MASK_PATH),
           output_dir=OUT / "ramp_stage0",
           fiber=FIBER,
       )


   summaries["UTR + Stage 0-2"] = run_step_grid(
       recover_ramp,
       name="UTR + Stage 0-2",
       title="BT-Settl lte040 (truth 1D -> UTR -> Stages 0-2)",
       out_path=OUT / "lte040_step_utr.png",
       bulk_rv_km_s=BULK_RV_KM_S,
   )

5. Compare steps
----------------

Mean residual, residual scatter, and mean formal ``RV_ERR`` for each completed
step. Increments (``Δ``) show what the next stage added. The largest increment
is the step that degrades the measurement the most.

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_rv_step_comparison
   from hispecdrp.simulations.rv_step_diagnostics import compare_step_summaries

   increments = compare_step_summaries(summaries)
   cmp = plot_rv_step_comparison(
       summaries,
       title="BT-Settl lte040: RV recovery by pipeline step",
       out_path=OUT / "lte040_step_comparison.png",
   )
   if cmp.get("out_path"):
       print("wrote", cmp["out_path"])

6. Separate WAVE association from extraction
--------------------------------------------

The rate-map jump is a nearly constant +33 m/s zeropoint. That can be a
wrong WAVE attached to a good extract, or an extract that shifts line
centroids. Re-extract the saved rate maps with two simulated wavecals:

1. **Line-ZP WAVE** — absorption-line zeropoint on the RV=0 rate map
   (``known_rv=0``). If the 0–50 km/s grid then matches the L2 residuals,
   the box extract is fine and geometric WAVE was sampled at deposit
   centers, not at the extracted PSF centroid.
2. **Comb WAVE** — paint an emission comb, box-extract it the same way as
   science, and fit ``λ(pixel)``. If this also removes the +33 m/s, the
   bias is wavelength association (PSF vs deposit), not a broken stellar
   extract.

.. code-block:: python

   import numpy as np
   from astropy.io import fits

   from hispecdrp.simulations.ideal_spectrum_rv import prepare_stage1_cals
   from hispecdrp.simulations.plotting import plot_rv_step_comparison
   from hispecdrp.simulations.rv_step_diagnostics import (
       compare_step_summaries,
       recover_rv_from_saved_rate_map,
       run_step_grid,
       saved_rate_map_path,
       wavesol_from_extracted_comb,
       wavesol_from_line_zeropoint,
   )

   RATE_PARENT = OUT / "rate_map_stage1"
   rate0_path = saved_rate_map_path(RATE_PARENT, 0.0)
   if not rate0_path.is_file():
       raise FileNotFoundError(f"Run section 3 first; missing {rate0_path}")
   rate0 = np.asarray(fits.getdata(rate0_path), dtype=float)

   print("Line-ZP WAVE from the RV=0 science rate map ...")
   wavesol_zp, zp_info = wavesol_from_line_zeropoint(
       rate0,
       fiber=FIBER,
       known_rv_m_s=0.0,
       poly_deg=0,
   )
   print(zp_info.get("per_fiber", zp_info))

   print("Comb WAVE from a 1D box-extracted emission comb ...")
   wavesol_comb, comb_info = wavesol_from_extracted_comb(fiber=FIBER)
   print(
       f"median PSF-centroid residual vs deposit WAVE = "
       f"{comb_info['median_centroid_residual_m_s']:.3f} m/s  "
       f"n_orders_fit={comb_info['n_orders_fit']}"
   )

   cals = prepare_stage1_cals(RATE_PARENT, rate0, fiber=FIBER)


   def recover_with_wavesol(wavesol):
       def recover(bulk):
           path = saved_rate_map_path(RATE_PARENT, bulk)
           rate = np.asarray(fits.getdata(path), dtype=float)
           return recover_rv_from_saved_rate_map(
               bulk,
               rate_map=rate,
               template=TEMPLATE,
               mask_path=str(MASK_PATH),
               output_dir=OUT / "wave_split",
               wavesol=wavesol,
               fiber=FIBER,
               stage1_cals=cals,
           )

       return recover


   summaries["rate map + line-ZP WAVE"] = run_step_grid(
       recover_with_wavesol(wavesol_zp),
       name="rate map + line-ZP WAVE",
       title="BT-Settl lte040 (rate map + absorption-line WAVE ZP)",
       out_path=OUT / "lte040_step_rate_map_linezp.png",
       bulk_rv_km_s=BULK_RV_KM_S,
   )
   summaries["rate map + comb WAVE"] = run_step_grid(
       recover_with_wavesol(wavesol_comb),
       name="rate map + comb WAVE",
       title="BT-Settl lte040 (rate map + extracted-comb WAVE)",
       out_path=OUT / "lte040_step_rate_map_comb.png",
       bulk_rv_km_s=BULK_RV_KM_S,
   )

   split = {
       k: summaries[k]
       for k in ("L2", "rate map", "rate map + line-ZP WAVE", "rate map + comb WAVE")
       if k in summaries
   }
   compare_step_summaries(split)
   cmp = plot_rv_step_comparison(
       split,
       title="WAVE vs extract: rate-map RV with alternate wavecals",
       out_path=OUT / "lte040_step_wave_vs_extract.png",
   )
   if cmp.get("out_path"):
       print("wrote", cmp["out_path"])

