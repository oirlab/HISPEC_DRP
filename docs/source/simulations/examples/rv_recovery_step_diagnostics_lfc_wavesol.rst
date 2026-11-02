RV recovery step diagnostics (LFC wavesol)
==========================================

Same densified BT-Settl lte040 recovery grid as
:doc:`rv_recovery_step_diagnostics` (0, 10, 20, 30, 40, 50 km/s; all finite
lines kept with ``outlier_sigma=1e30`` — no per-RV rejection). Run the preamble
and section 1 first, then sections 2–8 in order. The rate-map and UTR paths use
simulated calibrations so recovery stays free of the ~15–20 m/s continuum/CCF
bias:

1. **Truth 1D** (section 1): densify the rest-frame template, Doppler-shift it,
   and verify recovered RV ≈ 0 at injected RV = 0.
2. **LFC WAVE + LSF** (section 2):
   :func:`~hispecdrp.simulations.rv_step_diagnostics.derive_lfc_wavesol`
   (same ProcessLFC settings as :doc:`simulated_lfc_spectrum`) fits
   extract-matched WAVE and an empirical ``GaussianLSFModel``.
3. **Extract-matched ccfopt mask** (section 3):
   :func:`~hispecdrp.simulations.rv_step_diagnostics.build_extract_matched_ccfopt_mask`
   (paint RV=0 → extract under LFC WAVE + identity FF, after
   :func:`~hispecdrp.simulations.rv_step_diagnostics.check_lsf_wavesol_fiber_alignment`;
   product ``*.extract_matched.ccfopt.delta.mas``). Near-perfect at RV=0 on the
   rate-map path, but a bulk-dependent slope keeps **packaged** ccfopt for
   recovery (§5–7). LFC WAVE + identity FF remain the main ≪1 m/s bias fix.
4. **Identity fiberflat** (section 4):
   :func:`~hispecdrp.simulations.ideal_spectrum_rv.prepare_stage1_cals`
   with ``fiberflat_mode="identity"``.
5. **Native L2** (section 5): paint the shifted 1D onto the echelle WAVE grid.
6. **Rate map → Stage 1** (section 6): native rate map, LFC WAVE, identity FF.
7. **UTR → Stages 0–2** (section 7): same calibrations after a noiseless ramp /
   Stage 0.
8. **Compare** (section 8).

A paint-matched continuum fiberflat cancels PRF blaze shape but still reweights
the CCF by ~17 m/s on this grid. Keep ``FIBERFLAT_MODE = "identity"`` for
bias-free simulations; set ``"paint_matched"`` only to study blaze cancel
itself.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/rv_recovery_step_diagnostics_lfc_wavesol.ipynb"
        data-notebook-download="rv_recovery_step_diagnostics_lfc_wavesol.ipynb"
        download="rv_recovery_step_diagnostics_lfc_wavesol.ipynb">Download as notebook</a>
   </p>

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   OUT = Path("/path/to/rv_recovery_step_diagnostics_lfc_wavesol")
   OUT.mkdir(parents=True, exist_ok=True)
   print("output:", OUT.resolve())

See the notebook for the full walkthrough.
