Build a CCF-Opt Delta Mask from an Ideal-Spectrum Template
==========================================================

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

For an arbitrary stellar template (here **BT-Settl lte040**, 4000 K):

1. **Seed mask** — gradient-based line detection on the 1D template
   (``build_line_mask_from_template`` / ``RELAXED_TEMPLATE_MASK_BUILD_KW``).
2. **Ideal L2** — :func:`~hispecdrp.simulations.simulate_spectra.build_l2` paints the template
   onto a native BSPEC L2, then convolves each order with the HDF PSF→1-D LSF
   (``apply_lsf=True``) so line shapes match the instrument.
3. **Re-zero + reweight** — line-by-line ``StellarRVPrimitive`` on that
   LSF-matched L2 and the seed mask, then shift each line
   ``λ' = λ × (1 + RV/c)`` and set weights ∝ ``|Gaussian contrast|``
   (``build_ccfopt_line_mask_from_l2_line_by_line_rv``).

Output: ``<template_stem>.lsf.ccfopt.delta.mas``. The packaged
``lte040...lsf.ccfopt.delta.mas`` is a two-order smoke product; this example
rebuilds a **full-band** LSF mask (needs the pyechelle HDF model).

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/build_ccfopt_mask_ideal_spectrum.ipynb"
        data-notebook-download="build_ccfopt_mask_ideal_spectrum.ipynb"
        download="build_ccfopt_mask_ideal_spectrum.ipynb">Download as notebook</a>
   </p>

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   OUT = Path("/path/to/build_ccfopt_mask_ideal_spectrum")
   OUT.mkdir(parents=True, exist_ok=True)
   print("output:", OUT.resolve())

1. Configure template
---------------------

.. code-block:: python

   TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
   FIBER = 1

   print(f"template: {TEMPLATE}")

2. Build seed mask, ideal L2, and CCF-opt mask
----------------------------------------------

:func:`~hispecdrp.simulations.simulate_spectra.build_ccfopt_mask` with ``apply_lsf=True``
runs the three-step pipeline on an LSF-matched ideal L2 (full BSPEC echelle
range). Equivalent helper: :func:`~hispecdrp.simulations.simulate_spectra.build_ccfopt_mask_lsf_matched`.

.. code-block:: python

   from hispecdrp.driver.logging import quiet_pipeline_logging
   from hispecdrp.simulations.simulate_spectra import build_ccfopt_mask
   from hispecdrp.utils.line_mask_utils import read_mask_generation_meta

   # This suppresses logging from the pipeline so the output is cleaner
   with quiet_pipeline_logging():
       seed_path, ccfopt_path, l2_inj0 = build_ccfopt_mask(
           TEMPLATE,
           OUT,
           fiber=FIBER,
           apply_lsf=True,
       )

   meta = read_mask_generation_meta(ccfopt_path) or {}
   print(f"seed mask: {seed_path}")
   print(f"ccfopt mask: {ccfopt_path}")
   print(
       f"generator={meta.get('generator')}  n_seed={meta.get('n_seed_lines')}  "
       f"n_drop={meta.get('n_dropped_lines')}  n_lines={meta.get('n_lines')}"
   )

3. Closed-loop check at injected RV = 0
---------------------------------------

Run :class:`~hispecdrp.primitives.StellarRVPrimitive` on the LSF-matched L2 and
ccfopt mask. The ``RVCCF`` table has more rows than mask lines because echelle
orders can measure the same line more than once; aggregate RV should still be
near zero. The :doc:`ideal_spectrum_rv_recovery` example uses the packaged
native ``*.ccfopt.delta.mas`` with ``build_l2(..., apply_lsf=False)``. Use this
LSF mask with ``apply_lsf=True`` for a like-to-like LSF closed loop.

.. code-block:: python

   from hispecdrp.primitives import StellarRVPrimitive

   with quiet_pipeline_logging():
       out = StellarRVPrimitive.call(
           l2_inj0,
           mask_path=str(ccfopt_path),
           ccf_input_velocity=0.0,
       )

   avg = getattr(out, f"rvccfavg_fib{FIBER}")
   rv = float(avg["RV"][0])
   rv_err = float(avg["RV_ERR"][0])
   print(f"closed-loop RV = {rv:+.6f} +/- {rv_err:.6f} m/s")
