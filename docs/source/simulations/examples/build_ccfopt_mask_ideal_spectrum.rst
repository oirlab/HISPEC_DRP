Build a CCF-Opt Delta Mask from an Ideal-Spectrum Template
==========================================================

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

For an arbitrary stellar template:

1. **Seed mask** — gradient-based line detection on the 1D template
   (``build_line_mask_from_template`` / ``RELAXED_TEMPLATE_MASK_BUILD_KW``).
2. **Ideal L2** — :func:`~hispecdrp.simulations.spectra.build_l2` paints the template
   onto a native BSPEC L2 data model.
3. **Re-zero + reweight** — line-by-line ``StellarRVPrimitive`` on the seed mask,
   then shift each line ``λ' = λ × (1 + RV/c)`` and set weights ∝ ``|Gaussian contrast|``
   (``build_ccfopt_line_mask_from_l2_line_by_line_rv``).

Output: ``<template_stem>.ccfopt.delta.mas``.

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

:func:`~hispecdrp.simulations.spectra.build_ccfopt_mask` runs the three-step pipeline
with ccfopt line-by-line defaults on the full BSPEC echelle order range.

.. code-block:: python

   from hispecdrp.simulations.spectra import build_ccfopt_mask, quiet_pipeline_logging
   from hispecdrp.utils.line_mask_utils import read_mask_generation_meta

   # This suppresses logging from the pipeline so the output is cleaner
   with quiet_pipeline_logging():
       seed_path, ccfopt_path, l2_inj0 = build_ccfopt_mask(
           TEMPLATE,
           OUT,
           fiber=FIBER,
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

Run :class:`~hispecdrp.primitives.StellarRVPrimitive` on the ccfopt mask. The
``RVCCF`` table has more rows than mask lines because echelle orders can measure the
same line more than once; aggregate RV should still be near zero. Use the mask from
this example in :doc:`ideal_spectrum_rv_recovery` for bulk-RV recovery validation.

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
