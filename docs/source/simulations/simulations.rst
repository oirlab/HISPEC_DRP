Simulations
===========

The :mod:`hispecdrp.simulations` package provides forward models and an end-to-end (E2E)
exercise of HISPEC DRP: from a stellar template and pyechelle instrument model through a
synthetic up-the-ramp (UTR) exposure, full Stage 0–2 reduction, and radial-velocity (RV)
estimation.

Use these pages for the **E2E simulation pipeline** (``python -m hispecdrp.simulations.run_e2e_sim``).
For low-level rate-map and ramp builders without the full pipeline, see also
:doc:`../quickstart/create_simulations`.

.. toctree::
   :maxdepth: 2

   overview
   spectrum_and_rate_map
   calibrations
   pipeline_stages
   configuration
   outputs_and_previews
   running
   limitations
