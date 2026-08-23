Examples
========

Runnable worked examples for exploring HISPEC simulation workflows and pipeline
tests. Each page follows the same style as :doc:`../basic_usage`: short setup, a
focused simulation or validation, and products under a local output directory.

Published examples live under this documentation tree (including a
**Download as notebook** button on each page). Work-in-progress notebooks remain
under ``hispecdrp/simulations/notebooks/`` until they are ready to promote here.

Simulated Calibrations
----------------------

Examples that exercise simulated calibration sources and related validations
(for example laser frequency comb spots and WAVE checks).

.. toctree::
   :maxdepth: 1

   simulated_lfc_spectrum

Simulated Science Observations
------------------------------

Examples that simulate science targets and reduce them through Stage 0–2
(for example a Barnard's star PRV analog), plus Stage-0 rate/noise self-checks
on realistic UTRs.

.. toctree::
   :maxdepth: 1

   stage0_rate_noise_selfcheck
   barnards_star_prv_example
