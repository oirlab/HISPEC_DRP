Examples
========

Runnable worked examples for exploring HISPEC simulation workflows and pipeline
tests. Each page follows the same style as :doc:`../basic_usage`: short setup, a
focused simulation or validation, and products under a local output directory.

Simulated Calibrations
----------------------

Examples that exercise simulated calibration sources and related validations
(for example laser frequency comb spots and WAVE checks).

.. toctree::
   :maxdepth: 1

   simulated_lfc_spectrum
   koa_lfc_wavesol

Simulated Science Observations
------------------------------

Examples that simulate science targets and reduce them through Stage 0–2
(for example a Barnard's star PRV analog), ideal-spectrum line-mask RV
validation on native L2, rate-map → Stage 1, and rate-map → UTR → Stages 0–2,
step-by-step RV bias/precision through that same chain, building CCF-opt
delta masks from LSF-convolved ideal L2, plus Stage-0 rate/noise self-checks
on realistic UTRs.

.. toctree::
   :maxdepth: 1

   stage0_rate_noise_selfcheck
   build_ccfopt_mask_ideal_spectrum
   ideal_spectrum_rv_recovery
   rv_recovery_step_diagnostics
   barnards_star_prv_example
