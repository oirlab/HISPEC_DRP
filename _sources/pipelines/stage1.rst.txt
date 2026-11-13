================
Stage 1 Pipeline
================

Overview
--------
**Input:** Level 0 (rate map) data.

**Output:** Level 1 (uncalibrated spectra) data.

The stage 1 pipeline converts level 0 (rate map) data into level 1 (uncalibrated spectra) data. Since the Stage 0 pipeline handles all detector level calibrations, spectral extraction is the only task performed by the Stage 1 pipeline.


Documentation
-------------

.. autoclass:: hispecdrp.pipelines.stage1.stage1_pipeline.Stage1Pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: