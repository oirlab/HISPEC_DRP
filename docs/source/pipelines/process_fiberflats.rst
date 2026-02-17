==================
Process Fiberflats
==================

Overview
--------

**Input:** Level 0 fiberflat data (2D rate maps). Each fiberflat must have only a single fiber illuminated within an exposure. Different fibers are grouped together and coadded.

**Cal. Outputs:** Trace positions calibration, trace profile calibraiton, and 1D fiberflat calibration for blaze correction.

**Other Output:** Rate map in e-/s (Level 0).

Calibration pipeline to generate detector flatfield and gain calibration files from raw detector flat UTR data.
Also creates level 0 rate map data products for each flat exposure processed.

Documentation
-------------

.. autoclass:: hispecdrp.pipelines.calibration.process_fiberflats_pipeline.ProcessFiberflatsPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: