Stage 1
=======

Overview
--------

The stage 1 pipeline converts level 0 (rate map) data into level 1 (uncalibrated spectra) data. Since the Stage 0 pipeline handles all detector level calibrations, spectral extraction is the only task performed by the Stage 1 pipeline.


**Class**: :py:class:`~hispecdrp.pipelines.stage1.stage1_pipeline.Stage1Pipeline`


Primitives
----------

1. ``spec_extract`` : :doc:`Spectral Extraction <../primitives/spectral_extraction>`