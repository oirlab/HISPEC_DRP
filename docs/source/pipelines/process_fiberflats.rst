Process Fiberflats
==================

Overview
--------

Process the 2D fiberflat echellograms to generate the trace position and trace profile calibration files.

**Class**: :py:class:`~hispecdrp.pipelines.calibration.process_fiberflats_pipeline.ProcessFiberflatsPipeline`


Primitives
----------

1. ``coadd_frames`` : :doc:`Coadd Frames <../primitives/coadd_frames>`
2. ``trace_pos`` : :doc:`Trace Position <../primitives/trace_finding>`
3. ``trace_prof`` : :doc:`Trace Profile <../primitives/trace_profile>`
4. ``register_tracepos_cal`` : :doc:`Register Trace Position Calibration <../primitives/register_cal>`
5. ``register_traceprof_cal`` : :doc:`Register Trace Profile Calibration <../primitives/register_cal>`
6. ``spec_extract`` : :doc:`Spectral Extraction <../primitives/spectral_extraction>`
7. ``set_wavesol`` : :doc:`Set Initial Wavelength Solution <../primitives/set_wavesol>`
8. ``register_fiberflat_cal`` : :doc:`Register Fiberflat Calibration <../primitives/register_cal>`


Arguments
---------

None.