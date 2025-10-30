Process Fiberflats
==================

Overview
--------

Process the 2D fiberflat echellograms to generate the trace position and trace profile calibration files.

**Class**: :py:class:`~hispecdrp.pipelines.calibration.process_fiberflat_pipeline.ProcessFiberflatPipeline`


Primitives
----------

1. ``coadd_frames`` : :doc:`Coadd Frames <../primitives/coadd_frames>`
2. ``trace_pos`` : :doc:`Trace Position <../primitives/trace_finding>`
3. ``trace_prof`` : :doc:`Trace Profile <../primitives/trace_profile>`
4. ``register_tracepos_cal`` : :doc:`Register Trace Position Calibration <../primitives/register_cal>`
5. ``register_traceprof_cal`` : :doc:`Register Trace Profile Calibration <../primitives/register_cal>`

Arguments
---------

None.