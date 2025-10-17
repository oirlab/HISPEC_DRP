Stage 0
=======

Overview
--------

The stage 0 pipeline converts raw data (UTR ramp) into level 0 (rate map) data, applying all detector-level corrections/calibrations.


**Class**: :py:class:`~hispecdrp.pipelines.stage0.stage0_pipeline.Stage0Pipeline`


Primitives
----------

1. ``dq_init`` : :doc:`DQ Initialization <../primitives/dq_init>`
2. ``saturation`` : :doc:`Check Saturation <../primitives/saturation>`
3. ``bias`` : :doc:`Bias Subtraction <../primitives/bias_subtraction>`
4. ``nonlin_corr`` : :doc:`Non Linear Correction <../primitives/nonlinear_correction>`
5. ``jump_detec`` : :doc:`Jump Detection <../primitives/jump_detection>`
6. ``ramp_fit`` : :doc:`Ramp Fit <../primitives/ramp_fitting>`
7. ``dark`` : :doc:`Dark Subtraction <../primitives/dark_subtraction>`
8. ``det_gain`` : :doc:`Detector Gain <../primitives/detector_gain>`
9. ``det_flat`` : :doc:`Detector Flat <../primitives/detector_flat>`

Arguments
---------

None.