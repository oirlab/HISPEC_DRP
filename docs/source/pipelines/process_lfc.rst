Process LFC
===========

Overview
--------

Process the 1D laser frequency comb (LFC) data to generate a wavelength calibration model and line spread function (LSF) model.

**Class**: :py:class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`


Primitives
----------

1. ``fit_lfc_modes`` : :doc:`Fit LFC lines <../primitives/lfc_line_fitting>`
2. ``lfc_wavesol`` : :doc:`Calculate Wavelength Solution <../primitives/lfc_wavesol>`
3. ``lsf`` : :doc:`Register Wavelength Solution <../primitives/gauss_lsf>`
4. ``register_wave_cal`` : :doc:`Calculate Line Spread Function (LSF) <../primitives/gauss_lsf>`
5. ``register_lsf_cal`` : :doc:`Register LSF <../primitives/register_cal>`

Arguments
---------

None.