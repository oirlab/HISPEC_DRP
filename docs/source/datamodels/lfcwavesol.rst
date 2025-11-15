LFC Wavelength Solution
=======================

The LFC wavelength solution.

**Class:** :py:class:`~hispecdrp.datamodels.lfcwavesol.LFCChebyshevWavesolModel`

**Calibration type:** lfc_wavesol


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   SPEC_FIB#, Image, Float32, "Nord x 4096", "e-/s", "Corresponding LFC spectrum"
   SPEC_ERR_FIB#, Image, Float32, "Nord x 4096", "e-/s", "LFC spectrum error."
   SPEC_VAR_POISSON_FIB#, Image, Float32, "Nord x 4096", "e-/s", "LFC spectrum Poisson variance"
   SPEC_VAR_RNOISE_FIB#, Image, Float32, "Nord x 4096", "e-/s", "LFC spectrum read noise variance"
   SPEC_DQ_FIB#, Image, UInt32, "Nord x 4096", None, "LFC spectrum data quality"
   WAVE_FIB#, Image, Float64, "Nord x 4096", "nm", "LFC spectrum wavelength grid"
   :doc:`LFC_FITPARAMS_FIB# <lfcfitparams>`, Table, See docs, See docs, See docs, "LFC fit parameters for each LFC mode, order, and fiber"
   CC_COEFF_FIB#, Image, Float64, (deg_pix + 1) x (deg_ord + 1), None, "Chebyshev coefficients for the wavelength solution"