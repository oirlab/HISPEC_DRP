Line Spread Function (LSF)
==========================

The effective line spread function (LSF) is modeled with a Gaussian where the width is described with a 2D polynomial (pixel, order, width).

**Classes:**

* :py:class:`~hispecdrp.datamodels.gausslsf.GaussianLSFModel`
* :py:class:`~hispecdrp.datamodels.gausslsf.HISPECGaussianLSFModel`
* :py:class:`~hispecdrp.datamodels.gausslsf.PARVIGaussianLSFModel`

**Calibration type:** lsf


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
   LSF_PARAMS_FIB#, Image, Float64, "Nord x 4096 x 3", None, "LFC spectrum Gaussian LSF parameters (mu, sigma, offset)"