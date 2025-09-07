LFC Fit Parameters
==================

Contains the best fit Gaussian parameters for each LFC mode.

**Classes:**

* :py:class:`~hispecdrp.datamodels.lfcfitparams.LFCFitParamsModel`
* :py:class:`~hispecdrp.datamodels.lfcfitparams.HISPECLFCFitParamsModel`
* :py:class:`~hispecdrp.datamodels.lfcfitparams.PARVILFCFitParamsModel`

**Calibration type:** lfc_fitparams


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
   LFC_FITPARAMS_FIB#, Table, See below, See below, See below, "LFC fit parameters for each LFC mode, order, and fiber"

.. csv-table:: **LFC_FITPARAMS_FIB# Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "AMPLITUDE", "Int32", "Gaussian amplitude."
   "AMPLITUDE_ERR", "Float64", "Error in Gaussian amplitude."
   "MU_PIX", "Float64", "Gaussian center in detector pixels."
   "MU_ERR_PIX", "Float64", "Error in Gaussian center in detector pixels."
   "MU_ERR_WAVE", "Float64", "Error in Gaussian center in wavelength (microns)."
   "SIGMA_PIX", "Float64", "Gaussian standard deviation in detector pixels."
   "SIGMA_ERR_PIX", "Float64", "Error in Gaussian standard deviation in detector pixels."
   "SIGMA_WAVE", "Float64", "Gaussian standard deviation in wavelength (microns)."
   "SIGMA_ERR_WAVE", "Float64", "Error in Gaussian standard deviation in wavelength (microns)."
   "B", "Float64", "Offset."
   "B_ERR", "Float64", "Error in offset."
   "MODE_NUM", "Float64", "LFC mode number."
   "WAVE", "Float64", "Wavelength of the LFC mode."
   "ORDER", "Float64", "Echelle order of the LFC mode."