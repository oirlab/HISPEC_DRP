Reference Etalon
================

Contains the reference etalon and corresponding parameters for wavelength drift corrections.

**Classes:**

* :py:class:`~hispecdrp.datamodels.ref_etalon.ReferenceEtalonModel`
* :py:class:`~hispecdrp.datamodels.ref_etalon.HISPECReferenceEtalonModel`
* :py:class:`~hispecdrp.datamodels.ref_etalon.PARVIReferenceEtalonModel`

**Calibration type:** ref_etalon


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   SPEC_FIB#, Image, Float32, "Nord x 4096", "e-/s", "Corresponding etalon spectrum"
   SPEC_ERR_FIB#, Image, Float32, "Nord x 4096", "e-/s", "Etalon spectrum error."
   SPEC_VAR_POISSON_FIB#, Image, Float32, "Nord x 4096", "e-/s", "Etalon spectrum Poisson variance"
   SPEC_VAR_RNOISE_FIB#, Image, Float32, "Nord x 4096", "e-/s", "Etalon spectrum read noise variance"
   SPEC_DQ_FIB#, Image, UInt32, "Nord x 4096", None, "Etalon spectrum data quality"
   MID_TIME_FIB#, Image, Float64, "Nord x 4096", "s", "Mid time for each exposure."
   WAVE_FIB#, Image, Float64, "Nord x 4096", "nm", "Etalon spectrum wavelength grid"
   ETALON_FITPARAMS_FIB#, Table, See below, See below, See below, "Etalon fit parameters for each mode, order, and fiber"
   DRIFT_FIB#, Table, See below, See below, See below, "Drift correction table for each fiber"


.. csv-table:: **ETALON_FITPARAMS_FIB# Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "AMPLITUDE", "Float64", "Line amplitude."
   "AMPLITUDE_ERR", "Float64", "Error in amplitude."
   "MU_PIX", "Float64", "Line center in detector pixels."
   "MU_ERR_PIX", "Float64", "Error in line center in detector pixels."
   "MU_ERR_WAVE", "Float64", "Error in line center in wavelength (microns)."
   "FWHM_PIX", "Float64", "Line FWHM in detector pixels."
   "FWHM_ERR_PIX", "Float64", "Error in FWHM in detector pixels."
   "FWHM_WAVE", "Float64", "Line FWHM in wavelength (microns)."
   "FWHM_ERR_WAVE", "Float64", "Error in FWHM in wavelength (microns)."
   "B", "Float64", "Offset."
   "B_ERR", "Float64", "Error in offset."
   "MODE_NUM", "Int16", "Etalon mode number."
   "WAVE", "Float64", "Wavelength of the etalon mode."
   "ORDER", "UInt16", "Echelle order of the etalon mode."

.. csv-table:: **DRIFT_FIB# Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "PIXEL", "Float64", "Pixel position."
   "ORDER", "UInt16", "Echelle order."
   "WAVE", "Float64", "Wavelength."
   "DRIFT_PIX", "Float64", "Drift in pixels."
   "DRIFT_ERR_PIX", "Float64", "Error in drift in pixels."
   "DRIFT_WAVE", "Float64", "Drift in wavelength."
   "DRIFT_ERR_WAVE", "Float64", "Error in drift in wavelength."
   "DRIFT_VEL", "Float64", "Drift in velocity."
   "DRIFT_ERR_VEL", "Float64", "Error in drift in velocity."