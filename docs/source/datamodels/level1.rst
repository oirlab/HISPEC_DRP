Level 1 (Uncalibrated Spectra)
==============================

HISPEC level 1 (L1) data products consist of 1D spectra immediately following spectral extraction. These spectra include a wavelength solution derived from the most recent wavelength calibration available in the calibration database (likely from the LFC). However, they do not yet incorporate the simultaneous etalon data to account for any wavelength drift or any additional corrections.

* :py:class:`~hispecdrp.datamodels.hispec.level1.HISPECL1Model`
* :py:class:`~hispecdrp.datamodels.parvi.level1.PARVIL1Model`

**File suffix:** specuncal


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   "SPEC_UNCAL_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Extracted 1D spectra."
   "SPEC_UNCAL_ERR_SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Extracted 1D spectra errors (all sources)."
   "SPEC_UNCAL_VAR_POISSON_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Extracted 1D spectra variance (Poisson)."
   "SPEC_UNCAL_VAR_RNOISE_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Extracted 1D spectra variance (read noise)."
   "SPEC_DQ_FIB#", "Image", "UInt32", "Nord x 4096", "None", "Data quality."
   "MID_TIME_FIB#", "Image", "Float64", "Nord x 4096", "MJD", "Exposure mid-point times."
   "WAVE_FIB#", "Image", "Float64", "Nord x 4096", "micron", "Wavelength solution."
