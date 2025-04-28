HISPEC Level 1
==============

HISPEC level 1 (L1) data products consist of 1D spectra immediately following spectral extraction. These spectra include a wavelength solution derived from the most recent wavelength calibration available in the calibration database (likely from the LFC). However, they do not yet incorporate the simultaneous etalon data to account for any wavelength drift or any additional corrections.

Extensions
----------

.. csv-table::
   :header: "#", "Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   "1-3", "SPEC_UNCAL_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Extracted 1D spectra."
   "4-6", "SPEC_UNCAL_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Extracted 1D spectra errors (all sources)."
   "7-9", "SPEC_UNCAL_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Extracted 1D spectra variance (Poisson)."
   "10-12", "SPEC_UNCAL_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Extracted 1D spectra variance (read noise)."
   "13-15", "SPEC_DQ_[B/R]SPEC_FIB#", "Image", "Uint32", "Nord x 4096", "None", "Data quality."
   "16-18", "MID_TIME_[B/R]SPEC_FIB#", "Image", "Float64", "Nord x 4096", "MJD", "Exposure mid-point times."
   "19-21", "WS_[B/R]SPEC_FIB#", "Image", "Float64", "Nord x 4096", "micron", "Wavelength solution."
