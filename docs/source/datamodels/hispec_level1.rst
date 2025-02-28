HISPEC Level 1
==============

HISPEC level 1 (L1) data products consist of 1D spectra immediately following spectral extraction. These spectra include a wavelength solution derived from the most recent wavelength calibration available in the calibration database (likely from the LFC). However, they do not yet incorporate the simultaneous etalon data to account for any wavelength drift or any additional corrections.

Extensions
----------

.. list-table:: 
   :widths: 10 15 10 10 15 10
   :header-rows: 1

   * - HDU
     - EXTNAME
     - HDUTYPE
     - Data Type
     - Dimensions
     - Units
   * - 1-3
     - SPEC_UNCAL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 4-6
     - SPEC_UNCAL_ERR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 7-9
     - SPEC_UNCAL_VAR_POISSON_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - (e-/s)²
   * - 10-12
     - SPEC_UNCAL_VAR_RNOISE_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - (e-/s)²
   * - 13-15
     - SPEC_DQ_[B/R]SPEC_FIB#
     - Image
     - Uint32
     - Nord x 4096
     - None
   * - 16-18
     - MID_TIME_[B/R]SPEC_FIB#
     - Image
     - Float64
     - Nord x 4096
     - MJD
   * - 19-21
     - WS_[B/R]SPEC_FIB#
     - Image
     - Float64
     - Nord x 4096
     - micron


**1-3. SPEC_UNCAL_[B/R]SPEC_FIB#**: Extracted 1D spectra.

**4-6. SPEC_UNCAL_ERR_[B/R]SPEC_FIB#**: Extracted 1D spectra errors (all noise sources).

**7-9. SPEC_UNCAL_VAR_POISSON_[B/R]SPEC_FIB#**: Extracted 1D spectra errors (Poisson noise only).

**10-12. SPEC_UNCAL_VAR_RNOISE_[B/R]SPEC_FIB#**: Extracted 1D spectra errors (read noise only).

**13-15. SPEC_DQ_[B/R]SPEC_FIB#**: Data quality.

**16-18. MID_TIME_[B/R]SPEC_FIB#**: Exposure mid-point time for wavelength bin.

**19-21. WS_[B/R]SPEC_FIB#**: Wavelength solutions likely derived from that night's LFC.