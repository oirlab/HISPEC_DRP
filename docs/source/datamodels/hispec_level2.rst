HISPEC Level 2
==============

HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements. Data products from BSPEC and RSPEC are still kept separate.


Extensions
----------

.. list-table::
   :widths: 10 20 10 10 15 10
   :header-rows: 1

   * - HDU
     - EXTNAME
     - HDUTYPE
     - Data Type
     - Dimensions
     - Units
   * - 0
     - N/A
     - Primary
     - N/A
     - N/A
     - 
   * - 1-3
     - SPEC_UNCAL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 4-6
     - SPEC_UNCAL_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 7-9
     - SPEC_UNCAL_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 10-12
     - SPEC_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s (or W/m²/µm)
   * - 13-15
     - SPEC_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s (or W/m²/µm)
   * - 16-18
     - SPEC_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s (or W/m²/µm)
   * - 19-21
     - SPEC_DQ_[B/R]SPEC_FIB#
     - Image
     - Uint32
     - Nord x 4096
     - N/A
   * - 22-24
     - MID_TIME_[B/R]SPEC_FIB#
     - Image
     - Float64
     - Nord x 4096
     - MJD
   * - 25-27
     - BARY_CORR_[B/R]SPEC_FIB#
     - Table
     - See below
     - See below
     - See below
   * - 28-30
     - WS_[B/R]SPEC_FIB#
     - Image
     - Float64
     - Nord x 4096
     - micron
   * - 31-33
     - FIBERFLAT_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 34-36
     - FIBERFLAT_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 37-39
     - FIBERFLAT_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 40-42
     - SKY_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 43-45
     - SKY_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 46-48
     - SKY_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 49-51
     - STAR_MOD_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - W/m²/µm
   * - 52-54
     - E2E_TPUT_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 55-57
     - TEL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 58-60
     - TEL_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 61-63
     - TEL_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - N/A
   * - 64-66
     - FLUX_ABS_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - W/m²/µm/(e-/s)
   * - 67-69
     - FLUX_ABS_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - W/m²/µm/(e-/s)
   * - 70-72
     - FLUX_ABS_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - W/m²/µm/(e-/s)
   * - 73-75
     - ET_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 76-78
     - ET_ERR_FLOOR_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 79-81
     - ET_ERR_ALL_[B/R]SPEC_FIB#
     - Image
     - Float32
     - Nord x 4096
     - e-/s
   * - 82-84
     - ET_CEN_[B/R]SPEC_FIB#
     - Table
     - See below
     - See below
     - See below
   * - 85-87
     - RVCCF_[B/R]SPEC_FIB#
     - Table
     - See below
     - See below
     - See below
   * - 88-90
     - RVCCFWAVG_[B/R]SPEC_FIB#
     - Table
     - See below
     - See below
     - See below
   * - 91-93
     - CCF_[B/R]SPEC_FIB#
     - Table
     - See below
     - See below
     - See below


**1-3. SPEC_UNCAL_[B/R]SPEC_FIB#**: 1D original flux in e-/s from level 1 which remains unmodified throughout stage 2 processing.

**4-6. SPEC_UNCAL_ERR_ALL_[B/R]SPEC_FIB#**: 1D vectors of original errors from level 1 with all sources of noise, including correlated errors.

**7-9. SPEC_UNCAL_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors (photon noise & read noise), also from level 1.

**10-12. SPEC_[B/R]SPEC_FIB#**: Calibrated 1D spectrum. The nature of this extension depends on the stage 2 reduction steps that have been run. It can, for example, be sky subtracted, blaze corrected, telluric corrected, or a subset of those.

**13-15. SPEC_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the calibrated spectrum.

**16-18. SPEC_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the calibrated spectrum.

**19-21. SPEC_DQ_[B/R]SPEC_FIB#**: Data Quality flag (i.e., general “bad pixel map”).

**22-24. MID_TIME_[B/R]SPEC_FIB#**: Exposure mid-point time (for barycenter correction).

**25-27. BARY_CORR_[B/R]SPEC_FIB#**: Table of exposure mid-point times and barycentric corrections. The DRP computes both flux-weighted and non-weighted timing midpoints for an exposure from the UTR ramp fit. The correction is specific to each echelle order. Some of the values in this table are redundant with the table in the extension **RVCCF_[B/R]SPEC_FIB#** (see below). We will consider merging these tables in future updates.

**Columns:**

- ``ORDER (int32)``: Echelle order number.
- ``WAVE (float64)``: Wavelength of the middle of order in microns.
- ``JDUTC (float64)``: Julian date in universal coordinate time (UTC) of the non-flux-weighted midpoint.
- ``JDUTCW (float64)``: Julian date in universal coordinate time (UTC) of the flux-weighted midpoint.
- ``HJD (float64)``: Heliocentric Julian date of the non-flux-weighted midpoint.
- ``HJDW (float64)``: Heliocentric Julian date of the flux-weighted midpoint.
- ``BJD (float64)``: Barycentric Julian date of the non-flux-weighted midpoint.
- ``BJDW (float64)``: Barycentric Julian date of the flux-weighted midpoint.
- ``BERV (float64)``: Barycentric velicity correction for the non-flux-weighted midpoint in m/s.
- ``BERVW (float64)``: Barycentric velicity correction for the flux-weighted midpoint in m/s.

**28-30. WS_[B/R]SPEC_FIB#**: Wavelength in microns in rest frame of the instrument.

**31-33. FIBERFLAT_[B/R]SPEC_FIB#**: Instrumental transmission.

**34-36. FIBERFLAT_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the instrumental transmission.

**37-39. FIBERFLAT_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the instrumental transmission.

**40-42. SKY_[B/R]SPEC_FIB#**: Thermal background and OH lines from the sky fiber.

**43-45. SKY_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the sky background spectrum.

**46-48. SKY_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the sky background spectrum.

**49-51. STAR_MOD_[B/R]SPEC_FIB#**: The stellar model sampled on the detector and appropriately Doppler shifted to match the observed wavelengths.

**52-54. E2E_TPUT_[B/R]SPEC_FIB#**: End-to-end throughput derived from STAR_MOD.

**55-57. TEL_[B/R]SPEC_FIB#**: Telluric model sampled on the detector in normalized units.

**58-60. TEL_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the telluric model spectrum.

**61-63. TEL_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the telluric model spectrum.

**64-66. FLUX_ABS_[B/R]SPEC_FIB#**: Absolute flux calibration derived from the ATC images which provides the scaling factors to go from blaze corrected e-/s to W/m²/µm.

**67-69. FLUX_ABS_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the absolute flux calibration.

**70-72. FLUX_ABS_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the absolute flux calibration.

**73-75. ET_[B/R]SPEC_FIB#**: Simultaneous etalon spectrum.

**76-78. ET_ERR_ALL_[B/R]SPEC_FIB#**: All errors for the etalon spectrum.

**79-81. ET_ERR_FLOOR_[B/R]SPEC_FIB#**: Uncorrelated errors for the etalon spectrum.

**82-84. ET_CEN_[B/R]SPEC_FIB#**: Fit results for each etalon mode from the simultaneous etalon spectrum.

**Columns:**

- ``AMPLITUDE, AMPLITUDE_ERR, (float64)``: The amplitude of the etalon mode and its uncertainty in e-.
- ``MEAN, MEAN_ERR (float64)``: The center of the etalon mode and its uncertainty in pixels.
- ``SIGMA, SIGMA_ERR (float64)``: The standard deviation of the etalon mode and its uncertainty in pixels.
- ``B0, B0_ERR (float64)``: The background level and uncertainty in e-.
- ``MODE_NUM (int32)``: The mode number of this etalon peak. Etalon mode numbers are arbitrary, but are nice to keep track of.
- ``WAVE, WAVE_ERR (float64)``: The measured wavelength of the etalon peak and its uncertainty as calibrated by the LFC in microns.
- ``FINESSE, FINESSE_ERR (float64)``: The cavity finesse and its uncertainty.
- ``CAVLEN, CAVLEN_ERR (float64)``: The effective cavity length at this wavelength.

**85-87. RVCCF_[B/R]SPEC_FIB#**: The RVs and other relevant information for the individual orders. By default, all RVs are corrected by the weighted barycentric velocity BERVW.

**Columns:**

- ``BJDW (float64)``: Barycentric Julian date of the flux-weighted midpoint.
- ``BERVW (float64)``: Barycentric velicity correction for the flux-weighted midpoint in m/s.
- ``RV (float64)``: The RV in m/s.
- ``RV_ERR (float64)``: The RV uncertainty in m/s.
- ``WAVE (float64)``: Wavelength of the middle of order in microns.
- ``SPECRES (float64)``: Average spectral resolution in this order.
- ``TELL_ABS (float64)``: Average telluric absorption in this order.
- ``BIS (float64)``: Bisector inverse slope (BIS) in m/s.
- ``RV_ABS (float64)``: Absolute RV measured in m/s.
- ``RV_ABS_ERR (float64)``: Error in absolute RV in m/s.

**88-90. RVCCFWAVG_[B/R]SPEC_FIB#**: The RVs averaged together across chunks and other relevant information.

**Columns** there will likely only be one value per column, so we may consider an alternative approach for this extension:

- ``BJDW (float64)``: Average barycentric Julian date of the flux-weighted midpoint.
- ``BERVW (float64)``: Barycentric velicity correction for the flux-weighted midpoint in m/s.
- ``RV (float64)``: RV in m/s.
- ``RV_ERR_AGG (float64)``: RV uncertainty measured by adding the individual RV errors in quadrature.
- ``RV_ERR_DIST (float64)``: RV uncertainty measured from the distribution of the individual RVs assuming they are normally distributed.
- ``AIRMASS (float64)``: Airmass at the midpoint of this observation.
- ``BIS (float64)``: Bisector inverse slope (BIS) in m/s.
- ``RV_ABS (float64)``: Absolute RV measured in m/s.
- ``RV_ABS_ERR (float64)``: Error in absolute RV in m/s.

**91-93. CCF_[B/R]SPEC_FIB#**: The cross-correlation functions (CCFs) for each chunk. **Final format TBD**.

**Columns:**

- ``WAVE (float64)``: The average wavelength in this order (microns).
- ``VEL_GRID (float64)``: The relative wavelength or velocity grid (TBD).
- ``CCF (float64)``: The CCF function.