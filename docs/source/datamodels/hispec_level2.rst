HISPEC Level 2
==============

HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements.

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


**1-3. SPEC_UNCAL_[BSPEC/RSPEC]_FIB#**: 1D original flux in e-/s from level 1 which remains unmodified throughout stage 2 processing.

**4-6. SPEC_UNCAL_ERR_ALL_[BSPEC/RSPEC]_FIB#**: 1D vectors of original errors from level 1 with all sources of noise, including correlated errors.

**7-9. SPEC_UNCAL_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors (photon noise & read noise), also from level 1.

**10-12. SPEC_[BSPEC/RSPEC]_FIB#**: Calibrated 1D spectrum. The nature of this extension depends on the stage 2 reduction steps that have been run. It can, for example, be sky subtracted, blaze corrected, telluric corrected, or a subset of those.

**13-15. SPEC_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the calibrated spectrum.

**16-18. SPEC_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the calibrated spectrum.

**19-21. SPEC_DQ_[BSPEC/RSPEC]_FIB#**: Data Quality flag (i.e., general “bad pixel map”).

**22-24. MID_TIME_[BSPEC/RSPEC]_FIB#**: Exposure mid-point time (for barycenter correction).

**25-27. BARY_CORR_[BSPEC/RSPEC]_FIB#**: Table of exposure mid-point times (for barycenter correction) for each order (not per wavelength bin). We will compute the Julian dates in the UTC, heliocentric (HJD), and barycentric (BJD) reference frames for all targets for consistency. We also include uniformly weighted and flux weighted variations.

Columns:

- Echelle order
- Wavelength (microns) – middle of order
- JDUTC midpoint (unweighted)
- JDUTCW midpoint (weighted)
- HJD midpoint (unweighted)
- HJDW midpoint (weighted)
- BJD midpoint (unweighted)
- BJDW midpoint (weighted)
- BERV (unweighted)
- BERVW (weighted)

**28-30. WS_[BSPEC/RSPEC]_FIB#**: Wavelength in microns in rest frame of the instrument.

**31-33. FIBERFLAT_[BSPEC/RSPEC]_FIB#**: Instrumental transmission.

**34-36. FIBERFLAT_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the instrumental transmission.

**37-39. FIBERFLAT_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the instrumental transmission.

**40-42. SKY_[BSPEC/RSPEC]_FIB#**: Thermal background and OH lines from the sky fiber.

**43-45. SKY_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the sky background spectrum.

**46-48. SKY_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the sky background spectrum.

**49-51. STAR_MOD_[BSPEC/RSPEC]_FIB#**: The stellar model sampled on the detector and appropriately Doppler shifted to match the observed wavelengths.

**52-54. E2E_TPUT_[BSPEC/RSPEC]_FIB#**: End-to-end throughput derived from STAR_MOD.

**55-57. TEL_[BSPEC/RSPEC]_FIB#**: Telluric model sampled on the detector in normalized units.

**58-60. TEL_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the telluric model spectrum.

**61-63. TEL_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the telluric model spectrum.

**64-66. FLUX_ABS_[BSPEC/RSPEC]_FIB#**: Absolute flux calibration derived from the ATC images which provides the scaling factors to go from blaze corrected e-/s to W/m²/µm.

**67-69. FLUX_ABS_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the absolute flux calibration.

**70-72. FLUX_ABS_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the absolute flux calibration.

**73-75. ET_[BSPEC/RSPEC]_FIB#**: Simultaneous etalon spectrum.

**76-78. ET_ERR_ALL_[BSPEC/RSPEC]_FIB#**: All errors for the etalon spectrum.

**79-81. ET_ERR_FLOOR_[BSPEC/RSPEC]_FIB#**: Uncorrelated errors for the etalon spectrum.

**82-84. ET_CEN_[BSPEC/RSPEC]_FIB#**: Centroid (in dispersion/horizontal direction) of each etalon peak in pixel coordinates and their uncertainties.

Columns: See Section 9.3.12.

**85-87. RVCCF_[BSPEC/RSPEC]_FIB#**: The RVs and other relevant information for the individual chunks. By default, all RVs are corrected by the weighted barycentric velocity BERVW.

Columns:

- BJDUTCW midpoint (weighted)
- BERVW
- RV (m/s)
- RV_ERR_AGG (m/s)
- RV_ERR_DIST (m/s)
- WAVE (RV info. weighted avg. wavelength for this chunk in microns)
- RES (average spectral resolution in this chunk)
- TELL_ABS (average telluric absorption in this chunk)
- BIS (bisector inverse slope to probe activity; m/s)
- RV_ABS (absolute RV; m/s)
- RV_ABS_ERR

**88-90. RVCCFWAVG_[BSPEC/RSPEC]_FIB#**: The RVs averaged together across chunks and other relevant information.

Columns – there will likely only be two values per column (BSPEC, RSPEC), so we may consider an alternative approach for this extension:

- JDUTC midpoint (unweighted)
- JDUTC midpoint (weighted)
- HJDUTC midpoint (unweighted)
- HJDUTC midpoint (weighted)
- BJDUTC midpoint (unweighted)
- BJDUTC midpoint (weighted)
- BERV midpoint (unweighted)
- BERV midpoint (weighted)
- RV (m/s)
- RV_ERR (m/s)
- BIS (bisector inverse slope to probe activity; m/s)
- RV_ABS (absolute RV; m/s)

**91-93. CCF_[BSPEC/RSPEC]_FIB#**: The cross-correlation functions (CCFs) for each chunk.

Columns:

- The average wavelength in this chunk (microns).
- WAVE - the relative/effective wavelength grid in microns
- CCF - the CCF function