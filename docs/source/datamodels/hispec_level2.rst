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


**1-3. SPEC_UNCAL_[B/R]SPEC_FIB#**

**4-6. SPEC_UNCAL_ERR_ALL_[B/R]SPEC_FIB#**

**7-9. SPEC_UNCAL_ERR_FLOOR_[B/R]SPEC_FIB#**

**10-12. SPEC_[B/R]SPEC_FIB#**

**13-15. SPEC_ERR_ALL_[B/R]SPEC_FIB#**

**16-18. SPEC_ERR_FLOOR_[B/R]SPEC_FIB#**

**19-21. SPEC_DQ_[B/R]SPEC_FIB#**

**22-24. MID_TIME_[B/R]SPEC_FIB#**

**25-27. BARY_CORR_[B/R]SPEC_FIB#**

**28-30. WS_[B/R]SPEC_FIB#**

**31-33. FIBERFLAT_[B/R]SPEC_FIB#**

**34-36. FIBERFLAT_ERR_ALL_[B/R]SPEC_FIB#**

**37-39. FIBERFLAT_ERR_FLOOR_[B/R]SPEC_FIB#**

**40-42. SKY_[B/R]SPEC_FIB#**

**43-45. SKY_ERR_ALL_[B/R]SPEC_FIB#**

**46-48. SKY_ERR_FLOOR_[B/R]SPEC_FIB#**

**49-51. STAR_MOD_[B/R]SPEC_FIB#**

**52-54. E2E_TPUT_[B/R]SPEC_FIB#**

**55-57. TEL_[B/R]SPEC_FIB#**

**58-60. TEL_ERR_ALL_[B/R]SPEC_FIB#**

**61-63. TEL_ERR_FLOOR_[B/R]SPEC_FIB#**

**64-66. FLUX_ABS_[B/R]SPEC_FIB#**

**67-69. FLUX_ABS_ERR_ALL_[B/R]SPEC_FIB#**

**70-72. FLUX_ABS_ERR_FLOOR_[B/R]SPEC_FIB#**

**73-75. ET_[B/R]SPEC_FIB#**

**76-78. ET_ERR_FLOOR_[B/R]SPEC_FIB#**

**79-81. ET_ERR_ALL_[B/R]SPEC_FIB#**

**82-84. ET_CEN_[B/R]SPEC_FIB#**

**85-87. RVCCF_[B/R]SPEC_FIB#**

**88-90. RVCCFWAVG_[B/R]SPEC_FIB#**

**91-93. CCF_[B/R]SPEC_FIB#**