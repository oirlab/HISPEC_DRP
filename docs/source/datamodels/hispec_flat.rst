HISPEC Flat Model
=================

The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.


Extensions
----------

.. list-table::
   :widths: 10 10 10 10 10 10
   :header-rows: 1

   * - HDU
     - EXTNAME
     - HDUTYPE
     - Data Type
     - Dimensions
     - Units
   * - 1
     - DATA
     - Image
     - Float32
     - 4096 x 4096
     - None (normalized)
   * - 2
     - ERR
     - Image
     - Float32
     - 4096 x 4096
     - None
   * - 3
     - VAR_POISSON
     - Image
     - Float32
     - 4096 x 4096
     - None
   * - 4
     - VAR_RNOISE
     - Image
     - Float32
     - 4096 x 4096
     - None
   * - 5
     - DQ
     - Image
     - UInt32
     - 4096 x 4096
     - None



**1. DATA**: The raw UTR data.

**2. ERR**: The flatfield error (all noise sources).

**3. VAR_POISSON**: The estimated variance of the Poisson noise.

**4. VAR_RNOISE**: The effective read noise variance.

**5. DQ**: Data quality.