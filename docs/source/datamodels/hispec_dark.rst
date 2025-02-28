HISPEC Dark Model
=================

The HISPEC dark current (and any other contribution to the dark) is assumed to be linear with time.


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
     - e-
   * - 2
     - ERR
     - Image
     - Float32
     - 4096 x 4096
     - e-
   * - 3
     - VAR_POISSON
     - Image
     - Float32
     - 4096 x 4096
     - e-^2
   * - 4
     - VAR_RNOISE
     - Image
     - Float32
     - 4096 x 4096
     - e-^2
   * - 5
     - DQ
     - Image
     - UInt32
     - 4096 x 4096
     - None


**1. DATA**: The dark current data.

**2. ERR**: The dark current error (all noise sources).

**3. VAR_POISSON**: The estimated variance of the Poisson noise.

**4. VAR_RNOISE**: The effective read noise variance.

**5. DQ**: Data quality.
