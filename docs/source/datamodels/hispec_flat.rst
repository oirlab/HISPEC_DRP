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
     - e-
   * - 2
     - ERR_ALL
     - Image
     - Float32
     - 4096 x 4096
     - e-
   * - 3
     - ERR_FLOOR
     - Image
     - Float32
     - 4096 x 4096
     - e-
   * - 4
     - DQ
     - Image
     - UInt32
     - 4096 x 4096
     - None



**1. DATA**: The raw UTR data.

**2. ERR_ALL**: The dark current error (all noise sources).

**3. ERR_FLOOR**: The dark current error (photon and read noise only).

**4. DQ**: Data quality.
