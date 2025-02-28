HISPEC Raw (Ramp) Data
======================

The Teledyne Hawaii H4RG detectors up-the-ramp (UTR) non-destructive read modes, so HISPEC raw data is made of a set of UTR images.


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
     - UInt16
     - 4096 x 4096 x Nreads
     - ADU
   * - 2
     - TIMES
     - Image
     - Float64
     - Nreads
     - s


1. **DATA**: The raw UTR data.

2. **TIMES**: The exposure times for each read. *NOTE: This is a temporary solution until we choose how to store the minimum information rerequired to generate the exposure times dynamically.*