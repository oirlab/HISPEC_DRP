HISPEC Raw Data
===============

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
     - 4096 x 4096 x Nreads x Ngroups
     - ADU


1. **DATA**: The raw UTR data.
