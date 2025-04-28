============
Calibrations
============

Overview
--------

The :py:mod:`~hispecdrp.calibrations` module contains code to:

1. Access calibrations stored in a local cache directory.
2. Sync the local cache with KOA as new calibrations are ingested.
3. Query and filter calibration metadata through an SQLite database with logic to select the best calibration given an input data product.

.. toctree::
   :maxdepth: 2
   :glob:
    
   ./*