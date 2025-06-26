============
Calibrations
============

Overview
--------

The :py:mod:`~hispecdrp.calibrations` module contains code to:

1. Interact with a local calibration cache that stores calibration files and uses an SQLite database to interace with metadata.
2. Select the appropriate calibration file for a given input data product based on metadata and instrument configuration via SQL queries.
3. Sync the metadata DB with KOA as new calibrations are ingested.

.. toctree::
   :maxdepth: 2
   :glob:
    
   selectors
   database
   store
