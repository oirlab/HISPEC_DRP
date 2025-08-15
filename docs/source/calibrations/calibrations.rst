============
Calibrations
============

Overview
--------

The :py:mod:`~hispecdrp.calibrations` module contains code to:

1. Interact with a local calibration cache that stores calibration files and uses an SQLite database to interface with metadata.
2. Select the appropriate calibration file for a given input data product based on metadata and instrument configuration via SQL queries.
3. Sync the metadata DB with KOA as new calibrations are ingested.

In order to select and retrieve calibrations for processing, the HISPEC DRP interacts with the `KOA Middleware <https://oirlab.github.io/koa_middleware/>`_ package. This package can download science and calibration files from the Keck Observatory Archive (KOA) and provides a local cache for these files. `koa_middleware` also provides a framework for matching calibration files to science data products based on metadata.

See *<link TBD>* for how to implement new calibration selectors.

.. toctree::
   :maxdepth: 2
   :glob:

   cache
   database
   selectors