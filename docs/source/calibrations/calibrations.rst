============
Calibrations
============

Overview
--------

The :py:mod:`~hispecdrp.calibrations` module contains code to:

1. Interact with a local calibration cache that stores HISPEC and PARVI calibration files and uses an SQLite database to interact with the calibration metadata.
2. Select the appropriate calibration file for a given input based on the metadata and instrument configuration using SQL queries and Python filtering.
3. Download new calibration metadata entries from KOA as new calibrations are created from the main DRP running at Keck.
4. Register new custom calibrations to the local cache after they are created by the DRP.
5. Upload new calibration files and metadata to the remote KOA database after they are created by the DRP (*restricted access*).

In order to select and retrieve calibrations for processing, the HISPEC DRP interacts with the `KOA Middleware <https://oirlab.github.io/KOA_Middleware/>`_ package. This package can download science and calibration files from the Keck Observatory Archive (KOA) and provides a local cache for these files. `koa_middleware` also provides the framework for matching calibration files to science data products based on metadata.

See `KOA Middleware Documentation <https://oirlab.github.io/KOA_Middleware/index.html>`_ for more information.

**The KOA Middleware quickstart and HISPEC DRP quickstarts both present common use cases of working with calibration data:**

- `KOA Middleware Quickstart <https://oirlab.github.io/KOA_Middleware/quickstart.html>`_.

- :doc:`HISPEC DRP Calibration Examples <../quickstart/calibration_examples>`.

In the sections below, we provide an overview of using the calibration database in the context of the HISPEC DRP, and link to the KOA Middleware documentation where appropriate.

.. toctree::
   :maxdepth: 2
   :glob:

   configure
   calibration_database