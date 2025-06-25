=================
Calibration Store
=================

The class :py:class:`~hispecdrp.calibrations.store.CalibrationStore` manages the local (Keck or DRP user) and remote (KOA) databases.


Calibration Cache
-----------------

The local calibration cache must have the directory structure below. The sub-folders will be created if they do not exist. The local SQLite database must be named ``<instrument_name>_calibrations.db`` where ``instrument_name`` is currently either ``parvi`` or ``hispec`` and be located in the ``database/`` subdirectory. The calibration files are located in the ``calibrations/`` subdirectory.

.. code-block:: text

    /hispec_calibration_cache/
        ├── calibrations/
        │   ├── calibration_filename1.fits
        │   ├── calibration_filename2.fits
        │   ├── calibration_filename3.fits
        │   └── ...
        └── database/
            └── hispec_calibrations.db


Environment Variables
---------------------

The following environment variables can be set to configure the calibration store:

- **HISPECDRP_CALIBRATION_CACHE**: Required for normal operations. Path to the top-level directory for downloaded calibrations. If not provided, the cache_dir must be specified when creating the calibration store.
- **HISPECDRP_CALIBRATION_DATABASE_FILENAME_<INSTRUMENT>**: Optional. Name of the local SQLite database file. Default is 'hispec_calibrations.db' for HISPEC and 'parvi_calibrations.db' for PARVI. Different env vars can be used with different instrument names.
- **HISPECDRP_CALIBRATION_DATABASE_URL**: Optional. PostgreSQL URL for the remote database. Set to None for only local operations including PARVI. Default is None for now, eventually KOA once deployed.
- **HISPECDRP_CALIBRATION_URL**: Optional. URL where actual calibrations (FITS files) are stored.

Examples
--------

**1. Construct a local SQLite DB from an existing cache directory.**

.. code-block:: python

    from hispecdrp import get_calibration_store

    # Create calibration store
    # Both the location and base folder name can be anything
    store = get_calibration_store(cache_dir='/home/user/hispecdrp_calibration_cache/', instrument='hispec')

    # Update 'hispecdrp_calibration_cache/database/hispec_calibrations.db'
    # with the contents of 'hispecdrp_calibration_cache/calibrations/'
    store.update_local_db_from_cache()


**2. How to retrieve a dark calibration.**

.. code-block:: python

    from hispecdrp import DarkSubtractionPrimitive
    from hispecdrp import datamodels

    sci_model = datamodels.open(filepath)  # Load a science exposure model

    # Query for a dark calibration given a science exposure
    # Effectively how the dark primitive retrieves a dark frame from a science exposure.
    dark_prim = DarkSubtractionPrimitive()
    dark_path = dark_prim.get_calibration(sci_model, use_cached=True)

    # Alternatively, you can use the store directly if you know the selector to use.
    from hispecdrp import get_calibration_store
    from hispecdrp.calibrations import HISPECDarkSelector
    store = get_calibration_store(instrument='hispec')
    dark_path = store.get_calibration(sci_model, selector=HISPECDarkSelector(), use_cached=True)