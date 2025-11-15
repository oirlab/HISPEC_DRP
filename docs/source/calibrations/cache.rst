=================
Calibration Cache
=================

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

- **KOA_CALIBRATION_CACHE**: Required for normal operations. Path to the top-level directory for downloaded calibrations. If not provided, the cache_dir must be specified when creating the calibration store.
- **KOA_LOCAL_DATABASE_FILENAME**: Optional. Name of the local SQLite database file. Default is ``f'{instrument.lower()}_calibrations.db'``.
- **KOA_REMOTE_DATABASE_URL**: Optional. PostgreSQL URL for the remote database. Set to None for only local operations including PARVI. Default is None for now, eventually will point to KOA once deployed.
- **KOA_CALIBRATION_URL**: Optional. URL where actual calibrations (FITS files) are stored. Default is None for now, eventually will point to KOA once deployed.

Examples
--------

**1. Construct a local SQLite DB from an existing cache directory.**

.. code-block:: python

    from hispecdrp import RegisterCalibrationPrimitive
    from hispecdrp import datamodels
    
    import os

    # Open dark model
    dark_path = 'HB.20240924.0000.00.master-dark.fits'
    dark_model = datamodels.open(dark_path)

    # Set the cache path
    os.environ['KOA_CALIBRATION_CACHE'] = '/home/user/hispecdrp_calibration_cache/'

    # Store the dark model in the local cache
    prim = RegisterCalibrationPrimitive()
    prim.apply(dark_model)

**2. How to retrieve a dark calibration.**

.. code-block:: python

    from koa_middleware import CalibrationStore
    from hispecdrp import DarkSubtractionPrimitive
    from hispecdrp.calibrations import DarkSelector
    from hispecdrp import datamodels
    
    import os

    # Open science model
    sci_path = 'HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    sci_model = datamodels.open(sci_path)

    # Set the cache path
    os.environ['KOA_CALIBRATION_CACHE'] = '/home/user/hispecdrp_calibration_cache/'

    # Query for a dark calibration given a science exposure using the Dark Primitive
    dark_prim = DarkSubtractionPrimitive()
    dark_path = dark_prim.get_calibration(sci_model, use_cached=True)
    
    # Alternatively, query for a dark calibration using the CalibrationStore directly
    with CalibrationStore(orm_class=sci_model.get_calibration_orm_class()) as store:
        dark_path = store.get_calibration(sci_model, selector=DarkSelector(), use_cached=True)