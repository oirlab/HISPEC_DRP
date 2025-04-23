=================
Calibration Store
=================

The class :py:class:`~hispecdrp.calibrations.store.CalibrationStore` manages the local (Keck or DRP user) and remote (KOA) databases.


Calibration Cache
-----------------

The local calibration cache must have the directory structure below. All folders will be created if they do not exist. The local SQLite database must be named ``hispec_calibrations.db`` and be located in the ``database/`` subdirectory. The calibration files must be located in the ``calibrations/`` subdirectory.

.. code-block:: text

    /hispec_calibration_cache/
        ├── calibrations/
        │   ├── calibration_filename1.fits
        │   ├── calibration_filename2.fits
        │   ├── calibration_filename3.fits
        │   └── ...
        └── database/
            └── hispec_calibrations.db


Examples
--------

**1. Construct a local SQLite DB from an existing cache directory.**

.. code-block:: python

    from hispecdrp import get_calibration_store

    # Create calibration store
    # Both the location and base folder name can be anything
    store = get_calibration_store(cache_dir='/home/user/hispecdrp_calibration_cache/')

    # Update 'hispecdrp_calibration_cache/database/hispec_calibrations.db'
    # with the contents of 'hispecdrp_calibration_cache/calibrations/'
    store.update_local_db_from_cache()


**2. How to retrieve a dark calibration.**

.. code-block:: python

    from hispecdrp import DarkSubtractionPrimitive

    sci_model = ...  # Load a science exposure model

    # Query for a dark calibration given a science exposure
    # Effectively how the dark primitive retrieves a dark frame from a science exposure.
    dark_prim = DarkSubtractionPrimitive()
    dark_path = dark_prim.get_calibration(sci_model, use_cached=True)