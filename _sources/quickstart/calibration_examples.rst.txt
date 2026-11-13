=========================
Working with Calibrations
=========================

**Select and Retrieve a calibration with a Primitive.**

This will use the default calibration selector associated with the primitive (see primitive class).

.. code-block:: python

    from hispecdrp import DarkSubtractionPrimitive

    # Download and open raw science data
    sci_path = download_gdrive_file('HB.20240924.00000.00_raw.fits')
    sci_model = datamodels.open(sci_path)

    # Automatically select and retrieve a dark calibration using the DarkSubtractionPrimitive
    dark_prim = DarkSubtractionPrimitive(sci_model)
    dark_path, dark_record = dark_prim.get_calibration(sci_model, 'dark')


Calibration Store
-----------------

The class :py:class:`~hispecdrp.calibrations.hispec_store.HISPECCalibrationStore` provides an interface for registering, querying, and retrieving calibration files from the local cache and remote database on KOA. The store uses a local SQLite database to keep track of all available calibration metadata both local and from KOA.

Here we specify the calibration selector to use. See `Calibration Selectors Documentation <https://oirlab.github.io/KOA_Middleware/selectors.html>`_ for more details.

.. code-block:: python

    from hispecdrp.calibrations import HISPECCalibrationStore, DarkSelector

    # Open science model
    sci_path = download_gdrive_file('HB.20240924.00000.00_raw.fits')
    sci_model = datamodels.open(sci_path)
    
    # Alternatively, query for a dark calibration using the CalibrationStore directly using a custom selector
    with HISPECCalibrationStore() as store:
        dark_path = store.get_calibration(sci_model, selector=DarkSelector())


**Register a calibration to the local cache.**

.. code-block:: python

    from hispecdrp import RegisterCalibrationPrimitive
    from hispecdrp import datamodels

    # Create a dark model
    dark_model = datamodels.open('/path/to/dark_calibration.fits')

    # Store the dark model in the local cache
    prim = RegisterCalibrationPrimitive.call(dark_model)
