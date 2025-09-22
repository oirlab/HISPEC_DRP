===================
Running a Primitive
===================

The following examples demonstrate how to run a single primitive. Files are retrieved from the shared HISPEC Google Drive folder.

Subtract Dark Frame
-------------------

This example subtracts a dark frame from an input frame at level L0 (rate map). In this case, a dark frame is needed to run the dark subtraction primitive. Calibrations are typically associated and retrieved automatically by the DRP (see :doc:`calibrations <../calibrations/calibrations>` and :doc:`Primitive Design <../development/primitives>`). Below we show the different methods for associated calibrations through a primitive's configuraiton.

.. code-block:: python

    # Imports
    from hispecdrp import DarkSubtractionPrimitive
    from hispecdrp.utils import download_gdrive_file
    from hispecdrp import datamodels

    # Input files
    sci_path = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_rate-TELLURICS-FIB1.fits')
    dark_path = download_gdrive_file('HISPEC/Cals/HB.20240924.0000.00.master-dark.fits')
    dark_model = datamodels.open(dark_path)

    # Use default behavior
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(sci_path, dark=dark_model) # dark can be dark_path or dark_model


The associated dark calibration can be found with:

.. code-block:: python

    # Returns dictionary with calibration information
    dark_info = model_result.get_calibration_from_role('dark')


Fit 1D LFC Modes
----------------

This example fits the 1D LFC modes from a Level 2 LFC file.

.. code-block:: python

    from hispecdrp import FitLFCLinesPrimitive
    from hispecdrp.utils import download_gdrive_file

    import textwrap

    # Input files
    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits', use_cached=True)

    # Specify which orders and fibers to fit
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = [3]

    # Setup and call the LFC modes fitting primitive
    prim = FitLFCLinesPrimitive()
    mode_fit_result = prim.apply(
        lfc_path,
        do_orders=do_orders,
        do_fibers=do_fibers,
        sigma_bounds=(0.1, 2.5),
        mu_bounds=(-1, 1),
    )

    # Alternatively, one can set arguments as follows:
    prim = FitLFCLinesPrimitive()
    prim.do_orders = do_orders
    prim.do_fibers = do_fibers
    prim.sigma_bounds = (0.1, 2.5)
    prim.mu_bounds = (-1, 1)
    mode_fit_result = prim.apply(lfc_path)

    # Lastly, one can use a config file to set arguments as follows:
    # See https://oirlab.github.io/HISPEC_DRP/development/primitives.html for more information on using config files.
    config_raw = f"""
    %YAML 1.2
    ---

    logging:
      level: info
      log_path: None

    arguments:
      run: True
      save_result: False
      do_orders: {do_orders}
      do_fibers: {do_fibers}
      sigma_bounds: [0.1, 2.5] # Needs to be a list!
      mu_bounds: [-1, 1] # Needs to be a list!
    """
    config_file = 'test_config.yaml'
    with open(config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))
    prim = FitLFCLinesPrimitive(config=config_file)
    mode_fit_result = prim.apply(lfc_path)