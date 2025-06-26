===================
Running a Primitive
===================

The following examples demonstrate how to run a single primitive. Files are retrieved from the shared HISPEC Google Drive folder.

Subtract Dark Frame
-------------------

This example subtracts a dark frame from an input frame at level L0 (rate). In this case, a dark frame is needed to run the dark subtraction primitive. Eventually calibrations are pulled from a local calibration cache, but for now we explicitly pass a dark frame to the primitive.

.. code-block:: python

    # Imports
    from hispecdrp import DarkSubtractionPrimitive
    from hispecdrp.utils import download_gdrive_file
    from hispecdrp import datamodels
    import numpy as np

    # Input files
    sci_path = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits', use_cached=True)
    dark_path = download_gdrive_file('HISPEC/Cals/HB.20240924.0000.00.master-dark.fits', use_cached=True)

    # Setup and call the dark primitive
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(sci_path, dark=dark_path)



Fit 1D LFC Modes
----------------

This example fits the 1D LFC modes from a Level 2 LFC file.

.. code-block:: python

    from hispecdrp import FitLFCModesPrimitive
    from hispecdrp.utils import download_gdrive_file
    from hispecdrp import datamodels

    # Input files
    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits', use_cached=True)

    # Specify which orders and fibers to fit
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = [3]

    # Setup and call the LFC modes fitting primitive
    prim = FitLFCModesPrimitive()
    mode_fit_result = prim.apply(
        lfc_path,
        do_orders=do_orders,
        do_fibers=do_fibers,
        sigma_bounds=(0.1, 2.5),
        mu_bounds=(-1, 1),
        lfc_nu0=datamodels.parvi.LFC_NU_0,
        lfc_delta_nu=datamodels.parvi.LFC_DELTA_NU,
    )

    # Alternatively, one can set arguments as follows:
    prim = FitLFCModesPrimitive()
    prim.args.do_orders = do_orders
    prim.args.do_fibers = do_fibers
    prim.args.sigma_bounds = (0.1, 2.5)
    prim.args.mu_bounds = (-1, 1)
    prim.args.lfc_nu0 = datamodels.parvi.LFC_NU_0
    prim.args.lfc_delta_nu = datamodels.parvi.LFC_DELTA_NU
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
      lfc_nu0: {datamodels.parvi.LFC_NU_0}
      lfc_delta_nu: {datamodels.parvi.LFC_DELTA_NU}
    """
    config_file = 'test_config.yaml'
    import textwrap
    with open(config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))
    prim = FitLFCModesPrimitive(config=config_file)
    mode_fit_result = prim.apply(lfc_path)


