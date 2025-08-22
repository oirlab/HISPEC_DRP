==================
Running a Pipeline
==================

The following examples demonstrate how to run a pipeline, which is a chain of primitives. Files are retrieved from the shared HISPEC Google Drive folder.

Process LFC Pipeline
--------------------

This pipeline takes as input level 2 laser frequency comb (LFC) data, and does the following:
1. Fit the 1D LFC modes with a Gaussian to determine the location of LFC lines in detector pixels.
2. Fit the resulting locations with a 2D Chebyshev polynomial to determine the wavelength solution of every pixel.
3. Fit the resulting Gaussian sigmas with a 2D Chebyshev polynomial to determine the effective Gaussin line spread function at every pixel as a function of wavelength.
4. Save the wavelength solution and LSF calibrations to the calibration cache and register each in the local database.

.. code-block:: python

    from hispecdrp import datamodels
    from hispecdrp import ProcessLFCPipeline
    from hispecdrp.utils import download_gdrive_file

    import numpy as np
    import os
    import textwrap

    # Input files
    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits')

    # Path to the local calibration cache (home dir, can be anywhere)
    os.environ['HISPECDRP_CALIBRATION_CACHE'] = os.path.expanduser('~/HISPEC_CALIBRATION_CACHE')

    # Specify which fibers and orders to fit
    do_fibers = [3]
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]

    # Init pipeline object
    pipe = ProcessLFCPipeline()

    # Specify the primitive arguments here
    prim_args = {
        'fit_lfc_modes' : {
            'do_orders': do_orders,
            'do_fibers': do_fibers,
            'sigma_bounds': (0.1, 2.5),
            'mu_bounds': (-1, 1),
            'lfc_nu0': datamodels.parvi.LFC_NU_0,
            'lfc_delta_nu': datamodels.parvi.LFC_DELTA_NU,
        },
        'lfc_wavesol' : {
            'fit_orders': do_orders,
            'opt_weights': True,
            'do_fibers': do_fibers,
            'deg_inter_order': 6,
            'deg_intra_order': 9,
        },
        'lsf' : {
            'do_fibers': do_fibers,
            'deg_inter_order': 3,
            'deg_intra_order': 3,
            'fit_orders': do_orders,
        },
    }
    # Alternatively, one can also set an argument with:
    #   pipe.fit_lfc_modes.do_orders = do_orders

    # Call the pipeline with these arguments.
    model_result = pipe.apply(lfc_path, prim_args=prim_args)

    # Lastly, one can also configure the pipeline and primitives with a YAML file.
    config_raw = f"""
    %YAML 1.2
    ---

    logging:
      level: info
      log_path: None

    arguments:
      run: True
      save_result: False

    primitives:
      fit_lfc_modes:
        arguments:
          do_orders: {do_orders}
          do_fibers: {do_fibers}
          sigma_bounds: [0.1, 2.5]
          mu_bounds: [-1, 1]
          lfc_nu0: {datamodels.parvi.LFC_NU_0}
          lfc_delta_nu: {datamodels.parvi.LFC_DELTA_NU}
      lfc_wavesol:
        arguments:
          fit_orders: {do_orders}
          opt_weights: True
          do_fibers: {do_fibers}
          deg_inter_order: 6
          deg_intra_order: 9
      lsf:
        arguments:
          do_fibers: {do_fibers}
          deg_inter_order: 3
          deg_intra_order: 3
          fit_orders: {do_orders}
    """

    config_file = 'test_config.yaml'
    with open(config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))
    pipe = ProcessLFCPipeline(config=config_file)
    result = pipe.apply(lfc_path)