==================
Running a Pipeline
==================


.. code-block:: python

    # Imports
    from hispecdrp import DarkSubtractionPrimitive
    from tests.utils import download_github_lfs_file
    import astropy.utils.data
    import textwrap
    from hispecdrp import datamodels
    import numpy as np

    # Input paths to science and cals (from GitHub or local)
    cache_dir = astropy.utils.data._get_download_cache_loc() + '/'
    git_file_path_sci = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    git_file_path_dark = 'HISPEC/Cals/HS_BSPEC_20240924T000000_master-dark.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config YAML file can also be manually created beforehand.
    config_raw = """
    %YAML 1.2
    ---

    logging:
      # Logging args here
    
    arguments:
      # Other pipeline arguments here (see defaults in primitive.ARGUMENTS)
      # run: True
      # save_results: False
      # suffix: customsuffix

    primitives:
      # Specify any primitive args here
    """
    config_file = cache_dir + 'dark_config.yaml'
    with open(dark_config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))

    # Setup and call the pipeline with custom coeff model
    stage0_pipe = Stage0Pipeline(config_file=config_file)
    stage0_pipe.nonlin_corr.args.coeffs = nonlin_model
    result = stage0_pipe.apply(sci_path, output_dir=tmp_path)

