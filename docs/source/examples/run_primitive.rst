===================
Running a Primitive
===================


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
    dark_path = download_github_lfs_file(git_file_path_dark, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config file
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
    """
    config_file = cache_dir + 'dark_config.yaml'
    with open(config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))

    # Setup and call the dark primitive
    # Arguments can also be passed to the primitive's constructor, or to apply().
    # These additional args will override anything in config.
    # See the objects ARGUMENTS attribute for a list of available arguments.
    dark_prim = DarkSubtractionPrimitive(config_file=dark_config_file, dark=dark_path)
    model_result = dark_prim.apply(sci_path, output_dir=tmp_path)

