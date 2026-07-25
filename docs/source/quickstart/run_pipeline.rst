=================
Running Pipelines
=================

The following examples demonstrate how to run a pipeline, which is a chain of primitives.

**The examples require access to the calibration database hosted at Keck Observatory. See :doc:`installation <../installation>` for instructions.**

Run a Pipeline
--------------

This example demonstrates how to configure and run the Stage 0 pipeline.

.. code-block:: python

    import os
    from hispecdrp.pipelines import Stage0Pipeline
    from hispecdrp.utils import download_gdrive_file
    from hispecdrp import datamodels

    from dotenv import load_dotenv
    load_dotenv()  # Load credentials for calibration DB at Keck
    
    # Configure calibration cache directory if not already set in shell environment
    os.environ['KOA_CALIBRATION_CACHE'] = '/path/to/cache/'

    # Download simulated ramp from Google drive
    sci_path = download_gdrive_file('HB.20240924.00000.00_raw.fits')

    # Setup and call the pipeline
    pipe = Stage0Pipeline(output_dir='/path/to/output/dir/')
    result = pipe.apply(sci_path)


Pipeline Configuration
----------------------

Like primitives, pipelines are configurable with different parameters, however pipeline configuration is typically limited to output directories.

See :doc:`Running a Primitive <run_primitive>` for a full listing of all default pipeline parameters, or the docstring of the pipeline class for parameters specific to that pipeline.

Most of the remaining pipeline configuration is handled through the primitives that the pipeline runs. Individual primitive configuration can be set in a few ways.

Python Interface
++++++++++++++++

A special argument called **prim_args** can be used as follows:

.. code-block:: python

    # Configure each primitive using prim_args
    prim_args = {
        'dq_init': {
            'max_cores': max_cores,
        },
        'saturation_check': {
            'max_cores': max_cores,
        },
        'bias_sub': {
            'run': False,  # Skip bias subtraction
            'max_cores': max_cores,
        },
        'nonlin_corr': {
            'max_cores': max_cores,
        },
        'jump_detec': {
            'run': False,  # Skip standalone jump detection
            'output_model': 'input',
            'method': 'tpd',
            'max_cores': max_cores,
            'rejection_threshold': 4.0,
        },
        'ramp_fit': {
            'method': 'jwst_likely',  # Options: 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'
            'max_cores': max_cores,
            'jwst_jump_detection': True,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark_sub': {
            'max_cores': max_cores,
        },
        'gain_corr': {
            'max_cores': max_cores,
        },
        'detflat_corr': {
            'max_cores': max_cores,
        },
    }
    
    # Initialize the pipeline
    stage0_pipe = Stage0Pipeline(prim_args=prim_args)
    

Directly setting attributes is also fine:

.. code-block:: python

    # Set calibration files for each primitive
    stage0_pipe.dq_init.dq = '/path/to/dq_calibration.fits'


Configuration Files
+++++++++++++++++++

Pipeline configuration files are YAML files that specify pipeline and primitive behavior.

.. code-block:: yaml

    %YAML 1.2
    ---

    class: Stage0Pipeline

    logging:
      level: info
      log_path: /path/to/log_dir/
      enable_stdout: true
      enable_file: true

    arguments:
      save_result: True
      output_dir: /path/to/output/dir/

    primitives:
      dark_sub:
        arguments:
          max_cores: 4
          save_result: True
          output_dir: /path/to/output/dir/
        calibrations:
          dark: /path/to/dark_calibration.fits
      bias_sub:
        arguments:
          run: False

Primitive configuration files follow the same format as primitive config files (see :doc:`Running a Primitive <run_primitive>`), but with an additional top-level key called **primitives** that specifies the configuration for each primitive in the pipeline. The keys under **primitives** should match the primitive names used in the pipeline.

Pipeline config files can also specify primitive-specific config files as so. Anything additional specified in the pipeline config file will override the primitive config file.

.. code-block:: yaml

    %YAML 1.2
    ---

    class: Stage0Pipeline

    primitives:
      dark_sub:
        config_file: /path/to/dark_sub_config.yaml
        # Anything below will override dark_sub_config.yaml
        arguments:
          max_cores: 4
          save_result: True
          output_dir: /path/to/output/dir/
        calibrations:
          dark: /path/to/dark_calibration.fits
      bias_sub:
        arguments:
          run: False

The pipeline can be initialized with a path to the configuration file, or with a dictionary containing the configuration, both using the ``config`` kwarg.

.. code-block:: python

    pipe =Stage0Pipeline(config='/path/to/config.yaml')
    model_result = pipe.apply('/path/to/ramp_input_file.fits')


Optionally, the class name can be added to the top of the config file to specify which primitive the configuration is for. This is useful for invoking primitives from configuration files alone and will be used for automation (*under development*).

Logging
-------

Pipelines always pass all logging-specific config to each primitive it uses.

.. code-block:: python

    dark_prim = DarkSubtractionPrimitive(...)
    dark_prim.config['logging']['enable_stdout'] = False
    dark_prim.config['logging']['enable_file'] = True


See the logging section in the :doc:`Running a Primitive <run_primitive>` quickstart documentation for all logging options.