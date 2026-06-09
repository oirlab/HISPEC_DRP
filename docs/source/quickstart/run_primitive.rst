==================
Running Primitives
==================

The following examples demonstrate how to run a primitive, which is a single processing `step` in the data reduction pipeline. Files are retrieved from the shared HISPEC Google Drive folder.

Run a Primitive
---------------

This example subtracts a dark frame from an input frame at level L0 (rate map) using the :py:class:`~hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive`. For this primitive, a dark calibration is needed. Calibrations are typically associated and retrieved automatically by the DRP (see :doc:`calibrations <../calibrations/calibrations>` and :doc:`Primitive Design <../development/primitives>`). Below we show the different methods for associated calibrations through a primitive's configuraiton.

.. code-block:: python

    # Imports
    from hispecdrp import DarkSubtractionPrimitive
    from hispecdrp.utils import download_gdrive_file
    from hispecdrp import datamodels

    # Input files
    sci_path = download_gdrive_file('HB.20231006.00000.00_rate-TELLURICS-FIB1.fits')
    dark_path = download_gdrive_file('HB.20260212.00000.00.mastercal-dark.001.fits')
    dark_model = datamodels.open(dark_path)

    # Use default behavior
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(sci_path, dark=dark_model) # or dark=dark_path

The associated dark calibration can be found with:

.. code-block:: python

    # Returns dictionary with calibration information
    dark_info = model_result.get_calibration_from_type('dark')

Primitive Configuration
-----------------------

Primitives are configurable with different parameters for data processing to fully specify the data processing behavior, including primitive parameters, calibrations, and logging behavior. Primitive arguments and calibrations can be set in several ways.


Python Interface
++++++++++++++++

Passed in as keyword arguments to the class constructor:

.. code-block:: python

    dark_prim = DarkSubtractionPrimitive(
        dark=dark_path,
        output_dir='/path/to/output/dir/',
    )
    model_result = dark_prim.apply(sci_path)

Passed in as keyword arguments to the class constructor or ``apply`` method, set directly as class attributes:

.. code-block:: python

    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(
        sci_path,
        dark=dark_path, save_result=True, output_dir='/path/to/output/dir/'
    )

Configuration Files
+++++++++++++++++++

Primitive configuration files are YAML files that specify primitive behavior. Anything not set within the configuration file will be set to the default value specified in the primitive's ``ARGUMENTS`` dictionary.

.. code-block:: yaml

    %YAML 1.2
    ---

    class: DarkSubtractionPrimitive

    logging:
        level: info
        log_path: /path/to/log_dir/
        enable_stdout: true
        enable_file: true

    arguments:
        save_result: True
        output_dir: /path/to/output/dir/
        dark: /path/to/dark_calibration.fits

The primitive can be initialized with a path to the configuration file, or with a dictionary containing the configuration, both using the ``config`` kwarg.

.. code-block:: python

    dark_prim = DarkSubtractionPrimitive(config='/path/to/config.yaml')
    model_result = dark_prim.apply(sci_path)


Optionally, the class name can be added to the top of the config file to specify which primitive the configuration is for. This is useful for invoking primitives from configuration files alone and will be used for automation (*under development*).

ARGUMENTS
~~~~~~~~~

All primitive parameters are specified through a primitive's ``ARGUMENTS`` class attribute (dict). The keys are the argument names, and the values are the default value. All primitives use a set of common parameters defined within :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive`:

- **input**: The primary input. Can be anything a primitive expects, but should be a container of datamodels or filepaths to data products. Default is None.
- **run**: Whether or not to run the primitive. Default is True.
- **save_result**: Whether or not to save the results. Default is False.
- **output_model**: The output data model to use. See :py:meth:`~hispecdrp.datamodels.base_model.DataModel._get_output_model` for more information. Default is None.
- **output_dir**: The directory where output files will be saved. Default is None.
- **output_file**: The base filename where output will be saved. Default is None.
- **suffix**: A suffix to append to output filenames. Default is None.
- **save_preview**: Whether or not to save the preview of the results. Default is False.
- **show_preview**: Whether or not to display the preview of the results. Default is False.

If either of **save_preview** or **show_preview** is True, then the primitive will generate a preview of the results.

Primitives can also define additional arguments to fully specify behavior.

Logging
~~~~~~~

The HISPEC DRP uses the standard Python logging module to report messages to the user. Logging supports the following configuration options and can be set within a config's logging section.

- **level**: The logging level. Can be set to a string (e.g. `'INFO'`) or an integer (e.g. `20`). See Python's `logging documentation <https://docs.python.org/3/library/logging.html#logging-levels>`_ for more information.
- **enable_stdout**: Enable logging to standard output. Default is True.
- **enable_file**: Enable logging to a file. Default is True.
- **log_dir**: Path to the log directory. If `None`, a logs are stored at `~/hispecdrp_logs/`.
- **filename_prefix**: Prefix for log files (e.g., `'hispecdrp_log'`).
- **rotate**: Enable rotating log files. Default is False. (NOTE: TBD)
- **max_bytes**: Maximum file size in bytes before rotation occurs (e.g., `50000000`). (NOTE: TBD)
- **backup_count**: Number of rotated log files to keep. (NOTE: TBD)

To modify the logging behavavior dynamically within Python, one can modify logging configuration directly:

.. code-block:: python

    dark_prim = DarkSubtractionPrimitive(...)
    dark_prim.config['logging']['enable_stdout'] = False
    dark_prim.config['logging']['enable_file'] = True
    ...
