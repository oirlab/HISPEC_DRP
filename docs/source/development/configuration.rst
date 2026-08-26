=============
Configuration
=============

Every primitive and pipeline in the HISPEC DRP is configured through a **config file**: a YAML document that is loaded into a :py:class:`~liger_iris_pipeline.driver.config.ConfigFile` object.

Each step or pipeline instance always has exactly one ``ConfigFile`` object associated with it at ``self.config``.

Pre-made config files for all steps and pipelines are included with ``hispecdrp`` in ``hispecdrp/configs/``.

If no config file is specified, a primitive or pipeline will automatically choose a config file based on the input dataset and processing context.

Creating Config Files
---------------------

Primitive and pipeline config files have the following five top-level fields:

.. code-block:: yaml

    class: FitRampPrimitive
    name: FitRampPrimitive-Custom
    description: Custom ramp fitting configuration.
    parameters:
      output_dir: /path/to/output_dir/
      method: mcds
      mcds_num_coadd: 5


- **class** (required) -- The primitive or pipeline class to configure.
  
  -If the class is defined with the HISPEC DRP, either the class name (``FitRampStep``) or full module path + class name (``hispecdrp.ramp_fitting.fit_ramp_step.RampFitStep``) can be used.
  - If the class is defined outside of the HISPEC DRP, the full module path + class name must be used.

- **name** (optional) -- The name of the config file. By convention, it should match the filename (withouth ``.yaml`` extension) but this is not enforced. This is also the default behavior if not provided.

- **description** (optional) -- A description of the config file.

- **parameters** (optional) -- Parameters and their values for a primitive or pipeline. Many parameters have defaults specified within the primitive or pipeline, but can be overriden here.


**Pipeline** config files can additionally specify:

- **primitives** (optional, pipelines only) -- Per-primitive overrides for ``context`` or ``parameters``.

.. code-block:: yaml
    :caption: stage0_pipeline_config.yaml

    class: Stage0Pipeline
    name: Stage0Pipeline-Custom
    description: Custom stage 0 pipeline configuration.
    primitives:
      ramp_fit:
        output_dir: /path/to/output_dir/ # Save file immediately after ramp fitting.
        method: mcds
        mcds_num_coadd: 5

Using Config Files
------------------

A user-created config file can be passed to a primitive or pipeline constructor via the ``config`` kwarg.

Additional parameters can be passed to the constructor or to ``apply`` / ``call`` to further override parameter values.

.. code-block:: python

    result, prim = Stage0Pipeline.call(
        dataset=input_model,
        config='stage0_pipeline_config.yaml',
        method='ols',
        mcds_num_coadd=5,
        output_dir='/path/to/output/dir/',
        return_instance=True
    )
    
    print(prim.config) # The auto-selected or user-provided ConfigFile object.


The ``ConfigFile`` filename used for a primitive or pipeline is stored in the datamodel RECEIPT extension under the column ``CONFIG_FILE``.

See the :doc:`Run Pipelines and Primitives <../quickstart/pipelines_and_primitives>` for more examples on creating config files.

Auto-Configuration
------------------

The DRP defines at least one config file for every primitive and pipeline. These config files define an additional section called **context** to facilitate auto-selecting the correct config file based on the dataset's properties and the input processing context. This section is **not needed for user-created config files** and can be ommitted.

Config files are automatically selected based on the following:

- **class** -- Only config files whose ``class`` matches the class being constructed are considered.

A config file's ``context`` section can contain any of the following keys:

- **quicklook** (bool) -- Matched against ``context['quickoook']`` passed to the primitive (default ``False``).

- **processing_mode** (str) -- Matched against ``context['processing_mode']`` passed to the class. Default is ``null`` (no restriction).

- **instrument_era** (``[min, max]`` version strings) -- Matched against ``meta.instrument_era`` on the dataset. Use ``null`` to leave a bound open-ended. For example, ``[null, 1.0.0]`` matches any dataset with ``meta.instrument_era <= 1.0.0``. Default is ``null`` (no restriction).

- **instrument_mode** (str or list of str) -- Matched against ``meta.instrument_mode`` on the dataset. Default is ``null`` (no restriction).

- **spectrograph** (str) -- Matched against ``meta.spectrograph`` on the dataset. Default is ``null`` (no restriction).

Example of passing context to a pipeline:

.. code-block:: python

    result, prim = Stage0Pipeline.call(
        dataset=input_model,
        context=dict(quicklook=True),
        output_dir='/path/to/output/dir/',
        return_instance=True
    )

An individual primitive's context is determined by merging the following, from lowest to highest precedence:

1. The ``context`` passed in to the pipeline constructor.
2. The primitive's ``context`` block under ``primitives`` in the pipeline config file.
3. The primitive's ``context`` in the ``primitives`` dict passed in at construction time (see below).


Effective Configuration and Reproducing Processing
--------------------------------------------------

**!!!! Under development !!!!**

Once a step or pipeline has resolved its configuration and any parameter overrides have been applied, ``get_effective_config()`` returns a single ``ConfigFile`` snapshot of everything that was actually used to run it -- including, for a pipeline, the effective config of every step it contains.

This is useful for reproducing a run later:

.. code-block:: python

    result1, pipe = Stage1Pipeline.call(
      dataset=sci_path,
      primitives={'ramp_fit': {'method': 'mcds'}},
      return_instance=True
    )

    effective_config = pipe.get_effective_config()
    effective_config.save('effective_config.yaml')

    # Later, reproduce the exact same run:
    result2 = Stage1Pipeline.call(sci_path, config='effective_config.yaml')


Logging
-------

**Under development**

Logging configuration can be specified in a config file or within the Python interface.

Pipelines always pass all logging-specific config to each primitive it uses.

.. code-block:: yaml

    class: Stage0Pipeline
    logging:
      level: info
      log_path: /path/to/log_dir/
      enable_stdout: true
      enable_file: true


.. code-block:: python

    log_config = {
        'level': 'info',
        'log_path': '/path/to/log_dir/',
        'enable_stdout': True,
        'enable_file': True
    }

    dark_prim = DarkSubtractionPrimitive(..., log_config=log_config)

**Logging Parameters**

- **level** -- Minimum emitted log level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL. Can be overridden by HISPECDRP_LOG_LEVEL. Default is INFO.
- **break_level** -- Raise an exception when a log record is emitted at or above this level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL. Can be overridden by HISPECDRP_BREAK_LEVEL. Default is ERROR.
- **enable_stdout** -- Write log records to stdout. Can be overridden by HISPECDRP_LOG_ENABLE_STDOUT (true/false). Default is true.
- **enable_file** -- Write log records to a file. Can be overridden by HISPECDRP_LOG_ENABLE_FILE (true/false). Default is false.
- **log_dir** -- Directory where log files are written (only used when enable_file: true). Defaults to ./logs/ relative to the current working directory when unset. Can be overridden by HISPECDRP_LOG_DIR.
- **filename_prefix** -- Prefix for log filenames: <prefix>_<run_id>_<pid>.txt. Default is hispecdrp_log.
- **rotate** -- Rotate log files when they reach max_bytes. Only used when enable_file: true and rotate: true. Default is false.
- **max_bytes** -- Maximum log file size in bytes before rotation. Only used when rotate: true. Default is 50000000 (50 MB).
- **backup_count** -- Number of rotated backup files to keep. Only used when rotate: true. Default is 5.