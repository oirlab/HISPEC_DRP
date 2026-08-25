==========================
Run Pipelines & Primitives
==========================

**Examples require access to the calibration database hosted at Keck Observatory.** See :doc:`installation <../installation>` to configure access.

Run a Pipeline
--------------

**Pipelines** represent a sequence of primitives (individual processing steps) and are the typical objects users will interact with to process their data.

It is recommended to use the ``call`` method which configures and runs a pipeline in one function call.

.. code-block:: python

    from hispecdrp.pipelines import Stage0Pipeline
    from hispecdrp.utils import download_gdrive_file

    # Load credentials for calibration DB at Keck
    from dotenv import load_dotenv
    load_dotenv()

    # Download simulated ramp from Google drive
    sci_path_raw = download_gdrive_file('HB.20240924.00000.00_raw.fits')

    # Setup and call the pipeline in one go
    # Default output_dir is current working directory
    model_L0 = Stage0Pipeline.call(
        dataset=sci_path_raw,
        output_dir='/path/to/output/dir/'
    )

The pipeline **input dataset** can be a single file path or datamodel, or a list of multiple files/models processed in sequence.

The pipeline **output** depends on the input type and pipeline class. Science pipelines (Stages 0-2) return a data model or list of models corresponding to their stage name For example, the ``Stage0Pipeline`` returns an ``L0Model``.

Configuration
-------------

Parameters for pipelines and primitives are specified in each the class's ``PARAMETERS`` dictionary (class attribute).

**Common Primitive and Pipeline Parameters:**

- **dataset**: The primary input, which can be anything a primitive/pipeline expects. This is typically a filepath, a datamodel, or a list of either. Default is ``None``.

- **save_result**: Whether or not to save the results. Default is ``False`` for primitives, and ``True`` for pipelines.

- **output_dir**: The directory where output files will be saved. If ``None``, the output_dir is set to the input directory. If the input directory is also ``None``, the output_dir is set to the current working directory. Default is ``None``. If a primitive does not specify output_dir, it will adopt the value from the pipeline.

- **output_file**: The base filename where output will be saved. Default is ``None``.

- **max_cores**: The maximum number of CPU cores to use for parallel processing. The default value is 1. Primitives will inherit this value from the pipeline if not specified.

**Primitives only**

- **run**: Whether or not to run the primitive. Default is ``True``.

- **suffix**: A suffix to append to output filenames. Each primitive specifies a unique default suffix.

- **output_model**: The output data model to use. See :py:meth:`~hispecdrp.datamodels.base_model.DataModel._get_output_model` for more information. The default for many primitives is ``'input'``, which means the output will be the same as the input.

- **save_preview**: Whether or not to save the preview of the results. Default is ``False``. If either **save_preview** or **show_preview** is ``True``, the primitive will generate a preview summary of the results, (typically a figure). The exact behavior depends on the primitive.

- **show_preview**: Whether or not to display the preview of the results. Default is ``False``.

The appropriate configuration files for a pipeline and all of its primitives are automatically selected from ``hispecdrp/configs/``. See :doc:`DRP Pipeline and Primitive Configuration <../development/configuration>` for details on how configuration files are selected.

Users can override the defaults with their own config file. In this example, primitive config files are still automatically selected, but any parameters specified in the user config file will override those defaults.


User Config File
++++++++++++++++


.. code-block:: yaml
    :caption: stage0_pipeline_config.yaml

    class: Stage0Pipeline
    parameters:
      output_dir: /path/to/output_dir/
      max_cores: 4
    primitives:
      dq_init:
        suffix: proc-dq_init
      ramp_fit:
        method: ols
        jwst_jump_detection: False

**Using the config file:**

.. code-block:: python

    model_L0 = Stage0Pipeline.call(sci_path_raw, config='stage0_pipeline_config.yaml')



Python Interface
++++++++++++++++

Pipelines can also be configured within Python using the **primitives** argument:

.. code-block:: python

    primitives = {
        # Keys = primitive alias defined in the pipeline class.
        'nonlin_corr': {
            'run': False,  # Skip nonlinearity correction
        },
        'ramp_fit': {
            'method': 'ols',
        },
    }

    model_L0 = Stage0Pipeline.call(
        sci_path_raw,
        primitives=primitives,
        output_dir='/path/to/output/dir/'
    )

Users can also interact with the pipeline object directly:

.. code-block:: python

    pipe = Stage0Pipeline(dataset=sci_path_raw)

    # Change a parameter for a primitive
    pipe.ramp_fit.method = 'ols'

    model_L0 = pipe.apply(output_dir='/path/to/output/dir/')


Running Primitives Directly
---------------------------

Primitives can also be run on their own, outside of a pipeline, using the same ``call`` interface.

.. code-block:: python

    from hispecdrp.primitives import FitRampPrimitive

    model_L0 = FitRampPrimitive.call(dataset=ramp_model, method='ols')

The result ``model_L0`` returned from ``FitRampPrimitive`` is not-yet corrected for dark current, gain, or flat-fielding which are included in the Stage 0 pipeline after ramp fitting.
