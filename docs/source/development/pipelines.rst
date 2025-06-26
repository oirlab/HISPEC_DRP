=========
Pipelines
=========

Overview
--------

Pipelines are a collection of primitives that are executed in a specified order. The base class :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline` specifies the interface for all pipelines. The pipeline level interface largely matches the primitive interface.

Pipeline Arguments
------------------

Like primitives, pipelines also use an ``ARGUMENTS`` variable to parametrize their behavior and is accessable via ``pipeline.args``.


Pipeline Configuration Files
----------------------------

Pipelines can also utilize configuration files. An example configuration for the stage 1 pipeline is below. Some values are redundant with the default ``ARGUMENTS``, but are included here to be explicit.

.. code-block:: yaml
    
    %YAML 1.2
    ---

    logging:
      level: info
      log_path: None

    arguments:
      # Can specify pipeline level arguments here.

    primitives:
      dark_sub:
        config_file: 'dark_config.yaml' # Optional
        arguments: # Additional arguments will override those in 'dark_primitive.yaml'
          suffix: 'exampledarksuffix'
      flat_field:
        arguments:
          run: True
          save_results: True
          suffix: 'exampleflatsuffix'


Logging
+++++++

By default, pipelines will pass any logging-specific config to each primitive it uses.


Saving Results
++++++++++++++

The argument ``save_result`` can take on different values for a pipeline:

- ``all``: Pipeline will save results of every primitive.
- ``final``: Pipeline will only save the final result of the pipeline.
- ``None``: Primitives will manage their own saving behavior.

Primitives will also use the pipeline's ``output_dir`` argument if none is specified in the primitive's arguments.

Pipeline Methods
----------------

Below are most methods one can implement or override when creating custom pipelines. Note some are required for certain behavior. See the full API for full documentation of available methods.

* _perform(self) -> Arguments
    - Required. This method takes in no arguments. It contains the primary logic of the pipeline and returns an instance of ``Arguments`` where the attribute ``name`` contains the primary output model. Extra outputs may be stored in the output ``Arguments`` object. If ``full_output`` is ``True``, the entire ``Arguments`` object is returned, otherwise only the ``name`` attribute is returned. _perform will call each primitive's ``apply`` method in any order specified, but should reflect the ordering in ``PRIMITIVES``.


Implementing a Pipeline
-----------------------

Pipelines must extend either:

* :py:class:`hispecdrp.pipelines.base_pipeline.BasePipeline` for general science pipelines.
* :py:class:`hispecdrp.pipelines.calibration.base_calibration_pipeline.BaseCalibrationPipeline`. for pipelines which create calibration products.

Below is the implementation of the LFC processing pipeline.

.. literalinclude:: ../../../hispecdrp/pipelines/calibration/process_lfc_pipeline.py
   :language: python