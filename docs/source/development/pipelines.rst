=========
Pipelines
=========

Overview
--------

Pipelines are a collection of primitives that are executed in a specified order. The base class :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline` specifies the interface for all pipelines. The pipeline level interface largely matches the primitive interface.

Pipeline Arguments
------------------

Like primitives, pipelines also use an ``ARGUMENTS`` variable to parametrize their behavior and are available as attributes.


Pipeline Configuration Files
----------------------------

Pipelines may also utilize configuration files. An example configuration for the stage 1 pipeline is below. Some values are redundant with the default ``ARGUMENTS``, but are included here to be explicit.

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
        arguments: # Additional arguments will override those in 'dark_config.yaml'
          suffix: 'exampledarksuffix'
      flat_field:
        arguments:
          run: True
          save_results: True
          suffix: 'exampleflatsuffix'


Logging
+++++++

Pipelines will pass any logging-specific config to each primitive it uses.


Saving Results
++++++++++++++

The argument ``save_result`` behaves the same as in primitives and will save whatever output is returned from a pipeline's ``_perform`` method. This will typically be the output of the last primitive called (including a skipped primitive), but this is not enforced and a Pipeline can choose to return what is needs to just like primitives.

A primitive will also use the pipeline's ``output_dir`` argument if none is specified in the primitive's arguments and that primitive's ``save_result`` argument is ``True``.

Pipeline Methods
----------------

Below are most methods one can implement or override when creating custom pipelines. Note some are required for certain behavior. See the full API for full documentation of available methods.

* _perform(self) -> dict[str, Any]
    - Required. This method takes in no arguments. _perform should call each primitive's ``apply`` method in the necessary manner to needed, but should reflect the ordering in ``PRIMITIVES`` for consistency (not enforced). It contains the primary logic of the pipeline and returns a ``dict`` where the key ``output`` contains the primary output model or models. Extra outputs may be stored in the output ``dict``.


Implementing a Pipeline
-----------------------

Pipelines must extend either:

* :py:class:`hispecdrp.pipelines.base_pipeline.BasePipeline` for general science pipelines.
* :py:class:`hispecdrp.pipelines.calibration.base_calibration_pipeline.BaseCalibrationPipeline`. for pipelines which create calibration products.

Below is the implementation of the LFC processing pipeline.

.. literalinclude:: ../../../hispecdrp/pipelines/calibration/process_lfc_pipeline.py
   :language: python