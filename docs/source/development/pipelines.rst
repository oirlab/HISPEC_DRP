=========
Pipelines
=========

Overview
--------

Pipelines are a collection of :doc:`Primitives <primitives>`. that are executed in a specified order. The base class :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline` specifies the interface for all pipelines which inherits from :py:class:`~hispecdrp.processing_component.ProcessingComponent`. The pipeline level interface largely matches the primitive interface.


Pipeline Primitives
-------------------

Every pipeline must define a PRIMITIVE table as a class attribute ``PRIMITIVES`` which is a dictionary of primitive alias (key; ``str``) and their associated classes (value; ``type[BasePrimitive]``). The order of the primitives in this dictionary defines the order in which they are executed. Each primitive can also have its own configuration file and/or additional arguments that override the default arguments or those set in the configuration file.



Pipeline Arguments
------------------

Pipelines arguments behave identical to :doc:`Primitives <primitives>`. The base pipeline defines no new arguments.


Pipeline Configuration Files
----------------------------

Pipelines can also use configuration files. An example configuration for the stage 1 pipeline is below. Some values are redundant with the default ``ARGUMENTS``, but are included here to be explicit.

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
          suffix: 'darksub'
      flat_field:
        arguments:
          run: True
          save_result: True
          suffix: 'flat_field'


Logging
-------

Pipelines will pass any logging-specific config to each primitive it uses.


Saving Results
--------------

The argument ``save_result`` behaves the same as in primitives and will save whatever output is returned from a pipeline's ``_perform`` method. This will typically be the output of the last primitive called (including a skipped primitive), but this is not enforced and a Pipeline can choose to return what is needs to just like primitives.

A primitive will also use the pipeline's ``output_dir`` argument if none is specified in the primitive's arguments and that primitive's ``save_result`` argument is ``True``.


Pipeline Methods
----------------

Every pipeline must implement the following method:

* .. automethod:: hispecdrp.pipelines.base_pipeline.BasePipeline._perform
    :noindex:


Implementing a Pipeline
-----------------------

Pipelines must extend from :py:class:`hispecdrp.pipelines.base_pipeline.BasePipeline`.

Below is the implementation of the LFC processing pipeline.

.. literalinclude:: ../../../hispecdrp/pipelines/calibration/process_lfc_pipeline.py
   :language: python