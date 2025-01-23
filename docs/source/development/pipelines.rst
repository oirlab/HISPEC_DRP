Pipelines
=========

Pipelines are a collection of primitives that are executed in a specific order to process the data. The base class :py:class:`hispecdrp.pipelines.BasePipeline` specifies the interface for all pipelines.

Implementing a Pipeline
-----------------------

The Pipeline interface is built on top of the primitive interface and is used in nearly an identicaly manner. Pipeline-specific configuration is stored in ``pipeline.args``, and the primary filename or datamodel to process can be accessed with ``pipeline.args.name``. A class attribute called ``PRIMITIVES`` specifies which primitives are instantiated and available for the pipeline to execute.

For example, the stage 1 pipeline might be implemented as follows and stored in hispecdrp/pipelines/stage1/stage1_pipeline.py

.. code-block:: python
    
    from ..base_pipeline import BasePipeline
    from ... import datamodels
    from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive

    __all__ = ['Stage1Pipeline']

    class Stage1Pipeline(BasePipeline):

        class_alias = 'stage1'

        PRIMITIVES = {
            'dark_sub': DarkSubtractionPrimitive,
            'flat_field': FlatFieldPrimitive,
        }

        def _perform(self, name : datamodels.HISPECL0 | str):
            result = self.dark_sub.apply(name)
            result = self.flat_field.apply(result)
            return result



Pipeline Configuraiton Files
----------------------------

Like primitives, pipelines can also utilize config files. An example configuration for the stage 1 pipeline is below. Some values are redundant with the default ARGUMENTS, but are included here to be explicit.

.. code-block:: yaml
    
    %YAML 1.2
    ---

    class: 'Stage1Pipeline'

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None

    arguments:
      skip: False
      save_results: True
      output_dir: None
      suffix: 'examplesuffix'

    primitives:
      dark_sub:
        config_file: 'dark_config.yaml' # Optional
        arguments: # Additional arguments will override those in 'dark_primitive.yaml'
          skip: False
      flat_field:
        arguments:
          skip: False
          save_results: False
          suffix: 'examplesuffix'
