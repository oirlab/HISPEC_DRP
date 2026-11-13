=========
Pipelines
=========

Pipelines are a collection of :doc:`Primitives <primitives>`. that are executed in a specified order. The base class :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline` specifies the interface for all pipelines which inherits from :py:class:`~hispecdrp.processing_component.ProcessingComponent`. The pipeline interface is similar to primitives.

Implement New Pipeline
----------------------

1. Subclass BasePipeline
~~~~~~~~~~~~~~~~~~~~~~~~

To implement a new pipeline (``NewPipeline``), create a new Python class that inherits from :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline`. If formally part of the HISPEC DRP, this class should be added to the `pipelines` module in the code base, following the structure of existing pipelines.

2. Primitives Table
~~~~~~~~~~~~~~~~~~~

Every pipeline must define the class attribute ``PRIMITIVES`` which is a dictionary of primitives used by the pipeline. Keys are an alias for the primitive (strings) and are chosen by the user but should reflect the primitive's role. Values are the associated primitive classes (types derived from ``BasePrimitive``). The order of the primitives in this dictionary should reflect the order in which they are executed, although this is not enforced.

.. code-block:: python

    class NewPipeline(BasePipeline):

        PRIMITIVES = {
            'dark_sub': DarkSubtractionPrimitive,
            'detflat': DetectorFlatPrimitive,
            ... # Additional primitives as needed.
        }

3. Parameters
~~~~~~~~~~~~~

By default, ``NewPipeline`` will inherit the same parameters as :py:class:`~hispecdrp.pipelines.base_pipeline.BasePipeline`, which shares the same parameters as :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive`. It can specify any additional parameters in the class attribute ``PARAMETERS``. These parameters are only relevant for the pipeline itself and are not passed to the primitives it uses.

.. code-block:: python

    class NewPipeline(BasePipeline):

        PARAMETERS = {
            'arg_name': ... # <default_value>
        }


4. Perform Method
~~~~~~~~~~~~~~~~~

The pipeline must implement the required method :py:meth:`~hispecdrp.pipelines.base_pipeline.BasePipeline._perform`. This method contains the main processing code for the pipeline. It takes in no parameters and returns a dictionary, which must contain at least one key - ``output``. If additional / ancillary outputs are needed, they can be found with either ``pipeline._output[<key>]`` or simply ``pipeline.<key>`` if there is no collision with the explicit attributes.

.. code-block:: python

    class NewPipeline(BasePipeline):

        def _perform(self):
            # Main processing code here
            # Primary input is accessed with ``self.dataset``
            # Access parameters with ``self.<param_name>``
            # Call primitives with ``result = self.prim_alias.apply(<input>, **kwargs)``
            ...
            return dict(output=result, extra1=<extra_output1>, extra2=<extra_output2>)

5. Previews
~~~~~~~~~~~

(Optional) The pipeline can optionally implement the method :py:meth:`~hispecdrp.pipelines.base_pipeline.BasePipeline._create_preview` to generate a preview of the results if either of the arguments ``save_preview`` or ``show_preview`` is True.

6. Unit tests
~~~~~~~~~~~~~

After implementing a pipeline, please also create corresponding tests in the testing suite `tests <https://github.com/oirlab/HISPEC_DRP/blob/main/tests>`_ to at minimum validate the basic processing steps within the pipeline.

7. Documentation
~~~~~~~~~~~~~~~~

Implement appropriate docstrings following the structure for other primitives.

For pipelines, also create a new page in the documentation under `docs/source/pipelines/`. Populate this page according to the structure of existing pipeline documentation, and include a link to this page in the `pipelines/index.rst` file.