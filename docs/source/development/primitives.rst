Primitives
==========

Primitives are the algorithmic building blocks of the HISPEC_DRP which specify the processesing for a single processing step. The base class :py:class:`hispecdrp.primitives.BasePrimitive` specifies the interface for all primitives.

Implementing a Primitive
------------------------

A primitive is designed to take as input either a datamodel or ``DataModel`` instance. A primitive is also specified by additional keyword-arguments which may be passed to the primitive upon construction or processing. All primitive-specific configuration is stored in ``primitive.args``. This is a dictionary-like object with attribute-style access. The primary filename or datamodel to process can be accessed with ``primitive.args.name``.

Primitives should invoke the ``apply`` method when used after construction. This method is fully implemented by ``BasePrimitive`` and does the following:

#. Any additional ``kwargs`` are passed to ``primitive.args`.
#. Calls ``primitive._pre_condition(self) -> bool``
    * If ``_precondition`` returns ``False``, the primitive will be skipped. Default is ``True``.
#. Calls ``output = primitive._perform(self, name : DataModel | str) -> DataModel`
    * The primary logic of the primitive should be implemented in `_perform`.
    * Subtypes of `BasePrimitive` must implement this method.
#. Calls ``primitive._post_condition(self, output) -> bool`
    * If ``_post_condition` returns `False`, the output is stashed and nothing is returned. Default is `True`.
#. Optionally calls ``primitive.save_model(self, model : DataModel | str)``
    * The model is saved if ``primitive.save_results`` is ``True``. See additional configuration options in ``BasePrimitive`` for output_path construction.

``apply`` will store the output in ``primitive.output`` and return the output.


For example, the dark subtraction module might be implemented like so:


.. code-block:: python

    from ..base_primitive import BasePrimitive
    from ... import datamodels

    __all__ = ['DarkSubtractionPrimitive']

    # Name of class should end with 'Primitive'
    # and inherit from BasePrimitive (or a subclass of BasePrimitive)
    class DarkSubtractionPrimitive(BasePrimitive):
        """
        # Log any new arguments that the primitive takes.
        ARGUMENTS:
            dark (datamodels.DataModel | str | None): Dark model to apply to the input data.
                If None, the dark model will be queried.
        """

        class_alias = 'dark_sub' # Class alias for pipeline context

        # Specify and new arguments that the primitive takes or override any values from the parent class.
        ARGUMENTS = {
            'suffix' : 'darksub',
            'dark' : None
        }

        def _perform(
                self,
                name : datamodels.HISPECL0 | str,
            ):
            model_to_process = self.open_model(name, _copy=True) # Open the input model or copy if already open
            if self.args.dark is None:
                # get_calibration_file not implemented yet
                dark = self.get_calibration_file(model_to_process, 'dark')
            else:
                dark = self.open_model(dark)
            # Perform the dark subtraction, error propagation, and data quality flagging here
            ...

            # Return a model or list of models
            return model_to_process