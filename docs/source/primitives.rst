Primitives
==========

Primitives are the building blocks of the HISPEC_DRP.

Implementing a Primitive 
------------------------

`apply` is the first method called when a primitive is run. It sets any additional configuration options and then calls `_pre_condition`, `_perform`, and `_post_condition`.

.. code-block:: python

Perform is the primary method to implement in a new primitive:

.. code-block:: python

    _perform(self, name : datamodels.DataModel | str)

Any other arguments including dark models will be available through `self.args`.


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