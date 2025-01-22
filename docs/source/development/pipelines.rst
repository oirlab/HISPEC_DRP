Pipelines
=========

Perform is primary method to implement in a primitive.

.. code-block:: python

    _perform(self, name : datamodels.DataModel | str)


For example, the dark subtraction module might be implemented like so:

Implementing a Primitive 
------------------------


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
                name : datamodels.DataModel | str,
            ):
            model_to_process = self.open_model(name)
            if self.args.dark is None:
                # get_calibration_file not implemented yet
                dark = self.get_calibration_file(model_to_process, 'dark')
            else:
                dark = self.open_model(dark)
            # Perform the dark subtraction, error propagation, and data quality flagging here
            ...
            return model_to_process