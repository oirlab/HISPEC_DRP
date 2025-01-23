Primitives
==========

Primitives are the algorithmic building blocks of the HISPEC_DRP which specify the processesing for a single processing step. The base class :py:class:`hispecdrp.primitives.BasePrimitive` specifies the interface for all primitives.

Implementing a Primitive
------------------------

A primitive is designed to take as input either a datamodel or ``DataModel`` instance. A primitive is also specified by additional keyword-arguments which may be passed to the primitive upon construction or processing. All primitive-specific configuration is stored in ``primitive.args``. This is a dictionary-like object with attribute-style access. The primary filename or datamodel to process can be accessed with ``primitive.args.name``.

Primitives should invoke the ``apply`` method when used after construction. This method is fully implemented by ``BasePrimitive`` and does the following:

#. Any additional ``kwargs`` are passed to ``primitive.args``.
#. Calls ``primitive._pre_condition(self) -> bool``
    * If ``_precondition`` returns ``False``, the primitive will be skipped. Default is ``True``.
#. Calls ``output = primitive._perform(self, name : DataModel | str) -> DataModel``
    * The primary logic of the primitive should be implemented in ``_perform``.
    * Subtypes of ``BasePrimitive``` must implement this method.
#. Calls ``primitive._post_condition(self, output) -> bool``
    * If ``_post_condition`` returns ``False``, the output is stashed and nothing is returned. Default is ``True``.
#. Optionally calls ``primitive.save_model(self, model : DataModel | str)``
    * The model is saved if ``primitive.save_results`` is ``True``. See additional configuration options in ``BasePrimitive`` for output_path construction.

``apply`` will store the output in ``primitive.output`` and return the output.


For example, the dark subtraction module might be implemented as follows and stored in hispecdrp/primitives/dark_subtraction/dark_primitive.py


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
            ) -> datamodels.HISPECL0:
            model_to_process = self.open_model(name, _copy=True) # Open the input model or copy if already open
            if self.args.dark is None:
                # Get cal from the calibration database
                dark = self.get_calibration_file(model_to_process, 'dark')
            else:
                dark = self.open_model(self.args.dark)
            self.log.info(f"Using {dark} to process {model_to_process}") # Always use self.log to log messages.
            # Perform the dark subtraction, error propagation, and data quality flagging here
            ...

            # Return a model
            return model_to_process


Primitive Arguments
-------------------

**Priority Hierarchy**

1. **Keyword Arguments** ``**kwargs``
   - Keyword arguments passed directly to the method or function when it is called have the highest priority.

2. **``primitive.config['arguments']`` (Configuration Arguments)**
   - Arguments loaded from a configuration file or a pre-defined configuration object with key ``arguments``.

3. **``primitive.ARGUMENTS`` (Class-Level Default Arguments)**
   - Arguments defined in the class attribute ``ARGUMENTS`` are the starting values.


**Note** the final behavior of a primitive, including setting any additional arguments can be done within ``primitive._perform``.


Primitive Configuration Files
-----------------------------

Primitives can also be configured with config files. An example configuration for the dark primitive is below. Some values are redundant with the default ARGUMENTS, but are included here to be explicit.


.. code-block:: yaml
    
    %YAML 1.2
    ---
    class: 'DarkSubtraction'

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

