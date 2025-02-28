Primitives
==========

Overview
--------

Primitives are the algorithmic building blocks of the HISPEC_DRP which specify the processesing for a single processing step. The base class :py:class:`hispecdrp.primitives.base_primitive.BasePrimitive` specifies the interface for all primitives.

A primitive takes as input either a filename or ``DataModel`` instance. A primitive is also specified by additional keyword-arguments which may be passed to the primitive upon construction or processing. Additional parameters can be numbers, strings, or any valid Python object. The final primitive-specific configuration is stored in ``primitive.args``, which is a dictionary-like object with attribute-style access. The primary filename or datamodel to process can be accessed with ``primitive.args.name``.

Primitives should invoke the ``apply`` method when used after construction. This method is fully implemented by ``BasePrimitive`` and does the following:

#. Any additional ``kwargs`` are passed to ``primitive.args``.
#. Calls ``primitive._pre_condition(self) -> bool``
    * If ``_precondition`` returns ``False``, the primitive will be skipped. Default is ``True``.
#. Calls ``output = primitive._perform(self) -> DataModel``
    * The primary logic of the primitive should be implemented in ``_perform``.
    * Subtypes of ``BasePrimitive``` must implement this method.
#. Calls ``primitive._post_condition(self, output) -> bool``
    * If ``_post_condition`` returns ``False``, the output is stashed and nothing is returned. Default is ``True``.
#. Optionally calls ``primitive.save_model(self, model : DataModel | str)``
    * The model is saved if ``primitive.save_results`` is ``True``. See additional configuration options in ``BasePrimitive`` for output_path construction.

``apply`` will store the output in ``primitive.output`` and return the output.


Primitive Arguments
-------------------

**Priority Hierarchy**

1. **Keyword Arguments** ``kwargs``
   - Keyword arguments passed directly to the method or function when it is called have the highest priority.

2. **Configuration Arguments** ``primitive.config['arguments']``
   - Arguments loaded from a configuration file or a pre-defined configuration object with key ``arguments``.

3. **Class-Level Default Arguments** ``primitive.ARGUMENTS``
   - Arguments defined in the class attribute ``ARGUMENTS`` are the starting values.


**Note** the final behavior of a primitive, including setting any additional arguments can be done within ``primitive._perform``.


Primitive Configuration Files
-----------------------------

Primitives can also be configured with config files, but only simple arguments can be specified here. An example configuration for the dark primitive is below. Some values are redundant with the default ARGUMENTS, but are included here to be explicit.


.. code-block:: yaml
    
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
  
    arguments:
      run: True
      save_results: True
      output_dir: None
      suffix: 'examplesuffix'



Implementing a Primitive
------------------------

The dark subtraction primitive is implemented as follows and stored in ``hispecdrp/primitives/dark_subtraction/dark_primitive.py``. See `dark_primitive.py <https://github.com/oirlab/HISPEC_DRP/blob/main/hispecdrp/primitives/dark_subtraction/dark_primitive.py>`_ for the rest of the implementation.


.. code-block:: python

    class DarkSubtractionPrimitive(BasePrimitive):
        """
        ARGUMENTS:
            dark (DataModel | str | None): Dark model to apply to the input data.
                If None, the dark model will be queried.
        """

        class_alias = 'dark_sub'

        ARGUMENTS = {
            'suffix' : 'darksub',
            'dark' : None
        }


        def _perform(self):
            input_model = self.open_model(self.args.name, copy=False)
            if self.args.dark is None:
                dark_path = self.get_calibration_file(input_model, cal_type='dark')
            else:
                dark_path = self.args.dark
            dark_model = self.open_model(dark_path, copy=False)
            result = input_model.copy()
            self.log.info(f"Applying dark correction with {dark_model} to {input_model}")
            if isinstance(result, datamodels.HISPECRampModel):
                result = dark_subtraction_ramp(result, dark_model)
            elif isinstance(result, datamodels.HISPECL0Model):
                result = dark_subtraction_rate(result, dark_model)
            return result
