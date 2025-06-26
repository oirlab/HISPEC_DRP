Primitives
==========

Overview
--------

Primitives are the algorithmic building blocks of the HISPEC_DRP which specify the processesing for a single processing step. The base class :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` specifies a common interface for all primitives.

The primary input for a primitive is either a single or filepath to a ``DataModel`` or an instance of one. A primitive is also fully defined by additional keyword-arguments which may be passed to the primitive upon construction or processing. These additional parameters can be numbers, strings, or any valid Python object. The final primitive-specific configuration is stored in ``primitive.args``, which is a dictionary-like object with attribute-style access. The primary input is always stored in ``primitive.args.name``.

Primitives should invoke the ``apply`` method when used after construction. This method is fully implemented by ``BasePrimitive`` and does the following:

#. Sets additional arguments from ``kwargs`` in ``primitive.args``.
#. Checks if ``primitive._pre_condition(self) -> bool``
    * If ``_precondition`` returns ``False``, the primitive will be skipped. Default is ``True``.
#. Calls ``primitive._perform(self) -> Arguments``
    * Subtypes of ``BasePrimitive``` must implement this method.
    * The primary logic of the primitive should be implemented in ``_perform`` which takes in no additional arguments.
    * All live arguments are available in ``self.args``.
    * The return value (``output``) of ``_perform`` is always an instance of ``Arguments`` where the primary return value is stored at ``output.name`` and is always a ``DataModel``.
#. Checks for failures:
    * The return value from ``_perform`` must not be ``None``
    * ``primitive.status`` must be ``None``
    * If at least one of these conditions is not met, the primitive will be marked as failed but the DRP will continue processing if the failure does not limit further processing.
#. Optionally calls ``primitive.save_model(self, model : DataModel | str)``
    * The model is saved if ``primitive.save_result`` is ``True``.
    * See additional configuration options in :py:class:`~hispecdrp.processing_component.ProcessingComponent` and :py:class:`~hispecdrp.primitives.base_primitive.BasePrimitive` for output_path construction.
#. Return the result to the user. If ``args.full_output`` is ``True``, the original Arguments object is returned. Otherwise, only the primary output is returned from the output's ``name`` field.


Primitive Arguments
-------------------

**Priority Hierarchy**

1. **Class-Level Default Arguments** ``primitive.ARGUMENTS``
   - Arguments defined in the class attribute ``ARGUMENTS`` are the starting values and have the **lowest priority.**

2. **Configuration Arguments** ``primitive.config['arguments']``
   - Arguments loaded from a configuration file or a pre-defined configuration dictionary with key ``arguments``. These will override the default args.

3. **Keyword Arguments** ``kwargs``
   - Keyword arguments passed directly to Primitive's constructor or ``apply`` method have the **highest priority.**


Primitive Configuration Files
-----------------------------

Primitives can also be configured with config files, but only simple serializable arguments can be set here. An example configuration for the dark primitive is below. Some values are redundant with the default ARGUMENTS, but are included here to be explicit.


.. code-block:: yaml
    
    %YAML 1.2
    ---

    logging:
      level: info
      log_path: None
  
    arguments:
      run: True
      save_results: True
      output_dir: None
      suffix: 'examplesuffix'


Logging
+++++++

Each primitive creates a unique logger instance which is accessible via ``self.log``. The logger will log messages at the level specified ``level``. The log messages can be directed to a file if log_path is provided, and always logs to the console for now.


Primitive Methods
-----------------

Below are most methods one can implement or override when creating custom primitives. Note some are required for certain behavior. See the full API for full documentation of available methods.

* _perform(self) -> Arguments
    - Required. This method takes in no arguments. It contains the primary logic of the primitive and returns an instance of ``Arguments`` where the attribute ``name`` contains the primary output model. Extra outputs may be stored in the output ``Arguments`` object. If ``full_output`` is ``True``, the entire ``Arguments`` object is returned, otherwise only the ``name`` attribute is returned.
* _pre_condition(self) -> bool
    - Optional. This method can be overwritten or extended to implement a check before the primitive is executed. If it returns ``False``, the primitive will be skipped. This currently only checks if ``self.args.run`` is ``True``.
* get_calibration_selector(self, input_model : str | DataModel) -> CalibrationSelector
    - Required if primitive needs a calibration file. This method determines the appropriate calibration selector based on the input model type. It should return an instance of a calibration selector class, such as ``HISPECDarkSelector`` or ``PARVIDarkSelector``. Specification and configuration of the selector is subject to change.
* get_output_model(self, input_model : DataModel, model_suffix : str | None = None) -> DataModel
    - Optional. This method can be overridden to specify the output model type. An optional suffix can also be provided to tell the primitive which type of model to return which is fully specified by the instrument name from input_model (e.g. model_suffix='DarkModel' for HISPECDarkModel or PARVIDarkModel).
* on_skip(self)
    - Optional. This method can be overridden to implement custom behavior when the primitive is skipped due to the pre-condition check.


Implementing a Primitive
------------------------

The dark subtraction primitive is implemented as follows and stored in ``hispecdrp/primitives/dark_subtraction/dark_primitive.py``. See `dark_primitive.py <https://github.com/oirlab/HISPEC_DRP/blob/main/hispecdrp/primitives/dark_subtraction/dark_primitive.py>`_ for the rest of the implementation.


.. code-block:: python

    class DarkSubtractionPrimitive(BasePrimitive):
        """
        ARGUMENTS:
            suffix (str): Suffix to append to the output file name.
                Default is 'darksub'.
                
            dark (HISPECDarkModel | PARVIDarkModel | str | None): Dark model to apply to the input data.
                If None, the dark model will be queried.
        """

        ARGUMENTS = {
            'suffix' : 'darksub',
            'dark' : None
        }

        def _perform(self):
            input_model = datamodels.open(self.args.name)
            if self.args.dark is None:
                dark_path = self.get_calibration(input_model, use_cached=True)
            else:
                dark_path = self.args.dark
            dark_model = datamodels.open(dark_path)
            result = input_model.copy()
            self.log.info(f"Applying dark correction with {dark_model} to {input_model}")
            result = dark_subtraction(result, dark_model)
            return Arguments(name=result)

        def get_calibration_selector(self, name):
            input_model = datamodels.open(name, meta_only=True)
            if isinstance(input_model, datamodels.HISPECDataModel):
                return HISPECDarkSelector()
            elif isinstance(input_model, datamodels.PARVIDataModel):
                return PARVIDarkSelector()
            else:
                raise ValueError(f"Unsupported input model type: {type(input_model)}")


    def dark_subtraction(input_model, dark_model):
        """
        Actual dark subtraction logic here.
        """
        pass