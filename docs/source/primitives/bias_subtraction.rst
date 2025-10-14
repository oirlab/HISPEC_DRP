Bias Subtraction
================


Overview
--------

Subtracts the bias level from the input data.


**Class**: :py:class:`~hispecdrp.primitives.bias_subtraction.bias_primitive.BiasSubtractionPrimitive`


Algorithm
---------

The appropriate master bias calibration is subtracted from the input data. The input's ``err`` attribute is updated by adding the bias's ``err`` attribute in quadarature. DQ flags are updated with bitwise or.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The input file or data model to process. For now this is a raw (ramp) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'biassub').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.

Calibrations
------------

- :doc:`bias <../datamodels/bias>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.BiasSelector`