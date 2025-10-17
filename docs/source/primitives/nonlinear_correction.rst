Nonlinear Correction
====================


Overview
--------

Corrects for the non-linear response of the detector.


**Class**: :py:class:`~hispecdrp.primitives.nonlinear_correction.nonlin_primitive.NonlinearCorrectionPrimitive`


Algorithm
---------

Converts the nonlinear signal to a linear signal through a polynomial transformation. DQ flags are updated with bitwise or. No error attributes are set at this step.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The input file or data model to process. Should be a raw (UTR ramp) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'nonlincorr').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.

Calibrations
------------

- :doc:`nonlin <../datamodels/nonlin>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.NonlinearCorrectionSelector`