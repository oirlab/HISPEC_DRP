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

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.ramp.HISPECRampModel`
    The input file or data model to process. Should be a raw (UTR ramp) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'nonlin').

Calibrations
------------

- :doc:`nonlin <../datamodels/nonlin>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.NonlinearCorrectionSelector`