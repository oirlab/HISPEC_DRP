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

- **name** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.ramp.HISPECRampModel`
    The input file or data model to process. Should be a raw (UTR ramp) data product.
- **nonlin** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.nonlin.HISPECNonlinearCorrectionModel` | ``None``
    The name of the nonlinear correction file or data model. If not provided, the file is fetched from the calibration database.


Calibration Files
-----------------

- :doc:`nonlin <../datamodels/nonlin>`