Register Calibration
====================


Overview
--------

Register a calibration file in the local HISPEC DRP calibration database.


**Class**: :py:class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`


Algorithm
---------

None.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.base_model.CalibrationModel`
    The input file or data model to register in the local calibration database.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: None).