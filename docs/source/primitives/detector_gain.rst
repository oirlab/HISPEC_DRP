Detector Gain
=============


Overview
--------

Converts DN to e- using the detector gain calibration file.


**Class**: :py:class:`~hispecdrp.primitives.detector_gain.detgain_primitive.DetectorGainPrimitive`


Algorithm
---------

Multiplies the input data and error arrays by the per-pixel gain value from the gain calibration file to convert from DN to e-. The gain value is in units of e-/DN.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The input file or data model to process. For now this is a level 0 (rate map) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'checksat').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.

Calibrations
------------

- :doc:`saturation <../datamodels/saturation>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.SaturationSelector`