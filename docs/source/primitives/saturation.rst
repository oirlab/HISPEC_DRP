Check Saturation
================


Overview
--------

Flags any saturated reads.


**Class**: :py:class:`~hispecdrp.primitives.saturation.saturation_primitive.CheckSaturationPrimitive`


Algorithm
---------

The input's DQ `SATURATED` flag is set for any pixel in any read that exceeds the per-pixel saturation level defined in the saturation calibration file.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The input file or data model to process. For now this is a level 0 (rate map) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'saturation').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.

Calibrations
------------

- :doc:`saturation <../datamodels/saturation>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.SaturationSelector`