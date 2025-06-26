Dark Subtraction
================


Overview
--------

Subtracts the dark current from the input data.


**Class**: :py:class:`~hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive`


Algorithm
---------

The appropriate master dark calibration is subtracted from the input data. The input's ``err`` attribute is updated by adding the dark's ``err`` attribute in quadarature. DQ flags are updated with bitwise or.

Arguments
---------

- **name** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level0.HISPECL0Model`
    The input file or data model to process. For now this is a level 0 (rate map) data product.
- **dark** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.dark.HISPECDarkModel` | ``None``
    The name of the dark calibration file or data model. If not provided, the dark is fetched from the calibration database.


Calibration Files
-----------------

- :doc:`dark <../datamodels/dark>`