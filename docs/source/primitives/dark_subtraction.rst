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

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level0.HISPECL0Model`
    The input file or data model to process. For now this is a level 0 (rate map) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'darksub').

Calibrations
------------

- :doc:`dark <../datamodels/dark>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.DarkSelector`