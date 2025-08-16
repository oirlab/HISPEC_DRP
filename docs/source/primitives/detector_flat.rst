Detector Flat Correction
========================


Overview
--------

Corrects for the pixel to pixel response of the detector.

**Class**: :py:class:`~hispecdrp.primitives.detector_flat.detflat_primitive.DetectorFlatPrimitive`


Algorithm
---------

The appropriate master detector flat calibration is dividided into the input data rate map. The input's ``err`` attribute is updated by adding the detector flat error in quadarature with the appropriate scaling. DQ flags are updated with bitwise or.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level0.HISPECL0Model`
    The input file or data model to process. Should be a level 0 (rate map) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'detflat').


Calibrations
------------

- :doc:`detflat <../datamodels/detector_flat>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.DetectorFlatSelector`