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

- **name** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level0.HISPECL0Model`
    The input file or data model to process. Should be a level 0 (rate map) data product.
- **detflat** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.detector_flat.HISPECDetectorFlatModel` | ``None``
    The name of the detector flat field calibration file or data model. If not provided, the flat is fetched from the calibration database.


Calibration Files
-----------------

- :doc:`detflat <../datamodels/detector_flat>`