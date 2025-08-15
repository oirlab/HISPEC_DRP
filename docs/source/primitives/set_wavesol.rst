Set Wavelength Solution
=======================


Overview
--------

Sets the initial wavelength solution into the data model typically from the start or end of the night. This wavelength solution has not yet been corrected for any drifts specific to this observation.

**Class**: :py:class:`~hispecdrp.primitives.wavesol.set_wavesol_primitive.SetInitialWLSPrimitive`


Algorithm
---------

The appropriate wavelength solution is associated with the input data model. This can come from the Calibration database or a user specified wavelength solution calibration.


Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
    The input file or data model to process. Should be a level 0 (rate map) data product.


Calibrations
------------

*How to choose between different wavesol types TBD*

- :doc:`lfcwavesol <../datamodels/lfcwavesol>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`
    - Selector Kwargs:
        - ``cal_type (str)``: The type of calibration to select (e.g. "lfc_wavesol", "une_wavesol", "gascells...").