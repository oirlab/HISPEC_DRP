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

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.level1.L1Model` | :py:class:`~hispecdrp.datamodels.level2.L2Model`
    The input file or data model to process. Should be a level 1 or 2 (spectrum) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: None).


Calibrations
------------

*How to choose between different wavesol types TBD*

- :doc:`lfcwavesol <../datamodels/lfcwavesol>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`
    - Selector Kwargs:
        - ``cal_type (str)``: The type of calibration to select (e.g. "lfc_wavesol", "une_wavesol", "gascells...").