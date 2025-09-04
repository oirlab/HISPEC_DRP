Wavelength Drift Correction
===========================


Overview
--------

Corrects the wavelength drifts. Currently only the Fabry-Pérot etalons are used to measure and correct for any drifts. Drifts are also currently calculated on a per-order basis which is subject to change as the instrument is characterized.

*Calculation and applying the WLS are separate primitives while other WLS sources are under development.*

**Classes**:

- :py:class:`~hispecdrp.primitives.wavesol.drift_correction_primitive.CalculateEtalonDriftCorrectionPrimitive`

- :py:class:`~hispecdrp.primitives.wavesol.drift_correction_primitive.ApplyDriftCorrectionPrimitive`

Algorithm
---------

1. The primitive CalculateEtalonDriftCorrectionPrimitive determines the per-etalon-peak drift by matching each etalon peak based on a proximity threshold in wavelength space.

**For now, a per-order drift is calculated** as the weighted average of the drifts measured in detector pixels. The drift is then converted to effective drifts in wavelength and velocity units according to the local dispersion.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
    The input file or data model to process. Should be a level 2 data product.

Calibrations
------------

- :doc:`ref_etalon <../datamodels/ref_etalon>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.ReferenceEtalonSelector`


2. The ApplyDriftCorrectionPrimitive applies the calculated drift corrections to the input data's wavelength grid as a classical Doppler shift.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
    The input file or data model to process. Should be a level 2 data product.