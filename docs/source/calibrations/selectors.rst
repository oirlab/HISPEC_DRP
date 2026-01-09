Calibration Selectors
=====================

**Temporal Matching:**

The rules below result in a subset of possible calibration files.
The "best" calibration file is chosen from this subset by selecting the one closest in time to the observation using the meta key ``meta.mjd_start``.

Bias Selector
+++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.BiasSelector`

**Calibration Type**: ``'bias'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Dark Selector
+++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DarkSelector`

**Calibration Type**: ``'dark'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Detector Flat Selector
++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DetectorFlatSelector`

**Calibration Type**: ``'detflat'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Detector Gain Selector
++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DetectorGainSelector`

**Calibration Type**: ``'detgain'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


DQ Selector
+++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DQSelector`

**Calibration Type**: ``'dq'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Nonlinear Correction Selector
+++++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.NonlinearCorrectionSelector`

**Calibration Type**: ``'nonlin'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Read Noise Selector
+++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.ReadNoiseSelector`

**Calibration Type**: ``'rn'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Reference Etalon Selector
++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.ReferenceEtalonSelector`

**Calibration Type**: ``'ref_etalon'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Closest in observation time (``meta['mjd_start']``)


Saturation Selector
+++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.SaturationSelector`

**Calibration Type**: ``'saturation'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Trace Position Selector
++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.TracePosSelector`

**Calibration Type**: ``'tracepos'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Trace Profile Selector
+++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.TraceProfSelector`

**Calibration Type**: ``'traceprof'``

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Require master calibration (``master_cal == True``)
- Closest in observation time (``meta['mjd_start']``)


Wavelength Solution Selector
++++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`

**Calibration Type**: Configurable (default: ``'lfc_wavesol'``)

**Selection Logic:**

- Match instrument era (``meta['instrument_era']``)
- Match spectrograph (``meta['spectrograph']``)
- Closest in observation time (``meta['mjd_start']``)

**Arguments:**

- ``cal_type`` (str, optional): The type of wavelength calibration to select. Default: ``'lfc_wavesol'``