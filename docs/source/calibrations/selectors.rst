Calibration Selectors
=====================

Bias Selector
+++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.BiasSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


Detector Gain Selector
++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DetectorGainSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


DQ Selector
+++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DQSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)

Nonlinear Correction Selector
+++++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.NonlinearCorrectionSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


Saturation Selector
+++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.SaturationSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)

Dark Selector
+++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DarkSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


Nonlinear Correction Selector
+++++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.NonlinearCorrectionSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


Detector Flat Selector
++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.DetectorFlatSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)


Wavelength Solution Selector
++++++++++++++++++++++++++++

**Class**: :py:class:`~hispecdrp.calibrations.selectors.WavelengthSolutionSelector`

**Selection Logic:**

- match instrument era (``meta['instrument.era']``)
- match spectrograph (``meta['instrument.spectrograph']``)
- closest in observation time (``meta['exposure.mjd_start']``)

**Arguments**

- ``cal_type``: The type of wavelength calibration to select (e.g. 'lfc_wavesol').