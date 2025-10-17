Gain
====

The detector gain specifies the conversion factor between electrons and data numbers (DN) for each pixel in the detector.

**Classes:**

* :py:class:`~hispecdrp.datamodels.detector_gain.DetectorGainModel`
* :py:class:`~hispecdrp.datamodels.detector_gain.HISPECDetectorGainModel`
* :py:class:`~hispecdrp.datamodels.detector_gain.PARVIDetectorGainModel`

**Calibration type:** gain


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   GAIN, Image, Float32, "4096 x 4096", "e-/DN", "Gain data."
   GAIN_ERR, Image, Float32, "4096 x 4096", "e-/DN", "Gain error (all sources)."
   GAIN_VAR_POISSON, Image, Float32, "4096 x 4096", "(e-/DN)^2", "Gain variance (Poisson)."
   GAIN_VAR_RNOISE, Image, Float32, "4096 x 4096", "(e-/DN)^2", "Gain variance (read noise)."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."