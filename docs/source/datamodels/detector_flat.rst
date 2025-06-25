Detector Flat Model
===================

The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.

**Classes:**

* :py:class:`~hispecdrp.datamodels.hispec.detflat.HISPECDetectorFlatModel`
* :py:class:`~hispecdrp.datamodels.parvi.detflat.PARVIDetectorFlatModel`

**Calibration type:** detflat


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   DATA, Image, Float32, "4096 x 4096", "None (normalized)", "Flat data."
   ERR, Image, Float32, "4096 x 4096", None, "Flat field error (all sources)."
   VAR_POISSON, Image, Float32, "4096 x 4096", None, "Flat field variance (Poisson)."
   VAR_RNOISE, Image, Float32, "4096 x 4096", None, "Flat field variance (read noise)."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."