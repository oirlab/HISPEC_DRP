Dark
====

The dark current (and any other contribution to the dark) is assumed to be linear with time and is stored as a rate map product (e-/s).

**Classes:**

* :py:class:`~hispecdrp.datamodels.hispec.dark.HISPECDarkModel`
* :py:class:`~hispecdrp.datamodels.parvi.dark.PARVIDarkModel`

**Calibration type:** dark


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   DATA, Image, Float32, "4096 x 4096", "e-/s", "Dark current data."
   ERR, Image, Float32, "4096 x 4096", "e-/s", "Dark current error (all sources)."
   VAR_POISSON, Image, Float32, "4096 x 4096", "e-/s", "Dark current variance (Poisson)."
   VAR_RNOISE, Image, Float32, "4096 x 4096", "e-/s", "Dark current variance (read noise)."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."