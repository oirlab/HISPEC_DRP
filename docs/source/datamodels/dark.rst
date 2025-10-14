Dark
====

The dark current (and any other contribution to the dark) is assumed to be linear with time and is stored as a rate map product (e-/s).

**Classes:**

* :py:class:`~hispecdrp.datamodels.dark.DarkModel`
* :py:class:`~hispecdrp.datamodels.dark.HISPECDarkModel`
* :py:class:`~hispecdrp.datamodels.dark.PARVIDarkModel`

**Calibration type:** dark


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   DARK_RATE, Image, Float32, "4096 x 4096", "DN", "Dark current data."
   DARK_ERR, Image, Float32, "4096 x 4096", "DN", "Dark current error (all sources)."
   DARK_VAR_POISSON, Image, Float32, "4096 x 4096", "DN^2", "Dark current variance (Poisson)."
   DARK_VAR_RNOISE, Image, Float32, "4096 x 4096", "DN^2", "Dark current variance (read noise)."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."