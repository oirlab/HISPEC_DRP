Saturation
==========

The saturation specifies the saturation level associated with each pixel.

**Classes:**

* :py:class:`~hispecdrp.datamodels.saturation.SaturationModel`
* :py:class:`~hispecdrp.datamodels.saturation.HISPECSaturationModel`
* :py:class:`~hispecdrp.datamodels.saturation.PARVISaturationModel`

**Calibration type:** saturation


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   SAT_THRESH, Image, UInt16, "4096 x 4096", "DN", "Saturation threshold."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."