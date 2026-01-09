Bias
====

The bias is (to be updated) ...

**Class:** :py:class:`~hispecdrp.datamodels.bias.BiasModel`

**Calibration type:** bias


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   BIAS, Image, Float32, "4096 x 4096", "DN", "Bias level."
   BIAS_ERR, Image, Float32, "4096 x 4096", "DN", "Bias error."
   BIAS_KTC, Image, Float32, "4096 x 4096", "DN", "Bias kTC noise level."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."