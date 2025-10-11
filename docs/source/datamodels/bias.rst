Bias
====

The bias is (to be updated) ...

**Classes:**

* :py:class:`~hispecdrp.datamodels.bias.BiasModel`
* :py:class:`~hispecdrp.datamodels.bias.HISPECBiasModel`
* :py:class:`~hispecdrp.datamodels.bias.PARVIBiasModel`

**Calibration type:** bias


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   BIAS, Image, Float32, "4096 x 4096", "DN", "Bias level."
   BIAS_ERR, Image, Float32, "4096 x 4096", "DN", "Bias error."
   BIAS_kTC, Image, Float32, "4096 x 4096", "DN", "Bias kTC noise level."
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."