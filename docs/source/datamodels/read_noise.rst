Read Noise
==========

The read noise specifies the read noise associated with each pixel.

**Class:** :py:class:`~hispecdrp.datamodels.read_noise.ReadNoiseModel`

**Calibration type:** rn


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   RN, Image, Float32, "4096 x 4096", "e-", "Read noise data."
   RN_ERR, Image, Float32, "4096 x 4096", "e-", "Read noise error"
   DQ, Image, UInt32, "4096 x 4096", None, "Data quality."