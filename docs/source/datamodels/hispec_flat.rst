HISPEC Flat Model
=================

The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.


Extensions
----------

.. csv-table::
   :header: "#", "Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   1, DATA, Image, Float32, "4096 x 4096", "None (normalized)", "Flat data."
   2, ERR, Image, Float32, "4096 x 4096", None, "Flat field error (all sources)."
   3, VAR_POISSON, Image, Float32, "4096 x 4096", None, "Flat field variance (Poisson)."
   4, VAR_RNOISE, Image, Float32, "4096 x 4096", None, "Flat field variance (read noise)."
   5, DQ, Image, UInt32, "4096 x 4096", None, "Data quality."