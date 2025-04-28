HISPEC Dark Model
=================

The HISPEC dark current (and any other contribution to the dark) is assumed to be linear with time.


Extensions
----------

.. csv-table::
   :header: "#", "Name", "HDU TYPE", "Data Type", "Dimensions", "Units", "Description"

   1, DATA, Image, Float32, "4096 x 4096", "None (normalized)", "Dark current data."
   2, ERR, Image, Float32, "4096 x 4096", None, "Dark current error (all sources)."
   3, VAR_POISSON, Image, Float32, "4096 x 4096", None, "Dark current variance (Poisson)."
   4, VAR_RNOISE, Image, Float32, "4096 x 4096", None, "Dark current variance (read noise)."
   5, DQ, Image, UInt32, "4096 x 4096", None, "Data quality."