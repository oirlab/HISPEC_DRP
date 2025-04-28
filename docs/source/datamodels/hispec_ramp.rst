HISPEC Raw (Ramp) Data
======================

The Teledyne Hawaii H4RG detectors up-the-ramp (UTR) non-destructive read modes, so HISPEC raw data is made of a set of UTR images.


Extensions
----------

.. csv-table::
   :header: "#", "Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   1, DATA, Image, UInt16, "4096 x 4096 x Nreads", ADU, Raw UTR data
   2, TIMES, Image, Float64, Nreads, s, Exposure times for each read.