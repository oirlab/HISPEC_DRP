Raw (Ramp) Data
===============

The Teledyne Hawaii HXRG detectors use up-the-ramp (UTR) non-destructive reads, so raw data is a cube where each 2D slice is a single read.

**Classes:**

* :py:class:`~hispecdrp.datamodels.ramp.RampModel`
* :py:class:`~hispecdrp.datamodels.ramp.HISPECRampModel`
* :py:class:`~hispecdrp.datamodels.ramp.PARVIRampModel`

**File suffix:** raw


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   DATA, Image, UInt16, "Nreads x 4096 x 4096", DN, Raw UTR data.
   DQ_RAW, Image, UInt32, "4096 x 4096", None, Data quality flags for all reads.
   DQ, Image, UInt32, "Nreads x 4096 x 4096", None, Combined data quality flags for individual reads.