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

   DATA, Image, UInt16, "Nreads x 4096 x 4096", ADU or e-, Raw UTR data
   TIMES, Table, See below, See below, See below, Exposure time information for reach read.
   DQ, Image, UInt32, "Nreads x 4096 x 4096", None, Data quality flags for each read
   ERR, Image, Float32, "Nreads x 4096 x 4096", e-, Error for each read (all sources) (TBD)
   VAR_RNOISE, Image, Float32, "Nreads x 4096 x 4096", e-^2, Read noise variance for each read (TBD)
   VAR_POISSON, Image, Float32, "Nreads x 4096 x 4096", e-^2, Poisson variance for each read (TBD)
   MID_TIME, Image, Float64, "4096 x 4096", MJD, Flux-weighted exposure mid-point times for each pixel. (TBD)



.. csv-table:: **TIMES Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "READNUM", "Float64", "The read number in a UTR exposure where the first read has READNUM=1."
   "TIMES", "Float64", "The starting time of the read (MJD)."
   "EXPTIME", "Float64", "The accumulated exposure time (sec)."
