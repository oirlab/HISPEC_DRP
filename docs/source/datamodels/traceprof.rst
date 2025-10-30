Trace Profiles
==============

The trace profiles are determined from the fiber flat data. They are currently stored as-is with no modifications. The spectral extraction algorithm uses these weights appropriately in optimal extraction.

**Classes:**

* :py:class:`~hispecdrp.datamodels.traceprof.TraceProfModel`
* :py:class:`~hispecdrp.datamodels.traceprof.HISPECTraceProfModel`
* :py:class:`~hispecdrp.datamodels.traceprof.PARVITraceProfModel`

**Calibration type:** traceprof


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   PROF_FIB#, Image, Float64, "Ny x Nx", "None", "Profiles for fiber #."
   PROF_ERR_FIB#, Image, Float64, "Ny x Nx", "None", "Profile errors for fiber #."