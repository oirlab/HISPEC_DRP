Trace Positions
===============

The trace positions are determined from the fiber flat data. They are stored as polynomial coeffs and evaluated positions for each order/fiber.

**Class:** :py:class:`~hispecdrp.datamodels.tracepos.TracePosModel`

**Calibration type:** tracepos


Extensions
----------

.. csv-table::
   :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   COEFF_POS_FIB#, Image, Float64, "Nord x Ncoeff", "None", "Polynomial coefficients for the trace positions of fiber #."
   COEFF_FWHM_FIB#, Image, Float64, "Nord x Ncoeff", "None", "Polynomial coefficients for the trace widths (FWHM) of fiber #."
   POS_FIB#, Image, Float64, "Nord x Ncoeff", "Pixels", "Evaluated positions for fiber #."
   FWHM_FIB#, Image, Float64, "Nord x Ncoeff", "Pixels", "Evaluated fwhm for fiber #."