Non Linearity
=============

Specifies the nonlinearity of the detector readout response with a unique polynomial for each pixel.

**Class:** :py:class:`~hispecdrp.datamodels.nonlin.NonlinearCorrectionModel`

**Calibration type:** nonlin


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   COEFFS, Image, Float64, "4096 x 4096 x Ncoeff", "None", "Polynomial coeffs describing the nonlinearity"
   THRESH, Image, UInt16, "4096 x 4096", None, "TBD, from JWST."
   MAX, Image, UInt16, "4096 x 4096", None, "TBD, from JWST."