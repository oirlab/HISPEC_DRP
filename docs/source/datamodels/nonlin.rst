Non Linearity
=============

Specifies the nonlinearity of the detector readout response with a unique polynomial for each pixel.

**Classes:**

* :py:class:`~hispecdrp.datamodels.nonlin.NonlinearCorrectionModel`
* :py:class:`~hispecdrp.datamodels.nonlin.HISPECNonlinearCorrectionModel`
* :py:class:`~hispecdrp.datamodels.nonlin.PARVINonlinearCorrectionModel`

**Calibration type:** nonlin


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   COEFFS, Image, Float64, "4096 x 4096 x Ncoeff", "None", "Polynomial coeffs describing the nonlinearity"
   THRESH, Image, UInt16, "4096 x 4096", None, "TBD, from JWST."
   MAX, Image, UInt16, "4096 x 4096", None, "TBD, from JWST."