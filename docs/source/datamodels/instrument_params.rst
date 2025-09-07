Instrument Parameters
=====================

Parameters that describe the instrument such as a laser frequency comb pump line and spacing.

**Classes:**

* :py:class:`~hispecdrp.datamodels.instrument_params.InstrumentParametersModel`
* :py:class:`~hispecdrp.datamodels.instrument_params.HISPECInstrumentParametersModel`
* :py:class:`~hispecdrp.datamodels.instrument_params.PARVIInstrumentParametersModel`

**Calibration type:** instrument_params


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   PARAMS, Table, See below, See below, See below, "Instrument parameters."

.. csv-table:: **PARAMS Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "NAME", "string", "Name of the parameter."
   "VALUE", "float64", "Value of the parameter."
   "UNIT", "string", "Unit of the parameter."
   "NOTE", "string", "Note about the parameter."