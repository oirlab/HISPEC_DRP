Trace Profile
=============


Overview
--------

Calculates the trace profiles used for extracting spectra.

**Class**: :py:class:`~hispecdrp.primitives.trace_fitting.traceprof_primitive.TraceProfilePrimitive`


Algorithm
---------

High SNR fiber flats are used to determine the trace profiles. Currently this algorithm simply converts the input fiber flat data into a :py:class:`~hispecdrp.datamodels.traceprof.TraceProfModel`.

Arguments
---------

- **input** : dict[str, list[str | :py:class:`~hispecdrp.datamodels.level0.L0Model`]]
    The input fiber flat files or data model sorted by fiber.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'traceprof').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.traceprof.TraceProfModel`] | ``None``
    The DataModel class or instance to return. Defaults to the appropriate `TraceProfModel`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.