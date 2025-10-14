DQ Initialization
=================


Overview
--------

Initializes the data quality (DQ) array with known pixel flags.


**Class**: :py:class:`~hispecdrp.primitives.dq_init.dqinit_primitive.DQInitPrimitive`


Algorithm
---------

The DQ array is initialized with known flags from a DQ reference file.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The input file or data model to process. For now this is a raw (ramp) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'dqinit').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.

Calibrations
------------

- :doc:`dq <../datamodels/dq>`
    - Selector: :py:class:`~hispecdrp.calibrations.selectors.DQSelector`