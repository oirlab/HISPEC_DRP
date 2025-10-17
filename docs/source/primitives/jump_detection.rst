Jump Detection
==============


Overview
--------

Detects jumps in the UTR data and updates the DQ flags where jumps are found.


**Class**: :py:class:`~hispecdrp.primitives.jump_detection.jump_detection_primitive.JumpDetectionPrimitive`


Algorithm
---------

Two methods are currently supported, see below.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The raw (UTR) model to  find jumps.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'rampfit').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'RampModel'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **method** : ``str``
    The method to use to fit the ramps. Defaults to ``'jwst'``.
    
    - **tpd**: A simple two-point difference based algorithm.
    - **jwst**: The two-point difference method implemented by `stcal <https://stcal.readthedocs.io/en/latest/stcal/jump/description.html#jump-algorithm>`_.

- **max_cores** : ``int``
    Maximum number of cores to use for parallel processing for method. Defaults to 1 (no parallel processing).

- **rejection_threshold** : ``float``
    Rejection threshold used by jump detection algorithms (roughly in signal-to-noise) above which jumps are flagged.
