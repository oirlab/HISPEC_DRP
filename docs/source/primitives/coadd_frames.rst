Coadd Frames
============


Overview
--------

Combines multiple frames of the same kind to increase signal-to-noise.


**Class**: :py:class:`~hispecdrp.primitives.coadd.coadd_frames_primitive.CoaddFramesPrimitive`


Algorithm
---------

Data extensions are combined using a weighted average where the weights are the inverse variances (1 / model.err^2).
The output error and variance extensions are computed as the standard weighted error.
DQ flags are updated with bitwise or (Subject to change).

Arguments
---------

- **input** : list[``str`` | :py:class:`~hispecdrp.datamodels.base_model.DataModel`] | dict[str, ``str`` | :py:class:`~hispecdrp.datamodels.base_model.DataModel`] | :py:class:`~hispecdrp.datamodels.level0.L0Model`
    Input files or models as a list or a dictionary. If a dictionary is provideded, each group will be coadded separately.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'coadd').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the same kind of model as `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **mean_calc** : ``str``
    Method for calculating the output data. Currently only `'wmean'` (weighted means) is implemented (default).
- **err_calc** : ``str``
    Method for calculating the output error and variance. Currently only `'wse'` (weighted standard error) is implemented (default).