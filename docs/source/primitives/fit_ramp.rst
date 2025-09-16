Fit Ramp
========


Overview
--------

Fit the UTR slopes to produce a rate map.


**Class**: :py:class:`~hispecdrp.primitives.ramp_fitting.ramp_fit_primitive.FitRampPrimitive`


Algorithm
---------

Several methods are supported, see below.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.ramp.RampModel`
    The raw (UTR) model to fit slopes to.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'rampfit').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'L0Model'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **method** : ``str``
    The method to use to fit the ramps. Defaults to ``'jwst_ols'``.
    
    - **ols**: Ordinary Least Squares with numba. No weights.
    - **cds**: Correlated double sampling.
    - **mcds**: Multiple correlated double sampling (see argument num_coadd).
    - **jwst_gls**: Approximated generalized least squares method with `stcal <https://stcal.readthedocs.io/en/latest/stcal/ramp_fitting/description.html#ramp-slopes-and-variances>`_.
    - **jwst_likely**: Likelihood-based method with `stcal <https://stcal.readthedocs.io/en/latest/stcal/ramp_fitting/description.html#ramp-slopes-and-variances>`_.

- **num_coadd** : ``int``
    Number of coadds for ``method='mcds'``.
- **max_cores** : ``int``
    Maximum number of cores to use for parallel processing for method. Defaults to 1 (no parallel processing). Only supported by jwst methods for now.