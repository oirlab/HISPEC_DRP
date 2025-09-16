Etalon Line Fitting
===================


Overview
--------

Fit each Fabry-Pérot etalon line with a Gaussian or Lorentzian (default) profile. This is useful for measuring drifts in the wavelength solution.


**Class**: :py:class:`~hispecdrp.primitives.line_fitting.fit_etalon_lines_primitive.FitEtalonLinesPrimitive`


Algorithm
---------

Each etalon line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian or Lorentzian (default) profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``etalon_fitparams_fib#``.

Arguments
---------

- **input** : ``str`` | :py:class:`~hispecdrp.datamodels.level2.L2Model`
    The input file or data model to process. Should be a level 2 data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: None).
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.base_model.DataModel`] | ``None``
    The DataModel class or instance to return. Defaults to the string `'input'`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **fwhm_bounds** : ``tuple[float, float]``
    The bounds for the FWHM of the fit.
- **mu_bounds** : ``tuple[float, float]``
    The bounds for the mean of the fit (in pixels) relative the centroid.
- **max_fit_window** : ``float``
    The maximum number of pixels to fit for a mode. Defaults to 6xFWHM.
- **do_orders** : ``list[int]`` | ``None``
    Which orders(s) to fit. If ``None``, fit all available orders. Defaults to all orders.
- **do_fibers** : ``list[str]`` | ``None``
    Which fiber(s) to fit. Defaults to all fibers.
- **fsr_estimate** : ``float``
    The estimated free spectral range (FSR) of the etalon in GHz.
- **profile_type** : ``str``
    The profile type to use for fitting. Options are "gaussian" or "lorentzian". Defaults to "lorentzian".