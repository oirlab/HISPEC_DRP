Fit Etalon Lines
================


Overview
--------

Fit each Fabry-Pérot etalon line with a Gaussian or Lorentzian profile. This is useful for calculating the wavelength solution drifts.


**Class**: :py:class:`~hispecdrp.primitives.line_fitting.fit_etalon_lines_primitive.FitEtalonLinesPrimitive`


Algorithm
---------

Each Etalon line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian or Lorentzian profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``etalon_fitparams_fib#``.

Arguments
---------

- **name** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
    The input file or data model to process. Should be a level 0 (rate map) data product.
- **fwhm_bounds** : ``tuple[float, float]``
    The bounds for the full width at half maximum (FWHM) of the fit.
- **mu_bounds** : ``tuple[float, float]``
    The bounds for the mean of the Gaussian fit (in pixels) relative the centroid.
- **fsr_estimate** : ``float``
    The estimated free spectral range (FSR) of the etalon in GHz. Used to estimate the peak spacing.
- **max_fit_window** : ``float``
    The maximum number of pixels to fit for a mode.
- **do_orders** : ``list[int]`` | ``None``
    Which orders(s) to fit. If ``None``, fit all available orders.
- **do_fibers** : ``list[int]`` | ``None``
    Which fiber(s) to fit.
- **profile_type** : ``str``
    The type of profile to fit. Either 'gaussian' or 'lorentzian'. Defaults to 'lorentzian'.

Calibration Files
-----------------

None.