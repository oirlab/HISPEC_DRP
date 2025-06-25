Fit LFC Lines
=============


Overview
--------

Fit each laser frequency comb (LFC) line with a Gaussian profile. This is useful for calculating the wavelength solution and the the line spread function (LSF).


**Class**: :py:class:`~hispecdrp.primitives.line_fitting.fit_lfc_modes_primitive.FitLFCModesPrimitive`


Algorithm
---------

Each LFC line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``lfc_fitparams_fib#``.

Arguments
---------

- **name** : ``str`` | :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
    The input file or data model to process. Should be a level 0 (rate map) data product.
- **sigma_bounds** : ``tuple[float, float]``
    The bounds for the standard deviation of the Gaussian fit.
- **mu_bounds** : ``tuple[float, float]``
    The bounds for the mean of the Gaussian fit (in pixels) relative the centroid.
- **max_fit_window** : ``float``
    The maximum number of pixels to fit for a mode.
- **do_orders** : ``list[int]`` | ``None``
    Which orders(s) to fit. If ``None``, fit all available orders.
- **do_fibers** : ``list[int]`` | ``None``
    Which fiber(s) to fit.
- **lfc_nu0** : ``float``
    The central frequency (pump line) of the LFC in GHz.
- **lfc_delta_nu** : ``float``
    The frequency spacing of the LFC modes in GHz.


Calibration Files
-----------------

None.