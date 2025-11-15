Trace Finding
=============


Overview
--------

Finds the traces on the detector.


**Class**: :py:class:`~hispecdrp.primitives.trace_fitting.tracepos_primitive.TracePositionPrimitive`


Algorithm
---------

This algorithm finds echelle orders on the detector using high SNR fiber flats.
The algorithm assumes only one fiber is illuminated at a time and that orders are oriented mostly in the horizontal (x-axis) direction.
The algorithm works as follows:

1. Pick a 1D vertical slice in the cross-dispersion direction (y-axis) near the center of the image (with additional smoothing).
2. Crop this slice so that it only contains the desired orders to trace, specified by the input ``yrange = (y_start, y_end)``.
3. Find the integer location of peaks in the slice using `scipy.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_. Refine these initial positions with a centroid.
4. For a given trace, walk left (towards smaller x values) and compute the trace positions (centroid) of each column. The second moment is also computed.
5. Repeat step 4 walking right (towards larger x values).
6. Fit a polynomial to the trace positions and second moments as a function of x position.
7. Repeat steps 4-6 for all traces.

Arguments
---------

- **input** : dict[str, list[str | :py:class:`~hispecdrp.datamodels.level0.L0Model`]] | :py:class:`~hispecdrp.datamodels.level0.L0Model`
    The input file or data model to process. For now this is a level 0 (rate map) data product.
- **suffix** : ``str`` | ``None``
    Suffix to append to the output filename (default: 'tracepos').
- **output_model** : ``str`` | type[:py:class:`~hispecdrp.datamodels.level0.L0Model`] | ``None``
    The DataModel class or instance to return. Defaults to the appropriate `L0Model`. See method :py:meth:`~hispecdrp.primitives.base_primitive.BasePrimitive._get_output_model`.
- **deg_pos** : ``int``
    Polynomial order to use for fitting trace positions.
- **deg_width** : ``int``
    Polynomial order to use for fitting trace widths (fwhm).
- **window_size** : ``int``
    The size of the window in the cross-dispersion direction used to calculate the centroid and second moment. Defaults to 8 pixels.
- **xrange** : ``tuple[int, int]`` | ``None``
    The range of x values to use for fitting the traces. If ``None``, use the full range allowed by the data quality extension.
- **yrange** : ``tuple[int, int]`` | ``None``
    The range of y values to use for finding initial trace positions. If ``None``, use the full range allowed by the data quality extension.