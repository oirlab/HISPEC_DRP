=============
Trace Fitting
=============

Trace Positions
---------------

This algorithm identifies echelle orders on the detector using high-SNR fiber flats.
It assumes only one fiber is illuminated at a time and that orders are oriented approximately along the horizontal (x) axis.

The procedure is:

1. Select a 1D vertical slice in the cross-dispersion (y) direction near the image center, applying additional smoothing.
2. Crop the slice to the desired orders using the input ``yrange = (y_start, y_end)``.
3. Identify integer peak locations in the slice using `scipy.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_, then refine them via centroiding.
4. For each trace, step left (toward smaller x values) and compute the centroid position in each column; also compute the second moment.
5. Repeat step 4 stepping right (toward larger x values).
6. Fit polynomials to the trace positions and second moments as functions of x.
7. Repeat steps 4-6 for all traces.

Documentation
+++++++++++++

.. autoclass:: hispecdrp.primitives.trace_fitting.tracepos_primitive.TracePositionPrimitive
   :members:
   :undoc-members:
   :no-index:

Trace Profiles
--------------

High SNR fiber flats are used to determine the trace profiles. Currently this algorithm simply converts the input fiber flat data into a :py:class:`~hispecdrp.datamodels.traceprof.TraceProfModel`.

Documentation
+++++++++++++

.. autoclass:: hispecdrp.primitives.trace_fitting.traceprof_primitive.TraceProfilePrimitive
   :members:
   :undoc-members:
   :no-index: