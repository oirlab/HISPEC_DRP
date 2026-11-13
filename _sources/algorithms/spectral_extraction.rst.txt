===================
Spectral Extraction
===================

There are three methods of spectral extraction currently implemented.

1. **box**: Sum all pixels within a fixed aperture around the trace position.
2. **optimal**: Fit a 1D profile to the data.
3. **cross_talk**: Simultaneously fit 1D profiles for all fibers to account for cross-talk between fibers. Only applicable when multiple fibers are present.

All methods rely on the trace positions defined in the calibration file `tracepos` (:py:class:`~hispecdrp.datamodels.tracepos.TracePosModel`).

For optimal and cross-talk extraction, the required trace profiles are provided by the calibration file `traceprof` (:py:class:`~hispecdrp.datamodels.traceprof.TraceProfModel`).

In both methods, the vector :math:`M_y` is a mask that is 1 where the data is included in the extraction, and 0 where it is not (e.g. bad pixels).
The vector :math:`W_y` is a weight vector that defines the fraction of a pixel that overlaps with the extraction aperture. In this scheme, the location of a pixel is defined at its center, so a given pixel at location `y` spans from `y - 0.5` to `y + 0.5`. This weight is important when it is difficult to define a common normalization factor for the profile at different columns, as is the case with a non-analytic profile or no profile at all. This correction is especially important when the profile is not symmetric.


Boxcar Extraction
-----------------

This method sums the flux within a defined aperture around the central trace positions.

.. math::

   \begin{aligned}
    N &= \frac{\sum_y M_y W_y}{\sum_y W_y} \\[2mm]
    f &= \frac{\sum_y M_y W_y D_y}{N} \\[2mm]
    \sigma &= \frac{\sqrt{\sum_y M_y \sigma_y^2}}{N} \\[2mm]
    \sigma^2_{\text{RN}} &= \frac{\sum_y M_y W_y \sigma_{\text{RN},y}^2}{N^2} \\[2mm]
    \sigma^2_{\text{phot}} &= \frac{\sum_y M_y W_y \sigma_{\text{phot},y}^2}{N^2}
    \end{aligned}

Optimal Extraction
------------------

This method attempts to find the best fit scale factor for the expected spatial profile of the spectrum. Here, :math:`P_y` is the expected profile at pixel `y`, normalized such that :math:`\sum_y W_y P_y = 1` within the extraction aperture.

.. math::

    \begin{aligned}
    \tilde{P}_y &= \frac{P_y}{\sum_y P_y W_y} \\[2mm]
    f &= \frac{\sum_y M_y \tilde{P}_y D_y W_y / \sigma_y^2}{\sum_y M_y \tilde{P}_y^2 W_y / \sigma_y^2} \\[2mm]
    \sigma &= \sqrt{\frac{\sum_y M_y \tilde{P}_y W_y}{\sum_y M_y \tilde{P}_y^2 W_y / \sigma_y^2}} \\[2mm]
    \sigma_{\rm RN}^2 &= \frac{\sum_y M_y \tilde{P}_y W_y}{\sum_y M_y \tilde{P}_y^2 W_y / \sigma_{\rm RN,y}^2} \\[2mm]
    \sigma_{\rm phot}^2 &= \frac{\sum_y M_y \tilde{P}_y W_y}{\sum_y M_y \tilde{P}_y^2 W_y / \sigma_{\rm phot,y}^2}
    \end{aligned}


Cross Talk
----------

This method attemps to simultaneously extract spectral values from all fibers within a given order simultaneously to account for the cross-talk between fibers. The math is effectively identical to optimal extraction.


Documentation
-------------

.. autoclass:: hispecdrp.primitives.spectral_extraction.spectral_extraction_primitive.SpectralExtractionPrimitive
   :members:
   :undoc-members:
   :no-index: