=====================
Cross-correlation (CCF)
=====================

This page summarizes the CCF algorithms and fitting machinery implemented in
:mod:`hispecdrp.primitives.utils.ccf_utils` (and used by both stellar-RV algorithms).


CCF utilities overview (``ccf_utils``)
--------------------------------------

:mod:`hispecdrp.primitives.utils.ccf_utils` contains the core CCF and RV estimation machinery for both
stellar-RV algorithms:

- **Template CCF** (``ccf_stellar_template``): weighted-covariance CCF in :math:`\ln\lambda`,
  Numba multi-order execution, continuum normalization, robust chunk aggregation (median/MAD) and
  inverse-variance combination.

- **Line-mask CCF** (``ccf_line_mask``): SPIROU/KPF-style mask-line CCF driver
  (:func:`~hispecdrp.primitives.utils.ccf_utils.run_ccf_line_mask_for_fiber`) and supporting kernels.
  For the mask-line path, per-chunk ``RV``/``RV_ERR`` can be derived from either an order-mean
  Gaussian fit (when accepted) or a parabolic-vertex estimate.


Deriving velocities from a CCF
------------------------------

Once you have a CCF curve (velocity axis ``v`` and CCF samples ``CCF(v)``), HISPEC DRP provides
generalized routines in :mod:`hispecdrp.primitives.utils.ccf_utils` to infer an RV (and an
uncertainty) in a way that can be applied to CCFs produced by **either** algorithm
(``ccf_stellar_template`` or ``ccf_line_mask``).

Gaussian peak fitting
^^^^^^^^^^^^^^^^^^^^^

The primary peak-fit routine is :func:`~hispecdrp.primitives.utils.ccf_utils.fit_ccf_rv`, which fits
a Gaussian + constant baseline model to :math:`\mathrm{CCF}(v)` (Astropy ``Gaussian1D`` + ``Const1D``
with ``LevMarLSQFitter``). The implementation uses a two-pass fitting window and internal scaling
to improve Levenberg–Marquardt conditioning on typical CCF shapes.

The fitted Gaussian center provides the RV estimate, and the code derives a corresponding ``RV_ERR``
using a combination of fit covariance, local CCF sampling information, and configuration-driven
heuristics (see :class:`~hispecdrp.primitives.utils.ccf_utils.CCFFitKpfConfig`). Several of these
heuristics are labeled “KPF” in-code to reflect published/known behavior patterns; this repository
does **not** include Keck KPF pipeline source.

Parabolic vertex / refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline can estimate the RV using a local parabolic-vertex method around the CCF peak. This is
used in two situations:

- **Post-fit refinement (default behavior)**: both the stellar-template and line-mask workflows can
  take an accepted Gaussian-derived RV and then apply a local refinement step in a narrow window
  around the peak.
- **Fallback when Gaussian fitting is rejected**: if Gaussian fitting fails acceptance checks, the
  workflow falls back to a peak-local estimate rather than returning no RV.

In the line-mask workflow specifically, the order-mean Gaussian fit can be followed by a
mask-line parabolic refinement step (see
:func:`~hispecdrp.primitives.utils.ccf_utils._refine_ccf_rv_after_gaussian_fit_ccf_line_mask`).
If Gaussian fitting is skipped or rejected, the parabolic vertex estimate is used directly for
``RV``/``RV_ERR`` on the relevant CCF curve.

For the template-CCF workflow, parabolic refinement behavior is controlled by the primitive argument
``ccf_parabolic_vertex_n_samples``: when set to an odd integer, the Gaussian-derived RV can be
refined locally; setting it to ``None`` disables that local refinement step (Gaussian-only).

Per-chunk ``RV_ERR`` combines gradient- and covariance-based estimates (see
:class:`~hispecdrp.primitives.utils.ccf_utils.CCFFitKpfConfig`) and can optionally use a parametric
bootstrap on ``ccf_err`` in the final fit window. Formal LM errors can fall below one velocity pixel
for well-sampled, high-SNR CCFs; bootstrap (when enabled) provides a more conservative floor from
diagonal ``ccf_err`` perturbations.


References
----------

.. [1] KPF-Pipeline radial_velocity ``src`` (mask-line tables and RV context; template path differs):
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
.. [2] KPF ``alg.py`` (masking algorithms in KPF):
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg.py
.. [3] spirou-ccf (SPIRou spectra / CCF ecosystem):
       https://github.com/edermartioli/spirou-ccf

