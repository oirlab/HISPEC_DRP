=======================
Cross-correlation (CCF)
=======================

This page summarizes the CCF algorithms and fitting machinery implemented in
:mod:`hispecdrp.primitives.utils.ccf.ccf_utils` (and used by both stellar-RV algorithms).


CCF utilities overview (``ccf_utils``)
--------------------------------------

:mod:`hispecdrp.primitives.utils.ccf.ccf_utils` contains the core CCF and RV estimation machinery for both
stellar-RV algorithms:

- **Template CCF** (``ccf_stellar_template``): weighted-covariance CCF in :math:`\ln\lambda`,
  Numba multi-order execution, continuum normalization, robust chunk aggregation (median/MAD) and
  inverse-variance combination.

- **Line-mask CCF** (``ccf_line_mask``): SPIROU/KPF-style mask-line CCF driver
  (:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber`) and supporting kernels.
  For the mask-line path, per-chunk ``RV``/``RV_ERR`` use bisector refinement after an accepted
  Gaussian fit (when bisector succeeds); Gaussian parameters remain available in preview metadata.


Deriving velocities from a CCF
------------------------------

Once you have a CCF curve (velocity axis ``v`` and CCF samples ``CCF(v)``), HISPEC DRP provides
generalized routines in :mod:`hispecdrp.primitives.utils.ccf.ccf_utils` to infer an RV (and an
uncertainty) in a way that can be applied to CCFs produced by **either** algorithm
(``ccf_stellar_template`` or ``ccf_line_mask``).

Gaussian peak fitting
^^^^^^^^^^^^^^^^^^^^^

The peak-fit routine :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv` fits
a Gaussian + constant baseline model to :math:`\mathrm{CCF}(v)` (Astropy ``Gaussian1D`` + ``Const1D``
with ``LevMarLSQFitter``). The implementation uses a two-pass fitting window and internal scaling
to improve Levenberg–Marquardt conditioning on typical CCF shapes.

Gaussian parameters (center, width, amplitude, baseline) are always computed and stored in preview
and debug output. The **reported** chunk RV is not taken from the Gaussian center alone; see
bisector refinement below.

The Gaussian fit also derives an ``RV_ERR`` from the Levenberg–Marquardt
covariance of the Gaussian mean, using **interpolated science-pixel** CCF errors
(:math:`\\sigma_{\\mathrm{meas}}`) and an oversampling factor so velocity bins that
share a detector pixel are not treated as independent. Fiberflat / blaze terms in
``SPEC_ERR`` are omitted from those weights (they are common-mode in wavelength).
Inverse-variance weights are renormalized before the Hessian so ``σ_μ`` decreases
with S/N and converges to a numerical floor as ``σ_meas → 0`` (it does not
explode at infinite S/N).

Gaussian CCF centroid (final estimate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After an accepted Gaussian fit to the CCF
(:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv` or
:func:`~hispecdrp.primitives.utils.ccf.ccf_fitting.calc_ccf_metrics`), the Gaussian mean
(``v0`` / ``metrics.rv``) and its Gaussian uncertainty are the final ``RV``/``RV_ERR`` on each
chunk. Bisector quantities remain available for diagnostics only and do not replace the reported RV.

Per-chunk ``RV_ERR`` combines gradient- and covariance-based estimates (see
:class:`~hispecdrp.primitives.utils.ccf.ccf_utils.CCFFitConfig`) and can optionally use a parametric
bootstrap on ``ccf_err`` in the final fit window. Formal LM errors can fall below one velocity pixel
for well-sampled, high-SNR CCFs; bootstrap (when enabled) provides a more conservative floor from
diagonal ``ccf_err`` perturbations.


References
----------

* KPF-Pipeline radial_velocity ``src`` (mask-line tables and RV context; template path differs):
  https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
* KPF ``alg.py`` (masking algorithms in KPF):
  https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg.py
* spirou-ccf (SPIRou spectra / CCF ecosystem):
  https://github.com/edermartioli/spirou-ccf

