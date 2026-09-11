===========================
Stellar RV primitive (L2)
===========================

The HISPEC DRP supports two cross-correlation based stellar radial-velocity (RV) algorithms
dispatched by :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`:

``ccf_stellar_template``
   Cross-correlate an observed spectrum against a (rest-frame) **template spectrum** in
   :math:`\ln\lambda` and fit the CCF peak to estimate RV.

``ccf_line_mask``
   Cross-correlate an observed spectrum against a **line mask** (a weighted list of line
   locations, optionally with finite widths), producing a mask statistic as a function of trial
   velocity and fitting its peak to estimate RV.

For line-mask **file formats**, :class:`~hispecdrp.utils.line_mask_utils.MaskLines` conventions, and
template-to-mask **generation**, see :doc:`line_mask`.

For **shared CCF machinery** (template and mask-line paths, peak fitting, and deriving RV from a CCF
curve in :mod:`~hispecdrp.primitives.utils.ccf.ccf_utils`), see :doc:`ccf`.


Primitive overview and references
---------------------------------

At a high level, :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`:

- reads ``SPEC_FIB*``, ``SPEC_ERR_FIB*``, ``SPEC_DQ_FIB*``, and ``WAVE_FIB*`` from the input L2 model,
- runs the selected algorithm per fiber and (optionally) a subset of echelle orders,
- writes ``RVCCF_FIB*`` / ``RVCCFAVG_FIB*`` (and optional CCF supplement columns) on the output L2 clone, and
- optionally writes/opens Plotly previews and overlay-debug HTML via
  :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`.

The ``ccf_line_mask`` implementation is inspired by public mask-line CCF descriptions and open code
in the community; HISPEC DRP does not ship Keck KPF pipeline source.

References
^^^^^^^^^^

* KPF mask-line algorithm reference (``alg_rv_mask_line.py``):
  https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg_rv_mask_line.py
* KPF radial_velocity module source tree:
  https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
* spirou-ccf helper library (``spiroulib.py``):
  https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py
* spirou-ccf example masks:
  https://github.com/edermartioli/spirou-ccf/tree/main/ccf_masks


Units and conventions
---------------------

- **Velocities**: internally, RVs and CCF velocity axes are in **m/s** across both algorithms.
  Some preview surfaces may *display* km/s for readability (labels/hover text), but stored values
  and primitive inputs/outputs remain m/s unless explicitly documented otherwise.

- **Wavelengths**: wavelength arrays and chunk centers are in **micron** (µm), and chunk sizing uses
  ``window_size_um`` (µm).

- **Tables**: outputs are written to ``RVCCF_FIB*`` (per-chunk) and ``RVCCFAVG_FIB*`` (fiber summary).
  Column units follow the table schema (typically RV in m/s and wavelength in µm).


RV tables and data storage (RVCCF)
----------------------------------

L2 datamodel formatting (code references)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RV tables are defined as L2 datamodel extensions in
:mod:`hispecdrp.datamodels.extensions`. The helper
:func:`~hispecdrp.datamodels.extensions.create_rv_extensions` maps these HDU schemas to the
fiber-specific HDU names (e.g. ``RVCCF_FIB1``).

The canonical column definitions are the module-level dictionaries:

- :data:`~hispecdrp.datamodels.extensions.RVCCF_HDU`
- :data:`~hispecdrp.datamodels.extensions.RVCCFAVG_HDU`

The stellar RV primitive writes two main product tables per fiber:

``RVCCF_FIB*``
   Per-chunk measurements. Each row corresponds to a wavelength chunk (or line-window segment
   when line-by-line chunking is used). The table includes:

   - ``RV`` / ``RV_ERR``: per-chunk RV estimate and uncertainty (m/s)
   - ``ORDER``: echelle order number
   - ``WAVELENGTH``: chunk center wavelength (µm)
   - ``INLIER``: robust inlier mask used for aggregation
   - optional CCF supplement columns (velocity axis metadata and CCF arrays) when enabled

``RVCCFAVG_FIB*``
   Aggregate RV and uncertainty for the fiber. The RV combination uses robust outlier rejection
   (median/MAD) followed by an inverse-variance weighted mean; the reported uncertainty is
   :math:`\\max(\\sigma_{\\mathrm{IVW}},\\,\\mathrm{MAD}/\\sqrt{N})`. In the infinite-S/N
   limit this converges to the empirical ``MAD/√N`` floor.

For details of the combination logic, see
:func:`~hispecdrp.primitives.stellar_rv.rv_combine_utils.combine_rv_estimates_median_mad_weighted`.

Preview plots
-------------

The pipeline provides two preview surfaces:

Unified Plotly preview
   The standard stellar-RV preview (RV vs wavelength, CCF + Gaussian fits, residuals) is produced
   by :func:`~hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview.create_stellar_rv_preview`.

Overlay debug HTML
   Interactive spectrum + CCF overlay HTML for debugging mask-line (and template) behavior. For
   ``ccf_line_mask``, the spectrum panel shows the observed spectrum with shifted mask segments,
   and a dense Lagrange-interpolated spectrum trace using the same interpolation order as the mask
   CCF.

The overlay debug page is written by
:func:`~hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview.write_mask_overlay_debug_pages`.


Workflow outline (``ccf_line_mask``)
------------------------------------

This is a high-level outline of what :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`
does when ``algorithm="ccf_line_mask"`` for one fiber (the template-CCF algorithm follows a different path).

#. **Entry point**

   :meth:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive._run_algorithm_ccf_line_mask`
   resolves coarse velocity grid defaults (``v_min_coarse``, ``v_max_coarse``, ``v_step_coarse``),
   optional centered-window configuration around ``ccf_input_velocity`` (``v_half_window``),
   loads mask lines once, then calls the fiber driver:

   - :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber`

#. **Velocity axis**

   The fiber driver constructs a **uniform** velocity grid:

   - ``ccf_input_velocity ± v_half_window`` at ``v_step_coarse`` when a center is set.

   When ``ccf_input_velocity`` is unset, an **auto-center scout** runs one pass on the
   broad uniform grid to estimate the peak, then rebuilds a centered uniform window for
   the fiber.

#. **Per-order chunking and mask CCF**

   For each echelle order in ``do_orders``, the driver computes wavelength chunks:

   - fixed-width chunks via :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`, or
   - line-by-line Voronoi chunks when ``window_line_by_line=True`` and ``ccf_input_velocity`` is finite
     (see :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.build_line_mask_chunk_wavelength_intervals_line_by_line`).

   The per-order worker calls:

   - :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`

   which runs the Numba kernel over all chunks (parallel ``prange`` over chunk index when applicable):

   - :func:`~hispecdrp.primitives.utils.ccf.ccf_utils._calc_mask_ccf_chunks_numba`

   producing per-chunk arrays ``CCF(v)`` and (when requested) ``ccf_err(v)``.

#. **Per-chunk RV and uncertainty**

   Per-chunk RVs come from a Gaussian fit to the (stored-convention) CCF:

   - Line-mask path: :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`
     via :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`
     (Numba Lagrange sampler; ``mask_ccf_pixel_interp_order`` / ``max_cores``)
   - Legacy / template helpers: :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`

   The reported chunk ``RV`` / ``RV_ERR`` are the Gaussian centroid and its
   photon-weighted Gaussian uncertainty (interpolated science-pixel errors,
   oversampling-corrected). Fiberflat / blaze terms in ``SPEC_ERR`` are not
   used as independent CCF-bin noise. Per-chunk ``RV_ERR`` decreases with S/N
   and converges to a numerical floor in the noiseless limit.
   A discrete-peak Numba seed may be used only when the Gaussian fit does not succeed.

   **CCF sign convention:** stored CCF rows (RVCCF / preview slots) follow the negated convention
   so the absorption-like feature appears as a **peak** near the true RV (see
   :data:`~hispecdrp.primitives.utils.ccf.ccf_utils.LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`).

#. **Fiber-level aggregation**

   Chunk-level estimates are robustly combined into a fiber RV using:

   - :func:`~hispecdrp.primitives.stellar_rv.rv_combine_utils.combine_rv_estimates_median_mad_weighted`

   with fallback to per-chunk MAD rejection + inverse-variance mean when needed.

#. **Previews and RVCCF supplement**

   When previews are requested, the driver builds preview slots and runs per-slot Gaussian/refine:

   - :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.finalize_ccf_preview_dict`

   When requested for table outputs, the primitive may also build RVCCF supplement columns:

   - :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.build_rvccf_chunk_supplement`


Additional notes (primitive behavior and details)
-------------------------------------------------

- **Default coarse CCF step**: when ``v_step_coarse`` is unset, the default coarse step is
  **50 m/s** (``ccf_common.V_STEP_DEFAULT_M_S``). Very long velocity grids may be subsampled inside
  :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv` to keep fitting robust and fast.

- **Overlay debug HTML chunking and gating**: the mask/template overlay debug pages list wavelength
  chunks (matching ``window_size_um`` chunking, or line-by-line Voronoi bins when
  ``window_line_by_line=True``) and gate chunk inclusion with ``ccf_min_good_pixels`` and
  data quality flags (e.g. ``DO_NOT_USE``), consistent with the underlying chunk drivers.

- **Overlay embedded array downsampling (display-only)**: for the **overlay debug preview HTML**
  only, embedded arrays (spectrum / CCF / template traces) are downsampled before being written into
  the page to keep file sizes reasonable. This affects **what is plotted in the browser**, not the
  underlying pipeline arrays or RVCCF products. Limits are controlled by the environment variables
  ``HISPEC_DRP_MASK_OVERLAY_SPECTRUM_MAX_POINTS``,
  ``HISPEC_DRP_MASK_OVERLAY_CCF_MAX_POINTS``, and ``HISPEC_DRP_TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS``
  (see :mod:`~hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`).

- **Dense interpolated spectrum trace in overlay debug HTML**: the overlay debug page always
  includes a dense Lagrange-interpolated spectrum trace using the same interpolation order as the
  mask CCF (``mask_ccf_pixel_interp_order``). See “Preview plots” above for the dense grid formula.

- **Overlay initial velocity**: the overlay page’s initial velocity matches finite
  ``ccf_input_velocity`` when set; otherwise it defaults to ``0`` m/s.


Unit tests and command-line overrides
-------------------------------------

The main integration tests for stellar RV live in ``tests/stage2/test_stellar_rv_primitive.py``.

Useful command-line / environment overrides used by these tests include:

``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1``
   Single switch for optional stellar-RV previews in ``tests/stage2/test_stellar_rv_primitive.py``: after BT-Settl BSPEC
   integration tests, write and open preview HTML (mask-overlay + unified Plotly for ``ccf_line_mask``,
   and unified Plotly + template-overlay HTML for ``ccf_stellar_template`` when that optional leg runs).
   Also enables the unified Plotly preview in ``test_stellar_rv_primitive_recovers_rv_per_chunk``. Forces
   HTML-only behavior for the unified Plotly preview (skips Kaleido PNG export when this env is set;
   see :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`). When unset, those optional
   preview legs are skipped.

``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER=<int>``
   Override the mask CCF interpolation order used by the tests (and reflected in the overlay
   debug spectrum interpolation trace label).


``HISPEC_DRP_BSPEC_BTSETTL_ORDER``
   Which BSPEC echelle orders the BT-Settl BSPEC tests use. **Default** when unset is ``139``
   (single order). Set to ``all`` to use every BSPEC order from ``HISPECConfig`` (same as the former
   ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1``). Otherwise a comma-separated list of integer echelle
   orders (e.g. ``139,140``).

``HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM=<float>``
   Override the chunk width (µm) used in the BT-Settl BSPEC tests.

``HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS=1``
   Relax strict RV recovery / benchmark assertions for exploratory runs (previews still run).


See also
--------

- :mod:`hispecdrp.primitives.utils.ccf.ccf_utils` — template CCF + mask-line CCF
- :mod:`hispecdrp.primitives.utils.spectral_utils`
- :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview` — preview generation


API references
--------------

.. autoclass:: hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive
   :members:
   :undoc-members:
   :no-index:

