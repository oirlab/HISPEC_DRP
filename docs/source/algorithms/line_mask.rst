==========
Line masks
==========

This page documents the line-mask formats used by ``ccf_line_mask``, the corresponding utilities in
:mod:`hispecdrp.utils.line_mask_utils`, and the unit tests that validate mask I/O and mask-building
behaviors.


Line mask formats (``mask_style``)
----------------------------------

The line-mask algorithm uses :class:`~hispecdrp.utils.line_mask_utils.MaskLines` as its canonical
in-memory representation and mask file I/O layer. Masks are stored on disk as ``.mas`` files and
loaded via :meth:`~hispecdrp.utils.line_mask_utils.MaskLines.load`.

Mask metadata header (regeneration provenance)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HISPEC DRP ``.mas`` files may include a single-line JSON metadata header beginning with the prefix
``# HISPEC_DRP_MASK_META``. This metadata is intended to be **regeneration provenance**: it records
the key parameters used to build the mask from the input template (e.g. selection thresholds and
filters) along with essential format/interpretation fields such as ``schema_version``,
``mask_style``, and the wavelength-column conventions used on disk vs in-memory.

Two mask geometries are supported:

``mask_style="delta"``
   Each mask entry is a **zero-width** line at a single rest wavelength. In the mask CCF, each
   line contributes via interpolation of the spectrum onto the Doppler-shifted line center.

``mask_style="tophat"``
   Each mask entry is a **finite-width segment** with rest-frame edges
   :math:`[\lambda_{\rm lo}, \lambda_{\rm hi}]`. In the mask CCF, each segment contributes by
   evaluating the spectrum across the full shifted segment width (i.e. a segment-average
   statistic per trial velocity), rather than a single-point sample.

The bundled masks live under ``hispecdrp/data/masks/``. The packaged default mask used when
``mask_path=None`` is controlled by
:func:`~hispecdrp.utils.line_mask_utils.packaged_default_line_mask_path`.


Line-mask utilities (``line_mask_utils``)
-----------------------------------------

:mod:`hispecdrp.utils.line_mask_utils` provides the line-mask file I/O, in-memory representation,
and template-to-mask tooling used by ``ccf_line_mask``.

- **Template mask generation**: loads a stellar template (HDF5 via
  :mod:`hispecdrp.primitives.utils.spectral_utils`), detects absorption centers (e.g.
  :func:`~hispecdrp.utils.line_mask_utils.detect_template_line_centers_gradient_um`),
  applies depth/deblending/telluric filters, and writes a **delta** mask via
  :func:`~hispecdrp.utils.line_mask_utils.build_line_mask_from_template` (two on-disk columns:
  vacuum-nm center, weight; weights sum to 1). Generation parameters are stored as a JSON line with
  prefix ``# HISPEC_DRP_MASK_META`` (see :func:`~hispecdrp.utils.line_mask_utils.read_mask_generation_meta`).

- **In-memory mask model**: :class:`~hispecdrp.utils.line_mask_utils.MaskLines` holds either
  **delta** (zero-width) or **tophat** (finite segment) geometry in vacuum **µm**, with shared
  :attr:`~hispecdrp.utils.line_mask_utils.MaskLines.metadata` conventions.
  Use :meth:`~hispecdrp.utils.line_mask_utils.MaskLines.load` /
  :meth:`~hispecdrp.utils.line_mask_utils.MaskLines.save` for disk I/O (``.mas`` and two-column
  text). Saved masks can use the auto-naming pattern ``{basename}.{mask_style}.mas``.

- **CCF integration**: the CCF driver uses :attr:`~hispecdrp.utils.line_mask_utils.MaskLines.line_lambda_um`
  (rest centers in µm). Two-column ``.mas`` files store line-center nm; ``.mas`` files with ≥3 columns
  store segment edges. Plain ``.txt`` / ``.dat`` / ``.csv`` inputs are treated as KPF-style two-column
  lists (wavelength units inferred).

See ``hispecdrp/data/masks/README.md`` and ``hispecdrp/data/atmosphere/README.md`` for bundled-mask
and transmission-file notes.


Unit tests (``tests/stage2/test_line_mask_utils.py``)
----------------------------------------------

The unit tests in ``tests/stage2/test_line_mask_utils.py`` validate both small mathematical helpers and
end-to-end mask I/O behaviors. Highlights:

- **Bundled data access (optional)**

  - ``test_load_mktrans_bundled`` and ``test_mean_transmission_interval`` validate atmospheric
    transmission file loading and a simple velocity-interval mean.
  - ``test_fit_half_width_k2_all_matches_default_constant`` and
    ``test_masklines_tophat_geometry_from_bundled_template_tophat`` exercise bundled ``*.tophat.mas`` masks
    when present; tests skip when those optional files are not packaged.

- **Metadata conventions**

  - ``test_parse_mask_generation_meta_line`` validates parsing of the ``# HISPEC_DRP_MASK_META`` JSON line.

- **Line-center detection math and helpers**

  - ``test_parabolic_depth_uses_flux_continuum_ratio_not_interp_of_depth`` ensures refined depth is
    computed as \(1 - F(\lambda)/F_c(\lambda)\) with interpolated \(F\) and \(F_c\), not by
    interpolating the ratio-derived depth grid.
  - ``test_continuum_normalized_d2_matches_parabolic_curvature`` validates the curvature estimator
    for a known parabolic profile.
  - ``test_cluster_max_depth_prefers_deeper`` validates clustering behavior by keeping the deepest
    line in a blended group.
  - ``test_local_min_continuum_normalized_flux_um_window`` validates the local-min window logic.
  - ``test_segment_mean_depth_averages_over_interval`` and
    ``test_segment_mean_depth_empty_interval_is_nan`` validate segment averaging behavior.

- **MaskLines I/O and geometry**

  - ``test_masklines_delta_zero_width_and_restrict`` validates delta-mask zero widths and
    chunk restriction behavior.
  - ``test_masklines_load_two_column_mas`` validates the two-column ``.mas`` reader.
  - ``test_masklines_load_mask_style_tophat_requires_three_columns`` /
    ``test_masklines_load_mask_style_delta_requires_two_columns`` /
    ``test_masklines_load_unknown_mask_style_raises`` validate explicit ``mask_style`` metadata rules.

- **Template-to-mask build**

  - ``test_build_line_mask_from_template_two_columns`` validates mask writing, metadata presence,
    column schema, and weight normalization for template-built masks.

