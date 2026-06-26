"""
Stellar radial-velocity and cross-correlation helpers.

Most symbols come from ``ccf_utils``; robust chunk-RV combination helpers live in
``hispecdrp.primitives.stellar_rv.rv_combine_utils`` and are re-exported here for a
stable ``hispecdrp.primitives.utils`` surface.

This package surface is the **public API** for template CCF, mask-line CCF, chunk
combination, and RVCCF preview helpers. Velocities are **m s\\ :sup:`−1`** unless a
function states otherwise.

**Exports**

``CCFFitConfig``
    Gaussian CCF fit configuration.

``MaskLines``, ``MaskLineStyle`` (from :mod:`hispecdrp.utils.line_mask_utils`)
    Tabulated line-mask container (**delta** / **tophat** geometry) and style literal.
    Use :meth:`~MaskLines.load` for disk masks (``.mas`` or two-column text).

``fit_ccf_rv``, ``dv_per_pixel_m_s``, ``continuum_normalize``
    Gaussian fit to a CCF, rough per-pixel Δv, optional continuum division before CCF.

``masked_ccf_rv``, ``rv_per_chunks``, ``process_one_order_rvs``, ``process_orders_rvs_numba_parallel``, ``flat_chunk_echelle_orders``
    Log-λ weighted-covariance CCF paths (single spectrum, chunked, one order, many orders).

``combine_rvs_median_mad``, ``weighted_mean_rv``
    Median/MAD outlier mask and inverse-variance weighted mean on chunk RVs
    (implemented in :mod:`hispecdrp.primitives.stellar_rv.rv_combine_utils`).

``resolve_mask_path``, ``restrict_mask_to_chunk``, ``stellar_rv_ccf_mask_lines``, ``pixel_edges_from_wavelength``
    Packaged-mask resolution, chunk restriction, StellarRV line-mask inputs, pixel edges (from :mod:`hispecdrp.utils.line_mask_utils`).

``DEFAULT_PACKAGED_LINE_MASK_BASENAME``, ``packaged_default_line_mask_path``
    Packaged default **delta** line mask for ``mask_path=None`` (see ``hispecdrp/data/masks/``).
    Optional ``*.tophat.mas`` segment masks are produced by ``regenerate_template_masks.py`` only;
    they are not part of the default RV API.

``run_ccf_line_mask_for_fiber``, ``build_ccf_preview_dict``
    Line-mask CCF driver for one fiber (``StellarRVPrimitive(algorithm="ccf_line_mask")``) and Plotly preview payload builder.

``line_mask_parabolic_vertex_rv_err_levmar_m_s``, ``line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s``
    Parabolic-vertex σ: LevMar ``Polynomial1D`` (+ optional ``ccf_err`` weights) and homoscedastic LSTSQ fallback.

See :mod:`hispecdrp.primitives.utils.ccf_utils` for full docstrings and references.
"""

from hispecdrp.utils.line_mask_utils import (
    DEFAULT_PACKAGED_LINE_MASK_BASENAME,
    MaskLineStyle,
    MaskLines,
    packaged_default_line_mask_path,
    pixel_edges_from_wavelength,
    resolve_mask_path,
    restrict_mask_to_chunk,
    stellar_rv_ccf_mask_lines,
)

from .ccf_utils import (
    CCFBisectorConfig,
    CCFFitConfig,
    build_ccf_preview_dict,
    continuum_normalize,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    fit_ccf_rv_bisector,
    flat_chunk_echelle_orders,
    masked_ccf_rv,
    line_mask_parabolic_vertex_rv_err_levmar_m_s,
    line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    run_ccf_line_mask_for_fiber,
    rv_per_chunks,
)

from ..stellar_rv.rv_combine_utils import combine_rvs_median_mad, weighted_mean_rv

from .gaia_query import *

__all__ = [
    "CCFBisectorConfig",
    "CCFFitConfig",
    "DEFAULT_PACKAGED_LINE_MASK_BASENAME",
    "MaskLineStyle",
    "MaskLines",
    "combine_rvs_median_mad",
    "continuum_normalize",
    "dv_per_pixel_m_s",
    "fit_ccf_rv",
    "fit_ccf_rv_bisector",
    "flat_chunk_echelle_orders",
    "build_ccf_preview_dict",
    "masked_ccf_rv",
    "packaged_default_line_mask_path",
    "pixel_edges_from_wavelength",
    "line_mask_parabolic_vertex_rv_err_levmar_m_s",
    "line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s",
    "process_one_order_rvs",
    "process_orders_rvs_numba_parallel",
    "resolve_mask_path",
    "restrict_mask_to_chunk",
    "stellar_rv_ccf_mask_lines",
    "run_ccf_line_mask_for_fiber",
    "rv_per_chunks",
    "weighted_mean_rv",
]
