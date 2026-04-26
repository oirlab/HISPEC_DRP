"""
Stellar radial-velocity and cross-correlation helpers re-exported from ``ccf_utils``.

This package surface is the **public API** for template CCF, mask-line CCF, chunk
combination, and RVCCF preview helpers. Velocities are **m s\\ :sup:`−1`** unless a
function states otherwise.

**Exports**

``CCFFitKpfConfig``, ``MaskLines``
    Configuration and mask-line container types.

``fit_ccf_rv``, ``dv_per_pixel_m_s``, ``continuum_normalize``
    Gaussian fit to a CCF, rough per-pixel Δv, optional continuum division before CCF.

``masked_ccf_rv``, ``rv_per_chunks``, ``process_one_order_rvs``, ``process_orders_rvs_numba_parallel``, ``flat_chunk_echelle_orders``
    Log-λ weighted-covariance CCF paths (single spectrum, chunked, one order, many orders).

``combine_rvs_median_mad``, ``weighted_mean_rv``
    Median/MAD outlier mask and inverse-variance weighted mean on chunk RVs.

``load_mask``, ``resolve_mask_path``, ``restrict_mask_to_chunk``
    SPIROU / KPF-style mask loaders and wavelength restriction.

``run_ccf_mask_kpf_for_fiber``, ``build_ccf_preview_dict``
    Mask-line pipeline driver for one fiber and Plotly preview payload builder.

``mask_kpf_parabolic_vertex_rv_err_levmar_m_s``, ``mask_kpf_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s``
    Parabolic-vertex σ: LevMar ``Polynomial1D`` (+ optional ``ccf_err`` weights) and homoscedastic LSTSQ fallback.

See :mod:`hispecdrp.primitives.utils.ccf_utils` for full docstrings and references.
"""

from .ccf_utils import (
    CCFFitKpfConfig,
    MaskLines,
    build_ccf_preview_dict,
    combine_rvs_median_mad,
    continuum_normalize,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    flat_chunk_echelle_orders,
    load_mask,
    masked_ccf_rv,
    mask_kpf_parabolic_vertex_rv_err_levmar_m_s,
    mask_kpf_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    resolve_mask_path,
    restrict_mask_to_chunk,
    run_ccf_mask_kpf_for_fiber,
    rv_per_chunks,
    weighted_mean_rv,
)

__all__ = [
    "CCFFitKpfConfig",
    "MaskLines",
    "combine_rvs_median_mad",
    "continuum_normalize",
    "dv_per_pixel_m_s",
    "fit_ccf_rv",
    "flat_chunk_echelle_orders",
    "build_ccf_preview_dict",
    "load_mask",
    "masked_ccf_rv",
    "mask_kpf_parabolic_vertex_rv_err_levmar_m_s",
    "mask_kpf_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s",
    "process_one_order_rvs",
    "process_orders_rvs_numba_parallel",
    "resolve_mask_path",
    "restrict_mask_to_chunk",
    "run_ccf_mask_kpf_for_fiber",
    "rv_per_chunks",
    "weighted_mean_rv",
]
