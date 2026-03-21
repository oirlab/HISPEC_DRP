"""Utilities shared across primitives (e.g. CCF for cross-correlation)."""
# Provenance notes: see ccf_utils and ccf_mask_kpf_utils module top comments.

from .ccf_utils import (
    _HAS_NUMBA,
    CCFFitKpfConfig,
    combine_rvs_median_mad,
    continuum_normalize,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    masked_ccf_rv,
    masked_ccf_rv_diagnostics,
    masked_ccf_rv_full,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    rv_per_chunks,
    rv_per_chunks_with_ccf,
    weighted_mean_rv,
)

__all__ = [
    "_HAS_NUMBA",
    "CCFFitKpfConfig",
    "combine_rvs_median_mad",
    "continuum_normalize",
    "dv_per_pixel_m_s",
    "fit_ccf_rv",
    "masked_ccf_rv",
    "masked_ccf_rv_diagnostics",
    "masked_ccf_rv_full",
    "process_one_order_rvs",
    "process_orders_rvs_numba_parallel",
    "rv_per_chunks",
    "rv_per_chunks_with_ccf",
    "weighted_mean_rv",
]
