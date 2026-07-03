"""
Cross-correlation (CCF) and radial-velocity utilities.

Implementation lives in :mod:`~hispecdrp.primitives.utils.ccf.ccf_utils`.
See :mod:`hispecdrp.primitives.utils` for the stable re-export surface.
"""

from .ccf_rv_content import (
    compute_rv_content,
    doppler_shift_from_barycenter,
)
from .ccf_spline import compute_ccf_spline, cubic_spline_build, cubic_spline_eval_scalar
from .ccf_tophat import TOPHAT_MASK_CCF_SEGMENT_SAMPLES, compute_ccf_tophat
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

__all__ = [
    "CCFBisectorConfig",
    "CCFFitConfig",
    "build_ccf_preview_dict",
    "compute_ccf_spline",
    "compute_ccf_tophat",
    "compute_rv_content",
    "continuum_normalize",
    "cubic_spline_build",
    "cubic_spline_eval_scalar",
    "doppler_shift_from_barycenter",
    "dv_per_pixel_m_s",
    "fit_ccf_rv",
    "fit_ccf_rv_bisector",
    "flat_chunk_echelle_orders",
    "masked_ccf_rv",
    "line_mask_parabolic_vertex_rv_err_levmar_m_s",
    "line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s",
    "process_one_order_rvs",
    "process_orders_rvs_numba_parallel",
    "run_ccf_line_mask_for_fiber",
    "rv_per_chunks",
    "TOPHAT_MASK_CCF_SEGMENT_SAMPLES",
]
