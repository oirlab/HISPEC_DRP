"""
Cross-correlation (CCF) and radial-velocity utilities.

Package layout (implementation modules under :mod:`hispecdrp.primitives.utils.ccf`):

* :mod:`~hispecdrp.primitives.utils.ccf.ccf_common` — velocity grids, ``z_b``, shared stats
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_rv_content` — photon-limited RV content, Doppler λ map
* :mod:`~hispecdrp.primitives.utils.ccf.cubic_spline` — Numba not-a-knot cubic spline on spectra
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_spline` — delta-mask CCF via spline spectrum sampling
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_tophat` — finite-width top-hat mask CCF
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_bisector` — Queloz BIS + NEID weighted-percentile bisector RV
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_parabolic` — parabolic-vertex / coarse-grid CCF RV refinement
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_fitting` — Gaussian line metrics on a CCF (:class:`CCFMetrics`)
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_line_mask_spline` — stellar line-mask CCF via spline + :func:`calc_ccf_metrics`
* :mod:`~hispecdrp.primitives.utils.ccf.ccf_utils` — template/mask CCF orchestration, Gaussian CCF fit, previews

See :mod:`hispecdrp.primitives.utils` for the stable re-export surface.
"""

from .ccf_rv_content import (
    compute_rv_content,
    doppler_shift_from_barycenter,
)
from .ccf_spline import compute_ccf_spline
from .ccf_tophat import TOPHAT_MASK_CCF_SEGMENT_SAMPLES, compute_ccf_tophat
from .cubic_spline import (
    cubic_spline_build,
    cubic_spline_eval,
    cubic_spline_eval_scalar,
)
from .ccf_parabolic import (
    CCFParabolicConfig,
    fit_ccf_rv_parabolic,
    refine_ccf_rv_coarse_grid_parabolic,
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

__all__ = [
    "CCFBisectorConfig",
    "CCFFitConfig",
    "CCFParabolicConfig",
    "build_ccf_preview_dict",
    "compute_ccf_spline",
    "compute_ccf_tophat",
    "compute_rv_content",
    "continuum_normalize",
    "cubic_spline_build",
    "cubic_spline_eval",
    "cubic_spline_eval_scalar",
    "doppler_shift_from_barycenter",
    "dv_per_pixel_m_s",
    "fit_ccf_rv",
    "fit_ccf_rv_bisector",
    "fit_ccf_rv_parabolic",
    "flat_chunk_echelle_orders",
    "masked_ccf_rv",
    "line_mask_parabolic_vertex_rv_err_levmar_m_s",
    "line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s",
    "process_one_order_rvs",
    "process_orders_rvs_numba_parallel",
    "run_ccf_line_mask_for_fiber",
    "rv_per_chunks",
    "refine_ccf_rv_coarse_grid_parabolic",
    "TOPHAT_MASK_CCF_SEGMENT_SAMPLES",
]
