"""Validation / diagnostic plots for HISPEC simulation notebooks and scripts."""

from .ccfopt_bias_cull import (
    analyze_line_bias_and_plot,
    compute_line_bias_table,
    write_bias_cull_histograms,
)
from .comb_wavesol_geometry import (
    write_comb_wavesol_order_residual_png,
    write_comb_wavesol_per_fiber_summary_png,
    write_comb_wavesol_per_order_summary_png,
)
from .e2e_preview import (
    PREVIEWS_SUBDIR,
    REPORT_HTML_NAME,
    PreviewItem,
    generate_e2e_preview_report,
    infer_exposure_base_name,
)
from .lfc_wavesol_refit import write_lfc_elliptical_wave_report
from .rv_order_scan import write_per_order_rv_scan_report

__all__ = [
    "PREVIEWS_SUBDIR",
    "REPORT_HTML_NAME",
    "PreviewItem",
    "analyze_line_bias_and_plot",
    "compute_line_bias_table",
    "generate_e2e_preview_report",
    "infer_exposure_base_name",
    "write_bias_cull_histograms",
    "write_comb_wavesol_order_residual_png",
    "write_comb_wavesol_per_fiber_summary_png",
    "write_comb_wavesol_per_order_summary_png",
    "write_lfc_elliptical_wave_report",
    "write_per_order_rv_scan_report",
]
