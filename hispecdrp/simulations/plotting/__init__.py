"""Validation / diagnostic plots for HISPEC simulation notebooks and scripts."""

from .barnards_star_prv import plot_injected_vs_recovered_rv
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
from .stage0_rate_noise import (
    DEFAULT_STAGE0_PULL_DQ_EXCLUDE,
    DEFAULT_STAGE0_PULL_MIN_TRUTH,
    diagnose_stage0_rate_noise,
    plot_stage0_rate_pull_histogram,
    robust_gaussian_params,
    stage0_rate_pulls,
    stage0_truth_scaled_rate_err,
    summarize_stage0_rate_residuals,
)

__all__ = [
    "DEFAULT_STAGE0_PULL_DQ_EXCLUDE",
    "DEFAULT_STAGE0_PULL_MIN_TRUTH",
    "PREVIEWS_SUBDIR",
    "REPORT_HTML_NAME",
    "PreviewItem",
    "analyze_line_bias_and_plot",
    "compute_line_bias_table",
    "diagnose_stage0_rate_noise",
    "generate_e2e_preview_report",
    "infer_exposure_base_name",
    "plot_injected_vs_recovered_rv",
    "plot_stage0_rate_pull_histogram",
    "robust_gaussian_params",
    "stage0_rate_pulls",
    "stage0_truth_scaled_rate_err",
    "summarize_stage0_rate_residuals",
    "write_bias_cull_histograms",
    "write_comb_wavesol_order_residual_png",
    "write_comb_wavesol_per_fiber_summary_png",
    "write_comb_wavesol_per_order_summary_png",
    "write_lfc_elliptical_wave_report",
    "write_per_order_rv_scan_report",
]
