"""
Infinite signal-to-noise RV validation for end-to-end simulations.

Configures the E2E driver for noiseless rate maps, HDF-truth Stage-1 calibrations,
and high-precision ``StellarRVPrimitive`` settings, then evaluates recovery against
the injected Gaia barycentric RV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import datamodels
from ..utils.line_mask_utils import paired_line_mask_path_for_template
from .e2e_pipeline import E2EConfig
from .sim_hdf_calibrations import STAGE1_TRACE_SOURCE_SIM_HDF

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE1_TRACE_SOURCE_SIM_HDF",
    "InfiniteSnrRvThresholds",
    "DEFAULT_INFINITE_SNR_RV_THRESHOLDS",
    "HIGH_PRECISION_STELLAR_RV_ARGS",
    "apply_infinite_snr_rv_e2e_config",
    "merge_infinite_snr_stage1_prim_args",
    "merge_infinite_snr_stage2_prim_args",
    "evaluate_infinite_snr_rv_recovery",
    "assert_infinite_snr_rv_recovery",
    "log_infinite_snr_rv_evaluation",
    "default_infinite_snr_e2e_config",
]

HIGH_PRECISION_STELLAR_RV_ARGS: dict[str, Any] = {
    "algorithm": "ccf_line_mask",
    "mask_ccf_pixel_interp_order": 1,
    "mask_auto_center_scout": False,
    "window_line_by_line": True,
    "v_half_window": 500.0,
    "v_step_coarse": 0.01,
    "chunk_rv_prior_half_width_m_s": 3000.0,
    "outlier_sigma": 4.0,
    "preview_write_png": False,
    "save_overlay_debug": False,
    "save_preview": False,
}


@dataclass(frozen=True)
class InfiniteSnrRvThresholds:
    max_abs_bias_km_s: float = 0.20
    max_fiber_bias_spread_km_s: float = 0.15
    min_inlier_chunks: int = 3


DEFAULT_INFINITE_SNR_RV_THRESHOLDS = InfiniteSnrRvThresholds()


@dataclass
class InfiniteSnrRvFiberResult:
    fiber: int
    measured_rv_m_s: float | None
    rv_err_m_s: float | None
    bias_m_s: float | None
    bias_km_s: float | None
    n_inlier_chunks: int
    n_total_chunks: int
    inlier_median_bias_m_s: float | None


@dataclass
class InfiniteSnrRvEvaluation:
    injected_rv_km_s: float
    injected_rv_m_s: float
    fibers: list[InfiniteSnrRvFiberResult]
    max_abs_bias_km_s: float | None
    fiber_bias_spread_km_s: float | None

    @property
    def passed_default_thresholds(self) -> bool:
        th = DEFAULT_INFINITE_SNR_RV_THRESHOLDS
        if self.max_abs_bias_km_s is None or self.fiber_bias_spread_km_s is None:
            return False
        if self.max_abs_bias_km_s > th.max_abs_bias_km_s:
            return False
        if self.fiber_bias_spread_km_s > th.max_fiber_bias_spread_km_s:
            return False
        if any(r.n_inlier_chunks < th.min_inlier_chunks for r in self.fibers):
            return False
        return True


def apply_infinite_snr_rv_e2e_config(config: E2EConfig) -> E2EConfig:
    from dataclasses import replace

    mask_path = config.stellar_rv_mask_path
    if mask_path is None and config.template_filename:
        try:
            mask_path = paired_line_mask_path_for_template(config.template_filename)
        except FileNotFoundError:
            logger.warning(
                "Infinite-S/N RV: paired mask for template %s not found; using default mask.",
                config.template_filename,
            )

    return replace(
        config,
        infinite_snr_rv_validation=True,
        poisson_noise=False,
        convert_to_uint16=False,
        stage1_trace_source=STAGE1_TRACE_SOURCE_SIM_HDF,
        stellar_rv_mask_path=mask_path,
        make_preview_html=False,
    )


def default_infinite_snr_e2e_config(output_dir: str, **kwargs: Any) -> E2EConfig:
    base = E2EConfig(output_dir=output_dir, **kwargs)
    return apply_infinite_snr_rv_e2e_config(base)


def merge_infinite_snr_stage1_prim_args(
    prim_args: dict | None,
    *,
    extraction_aperture: int = 10,
) -> dict:
    import copy

    out = copy.deepcopy(prim_args or {})
    out.setdefault("spec_extract", {})
    out["spec_extract"]["method"] = "box"
    out["spec_extract"]["extraction_aperture"] = int(extraction_aperture)
    return out


def merge_infinite_snr_stage2_prim_args(prim_args: dict | None) -> dict:
    import copy

    out = copy.deepcopy(prim_args or {})
    out.setdefault("stellar_rv", {})
    out["stellar_rv"].update(HIGH_PRECISION_STELLAR_RV_ARGS)
    return out


def evaluate_infinite_snr_rv_recovery(
    stage2_model: datamodels.DataModel | None,
    *,
    injected_rv_km_s: float,
    fibers: list[int] | None = None,
) -> InfiniteSnrRvEvaluation:
    injected_m_s = float(injected_rv_km_s) * 1000.0
    if stage2_model is None:
        return InfiniteSnrRvEvaluation(
            injected_rv_km_s=float(injected_rv_km_s),
            injected_rv_m_s=injected_m_s,
            fibers=[],
            max_abs_bias_km_s=None,
            fiber_bias_spread_km_s=None,
        )
    if fibers is None:
        fibers = [1, 2, 3]
    rows: list[InfiniteSnrRvFiberResult] = []
    biases_km: list[float] = []
    for fib in fibers:
        avg = getattr(stage2_model, f"rvccfavg_fib{fib}", None)
        chunks = getattr(stage2_model, f"rvccf_fib{fib}", None)
        meas = err = bias_m_s = None
        n_in = n_tot = 0
        inlier_med = None
        if avg is not None and len(avg) > 0:
            meas = float(avg["RV"][0])
            err = float(avg["RV_ERR"][0])
            if np.isfinite(meas):
                bias_m_s = meas - injected_m_s
                biases_km.append(bias_m_s / 1000.0)
        if chunks is not None and len(chunks) > 0:
            rv_c = np.asarray(chunks["RV"], dtype=np.float64)
            n_tot = int(rv_c.size)
            chunk_names = getattr(chunks, "colnames", None)
            if chunk_names is None and hasattr(chunks, "dtype"):
                chunk_names = chunks.dtype.names
            if chunk_names and "INLIER" in chunk_names:
                inl = np.asarray(chunks["INLIER"], dtype=bool)
            else:
                inl = np.isfinite(rv_c)
            ok = inl & np.isfinite(rv_c)
            n_in = int(np.sum(ok))
            if n_in > 0:
                inlier_med = float(np.median(rv_c[ok] - injected_m_s))
        rows.append(
            InfiniteSnrRvFiberResult(
                fiber=int(fib),
                measured_rv_m_s=meas,
                rv_err_m_s=err,
                bias_m_s=bias_m_s,
                bias_km_s=(bias_m_s / 1000.0) if bias_m_s is not None else None,
                n_inlier_chunks=n_in,
                n_total_chunks=n_tot,
                inlier_median_bias_m_s=inlier_med,
            )
        )
    max_abs = max((abs(b) for b in biases_km), default=None)
    spread = (max(biases_km) - min(biases_km)) if len(biases_km) >= 2 else None
    return InfiniteSnrRvEvaluation(
        injected_rv_km_s=float(injected_rv_km_s),
        injected_rv_m_s=injected_m_s,
        fibers=rows,
        max_abs_bias_km_s=max_abs,
        fiber_bias_spread_km_s=spread,
    )


def assert_infinite_snr_rv_recovery(
    evaluation: InfiniteSnrRvEvaluation,
    *,
    thresholds: InfiniteSnrRvThresholds | None = None,
) -> None:
    th = thresholds or DEFAULT_INFINITE_SNR_RV_THRESHOLDS
    if evaluation.max_abs_bias_km_s is None:
        raise AssertionError("Infinite-S/N RV evaluation: no measured RVs (stage2 missing?).")
    if evaluation.max_abs_bias_km_s > th.max_abs_bias_km_s:
        raise AssertionError(
            f"Infinite-S/N RV max |bias|={evaluation.max_abs_bias_km_s:.4f} km/s "
            f"> {th.max_abs_bias_km_s:.4f} km/s"
        )
    if (
        evaluation.fiber_bias_spread_km_s is not None
        and evaluation.fiber_bias_spread_km_s > th.max_fiber_bias_spread_km_s
    ):
        raise AssertionError(
            f"Infinite-S/N RV fiber spread={evaluation.fiber_bias_spread_km_s:.4f} km/s "
            f"> {th.max_fiber_bias_spread_km_s:.4f} km/s"
        )
    for row in evaluation.fibers:
        if row.n_inlier_chunks < th.min_inlier_chunks:
            raise AssertionError(
                f"Infinite-S/N RV fiber {row.fiber}: n_inlier={row.n_inlier_chunks} "
                f"< {th.min_inlier_chunks}"
            )


def log_infinite_snr_rv_evaluation(evaluation: InfiniteSnrRvEvaluation) -> None:
    logger.info(
        "Infinite-S/N RV evaluation injected=%.3f km/s max|bias|=%s spread=%s passed=%s",
        evaluation.injected_rv_km_s,
        f"{evaluation.max_abs_bias_km_s:.4f} km/s" if evaluation.max_abs_bias_km_s is not None else "—",
        f"{evaluation.fiber_bias_spread_km_s:.4f} km/s"
        if evaluation.fiber_bias_spread_km_s is not None
        else "—",
        evaluation.passed_default_thresholds,
    )
    for row in evaluation.fibers:
        logger.info(
            "  fib%d: bias=%s km/s n_inlier=%d/%d inlier_med=%s m/s",
            row.fiber,
            f"{row.bias_km_s:+.4f}" if row.bias_km_s is not None else "—",
            row.n_inlier_chunks,
            row.n_total_chunks,
            f"{row.inlier_median_bias_m_s:.1f}" if row.inlier_median_bias_m_s is not None else "—",
        )
