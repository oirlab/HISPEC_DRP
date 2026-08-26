"""Line-mask stellar RV via cubic-spline mask CCF and :func:`calc_ccf_metrics`."""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ....utils.line_mask_utils import MaskLines, restrict_mask_to_chunk
from .ccf_common import (
    V_MAX_DEFAULT_M_S,
    V_MIN_DEFAULT_M_S,
    V_STEP_DEFAULT_M_S,
)
from .ccf_fitting import CCFMetrics, calc_ccf_metrics
from .ccf_spline import compute_ccf_spline
from .ccf_peak import _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s
from .ccf_utils import (
    LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    _ccf_preview_finalize_n_workers,
    _line_mask_peak_index_integral_and_prominence,
    _nanmean_axis0_ignore_empty_columns,
    _nan_stderr_of_mean_axis0,
    _smooth_uniform5_numba,
    build_line_mask_chunk_wavelength_intervals_line_by_line,
    ccf_order_n_chunks_for_span,
    line_mask_chunk_restrict_velocity_range_m_s,
    mask_line_window_spectrum_band_um,
)

__all__ = [
    "finalize_ccf_preview_dict_spline",
    "fit_line_mask_order_mean_rv_spline",
    "line_mask_scout_center_m_s",
    "rv_per_chunks_line_mask_spline",
]

logger = logging.getLogger(__name__)

# Half-width (m/s) of the local velocity window for :func:`calc_ccf_metrics` after the
# mask-line prominence peak seed (coarse uniform grids alias without this step).
# Alias troughs inside this window are handled by prior-aware recentering of the
# Gaussian fit window (not by replacing the fitted RV with a discrete peak).
_SPLINE_MASK_METRICS_FIT_HALF_WIDTH_M_S = 20_000.0

# If the prominence seed is farther than this from ``gaussian_seed_mean_m_s``
# (typically ``ccf_input_velocity``), prefer a competitive local peak near the prior
# as the metrics *window center* so the Gaussian fit does not lock onto a deeper
# alias trough. Reported chunk RV always remains the Gaussian vertex.
# Must sit below dense-mask alias offsets (~280–480 m/s on BSPEC 117/145).
_ALIAS_PRIOR_RECENTER_GATE_M_S = 200.0
# Half-width for prior-local peak search when recentering the metrics window.
_ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S = 250.0
# Prior-local peak must reach at least this fraction of the prominence-peak amplitude
# (stored peak-up convention) before we recenter away from prominence.
_ALIAS_PRIOR_MIN_DEPTH_FRAC = 0.90


def _line_mask_spline_peak_index(v_grid: np.ndarray, ccf_raw: np.ndarray) -> int:
    """Prominence-based peak index on ``-ccf_raw`` (mask-line convention)."""
    vg = np.asarray(v_grid, dtype=np.float64).ravel()
    raw = np.asarray(ccf_raw, dtype=np.float64).ravel()
    n = int(vg.size)
    if n < 3:
        return -1
    y_work = np.empty(n, dtype=np.float64)
    for i in range(n):
        yi = raw[i]
        y_work[i] = -yi if np.isfinite(yi) else np.nan
    ys = _smooth_uniform5_numba(y_work, n)
    w_prom = max(80, min(2000, n // 10))
    w_int = max(100, min(800, n // 12))
    return int(_line_mask_peak_index_integral_and_prominence(ys, n, w_prom, w_int))


def line_mask_scout_center_m_s(
    scout_rvs: np.ndarray,
    v_grid: np.ndarray,
    ccf_stored_mean: np.ndarray,
) -> float:
    """
    Velocity center from a coarse scout pass on mask CCFs.

    Prefer the RV already measured by the scout chunk worker (Gaussian centroid
    after the Numba seed). That estimate is typically within a coarse-grid step
    of the true peak.

    Do **not** use a global argmax on the mean stored CCF: with
    :data:`~hispecdrp.primitives.utils.ccf.ccf_utils.LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`
    the stored series is often entirely negative and the true alignment is a *minimum*
    of the stored values, so argmax-style scouts can land hundreds of km/s away from the
    real peak.

    Falls back to the same prominence peak used by :func:`_fit_spline_mask_ccf_metrics`,
    then to a sub-pixel extremum on the stored CCF if needed.
    """
    rvs = np.asarray(scout_rvs, dtype=np.float64).ravel()
    ok = np.isfinite(rvs)
    if np.any(ok):
        return float(np.nanmedian(rvs[ok]))

    vg = np.asarray(v_grid, dtype=np.float64).ravel()
    stored = np.asarray(ccf_stored_mean, dtype=np.float64).ravel()
    if vg.size < 3 or stored.size != vg.size:
        return float("nan")

    raw = -stored if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else stored
    i_peak = _line_mask_spline_peak_index(vg, raw)
    if 0 <= i_peak < int(vg.size) and math.isfinite(float(vg[i_peak])):
        return float(vg[i_peak])

    v_sub = _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg, stored)
    if math.isfinite(float(v_sub)):
        return float(v_sub)
    if np.any(np.isfinite(stored)):
        return float(vg[int(np.nanargmax(stored))])
    return float("nan")


def _stored_corr_strength_in_window(
    vel_grid: np.ndarray,
    ccf_stored: np.ndarray,
    *,
    anchor_m_s: float,
    half_width_m_s: float,
) -> float:
    """
    Correlation-peak strength in ``[anchor±half_width]`` (larger = stronger), or NaN.

    For :data:`LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`, stored series are often entirely
    negative and the alignment is a trough — use ``-min(stored)``. Otherwise use
    ``max(stored)`` (peak-up).
    """
    vg = np.asarray(vel_grid, dtype=np.float64).ravel()
    ys = np.asarray(ccf_stored, dtype=np.float64).ravel()
    if vg.size < 1 or ys.size != vg.size:
        return float("nan")
    hw = float(half_width_m_s)
    anc = float(anchor_m_s)
    m = (vg >= anc - hw) & (vg <= anc + hw) & np.isfinite(vg) & np.isfinite(ys)
    if not np.any(m):
        return float("nan")
    if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW:
        return float(-np.nanmin(ys[m]))
    return float(np.nanmax(ys[m]))


def _resolve_spline_mask_metrics_center_m_s(
    vel_grid: np.ndarray,
    ccf_stored: np.ndarray,
    *,
    prominence_center_m_s: float,
    prior_m_s: float | None,
) -> float:
    """
    Metrics-window center: prominence peak, optionally recentered near ``prior_m_s``.

    When the prominence seed lies farther than :data:`_ALIAS_PRIOR_RECENTER_GATE_M_S`
    from a finite prior (usually ``ccf_input_velocity``) and a local peak near that
    prior reaches at least :data:`_ALIAS_PRIOR_MIN_DEPTH_FRAC` of the prominence
    amplitude, return the prior-local peak so the Gaussian does not lock onto a
    deeper alias trough.
    """
    v_prom = float(prominence_center_m_s)
    if prior_m_s is None or not math.isfinite(float(prior_m_s)):
        return v_prom
    prior = float(prior_m_s)
    if not math.isfinite(v_prom):
        return prior
    if abs(v_prom - prior) <= float(_ALIAS_PRIOR_RECENTER_GATE_M_S):
        return v_prom

    search_hw = float(_ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S)
    vg = np.asarray(vel_grid, dtype=np.float64).ravel()
    ys = np.asarray(ccf_stored, dtype=np.float64).ravel()
    m_prior = (
        (vg >= prior - search_hw)
        & (vg <= prior + search_hw)
        & np.isfinite(vg)
        & np.isfinite(ys)
    )
    if int(np.count_nonzero(m_prior)) < 3:
        return v_prom
    # Peak-up disc max on stored; for negated-from-raw troughs, search on -stored.
    y_peak = -ys if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else ys
    v_prior = float(
        _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg[m_prior], y_peak[m_prior])
    )
    if not math.isfinite(v_prior):
        return v_prom

    str_prom = _stored_corr_strength_in_window(
        vg,
        ys,
        anchor_m_s=v_prom,
        half_width_m_s=search_hw,
    )
    str_prior = _stored_corr_strength_in_window(
        vg,
        ys,
        anchor_m_s=v_prior,
        half_width_m_s=search_hw,
    )
    if (
        math.isfinite(str_prom)
        and math.isfinite(str_prior)
        and str_prom > 0.0
        and (str_prior / str_prom) >= float(_ALIAS_PRIOR_MIN_DEPTH_FRAC)
    ):
        return v_prior
    return v_prom


def _fit_spline_mask_ccf_metrics(
    vel_grid: np.ndarray,
    ccf_raw: np.ndarray,
    ccf_err: np.ndarray | None,
    *,
    fit_half_width_m_s: float | None = None,
    gaussian_seed_mean_m_s: float | None = None,
) -> CCFMetrics:
    """
    Fit :class:`CCFMetrics` on a spline mask CCF with mask-line peak seeding.

    Uses a local velocity window around the prominence peak (optionally recentered
    toward ``gaussian_seed_mean_m_s`` when a competitive prior-local peak exists)
    so coarse-grid alias minima do not dominate the Gaussian fit. The seed is also
    forwarded as ``mu_guess`` for the Gaussian/bisector.
    """
    vg = np.asarray(vel_grid, dtype=np.float64).ravel()
    raw = np.asarray(ccf_raw, dtype=np.float64).ravel()
    ce = (
        np.asarray(ccf_err, dtype=np.float64).ravel()
        if ccf_err is not None
        else None
    )
    if ce is not None and ce.size != raw.size:
        ce = None
    stored = -raw if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else raw.copy()
    i_peak = _line_mask_spline_peak_index(vg, raw)
    if i_peak < 0 or i_peak >= vg.size:
        v_sub = _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg, stored)
        v_prom = float(v_sub) if math.isfinite(float(v_sub)) else float(vg[int(np.nanargmax(stored))])
    else:
        v_prom = float(vg[i_peak])
    prior = (
        float(gaussian_seed_mean_m_s)
        if gaussian_seed_mean_m_s is not None
        and math.isfinite(float(gaussian_seed_mean_m_s))
        else None
    )
    v_center = _resolve_spline_mask_metrics_center_m_s(
        vg,
        stored,
        prominence_center_m_s=v_prom,
        prior_m_s=prior,
    )
    if fit_half_width_m_s is None or not math.isfinite(float(fit_half_width_m_s)) or float(fit_half_width_m_s) <= 0.0:
        hw = _SPLINE_MASK_METRICS_FIT_HALF_WIDTH_M_S
    else:
        hw = float(fit_half_width_m_s)
    in_win = (vg >= v_center - hw) & (vg <= v_center + hw)
    if int(np.count_nonzero(in_win)) < 7:
        in_win = np.isfinite(stored)
    return calc_ccf_metrics(
        vg[in_win],
        stored[in_win],
        ce[in_win] if ce is not None else None,
        pix_size=None,
        absorption=not LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        calc_bis=True,
        mu_guess=prior,
    )


def _ccf_metrics_to_preview_fits(metrics: CCFMetrics) -> tuple[dict, dict | None]:
    """Map :class:`CCFMetrics` to stellar-RV preview ``gaussian_fit`` / ``bisector_fit`` slots."""
    fit_ok = bool(
        np.isfinite(metrics.rv)
        and np.isfinite(metrics.rv_err)
        and float(metrics.rv_err) > 0.0
    )
    params = {
        "amplitude": float(metrics.contrast),
        "v0": float(metrics.rv),
        "sigma": float(metrics.sigma),
        "baseline": float(metrics.offset),
    }
    gaussian_fit = {
        "rv": float(metrics.rv),
        "rv_err": float(metrics.rv_err),
        "params": params,
        "diagnostics": {
            "fit_ok": fit_ok,
            "profile": "emission" if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else "absorption",
            "ccf_dir_style": "up" if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else "down",
            "used_ccf_metrics": True,
        },
    }
    bisector_fit: dict | None = None
    if metrics.bis_rv is not None and np.isfinite(float(metrics.bis_rv)):
        bdiag: dict = {"used_ccf_metrics": True}
        # Prefer NEID cut diagnostics (cuts, bis_m_s, …) when the metrics fit retained them.
        if isinstance(metrics.bisector_diagnostics, dict):
            bdiag.update(metrics.bisector_diagnostics)
        # Queloz BIS slope + error from CCFMetrics when cut-based BIS is absent.
        if (
            (bdiag.get("bis_m_s") is None or not math.isfinite(float(bdiag.get("bis_m_s"))))
            and metrics.bis is not None
            and math.isfinite(float(metrics.bis))
        ):
            bdiag["bis_m_s"] = float(metrics.bis)
            if metrics.bis_err is not None and math.isfinite(float(metrics.bis_err)):
                bdiag["bis_err_m_s"] = float(metrics.bis_err)
        elif (
            (bdiag.get("bis_m_s") is None or not math.isfinite(float(bdiag.get("bis_m_s"))))
            and metrics.bis_shallow is not None
            and metrics.bis_core is not None
            and math.isfinite(float(metrics.bis_shallow))
            and math.isfinite(float(metrics.bis_core))
        ):
            bdiag["bis_m_s"] = float(metrics.bis_shallow) - float(metrics.bis_core)
            if (
                metrics.bis_shallow_err is not None
                and metrics.bis_core_err is not None
                and math.isfinite(float(metrics.bis_shallow_err))
                and math.isfinite(float(metrics.bis_core_err))
            ):
                bdiag["bis_err_m_s"] = float(
                    math.sqrt(
                        float(metrics.bis_shallow_err) ** 2
                        + float(metrics.bis_core_err) ** 2
                    )
                )
        if metrics.bis_shallow is not None and math.isfinite(float(metrics.bis_shallow)):
            bdiag["bis_shallow_m_s"] = float(metrics.bis_shallow)
        if metrics.bis_core is not None and math.isfinite(float(metrics.bis_core)):
            bdiag["bis_core_m_s"] = float(metrics.bis_core)
        if metrics.bisector is not None:
            bdiag["bisector"] = np.asarray(metrics.bisector, dtype=np.float64)
        if metrics.bis_levels is not None:
            bdiag["bis_levels"] = np.asarray(metrics.bis_levels, dtype=np.float64)
        bisector_fit = {
            "rv": float(metrics.bis_rv),
            "rv_err": (
                float(metrics.bis_rv_err)
                if metrics.bis_rv_err is not None
                and np.isfinite(float(metrics.bis_rv_err))
                and float(metrics.bis_rv_err) > 0.0
                else float("nan")
            ),
            "diagnostics": bdiag,
        }
    return gaussian_fit, bisector_fit


def _pipeline_rv_from_metrics(
    metrics: CCFMetrics,
    *,
    vel_grid: np.ndarray | None = None,
    ccf_raw: np.ndarray | None = None,
    peak_anchor_m_s: float | None = None,
) -> tuple[float, float]:
    """
    Chunk RV from :class:`CCFMetrics` Gaussian fit only.

    Always returns the Gaussian ``rv`` / ``rv_err``. Absolute bisector refine
    (``bis_rv``) stays on the metrics object for diagnostics. Keyword arguments
    ``vel_grid``, ``ccf_raw``, and ``peak_anchor_m_s`` are accepted for call-site
    compatibility and are not used to replace the fitted RV with a discrete peak
    (no snap-back onto the coarse velocity grid).
    """
    del vel_grid, ccf_raw, peak_anchor_m_s
    return float(metrics.rv), float(metrics.rv_err)


def _fit_ccf_metrics_on_stored_ccf(
    vel_grid: np.ndarray,
    ccf_stored: np.ndarray,
    ccf_err: np.ndarray | None,
    *,
    gaussian_seed_mean_m_s: float | None = None,
    metrics_fit_half_width_m_s: float | None = None,
) -> CCFMetrics:
    """Gaussian + bisector metrics on spline mask CCF rows."""
    raw = (
        -np.asarray(ccf_stored, dtype=np.float64).ravel()
        if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW
        else np.asarray(ccf_stored, dtype=np.float64).ravel()
    )
    return _fit_spline_mask_ccf_metrics(
        vel_grid,
        raw,
        ccf_err,
        fit_half_width_m_s=metrics_fit_half_width_m_s,
        gaussian_seed_mean_m_s=gaussian_seed_mean_m_s,
    )


def fit_line_mask_order_mean_rv_spline(
    vel_grid: np.ndarray,
    ccf_mean_stored: np.ndarray,
    ccf_err_mean: np.ndarray | None = None,
    *,
    gaussian_seed_mean_m_s: float | None = None,
    metrics_fit_half_width_m_s: float | None = None,
) -> tuple[float, float, bool]:
    """
    Order-mean RV for a single-chunk echelle row using :func:`calc_ccf_metrics`.

    Returns ``(rv, rv_err, used_order_mean)``.
    """
    metrics = _fit_ccf_metrics_on_stored_ccf(
        vel_grid,
        ccf_mean_stored,
        ccf_err_mean,
        gaussian_seed_mean_m_s=gaussian_seed_mean_m_s,
        metrics_fit_half_width_m_s=metrics_fit_half_width_m_s,
    )
    raw = (
        -np.asarray(ccf_mean_stored, dtype=np.float64).ravel()
        if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW
        else np.asarray(ccf_mean_stored, dtype=np.float64).ravel()
    )
    prior = (
        float(gaussian_seed_mean_m_s)
        if gaussian_seed_mean_m_s is not None
        and math.isfinite(float(gaussian_seed_mean_m_s))
        else None
    )
    rv, err = _pipeline_rv_from_metrics(
        metrics,
        vel_grid=vel_grid,
        ccf_raw=raw,
        peak_anchor_m_s=prior,
    )
    ok = (
        np.isfinite(rv)
        and np.isfinite(err)
        and float(err) > 0.0
    )
    return rv, err, ok



def finalize_ccf_preview_dict_spline(
    order_slots: list[dict],
    all_ccf_rows: list[np.ndarray],
    v_grid: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    all_ccf_err_rows: list[np.ndarray] | None = None,
    *,
    gaussian_seed_mean_m_s: float | None = None,
    metrics_fit_half_width_m_s: float | None = None,
) -> dict:
    """
    Build ``_ccf_preview_by_fiber`` using :func:`calc_ccf_metrics` per slot.

    Same layout as :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.finalize_ccf_preview_dict`.
    """
    if not all_ccf_rows:
        return {
            "v_step": float(v_step),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "slots": [],
            "ccf_line_mask_stored_negated": bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW),
        }
    stacked = np.vstack(all_ccf_rows)
    combined = _nanmean_axis0_ignore_empty_columns(stacked)
    combined_err: np.ndarray | None = None
    if (
        all_ccf_err_rows is not None
        and len(all_ccf_err_rows) == len(order_slots)
        and all(r.size > 0 for r in all_ccf_err_rows)
    ):
        stacked_e = np.vstack(all_ccf_err_rows)
        combined_err = _nan_stderr_of_mean_axis0(stacked_e, stacked)
    ng_mean = (
        max(1, int(np.sum([max(1, int(s.get("n_good", 1))) for s in order_slots])))
        if order_slots
        else 1
    )
    mean_slot = {
        "kind": "mean",
        "order": None,
        "v": np.asarray(v_grid, dtype=np.float64),
        "ccf": np.asarray(combined, dtype=np.float64),
        "wl_lo": float("nan"),
        "wl_hi": float("nan"),
        "n_good": ng_mean,
    }
    if combined_err is not None:
        mean_slot["ccf_err"] = np.asarray(combined_err, dtype=np.float64)
    slots = [mean_slot] + order_slots
    n_workers = _ccf_preview_finalize_n_workers(len(slots))
    logger.info(
        "CCF preview (spline): finalize n_slots=%d n_workers=%d",
        len(slots),
        n_workers,
    )

    def _fit_slot(sl: dict) -> tuple[dict, dict | None]:
        raw = (
            -np.asarray(sl["ccf"], dtype=np.float64).ravel()
            if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW
            else np.asarray(sl["ccf"], dtype=np.float64).ravel()
        )
        ce = sl.get("ccf_err")
        ccf_err_arg = np.asarray(ce, np.float64).ravel() if ce is not None else None
        metrics = _fit_spline_mask_ccf_metrics(
            np.asarray(sl["v"], dtype=np.float64),
            raw,
            ccf_err_arg,
            fit_half_width_m_s=metrics_fit_half_width_m_s,
            gaussian_seed_mean_m_s=gaussian_seed_mean_m_s,
        )
        return _ccf_metrics_to_preview_fits(metrics)

    if n_workers <= 1:
        results = [_fit_slot(sl) for sl in slots]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_fit_slot, slots))
    for sl, (gaussian_fit, bisector_fit) in zip(slots, results):
        sl["gaussian_fit"] = gaussian_fit
        if bisector_fit is not None:
            sl["bisector_fit"] = bisector_fit
    out: dict = {
        "v_step": float(v_step),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "slots": slots,
        "ccf_line_mask_stored_negated": bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW),
    }
    if (
        gaussian_seed_mean_m_s is not None
        and math.isfinite(float(gaussian_seed_mean_m_s))
    ):
        out["line_mask_gaussian_seed_mean_m_s"] = float(gaussian_seed_mean_m_s)
    return out


def rv_per_chunks_line_mask_spline(
    wave_um: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq_bad_mask: np.ndarray,
    mask_lines: MaskLines,
    *,
    window_size_um: float,
    v_min_m_s: float = V_MIN_DEFAULT_M_S,
    v_max_m_s: float = V_MAX_DEFAULT_M_S,
    v_step_m_s: float = V_STEP_DEFAULT_M_S,
    v_grid_m_s: np.ndarray | None = None,
    zb: float = 0.0,
    return_ccf: bool = False,
    min_good_pixels: int = 10,
    window_line_by_line: bool = False,
    line_mask_line_chunk_v_center_m_s: float | None = None,
    gaussian_seed_mean_m_s: float | None = None,
    metrics_fit_half_width_m_s: float | None = None,
    line_set_velocity_m_s: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]:
    """
    Per-chunk line-mask RV via :func:`compute_ccf_spline` + :func:`calc_ccf_metrics`.

    Chunking matches :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`.
    Stored CCF rows follow :data:`~hispecdrp.primitives.utils.ccf.ccf_utils.LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`.

    The spline CCF freezes its line set at ``line_set_velocity_m_s`` when finite;
    otherwise at ``gaussian_seed_mean_m_s``, then ``line_mask_line_chunk_v_center_m_s``.
    If none of those are set (broad scout with no input RV), membership is recomputed
    at each trial velocity — freezing at rest with :math:`F=0` for missing lines
    creates false CCF minima on wide grids.

    Returns also include per-chunk Queloz ``bis`` / ``bis_err`` (m/s) from
    :class:`~hispecdrp.primitives.utils.ccf.ccf_fitting.CCFMetrics`.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    def _empty(v_grid_local: np.ndarray):
        z = np.array([], dtype=np.float64)
        zi = np.array([], dtype=np.int32)
        if not return_ccf:
            return (z, z, z, v_grid_local, None, z, z, zi, None, None, z, z)
        ccf_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        std_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        return (z, z, z, v_grid_local, ccf_out, z, z, zi, std_out, ccf_out, z, z)

    if line_set_velocity_m_s is not None and math.isfinite(float(line_set_velocity_m_s)):
        v_line_set: float | None = float(line_set_velocity_m_s)
    elif gaussian_seed_mean_m_s is not None and math.isfinite(
        float(gaussian_seed_mean_m_s)
    ):
        v_line_set = float(gaussian_seed_mean_m_s)
    elif line_mask_line_chunk_v_center_m_s is not None and math.isfinite(
        float(line_mask_line_chunk_v_center_m_s)
    ):
        v_line_set = float(line_mask_line_chunk_v_center_m_s)
    else:
        v_line_set = None

    vr_lo = float(v_min_m_s)
    vr_hi = float(v_max_m_s)

    if v_grid_m_s is not None:
        v_grid = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    else:
        v_grid = np.arange(
            v_min_m_s,
            v_max_m_s + 0.5 * v_step_m_s,
            v_step_m_s,
            dtype=np.float64,
        )

    good_pix = ~d & np.isfinite(w) & np.isfinite(s) & np.isfinite(e)
    if np.sum(good_pix) < min_good_pixels:
        return _empty(v_grid)

    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return _empty(v_grid)

    span_um = float(w_max - w_min)
    use_line_chunks = bool(window_line_by_line)
    _w_edge_eps = max(1e-15, abs(float(w_max)) * 1e-12)
    if use_line_chunks:
        if line_mask_line_chunk_v_center_m_s is None or not math.isfinite(
            float(line_mask_line_chunk_v_center_m_s)
        ):
            raise ValueError(
                "rv_per_chunks_line_mask_spline: window_line_by_line requires finite "
                "line_mask_line_chunk_v_center_m_s"
            )
        wl_lo_b, wl_hi_b, wl_ctr_b = build_line_mask_chunk_wavelength_intervals_line_by_line(
            w,
            mask_lines,
            v_center_m_s=float(line_mask_line_chunk_v_center_m_s),
            zb=float(zb),
            min_good_pixels=int(min_good_pixels),
        )
        n_chunk = int(len(wl_lo_b))
        chunk_iter_fwd: list[tuple[float, float, float, bool]] = [
            (
                float(wl_lo_b[k]),
                float(wl_hi_b[k]),
                float(wl_ctr_b[k]),
                float(wl_hi_b[k]) >= float(w_max) - _w_edge_eps,
            )
            for k in range(n_chunk)
        ]
        chunk_iter = list(reversed(chunk_iter_fwd))
    else:
        n_chunk = ccf_order_n_chunks_for_span(span_um, float(window_size_um))
        chunk_iter_fwd = [
            (
                w_min + k * span_um / n_chunk,
                w_min + (k + 1) * span_um / n_chunk,
                0.5
                * (
                    w_min + k * span_um / n_chunk
                    + w_min + (k + 1) * span_um / n_chunk
                ),
                (w_min + (k + 1) * span_um / n_chunk) >= float(w_max) - _w_edge_eps,
            )
            for k in range(n_chunk)
        ]
        chunk_iter = list(reversed(chunk_iter_fwd))

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_centers: list[float] = []
    all_wl_lo: list[float] = []
    all_wl_hi: list[float] = []
    all_n_good: list[int] = []
    all_bis: list[float] = []
    all_bis_errs: list[float] = []
    all_ccf_rows: list[np.ndarray] = []
    all_ccf_preview_rows: list[np.ndarray] = []
    all_ccf_err_rows: list[np.ndarray] = []

    for lam_lo, lam_hi, wl_center, right_closed in chunk_iter:
        if right_closed:
            in_voronoi = (w >= lam_lo) & (w <= lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        else:
            in_voronoi = (w >= lam_lo) & (w < lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        v_span_lo_um = float(lam_lo)
        v_span_hi_um = float(lam_hi)
        if use_line_chunks:
            band_lo, band_hi = mask_line_window_spectrum_band_um(
                float(wl_center),
                float(window_size_um),
                w_min,
                w_max,
            )
            in_band = (w >= band_lo) & (w <= band_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
            in_chunk = in_voronoi & in_band
            if int(np.sum(in_chunk)) < int(min_good_pixels):
                in_chunk = in_voronoi
                v_span_lo_um, v_span_hi_um = float(lam_lo), float(lam_hi)
            else:
                v_span_lo_um, v_span_hi_um = float(band_lo), float(band_hi)
        else:
            in_chunk = in_voronoi

        if int(np.sum(in_chunk)) < min_good_pixels:
            continue

        w_c = w[in_chunk]
        s_c = s[in_chunk]
        e_c = e[in_chunk]
        d_c = d[in_chunk]
        spec_dq = np.where(~d_c, 0, 1).astype(np.int32)

        if use_line_chunks:
            v_r_lo, v_r_hi = line_mask_chunk_restrict_velocity_range_m_s(
                v_span_lo_um,
                v_span_hi_um,
                v_prior_m_s=float(line_mask_line_chunk_v_center_m_s),
                v_span_lo=vr_lo,
                v_span_hi=vr_hi,
            )
        else:
            v_r_lo, v_r_hi = vr_lo, vr_hi
        mask_c = restrict_mask_to_chunk(
            mask_lines,
            float(np.nanmin(w_c)),
            float(np.nanmax(w_c)),
            v_min_m_s=v_r_lo,
            v_max_m_s=v_r_hi,
            zb=zb,
        )
        if mask_c.line_lambda_um.size == 0:
            continue

        n_good = int(np.sum(in_chunk))
        vel_grid, ccf_raw, ccf_err = compute_ccf_spline(
            w_c,
            s_c,
            e_c,
            spec_dq,
            mask_c.line_lambda_um.astype(np.float64),
            mask_c.weight.astype(np.float64),
            vel_grid=v_grid,
            zb=float(zb),
            use_variance=True,
            line_set_velocity=v_line_set,
        )
        if not np.any(np.isfinite(ccf_raw)):
            continue

        seed_m_s = (
            float(gaussian_seed_mean_m_s)
            if gaussian_seed_mean_m_s is not None
            and math.isfinite(float(gaussian_seed_mean_m_s))
            else (
                float(line_mask_line_chunk_v_center_m_s)
                if line_mask_line_chunk_v_center_m_s is not None
                and math.isfinite(float(line_mask_line_chunk_v_center_m_s))
                else None
            )
        )
        metrics = _fit_spline_mask_ccf_metrics(
            vel_grid,
            ccf_raw,
            ccf_err,
            fit_half_width_m_s=metrics_fit_half_width_m_s,
            gaussian_seed_mean_m_s=seed_m_s,
        )
        rv, rv_err = _pipeline_rv_from_metrics(
            metrics,
            vel_grid=vel_grid,
            ccf_raw=ccf_raw,
            peak_anchor_m_s=seed_m_s,
        )
        if not (np.isfinite(rv) and np.isfinite(rv_err) and float(rv_err) > 0.0):
            continue

        ccf_preview = np.asarray(ccf_raw, dtype=np.float64)
        ccf_stored = (
            -ccf_preview if LINE_MASK_CCF_STORED_NEGATED_FROM_RAW else ccf_preview.copy()
        )

        bis_v = (
            float(metrics.bis)
            if metrics.bis is not None and math.isfinite(float(metrics.bis))
            else float("nan")
        )
        bis_e = (
            float(metrics.bis_err)
            if metrics.bis_err is not None and math.isfinite(float(metrics.bis_err))
            else float("nan")
        )
        all_rvs.append(float(rv))
        all_rv_errs.append(float(rv_err))
        all_centers.append(float(wl_center))
        all_wl_lo.append(float(np.nanmin(w_c)))
        all_wl_hi.append(float(np.nanmax(w_c)))
        all_n_good.append(n_good)
        all_bis.append(bis_v)
        all_bis_errs.append(bis_e)
        if return_ccf:
            all_ccf_rows.append(ccf_stored)
            all_ccf_preview_rows.append(ccf_preview)
            all_ccf_err_rows.append(np.asarray(ccf_err, dtype=np.float64))

    if not all_rvs:
        return _empty(v_grid)

    rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    centers_arr = np.asarray(all_centers, dtype=np.float64)
    wl_lo_arr = np.asarray(all_wl_lo, dtype=np.float64)
    wl_hi_arr = np.asarray(all_wl_hi, dtype=np.float64)
    n_good_arr = np.asarray(all_n_good, dtype=np.int32)
    bis_arr = np.asarray(all_bis, dtype=np.float64)
    bis_err_arr = np.asarray(all_bis_errs, dtype=np.float64)

    if not return_ccf:
        return (
            rvs_arr,
            rv_errs_arr,
            centers_arr,
            v_grid,
            None,
            wl_lo_arr,
            wl_hi_arr,
            n_good_arr,
            None,
            None,
            bis_arr,
            bis_err_arr,
        )

    ccf_2d = np.vstack(all_ccf_rows)
    ccf_preview_2d = np.vstack(all_ccf_preview_rows)
    ccf_std_2d = np.vstack(all_ccf_err_rows)
    return (
        rvs_arr,
        rv_errs_arr,
        centers_arr,
        v_grid,
        ccf_2d,
        wl_lo_arr,
        wl_hi_arr,
        n_good_arr,
        ccf_std_2d,
        ccf_preview_2d,
        bis_arr,
        bis_err_arr,
    )
