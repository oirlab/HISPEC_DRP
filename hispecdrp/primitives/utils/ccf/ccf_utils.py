"""Cross-correlation (CCF) and radial-velocity utilities.

The canonical overview of the CCF algorithms and conventions lives in the documentation:
``docs/source/algorithms/ccf.rst``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from typing import Any, Literal

from astropy import units as u
from astropy.table import Table
from astropy.modeling import fitting
from astropy.modeling.models import Const1D, Gaussian1D, Polynomial1D

try:
    from astropy.utils.exceptions import AstropyUserWarning
except ImportError:  # pragma: no cover
    AstropyUserWarning = UserWarning
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter
from astropy.stats import mad_std
from pathlib import Path

from hispecdrp.utils.physical_constants import C_M_S
from hispecdrp.utils.line_mask_utils import (
    MaskLineStyle,
    MaskLines,
    pixel_edges_from_wavelength,
    restrict_mask_to_chunk,
    stellar_rv_ccf_mask_lines,
)

from ....datamodels.dqflags import DQ_FLAGS
from ....datamodels.extensions import RVCCF_CCF_ELEM_DTYPE
from ....utils.parallelization_utils import (
    njit,
    numba_thread_scope,
    prange,
    resolve_numba_threads,
)
from .ccf_rv_content import doppler_shift_from_barycenter
from .ccf_bisector import (
    CCFBisectorConfig,
    _refine_ccf_rv_bisector_after_gaussian_fit,
    fit_ccf_rv_bisector,
)
from .ccf_parabolic import (
    CCFParabolicConfig,
    _line_mask_parabolic_vertex_lstsq_npt_numba,
    _refine_ccf_rv_after_gaussian_fit_ccf_line_mask,
    _refine_ccf_rv_parabolic_after_gaussian_fit,
    _require_ccf_parabolic_vertex_n_samples,
    ensure_line_mask_lstsq_preview_plot_metadata,
    fit_ccf_rv_parabolic,
    line_mask_parabolic_vertex_rv_err_levmar_m_s,
    line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s,
    refine_ccf_rv_coarse_grid_parabolic,
    refine_ccf_rv_line_mask_lstsq_from_stored,
    require_ccf_parabolic_vertex_n_samples,
    sweep_refine_ccf_rv_coarse_grid_min_abs_bias,
)
from .ccf_common import (
    V_MAX_DEFAULT_M_S,
    V_MIN_DEFAULT_M_S,
    V_STEP_DEFAULT_M_S,
    _ccf_background_sigma_clipped_median,
    _ccf_rv_err_numeric_floor,
    _make_velocity_grid,
    effective_ccf_v_step_for_fit,
    resolve_ccf_broad_grid_m_s,
)


logger = logging.getLogger(__name__)

# =============================================================================
# Module constants (algorithm IDs, line-mask CCF defaults, coarse velocity grid)
# =============================================================================

# Stellar-RV :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`
# ``algorithm`` values that use mask-line CCF (see callers for membership tests).
STELLAR_RV_LINE_MASK_ALGORITHMS: frozenset[str] = frozenset({"ccf_line_mask"})

# Line-mask CCF (``ccf_line_mask``): stored RVCCF / preview CCF rows are negated relative to the
# raw weighted mask statistic when ``flip_rows`` is 1 (always for the current Numba path) so
# the line appears as a maximum near the true RV. Pass ``ccf_line_mask_stored_negated=True``
# to :func:`fit_ccf_rv` when fitting those arrays; see ``diag["ccf_line_mask_stored_negated"]``.
LINE_MASK_CCF_STORED_NEGATED_FROM_RAW = True

# Max Lagrange degree+1 stencil for mask-line CCF sampling on L2 wavelength pixels
# (see ``mask_ccf_pixel_interp_order``).
MAX_MASK_CCF_PIXEL_INTERP_ORDER = 12

from .ccf_tophat import TOPHAT_MASK_CCF_SEGMENT_SAMPLES

# Default coarse CCF velocity grid (m/s) for :func:`resolve_ccf_broad_grid_m_s` when any of
# ``v_min``, ``v_max``, or ``v_step`` are ``None``, and for StellarRVPrimitive coarse kwargs when unset.
# Sub-step biases (~20 m/s) can occur from the grid + Gaussian fit; 50 m/s steps double N_v vs 100 m/s.

# Fractional padding on ``window_size_um`` **full width** for line-mask line-window spectral
# pixels (symmetric about predicted observed line λ at ``ccf_input_velocity``).
LINE_MASK_LINE_WINDOW_SPEC_REL_PAD = 0.02


# =============================================================================
# Line-mask chunking: echelle spans, observed λ windows, mask restriction helpers
# =============================================================================


def ccf_order_n_chunks_for_span(span_um: float, window_um: float) -> int:
    """
    Number of equal-wavelength segments for one echelle order given a target chunk width.

    If the order span does not exceed ``window_um``, returns 1 (one chunk = full order).
    Otherwise returns ``ceil(span_um / window_um)`` so each segment has width
    ``span_um / n_chunks``, which is at most ``window_um``.

    Used for mask-line CCF chunking (:func:`rv_per_chunks_line_mask`), template
    :func:`rv_per_chunks`, and Numba order layouts so chunk counts follow a single rule.

    See Also
    --------
    :func:`build_line_mask_chunk_wavelength_intervals_line_by_line` — alternate chunking for
    ``ccf_line_mask`` when ``window_line_by_line`` is true (see :func:`rv_per_chunks_line_mask`).
    """
    if not (math.isfinite(span_um) and math.isfinite(window_um)):
        return 1
    if span_um <= 0.0 or window_um <= 0.0:
        return 1
    if span_um <= window_um:
        return 1
    return int(math.ceil(span_um / window_um))


def validate_line_mask_line_window_chunking_config(
    *,
    algorithm: str,
    window_line_by_line: bool | None,
    ccf_input_velocity_m_s: float | None,
) -> None:
    """
    Validate ``window_line_by_line`` against ``algorithm`` and ``ccf_input_velocity``.

    Raises
    ------
    ValueError
        If line-by-line chunking is enabled but incompatible with the algorithm, or if
        ``ccf_input_velocity`` is missing when line windowing is requested.
    """
    if not bool(window_line_by_line):
        return
    algo = (algorithm or "").strip().lower()
    if algo not in STELLAR_RV_LINE_MASK_ALGORITHMS:
        raise ValueError(
            "window_line_by_line is only supported for algorithm='ccf_line_mask'; "
            f"got algorithm={algorithm!r}."
        )
    if ccf_input_velocity_m_s is None or not math.isfinite(float(ccf_input_velocity_m_s)):
        raise ValueError(
            "ccf_input_velocity must be set to a finite value (m/s) when "
            "window_line_by_line is true (hybrid / fine grid center defines the "
            "Doppler mapping used to place mask lines against the spectrum)."
        )


def line_mask_line_window_coarse_velocity_bounds_m_s(
    wave: np.ndarray,
    order_tasks: list[tuple[int, int]],
    mask_lines: MaskLines,
    *,
    v_center_m_s: float,
    zb: float,
    min_good_pixels: int,
    v_coarse_lo: float,
    v_coarse_hi: float,
    hybrid_half_width_m_s: float,
    v_step_m_s: float,
) -> tuple[float, float, float]:
    """
    Narrow the **broad** CCF velocity span for line-window line-mask.

    Uses the largest chunk wavelength span (in velocity units
    :math:`\\Delta v \\approx c\\,\\Delta\\lambda/\\lambda`) over all echelle rows in
    ``order_tasks``, together with ``v_center_m_s`` (``ccf_input_velocity``), to set a
    symmetric interval inside ``[v_coarse_lo, v_coarse_hi]`` that still contains the
    hybrid fine window (``v_center`` :math:`\\pm` ``hybrid_half_width_m_s``).

    Returns
    -------
    v_min_eff, v_max_eff, dv_chunk_max_m_s : float
        Effective broad bounds for :func:`build_hybrid_merged_velocity_grid` and the
        maximum :math:`c\\,\\Delta\\lambda/\\lambda` among chunks (m/s).
    """
    v_c = float(v_center_m_s)
    vb = float(v_coarse_lo)
    vx = float(v_coarse_hi)
    dv_max = 0.0
    for oi, _order in order_tasks:
        w_row = np.asarray(wave[oi], dtype=np.float64).ravel()
        wlo, whi, _ctr = build_line_mask_chunk_wavelength_intervals_line_by_line(
            w_row,
            mask_lines,
            v_center_m_s=v_c,
            zb=float(zb),
            min_good_pixels=int(min_good_pixels),
        )
        for k in range(int(len(wlo))):
            lam_m = 0.5 * (float(wlo[k]) + float(whi[k]))
            d_lam = float(whi[k]) - float(wlo[k])
            if lam_m > 0.0 and d_lam > 0.0 and math.isfinite(lam_m) and math.isfinite(d_lam):
                dv = C_M_S * d_lam / lam_m
                if math.isfinite(dv):
                    dv_max = max(dv_max, float(dv))
    half_w = float(hybrid_half_width_m_s)
    pad = max(half_w, 2.0 * dv_max, 5.0 * abs(float(v_step_m_s)))
    v_lo = max(vb, v_c - pad)
    v_hi = min(vx, v_c + pad)
    if not math.isfinite(v_lo) or not math.isfinite(v_hi) or v_hi <= v_lo:
        return vb, vx, dv_max
    return float(v_lo), float(v_hi), float(dv_max)


def line_mask_chunk_restrict_velocity_range_m_s(
    wl_lo_um: float,
    wl_hi_um: float,
    *,
    v_prior_m_s: float,
    v_span_lo: float,
    v_span_hi: float,
) -> tuple[float, float]:
    """
    Velocity bounds for :func:`~hispecdrp.utils.line_mask_utils.restrict_mask_to_chunk` on one line-window chunk.

    Combines the pipeline prior ``v_prior_m_s`` (``ccf_input_velocity``) with the chunk
    wavelength span :math:`[\\lambda_{\\rm lo},\\lambda_{\\rm hi}]` via
    :math:`\\Delta v \\approx c\\,(\\lambda_{\\rm hi}-\\lambda_{\\rm lo})/\\bar\\lambda`,
    intersected with ``[v_span_lo, v_span_hi]`` (already the tightened coarse span when
    line-window mode is active).
    """
    lam_m = 0.5 * (float(wl_lo_um) + float(wl_hi_um))
    d_lam = float(wl_hi_um) - float(wl_lo_um)
    span = float(v_span_hi) - float(v_span_lo)
    floor_pad = max(500.0, 0.02 * span) if math.isfinite(span) and span > 0.0 else 500.0
    if lam_m > 0.0 and d_lam > 0.0 and math.isfinite(lam_m) and math.isfinite(d_lam):
        dv = C_M_S * d_lam / lam_m
        pad = max(floor_pad, 2.0 * dv) if math.isfinite(dv) else floor_pad
    else:
        pad = floor_pad
    v_lo = max(float(v_span_lo), float(v_prior_m_s) - pad)
    v_hi = min(float(v_span_hi), float(v_prior_m_s) + pad)
    if not math.isfinite(v_lo) or not math.isfinite(v_hi) or v_hi <= v_lo:
        return float(v_span_lo), float(v_span_hi)
    return float(v_lo), float(v_hi)


def mask_line_window_spectrum_band_um(
    wl_center_um: float,
    window_size_um: float,
    w_min_um: float,
    w_max_um: float,
    *,
    rel_pad: float | None = None,
) -> tuple[float, float]:
    """
    Observed-frame µm interval for spectrum + mask-CCF pixels in line-window line-mask.

    Band is centered on ``wl_center_um`` (predicted observed mask-line λ at the chunking
    velocity) with total width ``window_size_um * (1 + rel_pad)``, clipped to the echelle
    pixel support ``[w_min_um, w_max_um]``.
    """
    wc = float(wl_center_um)
    wsz = float(window_size_um)
    if not math.isfinite(wsz) or wsz <= 0.0:
        raise ValueError("mask_line_window_spectrum_band_um: window_size_um must be finite and > 0")
    rp = float(rel_pad) if rel_pad is not None else float(LINE_MASK_LINE_WINDOW_SPEC_REL_PAD)
    if not math.isfinite(rp) or rp < 0.0:
        rp = 0.0
    half = 0.5 * wsz * (1.0 + rp)
    w0 = float(w_min_um)
    w1 = float(w_max_um)
    lo = max(w0, wc - half)
    hi = min(w1, wc + half)
    if hi <= lo:
        return wc, wc
    return float(lo), float(hi)


def count_mask_lines_in_observed_wavelength_interval(
    mask_lines: MaskLines,
    *,
    wl_lo_um: float,
    wl_hi_um: float,
    v_center_m_s: float,
    zb: float,
) -> int:
    """
    Count mask line segments with positive weight whose observed λ lies in ``[wl_lo_um, wl_hi_um]``.

    Uses the same Doppler mapping as :func:`doppler_shift_from_barycenter` at
    ``v_center_m_s``.
    """
    lam_r = np.asarray(mask_lines.line_lambda_um, dtype=np.float64).ravel()
    wt = np.asarray(mask_lines.weight, dtype=np.float64).ravel()
    if lam_r.size == 0:
        return 0
    lam_obs = doppler_shift_from_barycenter(lam_r, float(v_center_m_s), float(zb))
    ok = np.isfinite(lam_obs) & np.isfinite(wt) & (wt > 0.0)
    lam_obs = lam_obs[ok]
    return int(np.sum((lam_obs >= float(wl_lo_um)) & (lam_obs <= float(wl_hi_um))))


def build_line_mask_chunk_wavelength_intervals_line_by_line(
    wave_um: np.ndarray,
    mask_lines: MaskLines,
    *,
    v_center_m_s: float,
    zb: float,
    min_good_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build observed-wavelength chunks with **one mask line per chunk** (Voronoi cells in λ).

    Mask rest wavelengths are mapped to the observed frame at ``v_center_m_s`` (same as
    :func:`doppler_shift_from_barycenter`). Distinct observed positions are sorted; each line
    defines one chunk whose wavelength **window** is the **largest** interval that (i) contains that
    predicted line λ, (ii) does not overlap neighboring chunks, and (iii) stays inside the pixel
    wavelength support ``[w_min, w_max]`` of ``wave_um``. Neighbor boundaries are **midpoints**
    between adjacent predicted line wavelengths (1-D Voronoi cells). Spectrum outside the union of
    chunks is not analyzed.

    Chunk **center** wavelengths are each line’s predicted observed λ.

    The **interval** ``[wl_lo_um, wl_hi_um]`` is a 1-D Voronoi cell between neighbor lines (midpoint
    boundaries). It is **not** generally symmetric about the line’s predicted λ unless neighbor
    spacings happen to match.

    Parameters
    ----------
    wave_um
        1-D wavelength samples (µm) for the order.
    mask_lines
        Mask lines for the run.
    v_center_m_s
        Typically ``ccf_input_velocity`` (m/s).
    zb
        Barycentric factor for the Doppler mapping.
    min_good_pixels
        Reserved for future filtering when a chunk would contain too few spectral samples.

    Returns
    -------
    wl_lo_um, wl_hi_um, chunk_center_wavelength_um : np.ndarray
        Per-chunk bounds (µm) and chunk-center wavelengths (µm).
        Chunk wavelength half-open convention in :func:`rv_per_chunks_line_mask` matches the last
        chunk inclusive on the right.
    """
    _ = int(min_good_pixels)  # reserved for future pixel-count-based merging of tiny chunks
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    lam_r = np.asarray(mask_lines.line_lambda_um, dtype=np.float64).ravel()
    wt = np.asarray(mask_lines.weight, dtype=np.float64).ravel()
    if lam_r.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    lam_obs = doppler_shift_from_barycenter(lam_r, float(v_center_m_s), float(zb))
    ok = np.isfinite(lam_obs) & np.isfinite(wt) & (wt > 0.0)
    ok &= (lam_obs >= w_min) & (lam_obs <= w_max)
    lam_sel = lam_obs[ok]
    if lam_sel.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    # Merge lines that Doppler-map to the same observed wavelength (within tight rtol).
    lam_sorted = np.sort(lam_sel)
    if lam_sorted.size >= 2:
        sep = np.maximum(np.abs(np.diff(lam_sorted)), np.finfo(np.float64).tiny)
        rel = sep / np.maximum(np.abs(0.5 * (lam_sorted[1:] + lam_sorted[:-1])), 1e-15)
        keep = np.ones(lam_sorted.size, dtype=bool)
        keep[1:] &= rel > 1e-9
        lam_u = lam_sorted[keep]
    else:
        lam_u = lam_sorted

    n_line = int(lam_u.size)
    n_per = 1

    wl_lo_list: list[float] = []
    wl_hi_list: list[float] = []
    ctr_list: list[float] = []

    for i0 in range(0, n_line, n_per):
        i1 = min(n_line, i0 + n_per)
        group = lam_u[i0:i1]
        lam_first = float(group[0])
        lam_last = float(group[-1])
        if i0 == 0:
            lo = w_min
        else:
            lo = 0.5 * (float(lam_u[i0 - 1]) + lam_first)
        if i1 >= n_line:
            hi = w_max
        else:
            hi = 0.5 * (lam_last + float(lam_u[i1]))
        lo = max(lo, w_min)
        hi = min(hi, w_max)
        if hi > lo:
            wl_lo_list.append(lo)
            wl_hi_list.append(hi)
            # Barycenter of predicted observed λ in the group (single line → that line’s λ).
            ctr_list.append(float(np.mean(group)))

    return (
        np.asarray(wl_lo_list, dtype=np.float64),
        np.asarray(wl_hi_list, dtype=np.float64),
        np.asarray(ctr_list, dtype=np.float64),
    )


def line_mask_n_chunks_for_order_line_by_line(
    wave_um: np.ndarray,
    mask_lines: MaskLines,
    *,
    v_center_m_s: float,
    zb: float,
    min_good_pixels: int = 10,
) -> int:
    """Return ``n_chunk`` for one order under :func:`build_line_mask_chunk_wavelength_intervals_line_by_line`."""
    wl_lo, wl_hi, _ctr = build_line_mask_chunk_wavelength_intervals_line_by_line(
        wave_um,
        mask_lines,
        v_center_m_s=v_center_m_s,
        zb=zb,
        min_good_pixels=min_good_pixels,
    )
    return int(len(wl_lo))


# =============================================================================
# Broad velocity grid defaults, hybrid merge, and CCF error-propagation helpers
# =============================================================================










def _nanmean_axis0_ignore_empty_columns(arr: np.ndarray) -> np.ndarray:
    """
    Column-wise mean of finite values (axis 0). Columns with no finite entries are NaN.

    Stacked CCF arrays often have velocity bins where **every** chunk row is NaN;
    ``np.nanmean(..., axis=0)`` then emits ``RuntimeWarning: Mean of empty slice``
    once per such column. This implementation matches ``nanmean`` for mixed
    finite/NaN columns but avoids that warning.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("_nanmean_axis0_ignore_empty_columns: expected a 2-D array")
    n_row, n_col = a.shape
    if n_col == 0:
        return np.array([], dtype=np.float64)
    if n_row == 0:
        return np.full(n_col, np.nan, dtype=np.float64)
    fin = np.isfinite(a)
    cnt = np.sum(fin, axis=0, dtype=np.float64)
    s = np.nansum(a, axis=0)
    out = np.full(n_col, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


def _nan_stderr_of_mean_axis0(stacked_sigma: np.ndarray, stacked_ccf: np.ndarray) -> np.ndarray:
    """
    Standard error of ``nanmean(stacked_ccf, axis=0)`` when each row is an independent
    estimate with 1σ ``stacked_sigma[i, :]`` at each velocity.

    Uses :math:`\\sigma_{\\mu,j}=\\sqrt{\\sum_i \\sigma_{ij}^2}\\,/\\,n_j` where the sum
    and :math:`n_j` use the same rows as the column-wise mean (finite ``stacked_ccf``;
    only rows with finite nonnegative ``stacked_sigma`` contribute to the sum of
    variances). This matches the scale of the **mean** CCF; using
    ``sqrt(mean(σ²))`` instead would track a **typical single-row** σ and overstate
    uncertainty on the mean by ~ :math:`\\sqrt{n}` when rows are similar.
    """
    sig = np.asarray(stacked_sigma, dtype=np.float64)
    ccf = np.asarray(stacked_ccf, dtype=np.float64)
    if sig.shape != ccf.shape:
        raise ValueError("_nan_stderr_of_mean_axis0: sigma and ccf stacks must have the same shape")
    if ccf.ndim != 2:
        raise ValueError("_nan_stderr_of_mean_axis0: expected 2-D arrays")
    n_row, n_col = ccf.shape
    if n_col == 0:
        return np.array([], dtype=np.float64)
    if n_row == 0:
        return np.full(n_col, np.nan, dtype=np.float64)
    fin_ccf = np.isfinite(ccf)
    cnt = np.sum(fin_ccf, axis=0, dtype=np.float64)
    use = fin_ccf & np.isfinite(sig) & (sig >= 0)
    sum_sq = np.sum(np.where(use, sig**2, 0.0), axis=0)
    out = np.full(n_col, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = np.sqrt(np.maximum(sum_sq[ok], 0.0)) / cnt[ok]
    return out


def _rv_err_slope_propagation_m_s(
    v_m_s: np.ndarray,
    y_ccf: np.ndarray,
    sigma_ccf: np.ndarray | None,
) -> float:
    """
    Diagonal error propagation σ_RV ≈ σ_y / |d(CCF)/dv| (m/s), v in m/s.

    Uses a robust high-percentile of |dCCF/dv| in the central part of the window so a
    single noisy bin does not dominate. When ``sigma_ccf`` is missing, uses ``mad_std``
    of ``y_ccf`` as a noise scale. The result is **not** clipped to an arbitrary
    upper bound; very large values indicate a near-flat CCF or tiny |dCCF/dv|.
    """
    v0 = np.asarray(v_m_s, np.float64).ravel()
    y0 = np.asarray(y_ccf, np.float64).ravel()
    if v0.size != y0.size or v0.size < 5:
        return float("nan")
    s0 = None
    if sigma_ccf is not None:
        s0 = np.asarray(sigma_ccf, np.float64).ravel()
        if s0.size != v0.size:
            s0 = None

    m = np.isfinite(v0) & np.isfinite(y0)
    if s0 is not None:
        m &= np.isfinite(s0) & (s0 > 0)
    v, y = v0[m], y0[m]
    se = s0[m] if s0 is not None else None
    if v.size < 5:
        m2 = np.isfinite(v0) & np.isfinite(y0)
        v, y = v0[m2], y0[m2]
        se = s0[m2] if s0 is not None else None
    if v.size < 5:
        return float("nan")

    o = np.argsort(v)
    v = v[o]
    y = y[o]
    if se is not None:
        se = se[o]
    if float(np.nanmax(v)) <= float(np.nanmin(v)):
        return float("nan")
    dye = np.gradient(y, v)
    abs_g = np.abs(dye)
    lo, hi = np.nanquantile(v, [0.15, 0.85])
    cen = (v >= lo) & (v <= hi)
    if not np.any(cen):
        cen = np.ones_like(v, dtype=bool)
    g_use = abs_g[cen]
    g95 = float(np.nanpercentile(g_use, 95)) if g_use.size else float("nan")
    if not math.isfinite(g95) or g95 <= 0:
        g95 = float(np.nanmax(abs_g))
    if not math.isfinite(g95) or g95 <= 0:
        return float("nan")
    if se is not None and np.any(np.isfinite(se) & (se > 0)):
        sig_y = float(np.nanmedian(se[np.isfinite(se) & (se > 0)]))
    else:
        sig_y = float(mad_std(y, ignore_nan=True))
    if not math.isfinite(sig_y) or sig_y <= 0:
        return float("nan")
    est = sig_y / g95
    if not math.isfinite(est) or est <= 0:
        return float("nan")
    return float(est)


def build_hybrid_merged_velocity_grid(
    broad_min: float,
    broad_max: float,
    broad_step: float,
    center_m_s: float,
    half_width_m_s: float,
    refined_step_m_s: float,
) -> np.ndarray:
    """
    Single CCF velocity list: full ``[broad_min, broad_max]`` sampled at ``broad_step``,
    plus extra samples on ``[center ± half_width_m_s]`` every ``refined_step_m_s``.

    Sorted unique values (rounded) so the merged grid spans the entire broad
    range while increasing resolution around ``center_m_s``.
    """
    vb = float(broad_min)
    vx = float(broad_max)
    vsb = float(broad_step)
    vc = float(center_m_s)
    hw = float(half_width_m_s)
    vsr = float(refined_step_m_s)
    if not np.isfinite(vb) or not np.isfinite(vx) or vx <= vb:
        raise ValueError("build_hybrid_merged_velocity_grid: invalid broad bounds")
    if not np.isfinite(vsb) or vsb <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: broad_step must be finite and > 0")
    if not np.isfinite(vc) or not np.isfinite(hw) or hw <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: center and half-width must be finite, half-width > 0")
    if not np.isfinite(vsr) or vsr <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: refined_step must be finite and > 0")
    r_lo = vc - hw
    r_hi = vc + hw
    if r_hi <= r_lo:
        raise ValueError("build_hybrid_merged_velocity_grid: refined window is empty")
    broad = np.arange(vb, vx + 0.5 * vsb, vsb, dtype=np.float64)
    refined = np.arange(r_lo, r_hi + 0.5 * vsr, vsr, dtype=np.float64)
    merged = np.unique(np.round(np.concatenate((broad, refined)), decimals=6))
    return merged.astype(np.float64)


# =============================================================================
# RVCCF tables and flattened chunk wavelength bounds (L2 RVCCF_FIB* columns)
# =============================================================================


def build_rvccf_chunk_supplement(
    *,
    chunk_id: np.ndarray,
    n_good: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    ccf_2d: np.ndarray,
) -> Table:
    """
    Supplement table with raw CCF(v) per chunk (columns merged into ``RVCCF_FIB*``).

    Parameters
    ----------
    ccf_2d : np.ndarray
        Shape ``(n_chunk, n_vel)``; row ``i`` is the CCF for chunk ``i``.
    """
    n = int(ccf_2d.shape[0]) if ccf_2d.ndim >= 2 else 0
    if n == 0:
        return Table(
            names=["CHUNK_ID", "N_GOOD", "V_MIN", "V_MAX", "V_STEP", "N_VEL", "CCF"],
            dtype=[np.int32, np.int32, np.float64, np.float64, np.float64, np.int32, object],
        )
    if len(chunk_id) != n or len(n_good) != n:
        raise ValueError("build_rvccf_chunk_supplement: chunk_id / n_good length mismatch vs ccf_2d")
    n_vel = int(ccf_2d.shape[1])
    ccf_col = np.empty(n, dtype=object)
    for i in range(n):
        ccf_col[i] = np.asarray(ccf_2d[i], dtype=RVCCF_CCF_ELEM_DTYPE).ravel()
    v_min_f = float(v_min)
    v_max_f = float(v_max)
    v_step_f = float(v_step)
    return Table(
        {
            "CHUNK_ID": np.asarray(chunk_id, dtype=np.int32).ravel(),
            "N_GOOD": np.asarray(n_good, dtype=np.int32).ravel(),
            "V_MIN": np.full(n, v_min_f, dtype=np.float64),
            "V_MAX": np.full(n, v_max_f, dtype=np.float64),
            "V_STEP": np.full(n, v_step_f, dtype=np.float64),
            "N_VEL": np.full(n, n_vel, dtype=np.int32),
            "CCF": ccf_col,
        }
    )


def assemble_rvccf_fib_table(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    wavelengths: np.ndarray,
    orders: np.ndarray,
    inlier: np.ndarray,
    *,
    supplement: Table | None = None,
) -> Table:
    """
    Full per-chunk ``RVCCF_FIB*`` table: RV metrics plus optional CCF supplement columns.

    When ``supplement`` is None or has length 0, ``CCF`` rows are empty arrays
    (``RVCCF_CCF_ELEM_DTYPE``), ``N_GOOD`` is -1, and velocity metadata is NaN / ``N_VEL`` 0.
    """
    n = len(rvs)
    mbjd = np.zeros(n, dtype=np.float64)
    berv = np.zeros(n, dtype=np.float64)
    res = np.full(n, np.nan, dtype=np.float64)
    tell_abs = np.zeros(n, dtype=np.float64)
    bis = np.zeros(n, dtype=np.float64)
    rv_abs = np.zeros(n, dtype=np.float64)
    rv_abs_err = np.zeros(n, dtype=np.float64)
    order_col = np.asarray(orders, dtype=np.int64).ravel()
    if len(order_col) != n:
        order_col = np.full(n, -1, dtype=np.int64)
    order_i16 = np.clip(order_col, -32768, 32767).astype(np.int16)
    wl = np.asarray(wavelengths, dtype=np.float64)
    inlier_u8 = np.asarray(inlier, dtype=np.uint8)

    if supplement is None or len(supplement) == 0:
        chunk_id = np.arange(n, dtype=np.int32)
        n_good = np.full(n, -1, dtype=np.int32)
        v_min_a = np.full(n, np.nan, dtype=np.float64)
        v_max_a = np.full(n, np.nan, dtype=np.float64)
        v_step_a = np.full(n, np.nan, dtype=np.float64)
        n_vel = np.zeros(n, dtype=np.int32)
        ccf_col = np.empty(n, dtype=object)
        for i in range(n):
            ccf_col[i] = np.array([], dtype=RVCCF_CCF_ELEM_DTYPE)
    else:
        if len(supplement) != n:
            raise ValueError(
                f"assemble_rvccf_fib_table: supplement has {len(supplement)} rows, "
                f"expected {n} to match per-chunk RV arrays"
            )
        chunk_id = np.asarray(supplement["CHUNK_ID"], dtype=np.int32)
        n_good = np.asarray(supplement["N_GOOD"], dtype=np.int32)
        v_min_a = np.asarray(supplement["V_MIN"], dtype=np.float64)
        v_max_a = np.asarray(supplement["V_MAX"], dtype=np.float64)
        v_step_a = np.asarray(supplement["V_STEP"], dtype=np.float64)
        n_vel = np.asarray(supplement["N_VEL"], dtype=np.int32)
        ccf_col = np.empty(n, dtype=object)
        for i in range(n):
            raw = supplement["CCF"][i]
            if raw is None or (isinstance(raw, np.ndarray) and raw.size == 0):
                ccf_col[i] = np.array([], dtype=RVCCF_CCF_ELEM_DTYPE)
            else:
                ccf_col[i] = np.asarray(raw, dtype=RVCCF_CCF_ELEM_DTYPE).ravel()

    return Table(
        {
            "MBJDW": mbjd,
            "BERVW": berv,
            "RV": np.asarray(rvs, dtype=np.float64),
            "RV_ERR": np.asarray(rv_errs, dtype=np.float64),
            "ORDER": order_i16,
            "WAVELENGTH": wl,
            "RESOLUTION": res,
            "TELL_ABS": tell_abs,
            "BIS": bis,
            "RV_ABS": rv_abs,
            "RV_ABS_ERR": rv_abs_err,
            "INLIER": inlier_u8,
            "CHUNK_ID": chunk_id,
            "N_GOOD": n_good,
            "V_MIN": v_min_a,
            "V_MAX": v_max_a,
            "V_STEP": v_step_a,
            "N_VEL": n_vel,
            "CCF": ccf_col,
        }
    )


def flat_chunk_wl_lo_hi(
    wave_2d: np.ndarray,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    total_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wavelength bounds per flattened chunk index (matches Numba chunking in
    :func:`_process_all_orders_ccf_numba`).
    """
    wl_lo = np.full(total_chunks, np.nan, dtype=np.float64)
    wl_hi = np.full(total_chunks, np.nan, dtype=np.float64)
    for j in range(len(order_indices)):
        oi = int(order_indices[j])
        w = wave_2d[oi]
        w_min = float(np.nanmin(w))
        w_max = float(np.nanmax(w))
        n_chunks = int(n_chunks_per_order[j])
        if n_chunks < 1 or not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue
        start = int(chunk_start[j])
        for k in range(n_chunks):
            idx = start + k
            if idx >= total_chunks:
                break
            wl_lo[idx] = w_min + k * (w_max - w_min) / n_chunks
            wl_hi[idx] = w_min + (k + 1) * (w_max - w_min) / n_chunks
    return wl_lo, wl_hi


# =============================================================================
# Gaussian CCF fit (LM + σ heuristics; config in km/s, public API in m/s)
# =============================================================================


def _gaussian_ccf_model(v: np.ndarray, amplitude: float, v0: float, sigma: float, baseline: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((v - v0) / np.maximum(sigma, 1e-10)) ** 2) + baseline


def clip_gaussian_plot_velocity_range_m_s(
    lo_m_s: float,
    hi_m_s: float,
    *,
    v_grid_min_m_s: float | None = None,
    v_grid_max_m_s: float | None = None,
    v_samples_m_s: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Bound ``[lo_m_s, hi_m_s]`` for evaluating a Gaussian CCF model.

    The returned interval is the intersection of ``[lo_m_s, hi_m_s]`` with the span of
    the CCF velocity axis (``v_grid_*`` or ``v_samples_m_s``), further clipped to the
    module absolute limits ``V_MIN_DEFAULT_M_S`` … ``V_MAX_DEFAULT_M_S`` (m/s), matching the broad CCF grid
    used in :func:`resolve_ccf_broad_grid_m_s` and related paths.
    """
    if v_grid_min_m_s is None or v_grid_max_m_s is None:
        if v_samples_m_s is not None and np.size(v_samples_m_s) > 0:
            va = np.asarray(v_samples_m_s, dtype=np.float64).ravel()
            gmin = float(np.nanmin(va))
            gmax = float(np.nanmax(va))
        else:
            gmin, gmax = float(V_MIN_DEFAULT_M_S), float(V_MAX_DEFAULT_M_S)
    else:
        gmin = float(v_grid_min_m_s)
        gmax = float(v_grid_max_m_s)
    gmin = max(gmin, float(V_MIN_DEFAULT_M_S))
    gmax = min(gmax, float(V_MAX_DEFAULT_M_S))
    lo = max(float(lo_m_s), gmin)
    hi = min(float(hi_m_s), gmax)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        lo, hi = gmin, gmax
    return lo, hi


def estimate_ccf_uncertainty_highpass_residual(
    ccf: np.ndarray,
    *,
    lowpass_window: int = 51,
    savgol_polyorder: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate a **single** CCF uncertainty σ (same units as ``ccf``): Savitzky–Golay
    low-pass smooth, subtract to form a high-frequency residual, then take the RMS of
    successive differences in that residual (pixel-to-pixel scale).

    **Line-mask CCF path** (:func:`run_ccf_line_mask_for_fiber`; ``ccf_line_mask`` primitive): this is the primary model
    for ``ccf_err`` on the per-order mean CCF (uniform array filled with ``sigma``), and
    for the stacked rows passed into :func:`finalize_ccf_preview_dict` (replacing the
    former per-chunk weighted SD + stderr-of-mean combination).

    Parameters
    ----------
    ccf : ndarray
        CCF samples (same shape as used for Gaussian fitting).
    lowpass_window : int
        Savitzky–Golay window length (forced odd); capped to array length. Larger ⇒ stronger
        low-pass (more large-scale structure removed from the residual).
    savgol_polyorder : int
        Savitzky–Golay polynomial order (must be ``< window_length``).

    Returns
    -------
    sigma : float
        Scalar σ applied uniformly at each velocity bin when building ``ccf_err`` for
        :func:`fit_ccf_rv`.
    ccf_lowpass : ndarray
        Low-pass (smooth) component of ``ccf``.
    residual : ndarray
        ``ccf - ccf_lowpass`` (high-pass remainder).
    """
    y = np.asarray(ccf, dtype=np.float64).ravel()
    if np.any(~np.isfinite(y)):
        m = np.isfinite(y)
        med = float(np.nanmedian(y[m])) if np.any(m) else 0.0
        if not np.isfinite(med):
            med = 0.0
        y = np.where(np.isfinite(y), y, med)
    n = int(y.size)
    if n < 3:
        z = np.zeros_like(y)
        return float("nan"), np.asarray(y, dtype=np.float64), z

    wl = int(lowpass_window)
    if wl % 2 == 0:
        wl += 1
    wl = max(5, wl)
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
        wl = max(5, wl)
    po = int(savgol_polyorder)
    po = max(1, min(po, wl - 1))

    try:
        smooth = savgol_filter(y, wl, po, mode="interp")
    except Exception:
        size = max(3, min(wl, n if n % 2 == 1 else n - 1))
        if size % 2 == 0:
            size -= 1
        size = max(3, size)
        smooth = uniform_filter1d(y, size=size, mode="nearest")

    res = y - smooth
    dd = np.diff(res)
    sigma = float(np.sqrt(np.nanmean(dd * dd)))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(res))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = max(float(np.nanstd(y)), 1e-30)
    return sigma, smooth, res


@dataclass
class CCFFitConfig:
    """
    Tunables for :func:`fit_ccf_rv` two-pass Gaussian + constant fits to :math:`\\mathrm{CCF}(v)`.

    Velocity windows in the first and second pass are expressed in **km s\\ :sup:`−1`**
    (``*_km_s`` fields); RV results from :func:`fit_ccf_rv` are still returned in **m s\\ :sup:`−1`**.

    Attributes
    ----------
    pass1_half_width_km_s, pass2_half_width_km_s : float
        Half-width of the velocity window (km/s) around the peak for LM fit pass 1 and 2.
        Pass 1 is wider to stabilize the baseline; pass 2 is narrower for refinement.
    initial_stddev_km_s : float
        Initial Gaussian width (km/s) for the Astropy model before fitting.
    min_points : int
        Minimum number of finite CCF samples required in a window to attempt a fit.
    emission_strength_ratio : float
        Threshold for classifying absorption vs. emission line profiles (affects peak search).
    cal_mask_methods : frozenset
        ``mask_method`` values treated as calibration (etalon/LFC/ThAr) for fit behavior.
    levmar_maxiter : int
        Max Levenberg–Marquardt iterations per χ² minimization step in Astropy’s fitter.
    sigma_clip_unclipped_prefit_passes : int
        Repeated **unclipped** LM fits on the full window before sigma-clipping rounds.
    sigma_clip_enable : bool
        If False, skip residual-based clipping after the prefit passes.
    sigma_clip_max_iter : int
        Maximum fit → clip → refit rounds after prefit.
    sigma_clip_n_sigma, sigma_clip_n_sigma_relaxed : float
        MAD multipliers for residual clipping (strict vs. relaxed early rounds).
    sigma_clip_relax_first_clip_iters : int
        Number of initial clipping iterations using the relaxed multiplier.
    sigma_clip_min_keep_frac : float
        Minimum fraction of points to retain when clipping (guards over-aggressive removal).
    rv_err_cov_combine : str
        How to merge gradient-based CCF error with LevMar covariance (e.g. ``"max"``).
    rv_err_bootstrap_samples : int
        Parametric bootstrap draws on ``ccf`` noise; ``0`` disables.
    rv_err_bootstrap_seed : int or None
        RNG seed for bootstrap when ``rv_err_bootstrap_samples > 0``.
    """

    # Wider windows help the LM Gaussian recover a sensible baseline away from the peak.
    pass1_half_width_km_s: float = 800.0
    pass2_half_width_km_s: float = 120.0
    initial_stddev_km_s: float = 5.0
    min_points: int = 5
    emission_strength_ratio: float = 1.02
    cal_mask_methods: frozenset = frozenset({"thar", "lfc", "etalon"})
    # Levenberg–Marquardt iterations per Gaussian fit (χ² with optional inverse-variance weights).
    levmar_maxiter: int = 1500
    # Unclipped prefit passes on the **full** velocity window (no point removal): repeated
    # LM fits to drive χ² down before any sigma-clipping refinement.
    sigma_clip_unclipped_prefit_passes: int = 2
    # After prefit, this many iterations: fit → clip outliers → refit (each pass uses LM).
    sigma_clip_enable: bool = True
    sigma_clip_max_iter: int = 10
    # MAD multiplier for residual clipping after prefit (tighter than relaxed).
    sigma_clip_n_sigma: float = 4.0
    # Weaker clipping for the first few post-prefit rounds (keep more points while refining).
    sigma_clip_n_sigma_relaxed: float = 6.0
    sigma_clip_relax_first_clip_iters: int = 3
    sigma_clip_min_keep_frac: float = 0.30
    # How to merge gradient-based CCF error with LevMar covariance when both pass plausibility.
    # ``"max"`` is more conservative than historic ``"min"`` (often gave tiny σ_RV).
    rv_err_cov_combine: str = "max"
    # Parametric bootstrap: refit Gaussian on CCF + N(0, σ_ccf) noise (same window/weights as
    # the final pass). 0 disables. Typical 24–64 for science; each sample is one LM fit.
    rv_err_bootstrap_samples: int = 0
    rv_err_bootstrap_seed: int | None = None




# Lighter LevMar / σ-clip budget for :func:`finalize_ccf_preview_dict` only (Plotly metadata).
# Pipeline RV fits on order-mean CCFs keep the default :class:`CCFFitConfig`.
CCF_PREVIEW_SLOT_FIT_CONFIG = CCFFitConfig(
    levmar_maxiter=450,
    sigma_clip_unclipped_prefit_passes=1,
    sigma_clip_max_iter=6,
)


def _line_mask_ccf_fit_free_baseline_requested(
    *,
    fit_ccf_baseline_free: bool | None,
    line_mask_line_window_by_line: bool | None,
    ccf_line_mask_stored_negated: bool | None = None,
) -> bool:
    """
    True when :func:`fit_ccf_rv` should treat the CCF constant term as an LM parameter.

    For line-mask **line-by-line** chunking (``window_line_by_line``), the hybrid/fine velocity
    span is often only a few km s\\ :sup:`−1` wide, so a robust median baseline estimated from
    essentially the entire profile is a poor substitute for wings; fitting ``Const1D`` jointly
    with the Gaussian avoids that bias.

    For ``ccf_line_mask_stored_negated=True`` (line-mask / ``ccf_line_mask`` stored CCF rows), the baseline is
    **not** a free LM parameter: a free ``Gaussian1D + Const1D`` is χ²-degenerate (Gaussian
    amplitude collapses toward zero and the line is absorbed by the constant), which breaks
    Plotly Gaussian overlays. Pass half-width caps still apply when
    ``ccf_line_mask_stored_negated`` is true (see :func:`fit_ccf_rv`), so fixed sigma-clipped
    baselines remain local to the hybrid grid.
    """
    if fit_ccf_baseline_free is not None:
        return bool(fit_ccf_baseline_free)
    if not bool(line_mask_line_window_by_line):
        return False
    if bool(ccf_line_mask_stored_negated):
        return False
    return True




def _bootstrap_gaussian_rv_std_m_s(
    gx_km: np.ndarray,
    gy0: np.ndarray,
    ge: np.ndarray,
    *,
    seed_mean_km: float,
    seed_std_km: float,
    profile: str,
    n_samples: int,
    rng: np.random.Generator,
    maxiter: int = 400,
    free_baseline: bool = False,
    ccf_line_mask_stored_negated: bool | None = None,
) -> float:
    """
    Scatter of the fitted Gaussian mean (m/s) when each CCF sample is perturbed by its
    reported σ (i.i.d.); ignores bin-to-bin correlation. Uses weighted LM like the main fit.
    """
    if n_samples < 4 or gy0.size < 5:
        return float("nan")
    gx = np.asarray(gx_km, np.float64).ravel()
    y0 = np.asarray(gy0, np.float64).ravel()
    gerr = np.asarray(ge, np.float64).ravel()
    if gx.size != y0.size or gerr.size != y0.size:
        return float("nan")
    good = np.isfinite(gx) & np.isfinite(y0) & np.isfinite(gerr) & (gerr > 0)
    if int(np.count_nonzero(good)) < 5:
        return float("nan")
    gx = gx[good]
    y0 = y0[good]
    gerr = gerr[good]
    w = 1.0 / (gerr * gerr)
    ft = fitting.LevMarLSQFitter()
    means = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        yn = y0 + rng.standard_normal(y0.shape) * gerr
        ys = _levmar_ccf_y_scale(yn)
        yn_s = np.asarray(yn, dtype=np.float64) * ys
        if free_baseline:
            bg_raw = float(np.nanmedian(np.asarray(yn, dtype=np.float64)))
            if not np.isfinite(bg_raw):
                bg_raw = 0.0
        else:
            bg_raw = _ccf_background_sigma_clipped_median(np.asarray(yn, dtype=np.float64))
            if not np.isfinite(bg_raw):
                bg_raw = float(np.median(yn))
        bmed = float(bg_raw * ys)
        gy = yn_s - bmed
        span = abs(float(np.nanmax(gy) - np.nanmin(gy))) * 100.0
        amp = (
            max(-1e7, float(np.nanmin(gy)) - span)
            if profile == "emission"
            else min(1e7, float(np.nanmax(gy)) + span)
        )
        if bool(ccf_line_mask_stored_negated):
            amp = max(float(amp), 1e-30)
        amp_floor_s = 0.0
        if free_baseline and (bool(ccf_line_mask_stored_negated) or profile == "absorption"):
            amp_floor_s = _ccf_gaussian_amp_floor_free_baseline_scaled(gy, ys)
            amp = max(float(amp), 2.0 * amp_floor_s)
        try:
            init_b = Gaussian1D(
                amplitude=amp, mean=float(seed_mean_km), stddev=float(seed_std_km)
            ) + Const1D(amplitude=bmed)
            if free_baseline and (bool(ccf_line_mask_stored_negated) or profile == "absorption"):
                init_b.amplitude_0.min = float(amp_floor_s)
            elif bool(ccf_line_mask_stored_negated):
                init_b.amplitude_0.min = 0.0
            if not free_baseline:
                init_b.amplitude_1.fixed = True
            w_eff = np.asarray(w, dtype=np.float64) / (ys * ys)
            gfit = _levmar_lsq_fit(ft, init_b, gx, yn_s, weights=w_eff, maxiter=maxiter)
            means[i] = float(gfit.mean_0.value) * 1000.0
        except Exception:
            means[i] = float("nan")
    fin = np.isfinite(means)
    if int(np.count_nonzero(fin)) < max(4, n_samples // 4):
        return float("nan")
    mv = means[fin]
    return float(np.std(mv, ddof=1))


def dv_per_pixel_m_s(wl_um: float) -> float:
    """
    Rough echelle resolving element per detector pixel in velocity units (m/s).

    Uses the HISPEC-style mapping ``order ≈ 147 / λ (µm)`` and ``n_pix = 4096`` so
    ``Δv ≈ c / (order · n_pix)``. This is a **diagnostic** scale for CCF sampling
    and error heuristics, not a physical FWHM.

    With :math:`\\Delta v = c\\,(\\Delta\\lambda/\\lambda)`, center wavelength
    :math:`\\lambda \\approx 147/\\,{\\rm order}` µm, and pixel width
    :math:`\\Delta\\lambda \\approx (147/{\\rm order}^2)/4096` µm, this gives
    :math:`\\Delta v = c / ({\\rm order}\\cdot4096)` m/s. Typical L2 scales:
    BSPEC (orders ~110–148) ~494–665 m/s per pixel; RSPEC (orders ~60–97)
    ~755–1220 m/s per pixel.

    Parameters
    ----------
    wl_um : float
        Wavelength in micron (positive, finite).

    Returns
    -------
    float
        Δv per pixel (m/s), or NaN if ``wl_um`` is non-finite or non-positive.
    """
    if not np.isfinite(wl_um) or wl_um <= 0:
        return float("nan")
    return float(C_M_S / (max(147.0 / wl_um, 1.0) * 4096.0))


def _subsample_velocity_ccf_for_gaussian_fit(
    vm: np.ndarray,
    y: np.ndarray,
    y_err_ok: np.ndarray | None,
    v_step_m_s: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, int]:
    """
    Uniform stride subsample along velocity when the grid is very long (line-mask broad scans).

    Preserves ordering; scales ``v_step_m_s`` by the stride so error heuristics stay consistent
    with the effective sample spacing.
    """
    n = int(vm.size)
    ms = int(max_samples)
    if n <= ms or ms < 8:
        return vm, y, y_err_ok, float(v_step_m_s), 1
    stride = int(math.ceil(n / float(ms)))
    stride = max(1, stride)
    vm_s = np.asarray(vm[::stride], dtype=np.float64)
    y_s = np.asarray(y[::stride], dtype=np.float64)
    err_s = (
        np.asarray(y_err_ok[::stride], dtype=np.float64)
        if y_err_ok is not None and y_err_ok.size == n
        else None
    )
    vstep_s = float(v_step_m_s) * float(stride)
    return vm_s, y_s, err_s, vstep_s, stride


def _ccf_fit_debug_jsonable(v):
    """Reduce fit_ccf_rv debug context values to JSON-friendly types."""
    if isinstance(v, (bool, str, type(None))):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        xf = float(v)
        return xf if np.isfinite(xf) else None
    if isinstance(v, dict):
        return {str(k): _ccf_fit_debug_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_ccf_fit_debug_jsonable(x) for x in v]
    return str(v)


def _levmar_lsq_fit(ft, init, *args, **kwargs):
    """
    Run :class:`~astropy.modeling.fitting.LevMarLSQFitter`.

    Ill-conditioned fits can make ``scipy.optimize.leastsq`` overflow when forming
    ``cov_x``; downstream code already rejects non-physical covariances via ``_lm``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ft(init, *args, **kwargs)


def _levmar_ccf_y_scale(gy: np.ndarray) -> float:
    """
    Multiplicative scale so ``gy * scale`` has max(|·|) ≈ 1.

    ``scipy.optimize.leastsq`` (used by Astropy :class:`~astropy.modeling.fitting.LevMarLSQFitter`)
    reports ``ierr`` outside {1,…,4} when the problem is poorly scaled (common for mask-CCF
    amplitudes ~1e-18), which triggers ``AstropyUserWarning: The fit may be unsuccessful``.
    Scaling **y** (and inverse-variance weights consistently) preserves Gaussian **mean** and
    **stddev** in velocity while improving conditioning.
    """
    m = float(np.nanmax(np.abs(np.asarray(gy, dtype=np.float64))))
    if not math.isfinite(m) or m <= 0:
        return 1.0
    return 1.0 / m


def _ccf_gaussian_amp_floor_free_baseline_scaled(
    gy_det: np.ndarray,
    ys: float,
    *,
    frac_of_span: float = 0.005,
    frac_of_peak: float = 0.05,
    abs_scaled_floor: float = 0.0,
) -> float:
    """
    Positive lower bound on Gaussian amplitude (LevMar **scaled** ``gy`` units) when the
    compound model is ``Gaussian1D + Const1D`` with **free** ``Const1D``: otherwise the fitter
    can drive Gaussian amplitude to **exactly zero** and absorb the line in the constant
    (χ² degeneracy), yielding flat preview Gaussians.

    Uses both a fraction of the detrended peak-to-valley span and a fraction of the
    **positive peak** above the median-detrended baseline: span alone can be noise-dominated
    on narrow mask-CCF grids, leaving a tiny floor that still allows an effectively flat Gaussian.

    Optional ``abs_scaled_floor`` (same units as scaled ``gy``) forces a minimum when
    peak-derived floors would otherwise be numerically negligible.
    """
    contr = float(np.nanmax(gy_det) - np.nanmin(gy_det))
    if not math.isfinite(contr):
        contr = 0.0
    contr = max(0.0, contr)
    span_floor = max(1e-30 * float(ys), float(frac_of_span) * contr)
    pk = float(np.nanmax(gy_det))
    if not math.isfinite(pk):
        pk = 0.0
    peak_floor = max(0.0, float(frac_of_peak) * max(0.0, pk))
    return max(span_floor, peak_floor, float(abs_scaled_floor))






def fit_ccf_rv(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    n_good: int | None = None,
    return_params: bool = False,
    *,
    ccf_err: np.ndarray | None = None,
    mask_method: str | None = None,
    config: CCFFitConfig | None = None,
    vel_span_m_s_per_pixel: float | None = None,
    two_pass: bool = True,
    max_velocity_samples: int | None = 8192,
    ccf_line_mask_stored_negated: bool | None = None,
    fit_ccf_baseline_free: bool | None = None,
    line_mask_line_window_by_line: bool | None = None,
    gaussian_seed_mean_m_s: float | None = None,
):
    """
    Fit a Gaussian plus constant baseline to a precomputed CCF and return RV ± error in m s⁻¹.

    Uses a robust emission/absorption profile guess, :class:`~astropy.modeling.models.Gaussian1D`
    plus :class:`~astropy.modeling.models.Const1D` on the CCF in a sliding velocity
    window (values are scaled to ~unit magnitude before LM; RV parameters are unchanged). By
    default the constant baseline is **fixed** to :func:`_ccf_background_sigma_clipped_median`
    of the CCF samples in each fit window (not a free LM parameter). When
    ``fit_ccf_baseline_free=True``, or when ``line_mask_line_window_by_line`` is true (narrow
    line-mask line-by-line hybrid grids) **and** ``ccf_line_mask_stored_negated`` is not true, the
    baseline is a **free**
    :class:`~astropy.modeling.models.Const1D` term fit jointly with the Gaussian, and pass
    half-widths are capped to a fraction of the available velocity span so the fit is not
    dominated by an artificial “global median” baseline across the whole short profile.
    The same half-width caps apply when ``ccf_line_mask_stored_negated`` is true (mask-line CCF),
    so fixed-baseline fits on short hybrid grids do not inherit multi-hundred-km pass windows.
    One or two Levenberg–Marquardt passes each use a **wide** velocity
    window (see :class:`CCFFitConfig`) to stabilize the Gaussian **amplitude** and width.
    Within each window the fitter runs **unclipped** χ² passes on the full sample
    (repeated LM with ``levmar_maxiter``), then **sigma-clipping** refinement with relaxed
    thresholds on the first clipping rounds. Formal uncertainty may use a KPF-inspired
    ``ccf_error`` scaling when plausible, else covariance from the fitter, else a
    σ/√n fallback.

    Parameters
    ----------
    v_grid_m_s : np.ndarray
        Velocity samples (m s⁻¹), usually sorted (e.g. broad scan and/or hybrid merge).
    ccf : np.ndarray
        CCF values on the same grid.
    v_step_m_s : float
        Effective **finest** velocity spacing (m s⁻¹) for plausibility checks and
        ``n_sp`` scaling — typically :func:`effective_ccf_v_step_for_fit` on ``v_grid_m_s``.
    n_good : int or None, optional
        Number of good spectral pixels in the chunk (for fallback error scaling).
    return_params : bool, optional
        If True, return ``(rv, rv_err, params_dict, diag_dict)``; else ``(rv, rv_err)``.
    mask_method : str or None, optional
        If in ``CCFFitConfig.cal_mask_methods``, use absorption-style calibration
        peak logic.
    config : CCFFitConfig or None, optional
        Override fit windows and thresholds.
    vel_span_m_s_per_pixel : float or None, optional
        Spectral sampling scale passed into error heuristics.
    two_pass : bool, optional
        If True (default), run a second fit narrowed around the first mean.
    max_velocity_samples : int or None, optional
        When the input grid has more than this many finite samples, apply uniform stride
        subsampling for the Gaussian fit only (default ``8192``). Set ``None`` to always
        use the full grid. Does not change stored CCF arrays elsewhere — only speeds LM
        fitting on very long line-mask broad scans.
    ccf_err : np.ndarray or None, optional
        Per-sample CCF uncertainty (same shape as ``ccf``). When finite and
        predominantly positive, Levenberg–Marquardt fits use ``weights = 1/ccf_err²``
        on the raw CCF samples in each pass window.
    ccf_line_mask_stored_negated : bool or None, optional
        When not ``None``, recorded in the diagnostics dict (when ``return_params`` is
        True) as ``"ccf_line_mask_stored_negated"``. Use ``True`` for mask-line CCF rows
        produced by :func:`rv_per_chunks_line_mask` / :func:`run_ccf_line_mask_for_fiber`
        (stored convention: negated from the raw weighted mask statistic so the peak is
        near the true RV). When ``True``, the fitter **does not** use the data-driven
        emission/absorption heuristic: it assumes a **peak-up** line (positive
        :class:`~astropy.modeling.models.Gaussian1D` amplitude in scaled units, seed at the
        discrete maximum) and sets ``Gaussian1D.amplitude`` lower bound to ``0`` so LM
        cannot flip to a spurious negative-lobe solution on flat mask CCFs.
        ``None`` (default) omits the key — caller convention unknown.
    fit_ccf_baseline_free : bool or None, optional
        When ``True``, fit the CCF constant baseline in LM (do not fix ``Const1D``). When
        ``None`` (default), use :func:`_line_mask_ccf_fit_free_baseline_requested` with
        ``line_mask_line_window_by_line`` to decide.
    line_mask_line_window_by_line : bool or None, optional
        Line-mask line-by-line chunking flag (same meaning as ``window_line_by_line`` on the
        primitive). When ``fit_ccf_baseline_free`` is ``None`` and this is true, free-baseline
        mode is enabled automatically unless ``ccf_line_mask_stored_negated`` is true (see
        :func:`_line_mask_ccf_fit_free_baseline_requested`).
    gaussian_seed_mean_m_s : float or None, optional
        When ``ccf_line_mask_stored_negated`` is true and this lies on the velocity grid, the
        discrete peak seed uses **argmax(CCF)** restricted to samples within roughly the hybrid
        core around this value (typically ``ccf_input_velocity``), instead of the global maximum
        on the full grid. That avoids spurious seeds on stacked mean CCFs and aligns pass-1 with
        the injected search center.

    Returns
    -------
    tuple
        If ``return_params`` is False: ``(rv_ms, rv_err_ms)``.
        If True: ``(rv_ms, rv_err_ms, params, diag)`` with fitted Gaussian parameters
        and diagnostic metadata.

    Notes
    -----
    Design parallels public KPF PRV CCF-fitting ideas; see module ``References``.
    When both the gradient-based CCF error and the Levenberg–Marquardt covariance on
    the Gaussian mean pass plausibility checks, :class:`CCFFitConfig` field
    ``rv_err_cov_combine`` chooses **``min``** (historic) or **``max``**
    (more conservative; default).

    Optional **parametric bootstrap** (``rv_err_bootstrap_samples`` > 0): repeatedly
    add i.i.d. ``N(0, ccf_err)`` noise to the final-window CCF samples, refit the
    weighted Gaussian, and take the sample standard deviation of fitted means (m/s).
    The reported ``rv_err`` is **at least** that bootstrap scatter when bootstrap
    runs (still diagonal / uncorrelated noise; see Notes).

    **Self-consistency of ``ccf_err`` → ``rv_err`` (what is *not* done).**

    - ``ccf_err`` is used only as **diagonal** inverse-variance weights ``1/σ²`` on
      the **raw** CCF samples inside each fit window (Gaussian + constant model). For the line-mask primitive (``ccf_line_mask``),
      those σ come from the weighted SD of per-line :math:`F\\,\\mathrm{SN}` about the
      chunk CCF, then combined across chunks as a standard error of the mean; this is
      not a full propagation from pixel flux errors, and **bin-to-bin correlations** in
      :math:`v` are ignored.
    - The Levenberg–Marquardt **covariance** on the Gaussian mean (when returned by
      Astropy) is a **local** linearization error for the weighted nonlinear fit; it
      does not add systematic terms from model mismatch (true line profile ≠ Gaussian).
    - The **gradient-based** ``_gradient_ccf_err_km`` branch uses gradients of the **raw** CCF
      (``√|CCF|`` scaling) and **does not** use ``ccf_err``. It returns **0** if any
      CCF sample in the fit window is **negative**, so for strongly absorption-shaped
      curves the gradient branch may be inactive and ``rv_err`` relies on covariance or
      fallbacks. With ``rv_err_cov_combine="min"``, **``min(gradient, cov)``** can still
      yield an **optimistically small** ``rv_err``; default ``"max"`` is more conservative.
    - If the LM Gaussian **width** collapses toward zero, the formal covariance on the
      mean can reach ~1e-16 m/s while still passing ``_ok_err``; such fits set
      ``diag["lm_gaussian_width_plausible"]=False``, drop the covariance branch, and
      combine ``rv_err`` with :func:`_rv_err_slope_propagation_m_s` (CCF + ``ccf_err``)
      and a widened ``σ/√n`` fallback using ``v_step``.
    - When the LevMar **covariance** is used for ``rv_err``, slope propagation is still
      computed for ``diag["rv_err_slope_propagation_m_s"]`` but is **not** forced as a
      minimum on ``rv_err`` (no 200 km/s-style floor from ``max(rv_err, σ_slope)``).
    """
    cfg = config or CCFFitConfig()
    free_bl = _line_mask_ccf_fit_free_baseline_requested(
        fit_ccf_baseline_free=fit_ccf_baseline_free,
        line_mask_line_window_by_line=line_mask_line_window_by_line,
        ccf_line_mask_stored_negated=ccf_line_mask_stored_negated,
    )
    _ccf_line_mask_diag: dict[str, bool] = {}
    if ccf_line_mask_stored_negated is not None:
        _ccf_line_mask_diag["ccf_line_mask_stored_negated"] = bool(ccf_line_mask_stored_negated)

    def _fail(msg: str, **ctx):
        d = {
            "fit_ok": False,
            "used_covariance_for_error": False,
            "used_sigma_fallback": False,
            "used_gradient_ccf_error": False,
            "used_ccf_pointwise_error": False,
            "sigma_clip_enabled": bool(cfg.sigma_clip_enable),
            "sigma_clip_iterations": 0,
            "sigma_clip_n_points_initial": 0,
            "sigma_clip_n_points_final": 0,
            "rv_err_bootstrap_sigma_m_s": float("nan"),
            "used_rv_err_bootstrap": False,
            "rv_err_cov_combine": str(cfg.rv_err_cov_combine),
            "applied_slope_propagation_floor": False,
            "message": msg,
        }
        if ctx:
            dbg = {k: _ccf_fit_debug_jsonable(v) for k, v in ctx.items()}
            d["debug_context"] = dbg
            try:
                extra = json.dumps(dbg, sort_keys=True)
            except (TypeError, ValueError):
                extra = str(dbg)
            logger.warning("CCF fit: %s | %s", msg, extra)
        else:
            logger.warning("CCF fit: %s", msg)
        d.update(_ccf_line_mask_diag)
        return (np.nan, np.nan, {}, d) if return_params else (np.nan, np.nan)

    vg = np.asarray(v_grid_m_s, np.float64).ravel()
    y0 = np.asarray(ccf, np.float64).ravel()
    ok = np.isfinite(vg) & np.isfinite(y0)
    n_ok_pre = int(np.sum(ok))
    if n_ok_pre < cfg.min_points + 2:
        return _fail(
            "insufficient valid points",
            fit_stage="input_grid",
            reason="Too few finite (velocity, CCF) pairs after masking non-finite values",
            n_finite_pairs=n_ok_pre,
            min_pairs_required=int(cfg.min_points + 2),
            min_points_fit_config=int(cfg.min_points),
            raw_grid_size=int(vg.size),
        )
    vm, y = vg[ok], y0[ok]
    y_err_ok: np.ndarray | None = None
    if ccf_err is not None:
        er0 = np.asarray(ccf_err, np.float64).ravel()
        if er0.size == vg.size:
            y_err_ok = er0[ok]
    v_step_fit = float(v_step_m_s)
    subsample_stride = 1
    if max_velocity_samples is not None and int(max_velocity_samples) > 0:
        vm, y, y_err_ok, v_step_fit, subsample_stride = _subsample_velocity_ccf_for_gaussian_fit(
            vm, y, y_err_ok, v_step_fit, int(max_velocity_samples)
        )
    vk = vm / 1000.0
    mm = str(mask_method).lower() if mask_method else ""
    forced_line_mask_peak_up = False
    if mm in cfg.cal_mask_methods:
        profile, seed_ms = "absorption", float(vm[int(np.nanargmax(y))])
    elif bool(ccf_line_mask_stored_negated):
        # Mask-line CCF rows use :data:`LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`: the physical line
        # sits at a **maximum** on the stored grid. The generic emission/absorption heuristic
        # mislabels nearly-flat weighted mask CCFs and flips the Gaussian sign, so LM chases the
        # wrong extremum. Fix profile to the code path that uses a **positive** Gaussian
        # amplitude and seeds at the discrete maximum (same convention as cal-mask absorption).
        profile = "absorption"
        forced_line_mask_peak_up = True
        v_lo, v_hi = float(np.min(vm)), float(np.max(vm))
        v_span = float(v_hi - v_lo) if vm.size > 1 else 0.0
        g_seed = gaussian_seed_mean_m_s
        if (
            g_seed is not None
            and math.isfinite(float(g_seed))
            and v_lo <= float(g_seed) <= v_hi
            and v_span > 0.0
        ):
            g0 = float(g_seed)
            # Wide enough to cover the coarse hybrid leg around the injected RV, narrow enough
            # to reject a higher spurious extremum on the opposite end of a ~10 km/s grid.
            half_w = max(3500.0, 0.38 * v_span)
            msk = (vm >= g0 - half_w) & (vm <= g0 + half_w)
            if int(np.count_nonzero(msk)) >= int(cfg.min_points):
                yy, vv = y[msk], vm[msk]
                seed_ms = float(vv[int(np.nanargmax(yy))])
            else:
                seed_ms = float(vm[int(np.nanargmax(y))])
        else:
            seed_ms = float(vm[int(np.nanargmax(y))])
    else:
        med = float(np.median(y))
        imx, imn = int(np.nanargmax(y)), int(np.nanargmin(y))
        profile = (
            "emission"
            if (med - y[imn]) > cfg.emission_strength_ratio * max(y[imx] - med, 1e-30)
            else "absorption"
        )
        seed_ms = float(vm[imn if profile == "emission" else imx])
    rv_gk = seed_ms / 1000.0
    # Use caller's effective finest step (hybrid grids are mostly coarse in |v|;
    # recomputing mean spacing here biased gradient errors high).
    vstep_k = float(v_step_fit) / 1000.0
    vsp_k = (
        float(vel_span_m_s_per_pixel) / 1000.0
        if vel_span_m_s_per_pixel and np.isfinite(vel_span_m_s_per_pixel) and vel_span_m_s_per_pixel > 0
        else vstep_k
    )

    vk_span_km = float(np.nanmax(vk) - np.nanmin(vk)) if vk.size > 1 else 0.0
    # Short hybrid / line-window grids: cap LM pass half-widths to a fraction of the actual
    # velocity span.  Otherwise default pass widths (~hundreds of km) flatten the CCF in the
    # fit window when ``ccf_line_mask_stored_negated`` uses a fixed baseline, or bias free-baseline
    # fits that relied on this cap alone.
    narrow_v_for_hw_cap = (
        vk_span_km > 0.0
        and math.isfinite(vk_span_km)
        and (free_bl or bool(ccf_line_mask_stored_negated))
    )
    if narrow_v_for_hw_cap:
        eps_km = max(1e-6, 2.0 * vstep_k)
        hw_cap1 = max(eps_km, 0.48 * vk_span_km)
        hw_cap2 = max(eps_km, 0.42 * vk_span_km)
        hw1_km = min(float(cfg.pass1_half_width_km_s), float(hw_cap1))
        hw2_km = min(float(cfg.pass2_half_width_km_s), float(hw_cap2))
    else:
        hw1_km = float(cfg.pass1_half_width_km_s)
        hw2_km = float(cfg.pass2_half_width_km_s)

    def _lm(mo, ft):
        if hasattr(mo, "mean_0"):
            mk, sk = float(mo.mean_0.value), abs(float(mo.stddev_0.value))
            mean_pname = "mean_0"
        else:
            mk, sk = float(mo.mean.value), abs(float(mo.stddev.value))
            mean_pname = "mean"
        sm = None
        fi = getattr(ft, "fit_info", {}) or {}
        pc = fi.get("param_cov")
        if pc is None:
            pc = fi.get("cov_x")
        if pc is not None:
            try:
                pc = np.asarray(pc, np.float64)
                j = list(mo.param_names).index(mean_pname)
                if pc.shape[0] > j and pc[j, j] > 0:
                    sm = float(math.sqrt(pc[j, j]))
            except Exception:
                pass
        return mk, sk, sm

    def _pass(center_km: float, hw_km: float, sd0: float, ft):
        lo, hi = center_km - hw_km, center_km + hw_km
        sel = (vk >= lo) & (vk <= hi)
        if not np.any(sel):
            return None, np.array([]), float("nan"), float("nan"), 0.0, False, {}, np.array([]), np.array([])
        gx0 = vk[sel]
        gy00 = y[sel]
        fin = np.isfinite(gx0) & np.isfinite(gy00)
        gx0, gy00 = gx0[fin], gy00[fin]
        if gy00.size < cfg.min_points:
            return None, gx0, float("nan"), float("nan"), 0.0, False, {}, np.array([]), np.array([])
        wts0 = None
        used_pw = False
        if y_err_ok is not None:
            ge = np.asarray(y_err_ok[sel], np.float64)[fin]
            good_w = np.isfinite(ge) & (ge > 0)
            if int(np.count_nonzero(good_w)) >= cfg.min_points:
                med_e = float(np.nanmedian(ge[good_w]))
                floor = max(med_e * 1e-6, 1e-30)
                ge_safe = np.where(good_w, ge, floor)
                wts0 = 1.0 / (ge_safe * ge_safe)
                used_pw = True

        n_start = int(gx0.size)
        min_keep = max(cfg.min_points, int(math.ceil(cfg.sigma_clip_min_keep_frac * n_start)))
        clip_info: dict = {
            "sigma_clip": bool(cfg.sigma_clip_enable),
            "iterations": 0,
            "n_start": n_start,
            "n_final": n_start,
            "prefit_unclipped_passes": int(cfg.sigma_clip_unclipped_prefit_passes),
        }

        def _one_fit(gx: np.ndarray, gy0: np.ndarray, wts: np.ndarray | None):
            ys = _levmar_ccf_y_scale(gy0)
            gy = np.asarray(gy0, dtype=np.float64) * ys
            if free_bl:
                bg_raw = float(np.nanmedian(np.asarray(gy0, dtype=np.float64)))
                if not np.isfinite(bg_raw):
                    bg_raw = 0.0
            else:
                bg_raw = _ccf_background_sigma_clipped_median(
                    np.asarray(gy0, dtype=np.float64)
                )
                if not np.isfinite(bg_raw):
                    bg_raw = float(np.nanmedian(gy0))
            b0 = float(bg_raw * ys)
            gy_det = gy - b0
            span = abs(float(np.nanmax(gy_det) - np.nanmin(gy_det))) * 100.0
            amp = (
                max(-1e7, float(np.nanmin(gy_det)) - span)
                if profile == "emission"
                else min(1e7, float(np.nanmax(gy_det)) + span)
            )
            if bool(ccf_line_mask_stored_negated):
                amp = max(float(amp), 1e-30)
            amp_floor_s = 0.0
            if free_bl and (bool(ccf_line_mask_stored_negated) or profile == "absorption"):
                amp_floor_s = _ccf_gaussian_amp_floor_free_baseline_scaled(gy_det, ys)
                amp = max(float(amp), 2.0 * amp_floor_s)
            init = Gaussian1D(amplitude=amp, mean=center_km, stddev=sd0) + Const1D(amplitude=b0)
            if free_bl and (bool(ccf_line_mask_stored_negated) or profile == "absorption"):
                init.amplitude_0.min = float(amp_floor_s)
            elif bool(ccf_line_mask_stored_negated):
                init.amplitude_0.min = 0.0
            if not free_bl:
                init.amplitude_1.fixed = True
            gx_lo = float(np.min(gx))
            gx_hi = float(np.max(gx))
            if gx_hi > gx_lo and hasattr(init, "mean_0"):
                init.mean_0.min = gx_lo
                init.mean_0.max = gx_hi
            mi = int(max(50, min(int(cfg.levmar_maxiter), 50_000)))
            if wts is not None:
                w_eff = np.asarray(wts, dtype=np.float64) / (ys * ys)
                return _levmar_lsq_fit(ft, init, gx, gy, weights=w_eff, maxiter=mi)
            return _levmar_lsq_fit(ft, init, gx, gy, maxiter=mi)

        gx_w = gx0
        gy0_w = gy00
        wts_w = wts0
        best_g = None
        best_bmed = 0.0
        best_gx = gx_w
        snap_gy0 = np.array([], dtype=np.float64)
        snap_ge = np.array([], dtype=np.float64)

        try:
            prefit = max(0, int(cfg.sigma_clip_unclipped_prefit_passes))
            if cfg.sigma_clip_enable:
                clip_max = max(0, int(cfg.sigma_clip_max_iter))
                total_it = prefit + clip_max
            else:
                clip_max = 0
                total_it = max(1, prefit) if prefit > 0 else 1

            for it in range(total_it):
                if gx_w.size < cfg.min_points:
                    break
                g = _one_fit(gx_w, gy0_w, wts_w)
                pred = np.asarray(g(gx_w), dtype=np.float64)
                resid = gy0_w - pred
                best_g = g
                best_bmed = (
                    float(g.amplitude_1.value)
                    if hasattr(g, "amplitude_1")
                    else float(np.nanmedian(gy0_w))
                )
                best_gx = gx_w
                snap_gy0 = np.asarray(gy0_w, dtype=np.float64).copy()
                if wts_w is not None:
                    snap_ge = (1.0 / np.sqrt(np.maximum(wts_w, 1e-300))).astype(np.float64)
                else:
                    snap_ge = np.full(snap_gy0.shape, np.nan, dtype=np.float64)
                clip_info["iterations"] = it + 1
                if not cfg.sigma_clip_enable:
                    continue
                if it >= total_it - 1:
                    break
                # Unclipped χ² passes: refit on full window only (no point removal).
                if it < prefit:
                    continue
                # Sigma-clipping refinement (relaxed threshold for first few clip rounds).
                clip_idx = it - prefit
                sigma_mult = (
                    float(cfg.sigma_clip_n_sigma_relaxed)
                    if clip_idx < int(cfg.sigma_clip_relax_first_clip_iters)
                    else float(cfg.sigma_clip_n_sigma)
                )
                med_r = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med_r)))
                sig_rob = 1.4826 * mad
                span_y = float(np.nanmax(resid) - np.nanmin(resid)) if resid.size else 1.0
                sig_rob = max(sig_rob, 1e-9 * max(abs(span_y), 1.0))
                thr = float(sigma_mult) * sig_rob
                keep = np.abs(resid) <= thr
                n_kept = int(np.count_nonzero(keep))
                if n_kept >= gx_w.size:
                    break
                if n_kept < min_keep:
                    break
                gx_w = gx_w[keep]
                gy0_w = gy0_w[keep]
                if wts_w is not None:
                    wts_w = wts_w[keep]
            clip_info["n_final"] = int(best_gx.size) if best_g is not None else 0
            if best_g is None:
                return (
                    None,
                    gx0,
                    lo * 1000.0,
                    hi * 1000.0,
                    0.0,
                    used_pw,
                    clip_info,
                    np.array([]),
                    np.array([]),
                )
            return (
                best_g,
                best_gx,
                lo * 1000.0,
                hi * 1000.0,
                best_bmed,
                used_pw,
                clip_info,
                snap_gy0,
                snap_ge,
            )
        except Exception:
            return (
                None,
                gx0,
                lo * 1000.0,
                hi * 1000.0,
                0.0,
                used_pw,
                clip_info,
                np.array([]),
                np.array([]),
            )

    def _gradient_ccf_err_km(fw_km: float, rv0_km: float) -> float:
        v = np.asarray(vk, np.float64).ravel()
        c = np.asarray(y, np.float64).ravel()
        m = np.isfinite(v) & np.isfinite(c)
        v, c = v[m], c[m]
        if v.size < 3 or fw_km <= 0:
            return 0.0
        dv = np.diff(np.sort(v))
        pos = dv[(dv > 0) & np.isfinite(dv)]
        vs = float(np.min(pos)) if pos.size > 0 else 0.0
        if vs <= 0:
            return 0.0
        nsp = 1.0 if not (np.isfinite(vsp_k) and vsp_k > 0) else vs / vsp_k
        s2 = (v >= rv0_km - fw_km * 0.5) & (v <= rv0_km + fw_km * 0.5)
        vf, cf = v[s2], c[s2]
        if cf.size == 0 or np.any(cf < 0):
            return 0.0
        ns = np.sqrt(np.maximum(cf, 1e-30))
        g = np.gradient(cf, vf)
        top = float(np.sqrt(np.sum((g / ns) ** 2)))
        bot = float(np.sqrt(np.sum(cf)))
        if top <= 0 or bot <= 0:
            return 0.0
        return 1.0 / ((top / bot) * (nsp**0.5) * bot)

    def _ok_err(e: float, sig: float, vstep: float) -> bool:
        """
        Accept finite, positive formal errors.

        Reject only non-finite values, non-positive values, or numerical blow-ups.
        When the fitted Gaussian width ``sig`` (m/s) is known and positive, reject
        covariances vastly larger than ``max(100·σ, 25·vstep)`` (unphysical vs. the
        fitted line width). There is **no** upper cap at ~200 km/s tied to velocity
        grid spacing (that spacing used to reject ``e > 5000·vstep`` and masked valid
        large uncertainties).
        """
        if not (math.isfinite(e) and e > 0):
            return False
        if e > 1.0e15:
            return False
        if sig > 0 and e > max(100.0 * sig, 25.0 * max(vstep, 1e-9)):
            return False
        return True

    ft1 = fitting.LevMarLSQFitter()
    g1, gx1, vlo1, vhi1, med1, pw1, clip1, gy01, ge1 = _pass(
        rv_gk, hw1_km, cfg.initial_stddev_km_s, ft1
    )
    if g1 is None or gx1.size < cfg.min_points:
        fp_ctx: dict = {
            "fit_stage": "pass1",
            "reason": "LevMar did not return a usable model, or post-window sample count "
            "fell below min_points",
            "g1_is_none": g1 is None,
            "pass1_fit_sample_count": int(gx1.size),
            "min_points": int(cfg.min_points),
            "seed_mean_km_s": float(rv_gk),
            "pass1_half_width_km_s": float(hw1_km),
            "fit_window_m_s_lo": float(vlo1) if np.isfinite(vlo1) else None,
            "fit_window_m_s_hi": float(vhi1) if np.isfinite(vhi1) else None,
            "ccf_profile": profile,
            "velocity_grid_m_s_min": float(vm.min()),
            "velocity_grid_m_s_max": float(vm.max()),
            "n_velocity_points": int(vm.size),
            "v_step_effective_m_s": float(v_step_fit),
            "velocity_subsample_stride": int(subsample_stride),
            "ccf_fit_free_baseline": bool(free_bl),
        }
        if isinstance(clip1, dict) and clip1:
            fp_ctx["sigma_clip_iterations"] = int(clip1.get("iterations", 0))
            fp_ctx["sigma_clip_n_start"] = int(clip1.get("n_start", 0))
            fp_ctx["sigma_clip_n_final"] = int(clip1.get("n_final", 0))
        return _fail("first-pass fit failed", **fp_ctx)
    m1k, _, _ = _lm(g1, ft1)
    if not (float(gx1.min()) <= m1k <= float(gx1.max())):
        gx_min, gx_max = float(gx1.min()), float(gx1.max())
        ow_ctx = {
            "fit_stage": "pass1_post_lm",
            "reason": "LevMar Gaussian mean (km/s) lies outside the velocity abscissa of "
            "samples retained in the pass-1 window (after any sigma-clipping)",
            "fitted_mean_km_s": float(m1k),
            "fit_samples_v_km_min": gx_min,
            "fit_samples_v_km_max": gx_max,
            "fit_window_m_s_lo": float(vlo1),
            "fit_window_m_s_hi": float(vhi1),
            "pass1_half_width_km_s": float(hw1_km),
            "n_points_in_pass1_window": int(gx1.size),
            "seed_mean_km_s": float(rv_gk),
            "ccf_profile": profile,
            "velocity_grid_m_s_min": float(vm.min()),
            "velocity_grid_m_s_max": float(vm.max()),
            "n_velocity_points": int(vm.size),
            "v_step_effective_m_s": float(v_step_fit),
            "velocity_subsample_stride": int(subsample_stride),
            "ccf_y_min": float(np.nanmin(y)),
            "ccf_y_max": float(np.nanmax(y)),
            "ccf_fit_free_baseline": bool(free_bl),
        }
        if not math.isfinite(m1k):
            ow_ctx["mean_side"] = "non_finite_mean"
        elif m1k < gx_min:
            ow_ctx["mean_offset_km_s"] = float(gx_min - m1k)
            ow_ctx["mean_side"] = "below_sample_min"
        elif m1k > gx_max:
            ow_ctx["mean_offset_km_s"] = float(m1k - gx_max)
            ow_ctx["mean_side"] = "above_sample_max"
        if isinstance(clip1, dict):
            ow_ctx["sigma_clip_iterations"] = int(clip1.get("iterations", 0))
            ow_ctx["sigma_clip_n_start"] = int(clip1.get("n_start", 0))
            ow_ctx["sigma_clip_n_final"] = int(clip1.get("n_final", 0))
        return _fail("mean outside window", **ow_ctx)
    best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g1, ft1, gx1, med1, vlo1, vhi1
    used_pw_fit = pw1
    clip_used = clip1
    snap_y0, snap_ge = gy01, ge1
    pw = 1
    if two_pass:
        ft2 = fitting.LevMarLSQFitter()
        g2, gx2, vlo2, vhi2, med2, pw2, clip2, gy02, ge02 = _pass(
            m1k,
            hw2_km,
            float(np.clip(cfg.initial_stddev_km_s, 0.05, 200.0)),
            ft2,
        )
        if g2 is not None and gx2.size >= cfg.min_points:
            m2k, _, _ = _lm(g2, ft2)
            r2 = m2k * 1000.0
            if vlo2 <= r2 <= vhi2 and float(gx2.min()) <= m2k <= float(gx2.max()):
                best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g2, ft2, gx2, med2, vlo2, vhi2
                used_pw_fit = pw2
                clip_used = clip2
                snap_y0, snap_ge = gy02, ge02
                pw = 2

    mk, sk, sigc = _lm(best_g, best_ft)
    rv_ms, sig_ms = float(mk * 1000.0), float(sk * 1000.0)
    vstep = float(v_step_fit)
    # LM can drive Gaussian σ → ~0; formal covariance on the mean then hits ~1e-16 m/s while
    # _ok_err still accepts it (bound uses max(100·σ, 25·vstep)). Treat that as non-physical.
    lm_width_plausible = math.isfinite(sig_ms) and sig_ms >= max(1e-15, 1e-12 * vstep)
    sig_for_fb_ms = (
        float(sig_ms)
        if lm_width_plausible
        else max(vstep, 1e-12) if math.isfinite(vstep) and vstep > 0 else max(float(sig_ms), 1e-12)
    )
    fw_km = 2.0 * (hw2_km if pw == 2 else hw1_km)
    ek = _gradient_ccf_err_km(fw_km, mk)
    ek_ms = float(ek * 1000.0)
    ek_ok = ek > 0 and np.isfinite(ek_ms) and _ok_err(ek_ms, sig_ms, vstep)
    cov_ms = (
        float(sigc * 1000.0)
        if (sigc is not None and np.isfinite(sigc) and float(sigc) > 0)
        else float("nan")
    )
    cov_ok = math.isfinite(cov_ms) and cov_ms > 0 and _ok_err(cov_ms, sig_ms, vstep)
    if not lm_width_plausible:
        cov_ok = False

    n_fit = max(int(best_gx.size), 1)
    rv_err_fb = float(sig_for_fb_ms / math.sqrt(float(n_fit)))
    if n_good is not None and int(n_good) > 0:
        rv_err_fb = max(rv_err_fb, float(sig_for_fb_ms / math.sqrt(float(n_good))))

    comb = str(cfg.rv_err_cov_combine).lower().strip()
    if comb not in ("min", "max"):
        comb = "max"

    used_gradient = used_cov = used_fb = False
    if cov_ok and ek_ok:
        rv_err_ms = min(cov_ms, ek_ms) if comb == "min" else max(cov_ms, ek_ms)
        used_gradient = True
        used_cov = True
        msg = f"{'min' if comb == 'min' else 'max'}(gradient ccf_error, LevMar cov)"
    elif cov_ok:
        rv_err_ms = cov_ms
        used_cov = True
        msg = "LevMar cov"
    elif ek_ok:
        rv_err_ms = ek_ms
        used_gradient = True
        msg = "gradient ccf_error"
    else:
        rv_err_ms = rv_err_fb
        used_fb = True
        if ek > 0 and np.isfinite(ek):
            msg = "σ/√n (gradient ccf_error implausible)"
        elif sigc is not None and np.isfinite(float(sigc)) and float(sigc) > 0:
            msg = "σ/√n (LevMar cov implausible)"
        else:
            msg = "σ/√n fallback"

    used_boot = False
    boot_sig = float("nan")
    n_bs = int(cfg.rv_err_bootstrap_samples)
    if (
        n_bs > 0
        and snap_ge.size == snap_y0.size
        and snap_y0.size > 0
        and np.any(np.isfinite(snap_ge) & (snap_ge > 0))
    ):
        rng = np.random.default_rng(cfg.rv_err_bootstrap_seed)
        _sd_km = (
            abs(float(best_g.stddev_0.value))
            if hasattr(best_g, "stddev_0")
            else abs(float(best_g.stddev.value))
        )
        boot_sig = _bootstrap_gaussian_rv_std_m_s(
            best_gx,
            snap_y0,
            snap_ge,
            seed_mean_km=float(mk),
            seed_std_km=max(_sd_km, 1e-9),
            profile=profile,
            n_samples=n_bs,
            rng=rng,
            maxiter=int(max(50, min(int(cfg.levmar_maxiter), 50_000))),
            free_baseline=free_bl,
            ccf_line_mask_stored_negated=ccf_line_mask_stored_negated,
        )
        if math.isfinite(boot_sig) and boot_sig > 0:
            used_boot = True
            rv_err_ms = max(rv_err_ms, boot_sig)

    slope_prop_ms = float("nan")
    applied_slope_floor = False
    if snap_y0.size >= 5 and best_gx.size == snap_y0.size:
        se_slope = snap_ge if np.any(np.isfinite(snap_ge) & (snap_ge > 0)) else None
        slope_prop_ms = _rv_err_slope_propagation_m_s(best_gx * 1000.0, snap_y0, se_slope)
    if math.isfinite(slope_prop_ms) and slope_prop_ms > 0 and not used_cov:
        rv_err_ms = max(rv_err_ms, slope_prop_ms)
        applied_slope_floor = True

    rv_err_ms = max(rv_err_ms, _ccf_rv_err_numeric_floor(vstep))

    ys_snap = _levmar_ccf_y_scale(snap_y0)
    _bg_robust = float(_ccf_background_sigma_clipped_median(snap_y0))
    if hasattr(best_g, "amplitude_0"):
        _amp = float(best_g.amplitude_0.value) / ys_snap
        if free_bl and hasattr(best_g, "amplitude_1"):
            _baseline = float(best_g.amplitude_1.value) / ys_snap
        else:
            _baseline = _bg_robust
            if not np.isfinite(_baseline):
                _baseline = float(best_g.amplitude_1.value) / ys_snap
    else:
        _amp = float(best_g.amplitude.value)
        _baseline = float(best_med)
    amplitude_from_peak_fallback = False
    if (
        bool(ccf_line_mask_stored_negated)
        and _amp <= 0.0
        and snap_y0.size > 0
        and hasattr(best_g, "amplitude_0")
    ):
        peak = float(np.nanmax(snap_y0))
        base = float(_baseline) if np.isfinite(float(_baseline)) else peak
        ph = peak - base
        if math.isfinite(ph) and ph > 0.0:
            _amp = ph
            amplitude_from_peak_fallback = True
    params = {
        "amplitude": _amp,
        "v0": rv_ms,
        "sigma": sig_ms,
        "baseline": _baseline,
    }
    diag = {
        "fit_ok": True,
        "ccf_line_mask_forced_peak_up_profile": bool(forced_line_mask_peak_up),
        "used_covariance_for_error": used_cov,
        "used_sigma_fallback": used_fb,
        "used_gradient_ccf_error": used_gradient,
        "used_ccf_pointwise_error": bool(used_pw_fit),
        "message": msg,
        "ccf_velocity_grid_min_m_s": float(np.min(vm)),
        "ccf_velocity_grid_max_m_s": float(np.max(vm)),
        "fit_v_lo": float(best_vlo),
        "fit_v_hi": float(best_vhi),
        "profile": profile,
        "fit_pass_used": pw,
        "ccf_dir_style": "up" if profile == "absorption" else "down",
        "sigma_clip_enabled": bool(clip_used.get("sigma_clip", cfg.sigma_clip_enable)),
        "sigma_clip_iterations": int(clip_used.get("iterations", 0)),
        "sigma_clip_n_points_initial": int(clip_used.get("n_start", 0)),
        "sigma_clip_n_points_final": int(clip_used.get("n_final", best_gx.size)),
        "rv_err_bootstrap_sigma_m_s": boot_sig,
        "used_rv_err_bootstrap": used_boot,
        "rv_err_cov_combine": comb,
        "lm_gaussian_width_plausible": bool(lm_width_plausible),
        "rv_err_slope_propagation_m_s": slope_prop_ms,
        "applied_slope_propagation_floor": bool(applied_slope_floor),
        "velocity_fit_subsample_stride": int(subsample_stride),
        "ccf_background_sigma_clipped_median": (
            float(_bg_robust)
            if np.isfinite(_bg_robust) and hasattr(best_g, "amplitude_0")
            else float(_baseline)
        ),
        "ccf_fit_free_baseline": bool(free_bl),
        "ccf_amplitude_from_peak_fallback": bool(amplitude_from_peak_fallback),
    }
    if free_bl and hasattr(best_g, "amplitude_1") and np.isfinite(_bg_robust):
        diag["ccf_baseline_lm_median_space"] = float(_baseline)
    diag.update(_ccf_line_mask_diag)
    if gaussian_seed_mean_m_s is not None and math.isfinite(float(gaussian_seed_mean_m_s)):
        diag["gaussian_seed_mean_m_s"] = float(gaussian_seed_mean_m_s)
    return (rv_ms, rv_err_ms, params, diag) if return_params else (rv_ms, rv_err_ms)


# =============================================================================
# Numba per-chunk log-λ CCF + Python Gaussian / parabolic RV (many orders at once)
# =============================================================================



def _ccf_rv_from_ccf(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    v_step: float,
    n_good: int,
    vel_span_m_s_per_pixel: float | None = None,
    *,
    min_good_pixels: int = 10,
    ccf_parabolic_vertex_n_samples: int | None = None,
    ccf_bisector_config: CCFBisectorConfig | None = None,
) -> tuple[float, float]:
    """Gaussian fit to CCF; bisector refine for final RV; else grid peak / HWHM.

    ``ccf_parabolic_vertex_n_samples`` is retained for API compatibility but no longer
    selects the final RV (bisector is used instead when the Gaussian fit succeeds).
    """
    ccf = np.asarray(ccf, dtype=np.float64)
    n_vel = len(ccf)
    if n_good < int(min_good_pixels) or n_vel < 3 or not np.any(np.isfinite(ccf)):
        return np.nan, np.nan
    ccf_max = float(np.nanmax(ccf))
    ccf_min = float(np.nanmin(ccf))
    i_max = int(np.nanargmax(ccf))
    rv = float(v_grid[i_max])
    if not np.isfinite(ccf_max):
        return np.nan, np.nan
    if ccf_max <= ccf_min or not np.isfinite(ccf_min):
        return rv, float(max(v_step * 2.0, _ccf_rv_err_numeric_floor(v_step)))

    vsp = (
        float(vel_span_m_s_per_pixel)
        if vel_span_m_s_per_pixel is not None and np.isfinite(vel_span_m_s_per_pixel) and vel_span_m_s_per_pixel > 0
        else None
    )
    rv_fit, err_fit, params, _diag = fit_ccf_rv(
        v_grid,
        ccf,
        v_step,
        n_good=n_good,
        vel_span_m_s_per_pixel=vsp,
        return_params=True,
    )
    if np.isfinite(rv_fit):
        rv_bi, err_bi, _bmeta = _refine_ccf_rv_bisector_after_gaussian_fit(
            v_grid,
            ccf,
            float(v_step),
            ccf_err=None,
            config=ccf_bisector_config,
        )
        if rv_bi is not None and math.isfinite(float(rv_bi)):
            err_out = float(err_fit) if np.isfinite(err_fit) else float("nan")
            if err_bi is not None and math.isfinite(float(err_bi)) and float(err_bi) > 0.0:
                err_out = float(err_bi)
            return float(rv_bi), float(err_out)

    if np.isfinite(rv_fit) and np.isfinite(err_fit) and err_fit > 0.0:
        return float(rv_fit), float(err_fit)

    half = (ccf_max + ccf_min) / 2.0
    below = np.where(ccf[:i_max] <= half)[0]
    above = np.where(ccf[i_max:] <= half)[0]
    left = int(below[-1]) if len(below) else 0
    right = i_max + int(above[0]) if len(above) else n_vel - 1
    hwhm = (right - left) * v_step * 0.5
    ng = max(int(n_good), 1)
    return rv, float(max(hwhm / math.sqrt(float(ng)), _ccf_rv_err_numeric_floor(v_step)))


@njit(cache=True)
def _ccf_chunk_ccf_numba(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    ccf_out: np.ndarray,
    min_good_pixels: int,
    zb: float,
) -> int:
    """
    Weighted cross-correlation CCF(v) for one chunk. Writes ``ccf_out`` (length
    n_vel); returns count of good spectral pixels (for error scaling).
    """
    n_good = 0
    for i in range(len(good)):
        if good[i]:
            n_good += 1
    if n_good < min_good_pixels:
        return n_good

    n_pix = len(ln_lam)
    # Inverse-variance weights (same as 1/spec_err² with floor); reused for every v.
    inv_w = np.empty(n_pix)
    for i in range(n_pix):
        if good[i] and np.isfinite(spec_err[i]) and spec_err[i] > 0:
            inv_w[i] = 1.0 / max(spec_err[i] * spec_err[i], 1e-30)
        else:
            inv_w[i] = 0.0

    n_vel = len(ccf_out)
    for i in range(n_vel):
        ccf_out[i] = np.nan

    zb1 = 1.0 + zb
    if zb1 <= 0.0:
        zb1 = 1.0

    # One CCF sample per trial velocity: shift template in lnλ, interpolate to data pixels.
    for iv in range(n_vel):
        v = v_min + iv * v_step
        delta_ln = math.log((1.0 + v / C_M_S) / zb1)
        t_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(ln_lam, t_ln_shifted, template_flux)
        # Weighted means of data and template at this v (same pixels as masked_ccf_rv).
        sum_w = 0.0
        sum_d = 0.0
        sum_t = 0.0
        n_use = 0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                sum_w += w
                sum_d += spec[i] * w
                sum_t += t_at_data[i] * w
                n_use += 1
        if n_use < 5:
            continue
        d_mean = sum_d / sum_w
        t_mean = sum_t / sum_w
        # Weighted covariance (mean-subtracted); same definition as Python masked_ccf_rv.
        cov = 0.0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                cov += (spec[i] - d_mean) * (t_at_data[i] - t_mean) * w
        ccf_out[iv] = cov / sum_w

    return n_good


@njit(cache=True)
def _ccf_chunk_ccf_numba_vgrid(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_grid: np.ndarray,
    ccf_out: np.ndarray,
    min_good_pixels: int,
    zb: float,
) -> int:
    """Same as :func:`_ccf_chunk_ccf_numba` but samples at arbitrary ``v_grid`` (m/s)."""
    n_good = 0
    for i in range(len(good)):
        if good[i]:
            n_good += 1
    if n_good < min_good_pixels:
        return n_good

    n_pix = len(ln_lam)
    inv_w = np.empty(n_pix)
    for i in range(n_pix):
        if good[i] and np.isfinite(spec_err[i]) and spec_err[i] > 0:
            inv_w[i] = 1.0 / max(spec_err[i] * spec_err[i], 1e-30)
        else:
            inv_w[i] = 0.0

    n_vel = len(ccf_out)
    for i in range(n_vel):
        ccf_out[i] = np.nan

    zb1 = 1.0 + zb
    if zb1 <= 0.0:
        zb1 = 1.0

    for iv in range(n_vel):
        v = float(v_grid[iv])
        delta_ln = math.log((1.0 + v / C_M_S) / zb1)
        t_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(ln_lam, t_ln_shifted, template_flux)
        sum_w = 0.0
        sum_d = 0.0
        sum_t = 0.0
        n_use = 0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                sum_w += w
                sum_d += spec[i] * w
                sum_t += t_at_data[i] * w
                n_use += 1
        if n_use < 5:
            continue
        d_mean = sum_d / sum_w
        t_mean = sum_t / sum_w
        cov = 0.0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                cov += (spec[i] - d_mean) * (t_at_data[i] - t_mean) * w
        ccf_out[iv] = cov / sum_w

    return n_good


@njit(parallel=True, cache=True)
def _process_all_orders_ccf_numba(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    n_vel: int,
    all_ccf: np.ndarray,
    all_n_good: np.ndarray,
    all_wl: np.ndarray,
    min_good_pixels: int,
    zb: float,
) -> None:
    """
    Parallel per-order CCF fill: writes ``all_ccf[chunk, :]``, ``all_n_good[chunk]``,
    ``all_wl[chunk]``. RV fit in Python via :func:`fit_ccf_rv`.
    """
    n_processed = len(order_indices)

    for idx in prange(n_processed):
        oi = order_indices[idx]
        w = wave_2d[oi]
        s = spec_2d[oi]
        e = spec_err_2d[oi]
        d = dq_2d[oi]
        n_pix = len(w)
        # Pixel usable for CCF: not DQ-bad, finite flux and positive error.
        mask = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            mask[i] = (d[i] & do_not_use) != 0
        good = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            good[i] = not mask[i] and np.isfinite(s[i]) and np.isfinite(e[i]) and e[i] > 0

        w_min = np.nanmin(w)
        w_max = np.nanmax(w)
        ln_min = np.log(w_min) - 0.01
        ln_max = np.log(w_max) + 0.01
        n_tpl = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                n_tpl += 1
        if n_tpl < 20:
            continue  # skip order: template would be too sparse after clip
        t_ln = np.empty(n_tpl)
        t_flux = np.empty(n_tpl)
        j = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                t_ln[j] = template_ln[i]
                t_flux[j] = template_flux[i]
                j += 1

        ccf_row = np.empty(n_vel)
        n_chunks = n_chunks_per_order[idx]
        start = chunk_start[idx]
        # Wavelength bins along the order; each bin gets one CCF vector.
        for k in range(n_chunks):
            lam_lo = w_min + k * (w_max - w_min) / n_chunks
            lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunks
            in_chunk = np.empty(n_pix, dtype=np.bool_)
            chunk_len = 0
            for i in range(n_pix):
                if k == n_chunks - 1:
                    in_chunk[i] = w[i] >= lam_lo and w[i] <= lam_hi
                else:
                    in_chunk[i] = w[i] >= lam_lo and w[i] < lam_hi
                if in_chunk[i]:
                    chunk_len += 1
            wl_c = 0.5 * (lam_lo + lam_hi)
            all_wl[start + k] = wl_c
            if chunk_len < 20:
                all_n_good[start + k] = 0
                continue
            ln_lam_c = np.empty(chunk_len)
            spec_c = np.empty(chunk_len)
            spec_err_c = np.empty(chunk_len)
            good_c = np.empty(chunk_len, dtype=np.bool_)
            j = 0
            for i in range(n_pix):
                if in_chunk[i]:
                    ln_lam_c[j] = np.log(w[i])
                    spec_c[j] = s[i]
                    spec_err_c[j] = e[i]
                    good_c[j] = good[i]
                    j += 1
            ng = _ccf_chunk_ccf_numba(
                ln_lam_c,
                spec_c,
                spec_err_c,
                good_c,
                t_ln,
                t_flux,
                v_min,
                v_max,
                v_step,
                ccf_row,
                min_good_pixels,
                zb,
            )
            all_n_good[start + k] = ng
            for iv in range(n_vel):
                all_ccf[start + k, iv] = ccf_row[iv]


@njit(parallel=True, cache=True)
def _process_all_orders_ccf_numba_vgrid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_grid: np.ndarray,
    n_vel: int,
    all_ccf: np.ndarray,
    all_n_good: np.ndarray,
    all_wl: np.ndarray,
    min_good_pixels: int,
    zb: float,
) -> None:
    """Same as :func:`_process_all_orders_ccf_numba` but with an explicit ``v_grid`` (m/s)."""
    n_processed = len(order_indices)

    for idx in prange(n_processed):
        oi = order_indices[idx]
        w = wave_2d[oi]
        s = spec_2d[oi]
        e = spec_err_2d[oi]
        d = dq_2d[oi]
        n_pix = len(w)
        mask = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            mask[i] = (d[i] & do_not_use) != 0
        good = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            good[i] = not mask[i] and np.isfinite(s[i]) and np.isfinite(e[i]) and e[i] > 0

        w_min = np.nanmin(w)
        w_max = np.nanmax(w)
        ln_min = np.log(w_min) - 0.01
        ln_max = np.log(w_max) + 0.01
        n_tpl = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                n_tpl += 1
        if n_tpl < 20:
            continue
        t_ln = np.empty(n_tpl)
        t_flux = np.empty(n_tpl)
        j = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                t_ln[j] = template_ln[i]
                t_flux[j] = template_flux[i]
                j += 1

        ccf_row = np.empty(n_vel)
        n_chunks = n_chunks_per_order[idx]
        start = chunk_start[idx]
        for k in range(n_chunks):
            lam_lo = w_min + k * (w_max - w_min) / n_chunks
            lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunks
            in_chunk = np.empty(n_pix, dtype=np.bool_)
            chunk_len = 0
            for i in range(n_pix):
                if k == n_chunks - 1:
                    in_chunk[i] = w[i] >= lam_lo and w[i] <= lam_hi
                else:
                    in_chunk[i] = w[i] >= lam_lo and w[i] < lam_hi
                if in_chunk[i]:
                    chunk_len += 1
            wl_c = 0.5 * (lam_lo + lam_hi)
            all_wl[start + k] = wl_c
            if chunk_len < 20:
                all_n_good[start + k] = 0
                continue
            ln_lam_c = np.empty(chunk_len)
            spec_c = np.empty(chunk_len)
            spec_err_c = np.empty(chunk_len)
            good_c = np.empty(chunk_len, dtype=np.bool_)
            j = 0
            for i in range(n_pix):
                if in_chunk[i]:
                    ln_lam_c[j] = np.log(w[i])
                    spec_c[j] = s[i]
                    spec_err_c[j] = e[i]
                    good_c[j] = good[i]
                    j += 1
            ng = _ccf_chunk_ccf_numba_vgrid(
                ln_lam_c,
                spec_c,
                spec_err_c,
                good_c,
                t_ln,
                t_flux,
                v_grid,
                ccf_row,
                min_good_pixels,
                zb,
            )
            all_n_good[start + k] = ng
            for iv in range(n_vel):
                all_ccf[start + k, iv] = ccf_row[iv]


def _process_all_orders_rvs_numba_single_grid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
    min_good_pixels: int = 10,
    zb: float = 0.0,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Single uniform velocity grid: Numba CCF fill + per-chunk Gaussian RV."""
    n_processed = len(order_indices)
    total_chunks = int(chunk_start[n_processed - 1] + n_chunks_per_order[n_processed - 1])
    n_vel = int((v_max - v_min) / v_step) + 1
    # Phase A — Numba fills CCF matrix (memory: total_chunks × n_vel).
    all_ccf = np.full((total_chunks, n_vel), np.nan, dtype=np.float64)
    all_n_good = np.zeros(total_chunks, dtype=np.int32)
    all_wl = np.empty(total_chunks, dtype=np.float64)
    all_wl[:] = np.nan
    all_rvs = np.empty(total_chunks, dtype=np.float64)
    all_rvs[:] = np.nan
    all_rv_errs = np.empty(total_chunks, dtype=np.float64)
    all_rv_errs[:] = np.nan

    nt = numba_threads if numba_threads is not None else 1
    logger.info(
        "CCF (Numba): starting CCF matrix fill total_chunks=%d n_vel=%d numba_threads=%d",
        total_chunks,
        n_vel,
        nt,
    )
    _process_all_orders_ccf_numba(
        wave_2d,
        spec_2d,
        spec_err_2d,
        dq_2d,
        do_not_use,
        template_ln,
        template_flux,
        window_size_um,
        order_indices,
        n_chunks_per_order,
        chunk_start,
        v_min,
        v_max,
        v_step,
        n_vel,
        all_ccf,
        all_n_good,
        all_wl,
        int(min_good_pixels),
        float(zb),
    )
    logger.info(
        "CCF (Numba): CCF matrix fill complete total_chunks=%d n_vel=%d",
        total_chunks,
        n_vel,
    )

    # Phase B — Gaussian fit per chunk row in Python (cannot run inside Numba).
    v_grid = v_min + np.arange(n_vel, dtype=np.float64) * v_step
    logger.info(
        "CCF (Python): starting per-chunk Gaussian RV fits total_chunks=%d",
        total_chunks,
    )
    n_fit_rows = 0
    mg = int(min_good_pixels)
    for i in range(total_chunks):
        if all_n_good[i] < mg:
            continue
        n_fit_rows += 1
        wl_c = float(all_wl[i])
        vspan = dv_per_pixel_m_s(wl_c)
        if not np.isfinite(vspan):
            vspan = None
        rv, rv_err = _ccf_rv_from_ccf(
            v_grid,
            all_ccf[i],
            v_step,
            int(all_n_good[i]),
            vel_span_m_s_per_pixel=vspan,
            min_good_pixels=mg,
            ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
        )
        all_rvs[i] = rv
        all_rv_errs[i] = rv_err
    logger.info(
        "CCF (Python): per-chunk Gaussian fits complete rows_fitted=%d of total_chunks=%d",
        n_fit_rows,
        total_chunks,
    )

    if return_ccf:
        return all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good
    return all_rvs, all_rv_errs, all_wl


def _process_all_orders_rvs_numba_single_vgrid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_grid: np.ndarray,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
    min_good_pixels: int = 10,
    zb: float = 0.0,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Non-uniform ``v_grid`` (e.g. merged hybrid): Numba CCF fill + per-chunk Gaussian RV."""
    v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    n_vel = int(v_grid.shape[0])
    if n_vel < 2:
        raise ValueError("v_grid must contain at least two velocity samples (m/s)")
    v_step_fit = effective_ccf_v_step_for_fit(v_grid)

    n_processed = len(order_indices)
    total_chunks = int(chunk_start[n_processed - 1] + n_chunks_per_order[n_processed - 1])
    all_ccf = np.full((total_chunks, n_vel), np.nan, dtype=np.float64)
    all_n_good = np.zeros(total_chunks, dtype=np.int32)
    all_wl = np.empty(total_chunks, dtype=np.float64)
    all_wl[:] = np.nan
    all_rvs = np.empty(total_chunks, dtype=np.float64)
    all_rvs[:] = np.nan
    all_rv_errs = np.empty(total_chunks, dtype=np.float64)
    all_rv_errs[:] = np.nan

    nt = numba_threads if numba_threads is not None else 1
    logger.info(
        "CCF (Numba): starting CCF matrix fill (merged v_grid) total_chunks=%d n_vel=%d "
        "numba_threads=%d",
        total_chunks,
        n_vel,
        nt,
    )
    _process_all_orders_ccf_numba_vgrid(
        wave_2d,
        spec_2d,
        spec_err_2d,
        dq_2d,
        do_not_use,
        template_ln,
        template_flux,
        window_size_um,
        order_indices,
        n_chunks_per_order,
        chunk_start,
        v_grid,
        n_vel,
        all_ccf,
        all_n_good,
        all_wl,
        int(min_good_pixels),
        float(zb),
    )
    logger.info(
        "CCF (Numba): CCF matrix fill complete total_chunks=%d n_vel=%d",
        total_chunks,
        n_vel,
    )

    logger.info(
        "CCF (Python): starting per-chunk Gaussian RV fits total_chunks=%d",
        total_chunks,
    )
    n_fit_rows = 0
    mg = int(min_good_pixels)
    for i in range(total_chunks):
        if all_n_good[i] < mg:
            continue
        n_fit_rows += 1
        wl_c = float(all_wl[i])
        vspan = dv_per_pixel_m_s(wl_c)
        if not np.isfinite(vspan):
            vspan = None
        rv, rv_err = _ccf_rv_from_ccf(
            v_grid,
            all_ccf[i],
            v_step_fit,
            int(all_n_good[i]),
            vel_span_m_s_per_pixel=vspan,
            min_good_pixels=mg,
            ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
        )
        all_rvs[i] = rv
        all_rv_errs[i] = rv_err
    logger.info(
        "CCF (Python): per-chunk Gaussian fits complete rows_fitted=%d of total_chunks=%d",
        n_fit_rows,
        total_chunks,
    )

    if return_ccf:
        return all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good
    return all_rvs, all_rv_errs, all_wl


def process_all_orders_rvs_numba(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
    hybrid_center_m_s: float | None = None,
    hybrid_half_width_m_s: float = 300.0,
    hybrid_step_m_s: float = 0.1,
    min_good_pixels: int = 10,
    zb: float = 0.0,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    High-throughput per-chunk RVs: Numba CCF grid, then SciPy Gaussian per chunk.

    Workhorse behind :func:`process_orders_rvs_numba_parallel`.

    When ``hybrid_center_m_s`` is ``None``, uses a single grid ``(v_min, v_max, v_step)``.

    When ``hybrid_center_m_s`` is set, uses one merged velocity list spanning the broad
    range ``v_min`` … ``v_max`` at ``v_step``, with additional samples on
    ``center ± hybrid_half_width_m_s`` every ``hybrid_step_m_s`` (see
    :func:`build_hybrid_merged_velocity_grid`). A single CCF is built on that grid and
    per-chunk RVs are fit once. Optional CCF output uses the same merged ``v_grid``.
    """
    if hybrid_center_m_s is None:
        return _process_all_orders_rvs_numba_single_grid(
            wave_2d,
            spec_2d,
            spec_err_2d,
            dq_2d,
            do_not_use,
            template_ln,
            template_flux,
            window_size_um,
            order_indices,
            n_chunks_per_order,
            chunk_start,
            v_min,
            v_max,
            v_step,
            return_ccf=return_ccf,
            numba_threads=numba_threads,
            min_good_pixels=int(min_good_pixels),
            zb=float(zb),
            ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
        )

    vc = float(hybrid_center_m_s)
    hw = float(hybrid_half_width_m_s)
    vsf = float(hybrid_step_m_s)
    if not np.isfinite(vc) or not np.isfinite(hw) or hw <= 0:
        raise ValueError(
            "hybrid CCF: center and half-width must be finite and half-width > 0 (m/s)"
        )
    if not np.isfinite(vsf) or vsf <= 0:
        raise ValueError("hybrid CCF: refined step must be finite and > 0 (m/s)")
    if vc + hw <= vc - hw:
        raise ValueError("hybrid CCF: refined velocity window is empty")

    v_grid = build_hybrid_merged_velocity_grid(
        float(v_min),
        float(v_max),
        float(v_step),
        vc,
        hw,
        vsf,
    )
    logger.info(
        "CCF (Numba): hybrid merged grid — broad [%g, %g] step %g; denser [%g, %g] "
        "step %g → n_vel=%d",
        v_min,
        v_max,
        v_step,
        vc - hw,
        vc + hw,
        vsf,
        int(v_grid.shape[0]),
    )
    return _process_all_orders_rvs_numba_single_vgrid(
        wave_2d,
        spec_2d,
        spec_err_2d,
        dq_2d,
        do_not_use,
        template_ln,
        template_flux,
        window_size_um,
        order_indices,
        n_chunks_per_order,
        chunk_start,
        v_grid,
        return_ccf=return_ccf,
        numba_threads=numba_threads,
        min_good_pixels=int(min_good_pixels),
        zb=float(zb),
        ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
    )


def flat_chunk_echelle_orders(
    order_tasks: list,
    chunk_start: np.ndarray,
    n_flat: int,
) -> np.ndarray:
    """
    Echelle order number for each flattened chunk index (same packing as
    :func:`process_orders_rvs_numba_parallel`).

    ``chunk_start[j]`` is the global flat index of the first chunk of
    ``order_tasks[j]`` (``(order_row_index, echelle_order_number)``).

    Parameters
    ----------
    order_tasks : list of tuple
        ``(order_row_index, echelle_order_number)`` in the same order as used to build
        ``chunk_start`` and ``n_chunks_per_order``.
    chunk_start : np.ndarray
        Cumulative start index of each order’s chunks in the flattened ``all_rvs`` layout.
    n_flat : int
        Total number of chunk slots (rows) in the flattened arrays.

    Returns
    -------
    np.ndarray
        ``dtype`` int64, length ``n_flat``. For each flat chunk index ``i``, the
        echelle order number of the order that owns that chunk, or ``-1`` where
        ``i`` is out of range.
    """
    if n_flat <= 0 or not order_tasks:
        return np.array([], dtype=np.int64)
    chunk_start = np.asarray(chunk_start, dtype=np.int64)
    order_nums = np.array([int(t[1]) for t in order_tasks], dtype=np.int64)
    ii = np.arange(n_flat, dtype=np.int64)
    j = np.searchsorted(chunk_start, ii, side="right") - 1
    valid = (j >= 0) & (j < len(order_tasks))
    out = np.full(n_flat, -1, dtype=np.int64)
    out[valid] = order_nums[j[valid]]
    return out


def process_orders_rvs_numba_parallel(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_tasks: list,
    max_cores: int | None = None,
    *,
    return_ccf: bool = False,
    ccf_v_min: float | None = None,
    ccf_v_max: float | None = None,
    ccf_v_step: float | None = None,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
    min_good_pixels: int = 10,
    zb: float = 0.0,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """
    Stellar-RV fast path: many orders × wavelength chunks in one call.

    Flattens ``order_tasks`` into chunk index arrays, runs
    :func:`process_all_orders_rvs_numba` (Numba CCF + Python Gaussian fit per chunk),
    and optionally sets Numba thread count via ``max_cores``.

    Parameters
    ----------
    wave_2d : np.ndarray, shape (n_orders, n_pix)
        Wavelength in micron.
    spec_2d, spec_err_2d : np.ndarray
        Flux and uncertainty per order.
    dq_2d : np.ndarray, uint32
        Data quality flags.
    do_not_use : int
        Bit mask; pixels with ``(dq & do_not_use) != 0`` are excluded.
    template_ln, template_flux : np.ndarray
        Full stellar template.
    window_size_um : float
        Wavelength span per chunk (micron).
    order_tasks : list of tuple
        ``(order_row_index, order_number)`` for each order to include.
    max_cores : int, optional
        Numba parallel thread count for the order-parallel CCF loop. ``None`` or
        ``0`` uses all logical CPUs (see :func:`~hispecdrp.utils.parallelization_utils.resolve_numba_threads`).
        ``1`` forces single-threaded execution (reproducible, slower).
    return_ccf : bool, optional
        If ``True``, also return per-chunk CCF stacks, the velocity grid, per-chunk
        ``order_indices``, and per-chunk good-pixel counts for previews / RVCCF
        supplements (see implementation return tuple).
    ccf_v_min, ccf_v_max, ccf_v_step : float, optional
        Single-grid CCF velocity range (m/s). Ignored when ``ccf_hybrid_center_m_s``
        is set. Defaults are ``V_MIN_DEFAULT_M_S``, ``V_MAX_DEFAULT_M_S``, ``V_STEP_DEFAULT_M_S``.
    ccf_hybrid_center_m_s : float or None, optional
        When set, uses a single merged velocity grid (full ``_V_*`` range plus denser
        sampling around ``center ± ccf_hybrid_half_width_m_s``); see
        :func:`build_hybrid_merged_velocity_grid`.
    ccf_hybrid_half_width_m_s, ccf_hybrid_step_m_s : float, optional
        Refined half-width (m/s) and step (m/s); defaults match
        :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.
    min_good_pixels : int, optional
        Minimum good spectral pixels per chunk for CCF evaluation and Gaussian / parabolic
        RV fit (same role as ``ccf_min_good_pixels`` on :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`).
    zb : float, optional
        Barycentric factor :math:`z_b` in :math:`\\ln\\lambda` template shifts,
        :math:`\\Delta\\ln\\lambda = \\ln\\frac{1+v/c}{1+z_b}` (matches mask-line mapping when
        ``z_b`` is small).
    ccf_parabolic_vertex_n_samples : int or None, optional
        When set (odd, :math:`\\ge 3`), after each chunk’s Gaussian fit, apply the same
        LSTSQ parabolic vertex refinement used for mask-line CCF (see
        :func:`_refine_ccf_rv_after_gaussian_fit_ccf_line_mask`). ``None`` skips that step
        (Gaussian-only RV per chunk).

    Returns
    -------
    all_rvs : np.ndarray
        Per-chunk RV (m/s), 1-D flattened over all orders.
    all_rv_errs : np.ndarray
        Per-chunk uncertainty (m/s).
    all_wl : np.ndarray
        Chunk center wavelength (micron).
    n_chunks_per_order, chunk_start : np.ndarray
        Chunk layout (always returned). Use with :func:`flat_chunk_echelle_orders`
        to map flat chunk index → echelle order number.
    all_ccf, v_grid, order_indices, all_n_good : np.ndarray, optional
        Extra arrays when ``return_ccf`` is True (for preview / RVCCF supplement columns).

    Notes
    -----
    **Performance:** Phase A (Numba ``prange`` over orders) fills
    ``total_chunks × n_vel`` with ``n_vel ≈ (v_max - v_min) / v_step``. Phase B
    runs :func:`fit_ccf_rv` per chunk in a serial Python loop. Dominant levers:
    ``numba_threads`` (``max_cores``) for Phase A, and optional coarser
    ``ccf_v_step`` if acceptable for the use case.
    """
    if not order_tasks:
        empty = (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        z64 = np.array([], dtype=np.int64)
        if return_ccf:
            return empty + (
                np.zeros((0, 0), dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                z64,
                z64,
                z64,
            )
        return empty + (z64, z64)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    spec_2d = np.asarray(spec_2d, dtype=np.float64)
    spec_err_2d = np.asarray(spec_err_2d, dtype=np.float64)
    dq_2d = np.asarray(dq_2d, dtype=np.uint32)
    template_ln = np.asarray(template_ln, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    order_indices = np.array([oi for oi, _ in order_tasks], dtype=np.int64)
    n_chunks_per_order = np.array([
        ccf_order_n_chunks_for_span(
            float(np.nanmax(wave_2d[oi]) - np.nanmin(wave_2d[oi])),
            float(window_size_um),
        )
        for oi, _ in order_tasks
    ], dtype=np.int64)
    chunk_start = np.zeros(len(order_indices), dtype=np.int64)
    for i in range(1, len(order_indices)):
        chunk_start[i] = chunk_start[i - 1] + n_chunks_per_order[i - 1]

    nthr = resolve_numba_threads(max_cores)
    vm = float(V_MIN_DEFAULT_M_S if ccf_v_min is None else ccf_v_min)
    vx = float(V_MAX_DEFAULT_M_S if ccf_v_max is None else ccf_v_max)
    vs = float(V_STEP_DEFAULT_M_S if ccf_v_step is None else ccf_v_step)
    hyc = ccf_hybrid_center_m_s
    hybrid = hyc is not None and np.isfinite(float(hyc))
    if not hybrid:
        if not np.isfinite(vm) or not np.isfinite(vx) or not np.isfinite(vs) or vs <= 0:
            raise ValueError(
                "ccf_v_min, ccf_v_max must be finite and ccf_v_step must be finite and > 0"
            )
        if vx <= vm:
            raise ValueError("ccf_v_max must be greater than ccf_v_min")
    total_chunks_layout = int(chunk_start[-1] + n_chunks_per_order[-1])
    if hybrid:
        logger.info(
            "CCF: process_orders_rvs_numba_parallel HYBRID mode n_order_tasks=%d "
            "total_chunks=%d window_um=%.4f return_ccf=%s numba_threads=%d "
            "center=%.2f m/s half_width=%.2f refined_step=%.4f",
            len(order_tasks),
            total_chunks_layout,
            window_size_um,
            return_ccf,
            nthr,
            float(hyc),
            float(ccf_hybrid_half_width_m_s),
            float(ccf_hybrid_step_m_s),
        )
    else:
        logger.info(
            "CCF: process_orders_rvs_numba_parallel n_order_tasks=%d total_chunks=%d "
            "window_um=%.4f return_ccf=%s numba_threads=%d "
            "v_grid_m_s=[%.0f, %.0f] step=%.1f",
            len(order_tasks),
            total_chunks_layout,
            window_size_um,
            return_ccf,
            nthr,
            vm,
            vx,
            vs,
        )
    # Flattened (order, chunk) RV list; parallelized over orders inside Numba.
    with numba_thread_scope(max_cores):
        if hybrid:
            kw_h = dict(
                hybrid_center_m_s=float(hyc),
                hybrid_half_width_m_s=float(ccf_hybrid_half_width_m_s),
                hybrid_step_m_s=float(ccf_hybrid_step_m_s),
            )
        else:
            kw_h = {}
        kw_ccf = {
            **kw_h,
            "min_good_pixels": int(min_good_pixels),
            "zb": float(zb),
            "ccf_parabolic_vertex_n_samples": ccf_parabolic_vertex_n_samples,
        }
        if return_ccf:
            all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good = process_all_orders_rvs_numba(
                wave_2d,
                spec_2d,
                spec_err_2d,
                dq_2d,
                do_not_use,
                template_ln,
                template_flux,
                window_size_um,
                order_indices,
                n_chunks_per_order,
                chunk_start,
                vm,
                vx,
                vs,
                return_ccf=True,
                numba_threads=nthr,
                **kw_ccf,
            )
            return (
                all_rvs,
                all_rv_errs,
                all_wl,
                all_ccf,
                v_grid,
                all_n_good,
                n_chunks_per_order,
                chunk_start,
                order_indices,
            )
        all_rvs, all_rv_errs, all_wl = process_all_orders_rvs_numba(
            wave_2d,
            spec_2d,
            spec_err_2d,
            dq_2d,
            do_not_use,
            template_ln,
            template_flux,
            window_size_um,
            order_indices,
            n_chunks_per_order,
            chunk_start,
            vm,
            vx,
            vs,
            return_ccf=False,
            numba_threads=nthr,
            **kw_ccf,
        )
        return all_rvs, all_rv_errs, all_wl, n_chunks_per_order, chunk_start


# =============================================================================
# Parallel CCF preview slot fits and Plotly / RVCCF preview payload assembly
# =============================================================================


def _ccf_preview_finalize_n_workers(n_slots: int) -> int:
    """
    Worker count for parallel Gaussian + refine in :func:`finalize_ccf_preview_dict`.

    Set ``HISPEC_DRP_CCF_PREVIEW_SLOT_THREADS`` to a positive int to cap workers; ``1`` forces
    sequential (legacy behavior). Unset or invalid: ``min(n_slots, cpu_count)``.
    """
    raw = os.environ.get("HISPEC_DRP_CCF_PREVIEW_SLOT_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n < 1:
            return max(1, min(n_slots, os.cpu_count() or 4))
        return max(1, min(n, n_slots))
    return max(1, min(n_slots, os.cpu_count() or 4))


def _fit_ccf_preview_slot_worker(
    sl: dict,
    vs: float,
    *,
    ccf_line_mask_stored_negated: bool | None = None,
    ccf_parabolic_vertex_n_samples: int = 11,
    line_mask_line_window_by_line: bool | None = None,
    line_mask_gaussian_seed_mean_m_s: float | None = None,
    ccf_bisector_config: CCFBisectorConfig | None = None,
) -> tuple[dict, dict | None]:
    """
    One slot: :func:`fit_ccf_rv` with :data:`CCF_PREVIEW_SLOT_FIT_CONFIG`, then bisector
    RV for pipeline/preview metadata (:func:`_refine_ccf_rv_bisector_after_gaussian_fit`).
    Thread-safe (no shared mutable state).

    ``ccf_parabolic_vertex_n_samples`` is ignored (retained for call-site compatibility).
    """
    wl = sl.get("wl_lo", np.nan)
    if not np.isfinite(wl):
        wl = sl.get("wl_hi", np.nan)
    vsp = dv_per_pixel_m_s(float(wl))
    ce = sl.get("ccf_err")
    ccf_err_arg = np.asarray(ce, np.float64) if ce is not None else None
    y_slot = np.asarray(sl["ccf"], np.float64)
    v_slot = np.asarray(sl["v"], np.float64)
    rv, rv_err, params, diag = fit_ccf_rv(
        v_slot,
        y_slot,
        vs,
        return_params=True,
        vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
        ccf_err=ccf_err_arg,
        config=CCF_PREVIEW_SLOT_FIT_CONFIG,
        ccf_line_mask_stored_negated=ccf_line_mask_stored_negated,
        line_mask_line_window_by_line=line_mask_line_window_by_line,
        gaussian_seed_mean_m_s=line_mask_gaussian_seed_mean_m_s,
    )
    gaussian_fit = {
        "rv": rv,
        "rv_err": rv_err,
        "params": params,
        "diagnostics": diag,
    }
    bisector_fit: dict | None = None
    if bool(diag.get("fit_ok")) and np.isfinite(rv):
        rv_bi, err_bi, bmeta = _refine_ccf_rv_bisector_after_gaussian_fit(
            v_slot,
            y_slot,
            vs,
            ccf_err=ccf_err_arg,
            config=ccf_bisector_config,
        )
        if rv_bi is not None and math.isfinite(float(rv_bi)):
            bisector_fit = {
                "rv": float(rv_bi),
                "rv_err": (
                    float(err_bi)
                    if err_bi is not None and math.isfinite(float(err_bi)) and float(err_bi) > 0
                    else float("nan")
                ),
                "diagnostics": bmeta if isinstance(bmeta, dict) else {},
            }
    return gaussian_fit, bisector_fit


def finalize_ccf_preview_dict(
    order_slots: list[dict],
    all_ccf_rows: list[np.ndarray],
    v_grid: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    all_ccf_err_rows: list[np.ndarray] | None = None,
    *,
    ccf_line_mask_stored_negated: bool | None = None,
    ccf_parabolic_vertex_n_samples: int = 11,
    line_mask_line_window_by_line: bool | None = None,
    line_mask_gaussian_seed_mean_m_s: float | None = None,
) -> dict:
    """
    Build the ``_ccf_preview_by_fiber`` structure from per-order CCF 2-D arrays.

    ``order_slots`` entries are typically one slot per line-mask chunk (or one
    per echelle order when a row is a single chunk); this prepends the global
    mean slot (mean over all stacked chunk rows in ``all_ccf_rows``).

    When ``all_ccf_err_rows`` is provided (one stacked array per order, aligned with
    ``all_ccf_rows``), the **mean** slot gets ``ccf_err`` = standard error of the
    column-wise mean CCF (``sqrt(sum σ²)/n``, not ``sqrt(mean(σ²))``) and
    :func:`fit_ccf_rv` uses them as inverse-variance weights.

    Parameters
    ----------
    ccf_line_mask_stored_negated : bool or None, optional
        When not ``None``, forwarded to each slot’s :func:`fit_ccf_rv` call and copied to
        the returned dict as ``"ccf_line_mask_stored_negated"`` (mask-line preview only).
    ccf_parabolic_vertex_n_samples : int, optional
        **Odd** integer :math:`\\ge 3` for LSTSQ parabolic refinement after Gaussian fit
        (mask-line slots only; raises if even). Must match :func:`run_ccf_line_mask_for_fiber`.
    line_mask_line_window_by_line : bool or None, optional
        Forwarded to :func:`fit_ccf_rv` (see :func:`_line_mask_ccf_fit_free_baseline_requested`).
    line_mask_gaussian_seed_mean_m_s : float or None, optional
        When set (typically ``ccf_input_velocity`` / hybrid center), passed to :func:`fit_ccf_rv`
        as ``gaussian_seed_mean_m_s`` for mask-line stored CCFs so the discrete peak seed is
        taken near the search center instead of a global maximum on a nearly flat curve.
    """
    if not all_ccf_rows:
        out_empty: dict = {
            "v_step": float(v_step),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "slots": [],
        }
        if ccf_line_mask_stored_negated is not None:
            out_empty["ccf_line_mask_stored_negated"] = bool(ccf_line_mask_stored_negated)
        return out_empty
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
    vs = float(v_step)
    n_workers = _ccf_preview_finalize_n_workers(len(slots))
    logger.info(
        "CCF preview: finalize_ccf_preview_dict fitting Gaussians n_slots=%d n_workers=%d "
        "(mean + orders)",
        len(slots),
        n_workers,
    )

    def _run_slot(s: dict) -> tuple[dict, dict | None]:
        return _fit_ccf_preview_slot_worker(
            s,
            vs,
            ccf_line_mask_stored_negated=ccf_line_mask_stored_negated,
            ccf_parabolic_vertex_n_samples=int(ccf_parabolic_vertex_n_samples),
            line_mask_line_window_by_line=line_mask_line_window_by_line,
            line_mask_gaussian_seed_mean_m_s=line_mask_gaussian_seed_mean_m_s,
        )

    if n_workers <= 1:
        results = [_run_slot(sl) for sl in slots]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_slot, slots))
    for sl, (gaussian_fit, bisector_fit) in zip(slots, results):
        sl["gaussian_fit"] = gaussian_fit
        if bisector_fit is not None:
            sl["bisector_fit"] = bisector_fit
    logger.info(
        "CCF preview: finalize_ccf_preview_dict Gaussian + bisector fits done n_slots=%d",
        len(slots),
    )
    out: dict = {
        "v_step": float(v_step),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "slots": slots,
    }
    if ccf_line_mask_stored_negated is not None:
        out["ccf_line_mask_stored_negated"] = bool(ccf_line_mask_stored_negated)
    if (
        line_mask_gaussian_seed_mean_m_s is not None
        and math.isfinite(float(line_mask_gaussian_seed_mean_m_s))
    ):
        out["line_mask_gaussian_seed_mean_m_s"] = float(line_mask_gaussian_seed_mean_m_s)
    return out


def build_ccf_preview_dict(
    kind: Literal["numba_flat", "spectra_orders"],
    order_tasks: list[tuple[int, int]],
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    order_indices: np.ndarray | None = None,
    n_chunks_per_order: np.ndarray | None = None,
    chunk_start: np.ndarray | None = None,
    all_ccf: np.ndarray | None = None,
    v_grid: np.ndarray | None = None,
    wave_2d: np.ndarray | None = None,
    wave: np.ndarray | None = None,
    spec: np.ndarray | None = None,
    err: np.ndarray | None = None,
    dq: np.ndarray | None = None,
    do_not_use: int | None = None,
    template_ln: np.ndarray | None = None,
    template_flux: np.ndarray | None = None,
    window_size_um: float | None = None,
) -> dict | None:
    """
    Build the stellar-RV CCF preview JSON-like structure for Plotly (per-order slots, mean CCF, Gaussian).

    Parameters
    ----------
    kind : {'numba_flat', 'spectra_orders'}
        Which construction path to use:

        * ``numba_flat`` — consume flattened Numba outputs from
          :func:`process_orders_rvs_numba_parallel` with ``return_ccf=True`` (``all_ccf``,
          ``chunk_start``, ``n_chunks_per_order``, ``order_indices``, ``v_grid``, ``wave_2d``).
        * ``spectra_orders`` — run :func:`rv_per_chunks` per order in pure Python (slower;
          useful when debugging or avoiding the Numba stack).
    order_tasks : list of tuple
        ``(order_row_index, echelle_order_number)`` for each order included in the preview.
    v_min, v_max, v_step : float
        CCF velocity grid bounds and step (m/s) used for logging and passed through to
        :func:`finalize_ccf_preview_dict` for axis metadata and Gaussian reference curves.
    order_indices, n_chunks_per_order, chunk_start : ndarray, optional
        Required for ``kind='numba_flat'``. Layout matches
        :func:`process_orders_rvs_numba_parallel`: ``chunk_start`` indexes into the
        flattened ``all_ccf`` rows per order block.
    all_ccf : ndarray, optional
        2-D array of CCF rows (chunks × velocity), required for ``numba_flat``.
    v_grid : ndarray, optional
        1-D velocity samples (m/s), required for ``numba_flat``.
    wave_2d : ndarray, optional
        Shape ``(n_orders, n_pix)`` wavelength grid (µm); required for ``numba_flat`` to
        annotate wavelength span per order slot.
    wave, spec, err, dq : ndarray, optional
        Required for ``spectra_orders``: 2-D extracted-order arrays (same axis 0 as
        ``order_tasks`` row indices).
    do_not_use : int or None, optional
        DQ bitmask; pixels with ``(dq & do_not_use) != 0`` are treated as bad for masking.
    template_ln, template_flux : ndarray, optional
        Full template in ln(micron) and flux for ``spectra_orders``.
    window_size_um : float or None, optional
        Chunk width (µm) for :func:`rv_per_chunks` in ``spectra_orders`` mode.

    Returns
    -------
    dict or None
        Structure consumed by :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`
        (order slots + aggregated rows + fit metadata), or ``None`` if there is nothing
        to plot (no valid orders, empty CCF, or length mismatch between orders).

    Raises
    ------
    ValueError
        If required arguments for the selected ``kind`` are missing, or ``kind`` is unknown.
    """
    if kind == "numba_flat":
        if (
            order_indices is None
            or n_chunks_per_order is None
            or chunk_start is None
            or all_ccf is None
            or v_grid is None
            or wave_2d is None
        ):
            raise ValueError(
                "build_ccf_preview_dict(kind='numba_flat') requires order_indices, "
                "n_chunks_per_order, chunk_start, all_ccf, v_grid, wave_2d"
            )
        if not order_tasks or len(v_grid) == 0:
            return None
        order_slots: list[dict] = []
        all_rows: list[np.ndarray] = []
        for idx in range(len(order_tasks)):
            oi = int(order_indices[idx])
            order_num = int(order_tasks[idx][1])
            start = int(chunk_start[idx])
            nch = int(n_chunks_per_order[idx])
            block = all_ccf[start : start + nch]
            if block.size == 0:
                continue
            ccf_m = _nanmean_axis0_ignore_empty_columns(np.asarray(block, dtype=np.float64))
            w = wave_2d[oi]
            wl_lo, wl_hi = float(np.nanmin(w)), float(np.nanmax(w))
            order_slots.append(
                {
                    "kind": "order",
                    "order": order_num,
                    "v": np.asarray(v_grid, dtype=np.float64),
                    "ccf": np.asarray(ccf_m, dtype=np.float64),
                    "wl_lo": wl_lo,
                    "wl_hi": wl_hi,
                }
            )
            all_rows.append(block)
        if not order_slots:
            return None
        logger.info(
            "CCF preview: build_ccf_preview_dict numba_flat n_orders=%d (mean CCF + finalize)",
            len(order_slots),
        )
        return finalize_ccf_preview_dict(order_slots, all_rows, v_grid, v_min, v_max, v_step)

    if kind == "spectra_orders":
        if (
            wave is None
            or spec is None
            or err is None
            or dq is None
            or do_not_use is None
            or template_ln is None
            or template_flux is None
            or window_size_um is None
        ):
            raise ValueError(
                "build_ccf_preview_dict(kind='spectra_orders') requires wave, spec, err, dq, "
                "do_not_use, template_ln, template_flux, window_size_um"
            )
        if not order_tasks:
            return None
        logger.info(
            "CCF preview: build_ccf_preview_dict spectra_orders n_order_tasks=%d "
            "window_um=%.4f v_grid=[%.0f, %.0f] m/s step=%.1f",
            len(order_tasks),
            window_size_um,
            v_min,
            v_max,
            v_step,
        )
        order_slots_so: list[dict] = []
        all_rows_so: list[np.ndarray] = []
        v_grid_ref = None
        for oi, order in order_tasks:
            w = np.asarray(wave[oi], dtype=np.float64)
            s = np.asarray(spec[oi], dtype=np.float64).copy()
            e = np.asarray(err[oi] if err.ndim >= 2 else err, dtype=np.float64)
            d = np.asarray(dq[oi] if dq.ndim >= 2 else dq, dtype=np.uint32)
            mask = (d & do_not_use) != 0
            w_min, w_max = np.nanmin(w), np.nanmax(w)
            ln_min, ln_max = np.log(w_min), np.log(w_max)
            in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
            if np.sum(in_range) < 20:
                continue
            t_ln = template_ln[in_range]
            t_flux = template_flux[in_range]
            _rvs, _rv_errs, _centers, v_grid_o, ccf_2d = rv_per_chunks(
                w,
                s,
                e,
                mask,
                t_ln,
                t_flux,
                window_size_um,
                v_min=v_min,
                v_max=v_max,
                v_step=v_step,
                return_ccf=True,
            )
            if v_grid_ref is None:
                v_grid_ref = v_grid_o
            elif v_grid_ref.shape != v_grid_o.shape or not np.allclose(v_grid_ref, v_grid_o):
                continue
            ccf_m = _nanmean_axis0_ignore_empty_columns(np.asarray(ccf_2d, dtype=np.float64))
            order_slots_so.append(
                {
                    "kind": "order",
                    "order": int(order),
                    "v": np.asarray(v_grid_o, dtype=np.float64),
                    "ccf": np.asarray(ccf_m, dtype=np.float64),
                    "wl_lo": float(np.nanmin(w)),
                    "wl_hi": float(np.nanmax(w)),
                }
            )
            all_rows_so.append(ccf_2d)
        if not order_slots_so or v_grid_ref is None:
            logger.info(
                "CCF preview: build_ccf_preview_dict spectra_orders no valid orders after CCF pass"
            )
            return None
        logger.info(
            "CCF preview: build_ccf_preview_dict spectra_orders built %d order slots; finalize",
            len(order_slots_so),
        )
        return finalize_ccf_preview_dict(
            order_slots_so, all_rows_so, v_grid_ref, v_min, v_max, v_step
        )

    raise ValueError(f"unknown build_ccf_preview_dict kind: {kind!r}")


# =============================================================================
# Continuum normalization (optional pre-step before CCF on extracted orders)
# =============================================================================


def continuum_normalize(
    flux: np.ndarray,
    wavelength: np.ndarray | None = None,
    lowpass_window: int | None = None,
    poly_order: int | None = None,
) -> np.ndarray:
    """
    Divide flux by an estimated continuum (median filter and/or polynomial).

    Used optionally before CCF in :func:`process_one_order_rvs` so line depths are
    less sensitive to slow spectral slope.

    Parameters
    ----------
    flux : np.ndarray
        1-D flux.
    wavelength : np.ndarray or None, optional
        Abscissa for polynomial continuum; if None, pixel index is used.
    lowpass_window : int or None, optional
        Odd median-filter window; even values are incremented. None skips filtering.
    poly_order : int or None, optional
        If set, fit continuum as a polynomial of this order (after median step if any).

    Returns
    -------
    np.ndarray
        ``flux / continuum``, same shape as ``flux``.

    Notes
    -----
    If both ``lowpass_window`` and ``poly_order`` are None, returns ``flux`` unchanged.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    if n == 0 or (lowpass_window is None and poly_order is None):
        return flux
    x = np.asarray(wavelength, dtype=np.float64) if wavelength is not None else np.arange(n, dtype=np.float64)
    if len(x) != n:
        x = np.arange(n, dtype=np.float64)

    continuum = np.full_like(flux, np.nan)
    if lowpass_window is not None and lowpass_window > 0:
        w = int(lowpass_window)
        if w % 2 == 0:
            w += 1
        w = min(w, n)
        if w >= 3:
            continuum = median_filter(flux, size=w, mode="nearest")
        else:
            continuum = np.where(np.isfinite(flux), flux, np.nan)
    else:
        continuum = np.where(np.isfinite(flux), flux, np.nan)

    if poly_order is not None and poly_order >= 0:
        ok = np.isfinite(x) & np.isfinite(continuum) & (continuum > 0)
        if np.sum(ok) > max(poly_order + 1, 5):
            try:
                p = np.polyfit(x[ok], continuum[ok], poly_order)
                continuum = np.polyval(p, x)
            except Exception:
                pass

    continuum = np.maximum(continuum, 1e-30)
    return np.where(np.isfinite(flux) & np.isfinite(continuum), flux / continuum, flux)


# =============================================================================
# Masked log-λ template CCF (single order or chunk): weighted covariance + Gaussian fit
# =============================================================================


def _ln_lam(wave: np.ndarray) -> np.ndarray:
    """Natural log of wavelength (micron → log micron). Internal helper."""
    return np.log(np.asarray(wave, dtype=np.float64))




def _ccf_weighted_covariance_scan(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_grid: np.ndarray,
    zb: float = 0.0,
) -> np.ndarray:
    """Python CCF(v); same statistic as ``_ccf_chunk_ccf_numba`` (fallback when Numba off)."""
    ccf = np.zeros_like(v_grid, dtype=np.float64)
    zb1 = 1.0 + float(zb)
    if zb1 <= 0.0:
        zb1 = 1.0
    for i, v in enumerate(v_grid):
        delta_ln = math.log((1.0 + float(v) / C_M_S) / zb1)
        t_at_data = np.interp(
            ln_lam,
            template_ln + delta_ln,
            template_flux,
            left=0.0,
            right=0.0,
        )
        use = good & np.isfinite(t_at_data) & (t_at_data != 0)
        if np.sum(use) < 5:
            ccf[i] = np.nan
            continue
        d = spec[use]
        t = t_at_data[use]
        w = 1.0 / np.maximum(spec_err[use] ** 2, 1e-30)
        d_norm = d - np.average(d, weights=w)
        t_norm = t - np.average(t, weights=w)
        ccf[i] = np.average(d_norm * t_norm, weights=w)
    return ccf


def _masked_ccf_rv_core(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    diagnostics: bool,
    v_grid: np.ndarray | None = None,
    zb: float = 0.0,
    min_good_pixels: int = 10,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, dict | None]:
    """
    Single CCF build + fit; optionally build diagnostics dict (no duplicate CCF work).

    Returns
    -------
    rv, rv_err, v_grid, ccf, diagnostics_or_none
    """
    good = ~mask & np.isfinite(spec) & np.isfinite(spec_err) & (spec_err > 0)
    n_good = int(np.sum(good))
    wl_min = float(np.nanmin(wave))
    wl_max = float(np.nanmax(wave))
    spec_mean = float(np.nanmean(spec[good])) if n_good > 0 else np.nan
    spec_std = float(np.nanstd(spec[good])) if n_good > 1 else 0.0
    empty_diag = {
        "ccf_max": np.nan,
        "ccf_min": np.nan,
        "n_good": n_good,
        "n_use_at_peak": 0,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "template_mean_at_peak": np.nan,
        "template_std_at_peak": np.nan,
        "wl_min": wl_min,
        "wl_max": wl_max,
    }

    if v_grid is not None:
        v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    else:
        v_grid = _make_velocity_grid(v_min, v_max, v_step)
    mg = int(min_good_pixels)
    if n_good < mg:
        nan_ccf = np.full_like(v_grid, np.nan)
        if diagnostics:
            return np.nan, np.nan, v_grid, nan_ccf, empty_diag
        return np.nan, np.nan, v_grid, nan_ccf, None

    ln_lam = _ln_lam(wave)
    ccf = np.full(len(v_grid), np.nan, dtype=np.float64)
    v_step_fit = effective_ccf_v_step_for_fit(v_grid)
    if (
        len(v_grid) >= 2
        and np.allclose(np.diff(v_grid), v_grid[1] - v_grid[0], rtol=1e-5, atol=1e-3)
    ):
        _ccf_chunk_ccf_numba(
            ln_lam,
            spec,
            spec_err,
            good,
            template_ln,
            template_flux,
            float(v_grid[0]),
            float(v_grid[-1]),
            float(v_grid[1] - v_grid[0]),
            ccf,
            mg,
            float(zb),
        )
    else:
        ccf[:] = _ccf_weighted_covariance_scan(
            ln_lam,
            spec,
            spec_err,
            good,
            template_ln,
            template_flux,
            v_grid,
            zb=float(zb),
        )
    if not np.any(np.isfinite(ccf)):
        if diagnostics:
            return np.nan, np.nan, v_grid, ccf, empty_diag
        return np.nan, np.nan, v_grid, ccf, None

    rv, rv_err = _ccf_rv_from_ccf(
        v_grid,
        ccf,
        v_step_fit,
        n_good,
        min_good_pixels=mg,
        ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
    )
    diag_out: dict | None = None
    if diagnostics:
        ccf_max = float(np.nanmax(ccf))
        ccf_min = float(np.nanmin(ccf))
        zb1 = 1.0 + float(zb)
        if zb1 <= 0.0:
            zb1 = 1.0
        delta_ln_peak = math.log((1.0 + float(rv) / C_M_S) / zb1)
        template_ln_shifted = template_ln + delta_ln_peak
        t_at_data = np.interp(
            ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
        )
        use_peak = good & np.isfinite(t_at_data) & (t_at_data != 0)
        n_use_at_peak = int(np.sum(use_peak))
        t_mean = float(np.mean(t_at_data[use_peak])) if n_use_at_peak > 0 else np.nan
        t_std = float(np.std(t_at_data[use_peak])) if n_use_at_peak > 1 else 0.0
        diag_out = {
            "ccf_max": ccf_max,
            "ccf_min": ccf_min,
            "n_good": n_good,
            "n_use_at_peak": n_use_at_peak,
            "spec_mean": spec_mean,
            "spec_std": spec_std,
            "template_mean_at_peak": t_mean,
            "template_std_at_peak": t_std,
            "wl_min": wl_min,
            "wl_max": wl_max,
        }
    return rv, rv_err, v_grid, ccf, diag_out


def masked_ccf_rv(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
    *,
    v_grid: np.ndarray | None = None,
    zb: float = 0.0,
    min_good_pixels: int = 10,
    ccf_parabolic_vertex_n_samples: int | None = None,
    return_ccf: bool = False,
    diagnostics: bool = False,
) -> (
    tuple[float, float]
    | tuple[float, float, np.ndarray, np.ndarray]
    | tuple[float, float, dict]
    | tuple[float, float, np.ndarray, np.ndarray, dict]
):
    """
    Radial velocity from a single masked spectrum via log-λ CCF and Gaussian peak fit.

    Builds a weighted covariance CCF on a velocity grid, then calls
    :func:`fit_ccf_rv` (sub-grid RV); on failure uses the grid peak and a
    HWHM-based error.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength per pixel, micron.
    spec, spec_err : np.ndarray
        Flux and uncertainty.
    mask : np.ndarray
        Boolean bad-pixel mask; **``True`` excludes** the pixel from the weighted CCF.
    template_ln, template_flux : np.ndarray
        Stellar template sampled on **natural log wavelength** (ln µm) and flux, covering
        the observed wavelength interval.
    v_min, v_max, v_step : float, optional
        Uniform CCF velocity range and step (m/s) when ``v_grid`` is not passed. Ignored
        if ``v_grid`` is given.
    v_grid : np.ndarray or None, optional
        Explicit velocity samples (m/s), e.g. from :func:`build_hybrid_merged_velocity_grid`.
        When passed, ``v_min`` / ``v_max`` / ``v_step`` are not used to build the grid.
    zb : float, optional
        Barycentric factor in the template :math:`\\ln\\lambda` shift (see
        :func:`_ccf_chunk_ccf_numba`).
    min_good_pixels : int, optional
        Minimum good pixels required before building the CCF and fitting.
    ccf_parabolic_vertex_n_samples : int or None, optional
        Deprecated; ignored for the returned RV. After the Gaussian fit, final RV and
        :math:`\\sigma_{\\rm RV}` come from :func:`fit_ccf_rv_bisector` when that step succeeds.
    return_ccf : bool, optional
        If ``True``, also return ``v_grid`` and the 1-D CCF array on that grid.
    diagnostics : bool, optional
        If ``True``, also return a diagnostics dict with template/CCF statistics at the
        fitted peak (produced by the internal CCF core).

    Returns
    -------
    tuple
        * Default: ``(rv, rv_err)`` with RV and uncertainty (m/s).
        * ``return_ccf=True``: ``(rv, rv_err, v_grid, ccf)``.
        * ``diagnostics=True``: ``(rv, rv_err, diagnostics)``.
        * Both flags: ``(rv, rv_err, v_grid, ccf, diagnostics)``.

    Notes
    -----
    Calls :func:`fit_ccf_rv` for sub-grid Gaussian RV when the weighted CCF is valid;
    otherwise falls back to the discrete peak with a HWHM-style error estimate.

    See Also
    --------
    rv_per_chunks : Per-chunk CCFs along an order.
    """
    rv, rv_err, vg, ccf, diag = _masked_ccf_rv_core(
        wave,
        spec,
        spec_err,
        mask,
        template_ln,
        template_flux,
        v_min,
        v_max,
        v_step,
        diagnostics=diagnostics,
        v_grid=v_grid,
        zb=float(zb),
        min_good_pixels=int(min_good_pixels),
        ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
    )
    if return_ccf and diagnostics:
        assert diag is not None
        return rv, rv_err, vg, ccf, diag
    if return_ccf:
        return rv, rv_err, vg, ccf
    if diagnostics:
        assert diag is not None
        return rv, rv_err, diag
    return rv, rv_err


# =============================================================================
# Split spectrum into wavelength chunks; each chunk → masked_ccf_rv
# =============================================================================


def rv_per_chunks(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
    *,
    return_ccf: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Split one echelle spectrum into wavelength chunks; one RV per chunk.

    Each chunk with at least 20 finite pixels uses :func:`_masked_ccf_rv_core` once
    (Numba CCF when available). Chunks with fewer valid pixels are left as NaN RVs.

    Parameters
    ----------
    wave : np.ndarray
        1-D wavelength samples (µm), same length as ``spec``.
    spec, spec_err : np.ndarray
        Flux and per-pixel uncertainty.
    mask : np.ndarray
        Boolean; **``True`` excludes** pixels from the CCF (same convention as
        :func:`masked_ccf_rv`).
    template_ln, template_flux : np.ndarray
        Template on ln(µm) and flux, covering the order’s wavelength span.
    window_size_um : float
        Target chunk width (µm); chunk count is :func:`ccf_order_n_chunks_for_span`
        applied to the order wavelength span.
    v_min, v_max, v_step : float, optional
        Uniform velocity grid (m/s) for the CCF inside each chunk.
    return_ccf : bool, optional
        If ``True``, also return ``v_grid`` and a 2-D array ``ccf_2d`` with shape
        ``(n_chunk, n_vel)`` (NaN rows where a chunk did not run).

    Returns
    -------
    rvs, rv_errs, chunk_centers : np.ndarray
        Per-chunk RV (m/s), uncertainty (m/s), and chunk center wavelength (µm).
    v_grid, ccf_2d : np.ndarray, optional
        Returned only when ``return_ccf=True``: shared velocity axis and stacked CCF rows.

    See Also
    --------
    masked_ccf_rv : Single-spectrum masked CCF on one segment.
    """
    w_min, w_max = np.nanmin(wave), np.nanmax(wave)
    span = float(w_max - w_min)
    n_chunk = ccf_order_n_chunks_for_span(span, float(window_size_um))
    rvs = np.full(n_chunk, np.nan, dtype=np.float64)
    rv_errs = np.full(n_chunk, np.nan, dtype=np.float64)
    v_grid = _make_velocity_grid(v_min, v_max, v_step)
    ccf_2d = (
        np.full((n_chunk, len(v_grid)), np.nan, dtype=np.float64) if return_ccf else None
    )
    chunk_centers = np.full(n_chunk, np.nan, dtype=np.float64)

    for k in range(n_chunk):
        lam_lo = w_min + k * span / n_chunk
        lam_hi = w_min + (k + 1) * span / n_chunk
        if k == n_chunk - 1:
            in_chunk = (wave >= lam_lo) & (wave <= lam_hi)
        else:
            in_chunk = (wave >= lam_lo) & (wave < lam_hi)
        chunk_centers[k] = 0.5 * (lam_lo + lam_hi)
        if np.sum(in_chunk) < 20:
            continue
        rv, rv_err, vg, ccf, _ = _masked_ccf_rv_core(
            wave[in_chunk],
            spec[in_chunk],
            spec_err[in_chunk],
            mask[in_chunk],
            template_ln,
            template_flux,
            v_min,
            v_max,
            v_step,
            diagnostics=False,
        )
        rvs[k] = rv
        rv_errs[k] = rv_err
        if return_ccf and ccf_2d is not None and vg.shape == v_grid.shape and np.allclose(vg, v_grid):
            ccf_2d[k, :] = ccf

    if return_ccf:
        assert ccf_2d is not None
        return rvs, rv_errs, chunk_centers, v_grid, ccf_2d
    return rvs, rv_errs, chunk_centers


# =============================================================================
# Single-order helper + multi-chunk CCF export (preview / diagnostics)
# =============================================================================


def process_one_order_rvs(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    do_continuum_normalize: bool = False,
    continuum_lowpass_window: int | None = None,
    continuum_poly_order: int | None = None,
    *,
    return_ccf: bool = False,
    ccf_v_min: float | None = None,
    ccf_v_max: float | None = None,
    ccf_v_step: float | None = None,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    One echelle order: clip template, optional continuum norm, chunk RVs.

    Serial-friendly (picklable) path used when not using the Numba multi-order
    driver.  Returns empty arrays if the template has fewer than 20 points in the
    order wavelength range. When ``ccf_hybrid_center_m_s`` is set, delegates to
    :func:`process_orders_rvs_numba_parallel` (broad + refined hybrid).

    Parameters
    ----------
    wave, spec, spec_err : np.ndarray
        1-D wavelength (micron), flux, uncertainty.
    dq : np.ndarray, uint32
        Data quality; pixels with ``(dq & do_not_use) != 0`` are masked.
    do_not_use : int
        DQ bitmask for bad pixels.
    template_ln, template_flux : np.ndarray
        Full-template arrays (log micron, flux).
    window_size_um : float
        Wavelength chunk width (micron).
    do_continuum_normalize : bool, optional
        If True and a continuum option is set, normalize flux before CCF.
    continuum_lowpass_window, continuum_poly_order : int or None, optional
        Passed to :func:`continuum_normalize`.
    return_ccf : bool, optional
        If True, also return ``v_grid`` and per-chunk CCF ``ccf_2d`` (uses
        ``ccf_v_min`` / ``ccf_v_max`` / ``ccf_v_step`` or defaults ``_V_*``).
    ccf_v_min, ccf_v_max, ccf_v_step : float or None, optional
        Broad uniform velocity grid (m/s). When ``ccf_hybrid_center_m_s`` is set, these
        define the coarse leg of the merged grid; otherwise they define the sole grid
        for :func:`rv_per_chunks`. Defaults use module ``V_MIN_DEFAULT_M_S``, ``V_MAX_DEFAULT_M_S``, ``V_STEP_DEFAULT_M_S``.
    ccf_hybrid_center_m_s : float or None, optional
        When **finite**, delegates to :func:`process_orders_rvs_numba_parallel` on this
        single order with hybrid velocity sampling (broad scan plus dense window around
        the center). When ``None``, uses the uniform grid only (via ``rv_per_chunks``).
    ccf_hybrid_half_width_m_s, ccf_hybrid_step_m_s : float, optional
        Half-width (m/s) and step (m/s) for the refined hybrid leg when
        ``ccf_hybrid_center_m_s`` is set.

    Returns
    -------
    rvs : np.ndarray
        Per-chunk RV (m/s).
    rv_errs : np.ndarray
        Per-chunk uncertainties (m/s).
    chunk_centers : np.ndarray
        Chunk center wavelengths (micron).
    v_grid, ccf_2d : np.ndarray, optional
        Returned only when ``return_ccf`` is True.

    See Also
    --------
    process_orders_rvs_numba_parallel : Faster path for many orders at once.
    """
    w = np.asarray(wave, dtype=np.float64)
    s = np.asarray(spec, dtype=np.float64).copy()
    e = np.asarray(spec_err, dtype=np.float64)
    mask = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    if do_continuum_normalize and (continuum_lowpass_window is not None or continuum_poly_order is not None):
        s = continuum_normalize(
            s,
            wavelength=w,
            lowpass_window=continuum_lowpass_window,
            poly_order=continuum_poly_order,
        )

    w_min, w_max = np.nanmin(w), np.nanmax(w)
    ln_min, ln_max = np.log(w_min), np.log(w_max)
    in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
    if np.sum(in_range) < 20:
        empty3 = (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        if return_ccf:
            vm = V_MIN_DEFAULT_M_S if ccf_v_min is None else ccf_v_min
            vx = V_MAX_DEFAULT_M_S if ccf_v_max is None else ccf_v_max
            vs = V_STEP_DEFAULT_M_S if ccf_v_step is None else ccf_v_step
            vg = _make_velocity_grid(vm, vx, vs)
            return empty3 + (vg, np.zeros((0, len(vg)), dtype=np.float64))
        return empty3

    t_ln = template_ln[in_range]
    t_flux = template_flux[in_range]

    vm = V_MIN_DEFAULT_M_S if ccf_v_min is None else ccf_v_min
    vx = V_MAX_DEFAULT_M_S if ccf_v_max is None else ccf_v_max
    vs = V_STEP_DEFAULT_M_S if ccf_v_step is None else ccf_v_step

    if ccf_hybrid_center_m_s is not None:
        dq_1d = np.where(mask, np.uint32(do_not_use), np.uint32(0))
        wave_2d = w.reshape(1, -1)
        spec_2d = s.reshape(1, -1)
        err_2d = e.reshape(1, -1)
        dq_2d = dq_1d.reshape(1, -1)
        out = process_orders_rvs_numba_parallel(
            wave_2d,
            spec_2d,
            err_2d,
            dq_2d,
            do_not_use,
            t_ln,
            t_flux,
            window_size_um,
            [(0, 0)],
            max_cores=1,
            return_ccf=return_ccf,
            ccf_v_min=vm,
            ccf_v_max=vx,
            ccf_v_step=vs,
            ccf_hybrid_center_m_s=float(ccf_hybrid_center_m_s),
            ccf_hybrid_half_width_m_s=float(ccf_hybrid_half_width_m_s),
            ccf_hybrid_step_m_s=float(ccf_hybrid_step_m_s),
        )
        if return_ccf:
            arv, are, awl, accf, vg, ang, _nc, _cs, _oi = out
            return arv, are, awl, vg, accf
        arv, are, awl, _nc, _cs = out
        return arv, are, awl

    return rv_per_chunks(
        w,
        s,
        e,
        mask,
        t_ln,
        t_flux,
        window_size_um,
        v_min=vm,
        v_max=vx,
        v_step=vs,
        return_ccf=return_ccf,
    )


# =============================================================================
# Mask-shift line CCF (SPIROU .mas / KPF two-column)
# =============================================================================

@njit(cache=True)
def _linear_interp_sorted_1d(xq: float, x: np.ndarray, y: np.ndarray, n: int) -> float:
    """Linear interpolation on sorted ``x[:n]`` (ascending); clamps outside range."""
    if n <= 0:
        return np.nan
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[n - 1]:
        return float(y[n - 1])
    i = int(np.searchsorted(x, xq, side="right"))
    i0 = i - 1
    i1 = i0 + 1
    if i1 >= n:
        return float(y[n - 1])
    x0 = float(x[i0])
    x1 = float(x[i1])
    if x1 <= x0:
        return float(y[i0])
    t = (xq - x0) / (x1 - x0)
    return (1.0 - t) * float(y[i0]) + t * float(y[i1])


@njit(cache=True)
def _poly_interp_sorted_1d(
    xq: float,
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    order: int,
) -> float:
    """
    Lagrange interpolation on sorted ``x[:n]`` (ascending); clamps outside ``[x[0], x[n-1]]``.

    Uses ``k = min(order + 1, n)`` consecutive samples centered on the ``searchsorted``
    bracket (degree at most ``k - 1``). ``order == 1`` matches :func:`_linear_interp_sorted_1d`
    on interior points.
    """
    if n <= 0:
        return np.nan
    if n == 1:
        return float(y[0])

    o = int(order)
    if o < 1:
        o = 1
    if o > MAX_MASK_CCF_PIXEL_INTERP_ORDER:
        o = MAX_MASK_CCF_PIXEL_INTERP_ORDER

    k = o + 1
    if k > n:
        k = n

    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[n - 1]):
        return float(y[n - 1])

    j = int(np.searchsorted(x, xq, side="right")) - 1
    if j < 0:
        j = 0
    if j >= n - 1:
        j = n - 2

    s = j - (k - 1) // 2
    if s < 0:
        s = 0
    if s + k > n:
        s = n - k

    acc = 0.0
    for m in range(k):
        xm = float(x[s + m])
        ym = float(y[s + m])
        lm = 1.0
        for t in range(k):
            if t == m:
                continue
            xt = float(x[s + t])
            d = xm - xt
            if d == 0.0:
                return ym
            lm *= (xq - xt) / d
        acc += ym * lm
    return acc


@njit(cache=True)
def _poly_interp_sorted_1d_value_and_flux_variance(
    xq: float,
    x: np.ndarray,
    y: np.ndarray,
    e_sig: np.ndarray,
    n: int,
    order: int,
) -> tuple[float, float]:
    """
    Lagrange interpolation of ``y`` at ``xq`` plus propagated variance from pixel σ.

    On the same stencil as ``_poly_interp_sorted_1d``, with independent pixel
    uncertainties ``e_sig[m]`` (standard deviations), returns
    ``(ŷ, \\sum_m L_m^2 e_sig[m]^2)``. Edge clamps return the endpoint sample and its
    ``e_sig^2``. Returns ``(nan, nan)`` if any stencil ``e_sig`` is non-finite or
    non-positive (caller should fall back to flux-only interpolation).
    """
    if n <= 0:
        return np.nan, np.nan
    if n == 1:
        es = float(e_sig[0])
        if not np.isfinite(es) or es <= 0.0:
            return np.nan, np.nan
        return float(y[0]), es * es

    o = int(order)
    if o < 1:
        o = 1
    if o > MAX_MASK_CCF_PIXEL_INTERP_ORDER:
        o = MAX_MASK_CCF_PIXEL_INTERP_ORDER

    k = o + 1
    if k > n:
        k = n

    if xq <= float(x[0]):
        es = float(e_sig[0])
        if not np.isfinite(es) or es <= 0.0:
            return np.nan, np.nan
        return float(y[0]), es * es
    if xq >= float(x[n - 1]):
        es = float(e_sig[n - 1])
        if not np.isfinite(es) or es <= 0.0:
            return np.nan, np.nan
        return float(y[n - 1]), es * es

    j = int(np.searchsorted(x, xq, side="right")) - 1
    if j < 0:
        j = 0
    if j >= n - 1:
        j = n - 2

    s = j - (k - 1) // 2
    if s < 0:
        s = 0
    if s + k > n:
        s = n - k

    acc_y = 0.0
    acc_var = 0.0
    for m in range(k):
        xm = float(x[s + m])
        ym = float(y[s + m])
        es_m = float(e_sig[s + m])
        if not np.isfinite(es_m) or es_m <= 0.0:
            return np.nan, np.nan
        lm = 1.0
        for t in range(k):
            if t == m:
                continue
            xt = float(x[s + t])
            d = xm - xt
            if d == 0.0:
                # Degenerate stencil; variance from duplicated abscissa is undefined.
                return float(ym), float(np.nan)
            lm *= (xq - xt) / d
        acc_y += ym * lm
        acc_var += lm * lm * es_m * es_m
    return acc_y, acc_var


@njit(cache=True)
def _lagrange_stencil_index_and_weights(
    xq: float,
    x: np.ndarray,
    n: int,
    order: int,
    out_L: np.ndarray,
) -> tuple[int, int]:
    """
    Lagrange basis ``L_m`` on the same stencil as ``_poly_interp_sorted_1d``.

    Writes ``out_L[0:k]`` and returns ``(s, k)`` where pixel indices are ``s, s+1, …, s+k-1``.
    Clamped endpoints use a one-point stencil with ``L[0]=1``. Returns ``(-1, 0)`` if invalid.
    """
    if n <= 0:
        return -1, 0
    if n == 1:
        out_L[0] = 1.0
        return 0, 1

    o = int(order)
    if o < 1:
        o = 1
    if o > MAX_MASK_CCF_PIXEL_INTERP_ORDER:
        o = MAX_MASK_CCF_PIXEL_INTERP_ORDER

    k = o + 1
    if k > n:
        k = n

    if xq <= float(x[0]):
        out_L[0] = 1.0
        return 0, 1
    if xq >= float(x[n - 1]):
        out_L[0] = 1.0
        return n - 1, 1

    j = int(np.searchsorted(x, xq, side="right")) - 1
    if j < 0:
        j = 0
    if j >= n - 1:
        j = n - 2

    s = j - (k - 1) // 2
    if s < 0:
        s = 0
    if s + k > n:
        s = n - k

    for m in range(k):
        xm = float(x[s + m])
        lm = 1.0
        for t in range(k):
            if t == m:
                continue
            xt = float(x[s + t])
            d = xm - xt
            if d == 0.0:
                return -1, 0
            lm *= (xq - xt) / d
        out_L[m] = lm
    return s, k


@njit(cache=True, parallel=True)
def _calc_mask_ccf_chunks_numba(
    left_edges_2d: np.ndarray,
    right_edges_2d: np.ndarray,
    s_use_2d: np.ndarray,
    e_use_2d: np.ndarray,
    sn_2d: np.ndarray,
    n_pix_arr: np.ndarray,
    v_grid_m_s: np.ndarray,
    zb: float,
    line_lambda_rest_2d: np.ndarray,
    line_left_rest_2d: np.ndarray,
    line_right_rest_2d: np.ndarray,
    line_w_2d: np.ndarray,
    n_line_arr: np.ndarray,
    compute_line_std: bool,
    pixel_interp_order: int,
    tophat_segment_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chunk-batched mask CCF(v) with parallelism over chunks.

    **Delta-line model:** each mask line is a rest wavelength :math:`\\lambda_{\\rm rest}`.
    At trial velocity :math:`v`, the observed wavelength is
    :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}\\,(1+v/c)/(1+z_b)`.

    **Tophat segment model (finite width):** each mask entry has rest edges
    :math:`[\\lambda_{\\rm lo},\\lambda_{\\rm hi}]`. At trial velocity the observed segment is
    shifted by the same factor as above. The segment’s contribution is approximated by
    averaging :math:`F\\cdot\\mathrm{SN}` over ``tophat_segment_samples`` uniformly spaced
    points inside the shifted segment.

    For each velocity, let :math:`x_j = F_j\\,\\mathrm{SN}_j` at each contributing line
    (interpolated on pixel centers). Accumulate :math:`\\sum_j w_j x_j` and
    :math:`\\sum_j w_j x_j^2` over lines whose shifted :math:`\\lambda_{\\rm obs}` lies in
    the chunk with finite :math:`F` and :math:`\\mathrm{SN}`. The CCF is the weighted
    mean :math:`\\mathrm{CCF}(v)=\\sum_j w_j x_j\\,/\\,\\sum_j w_j`.

    **Flux errors:** ``e_use_2d`` holds per-pixel flux :math:`\\sigma` on the chunk grid.
    The mask statistic is linear in pixel values
    :math:`G_m` (Lagrange interpolation of flux per line, :math:`\\mathrm{SN}_j` treated as
    exact on the stencil):
    :math:`C = \\sum_m c_m G_m` with
    :math:`c_m = \\sum_j (w_j/W)\\,\\mathrm{SN}_j\\,L^{(j)}_{m}` where :math:`W=\\sum_j w_j`
    and :math:`L^{(j)}_{m}` is the Lagrange weight of pixel :math:`m` for line :math:`j`
    (zero outside that line's stencil). Under independent pixel noise,
    :math:`\\sigma_{\\mathrm{meas}}^2 = \\sum_m c_m^2 \\sigma_m^2`. This **co-adds** lines
    that share stencil pixels and replaces the earlier :math:`\\sum_j w_j^2\\mathrm{Var}(F_j)/W^2`
    independent-line sum, which **inflated** :math:`\\sigma_{\\mathrm{meas}}` when many lines
    reused the same pixels.

    The **systematic / excess line-to-line** component is the usual weighted dispersion
    of :math:`x_j` about :math:`C`:

    .. math::

        \\sigma_{\\mathrm{line}}^2 = \\max\\left(0,\\;
          \\frac{\\sum_j w_j x_j^2}{\\sum_j w_j} - C^2\\right)\\,.

    When ``compute_line_std`` is True, the second output combines both in quadrature:

    .. math::

        \\sigma_{\\mathrm{CCF}} = \\sqrt{\\sigma_{\\mathrm{meas}}^2 + \\sigma_{\\mathrm{line}}^2}\\,.

    The third output stores :math:`\\sigma_{\\mathrm{meas}}` (``nan`` when unavailable).

    Input arrays are padded to fixed widths; ``n_pix_arr`` and ``n_line_arr`` give
    valid lengths for each chunk row.
    """
    n_chunk = left_edges_2d.shape[0]
    n_v = v_grid_m_s.shape[0]
    n_line_pad = line_lambda_rest_2d.shape[1]
    out_ccf = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
    if compute_line_std:
        out_std = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
        out_sigma_flux = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
    else:
        out_std = np.empty((0, 0), dtype=np.float64)
        out_sigma_flux = np.empty((0, 0), dtype=np.float64)
    zb1 = 1.0 + zb

    for ic in prange(n_chunk):
        n_pix = int(n_pix_arr[ic])
        n_ln = int(n_line_arr[ic])
        if n_pix <= 0 or n_ln <= 0:
            continue
        chunk_lo = left_edges_2d[ic, 0]
        chunk_hi = right_edges_2d[ic, n_pix - 1]
        centers = np.empty(n_pix, dtype=np.float64)
        for n in range(n_pix):
            centers[n] = 0.5 * (
                left_edges_2d[ic, n] + right_edges_2d[ic, n]
            )
        lag_L_buf = np.empty(MAX_MASK_CCF_PIXEL_INTERP_ORDER + 1, dtype=np.float64)
        coef_pix = np.zeros(n_pix, dtype=np.float64)
        n_samp = int(tophat_segment_samples)
        if n_samp < 1:
            n_samp = 1
        for i_v in range(n_v):
            sh = (1.0 + v_grid_m_s[i_v] / C_M_S) / zb1
            total = 0.0
            w_sum = 0.0
            sum_wx2 = 0.0
            for j in range(n_ln):
                lam_rest = line_lambda_rest_2d[ic, j]
                if not np.isfinite(lam_rest) or lam_rest <= 0.0:
                    continue
                wj = line_w_2d[ic, j]
                if not np.isfinite(wj) or wj <= 0.0:
                    continue

                lam_lo_r = line_left_rest_2d[ic, j]
                lam_hi_r = line_right_rest_2d[ic, j]
                finite_edges = (
                    np.isfinite(lam_lo_r)
                    and np.isfinite(lam_hi_r)
                    and lam_hi_r >= lam_lo_r
                    and lam_lo_r > 0.0
                )
                if finite_edges and lam_hi_r > lam_lo_r:
                    # Finite-width tophat segment: average samples across the segment.
                    lam_lo = lam_lo_r * sh
                    lam_hi = lam_hi_r * sh
                    if lam_hi < chunk_lo or lam_lo > chunk_hi:
                        continue
                    # Clamp to chunk support to avoid sampling outside pixels.
                    if lam_lo < chunk_lo:
                        lam_lo = chunk_lo
                    if lam_hi > chunk_hi:
                        lam_hi = chunk_hi
                    if lam_hi <= lam_lo:
                        continue
                    acc = 0.0
                    n_ok = 0
                    for t0 in range(n_samp):
                        frac = (t0 + 0.5) / float(n_samp)
                        lam0 = lam_lo + frac * (lam_hi - lam_lo)
                        fv = _poly_interp_sorted_1d(
                            lam0, centers, s_use_2d[ic, :], n_pix, pixel_interp_order
                        )
                        snv = _poly_interp_sorted_1d(
                            lam0, centers, sn_2d[ic, :], n_pix, pixel_interp_order
                        )
                        if not (np.isfinite(fv) and np.isfinite(snv)):
                            continue
                        acc += fv * snv
                        n_ok += 1
                    if n_ok <= 0:
                        continue
                    xj = acc / float(n_ok)
                else:
                    # Zero-width line (delta): sample at the shifted center.
                    lam0 = lam_rest * sh
                    if lam0 < chunk_lo or lam0 > chunk_hi:
                        continue
                    fv = _poly_interp_sorted_1d(
                        lam0, centers, s_use_2d[ic, :], n_pix, pixel_interp_order
                    )
                    snv = _poly_interp_sorted_1d(
                        lam0, centers, sn_2d[ic, :], n_pix, pixel_interp_order
                    )
                    if not (np.isfinite(fv) and np.isfinite(snv)):
                        continue
                    xj = fv * snv
                partial = wj * xj
                total += partial
                w_sum += wj
                sum_wx2 += wj * xj * xj
            if w_sum > 0.0:
                inv_w = 1.0 / w_sum
                ccf = total * inv_w
                out_ccf[ic, i_v] = ccf
                if compute_line_std:
                    mean_x2 = sum_wx2 * inv_w
                    var_line = mean_x2 - ccf * ccf
                    if var_line < 0.0:
                        var_line = 0.0
                    sig_line = math.sqrt(var_line)
                    for mm in range(n_pix):
                        coef_pix[mm] = 0.0
                    for j in range(n_ln):
                        lam_rest = line_lambda_rest_2d[ic, j]
                        if not np.isfinite(lam_rest) or lam_rest <= 0.0:
                            continue
                        wj = line_w_2d[ic, j]
                        if not np.isfinite(wj) or wj <= 0.0:
                            continue

                        lam_lo_r = line_left_rest_2d[ic, j]
                        lam_hi_r = line_right_rest_2d[ic, j]
                        finite_edges = (
                            np.isfinite(lam_lo_r)
                            and np.isfinite(lam_hi_r)
                            and lam_hi_r >= lam_lo_r
                            and lam_lo_r > 0.0
                        )
                        if finite_edges and lam_hi_r > lam_lo_r:
                            lam_lo = lam_lo_r * sh
                            lam_hi = lam_hi_r * sh
                            if lam_hi < chunk_lo or lam_lo > chunk_hi:
                                continue
                            if lam_lo < chunk_lo:
                                lam_lo = chunk_lo
                            if lam_hi > chunk_hi:
                                lam_hi = chunk_hi
                            if lam_hi <= lam_lo:
                                continue
                            # Average coefficient contributions across segment samples.
                            for t0 in range(n_samp):
                                frac = (t0 + 0.5) / float(n_samp)
                                lam0 = lam_lo + frac * (lam_hi - lam_lo)
                                s_l, k_l = _lagrange_stencil_index_and_weights(
                                    lam0,
                                    centers,
                                    n_pix,
                                    pixel_interp_order,
                                    lag_L_buf,
                                )
                                if s_l < 0 or k_l <= 0:
                                    continue
                                snv = _poly_interp_sorted_1d(
                                    lam0, centers, sn_2d[ic, :], n_pix, pixel_interp_order
                                )
                                if not np.isfinite(snv):
                                    continue
                                pref = (wj * inv_w * snv) / float(n_samp)
                                for t in range(k_l):
                                    coef_pix[s_l + t] += pref * lag_L_buf[t]
                        else:
                            lam0 = lam_rest * sh
                            if lam0 < chunk_lo or lam0 > chunk_hi:
                                continue
                            s_l, k_l = _lagrange_stencil_index_and_weights(
                                lam0,
                                centers,
                                n_pix,
                                pixel_interp_order,
                                lag_L_buf,
                            )
                            if s_l < 0 or k_l <= 0:
                                continue
                            snv = _poly_interp_sorted_1d(
                                lam0, centers, sn_2d[ic, :], n_pix, pixel_interp_order
                            )
                            if not np.isfinite(snv):
                                continue
                            pref = wj * inv_w * snv
                            for t in range(k_l):
                                coef_pix[s_l + t] += pref * lag_L_buf[t]
                    sig_meas_sq = 0.0
                    for m in range(n_pix):
                        cm = float(coef_pix[m])
                        if cm != 0.0:
                            em = float(e_use_2d[ic, m])
                            if np.isfinite(em) and em > 0.0:
                                sig_meas_sq += cm * cm * em * em
                    sig_meas = math.sqrt(sig_meas_sq)
                    out_sigma_flux[ic, i_v] = sig_meas
                    out_std[ic, i_v] = math.sqrt(sig_meas * sig_meas + sig_line * sig_line)
            else:
                out_ccf[ic, i_v] = np.nan
                if compute_line_std:
                    out_sigma_flux[ic, i_v] = np.nan
                    out_std[ic, i_v] = np.nan
    return out_ccf, out_std, out_sigma_flux


@njit(cache=True)
def _ccf_rv_numba_err_floor(vs: float) -> float:
    """Match :func:`_ccf_rv_err_numeric_floor` (m/s); sub-grid σ_RV allowed."""
    ax = abs(vs)
    if ax < 1.0e-30:
        ax = 1.0
    return max(1.0e-9 * ax, 1.0e-12)


@njit(cache=True)
def _nanmedian_numba_1d(x: np.ndarray) -> float:
    vals = np.empty(x.size, dtype=np.float64)
    n = 0
    for i in range(x.size):
        xi = x[i]
        if np.isfinite(xi):
            vals[n] = xi
            n += 1
    if n == 0:
        return np.nan
    vv = np.sort(vals[:n])
    mid = n // 2
    if (n % 2) == 1:
        return float(vv[mid])
    return float(0.5 * (vv[mid - 1] + vv[mid]))


@njit(cache=True)
def _smooth_uniform5_numba(y: np.ndarray, n: int) -> np.ndarray:
    """5-point moving average; non-finite samples excluded from each local mean."""
    ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        c = 0.0
        for k in range(i - 2, i + 3):
            if 0 <= k < n:
                yk = float(y[k])
                if np.isfinite(yk):
                    s += yk
                    c += 1.0
        if c > 0.0:
            ys[i] = s / c
        else:
            ys[i] = np.nan
    return ys


@njit(cache=True)
def _line_mask_peak_index_integral_and_prominence(
    ys: np.ndarray,
    n: int,
    prom_half_width: int,
    integral_half_width: int,
) -> int:
    """
    Choose among strict local maxima of the smoothed CCF using **integrated excess**
    (sum of positive deviation from the global median in a wide window) times
    ``sqrt(prominence + 1)``.

    Pure prominence favors **narrow** spikes (large drop to a nearby valley). The physical
    stellar-RV peak from thousands of mask lines is **broad**; its prominence can be modest
    while the integrated contrast in ±~20–40 km/s is large. Combining integral × √prominence
    prefers that coherent peak over spurious narrow maxima (e.g. grid artifacts near
    unrelated velocities).
    """
    W = prom_half_width
    if W < 3:
        W = 3
    Wi = integral_half_width
    if Wi < 10:
        Wi = 10

    y_med = _nanmedian_numba_1d(ys)
    if not np.isfinite(y_med):
        y_med = 0.0

    best_i = -1
    best_score = -np.inf
    best_y = -np.inf
    for i in range(1, n - 1):
        yi = ys[i]
        ym = ys[i - 1]
        yp = ys[i + 1]
        if not (np.isfinite(yi) and np.isfinite(ym) and np.isfinite(yp)):
            continue
        if not (yi > ym and yi > yp):
            continue
        lo = i - W
        if lo < 0:
            lo = 0
        hi = i + W + 1
        if hi > n:
            hi = n
        left_min = np.inf
        for j in range(lo, i):
            yj = ys[j]
            if np.isfinite(yj) and yj < left_min:
                left_min = yj
        right_min = np.inf
        for j in range(i + 1, hi):
            yj = ys[j]
            if np.isfinite(yj) and yj < right_min:
                right_min = yj
        if not np.isfinite(left_min) or not np.isfinite(right_min):
            continue
        base = left_min
        if right_min > base:
            base = right_min
        prom = yi - base
        if prom <= 0.0:
            continue

        lo2 = i - Wi
        if lo2 < 0:
            lo2 = 0
        hi2 = i + Wi + 1
        if hi2 > n:
            hi2 = n
        integral = 0.0
        for j in range(lo2, hi2):
            yj = ys[j]
            if np.isfinite(yj) and yj > y_med:
                integral += yj - y_med

        score = integral * np.sqrt(prom + 1.0)
        if score > best_score or (score == best_score and yi > best_y):
            best_score = score
            best_i = i
            best_y = yi

    if best_i >= 0:
        return best_i
    # Fallback: global maximum of smoothed profile
    ig = -1
    yg = -np.inf
    for i in range(n):
        yi = ys[i]
        if np.isfinite(yi) and yi > yg:
            yg = yi
            ig = i
    return ig




@njit(cache=True)
def _rv_from_ccf_line_mask_peak_index_numba(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    i_peak_prior: int,
) -> int:
    """
    Peak index used by :func:`_rv_from_ccf_line_mask_numba_single` for LSTSQ / fallback paths.

    Returns ``-1`` if the grid is invalid or no usable peak is found before edge handling.
    """
    n = v_grid_m_s.size
    if ccf_raw.size != n:
        return -1
    baseline_raw = _nanmedian_numba_1d(ccf_raw)
    if not np.isfinite(baseline_raw):
        return -1
    sign = -1.0
    y_work = np.empty(n, dtype=np.float64)
    for i in range(n):
        yi = ccf_raw[i]
        if np.isfinite(yi):
            y_work[i] = sign * yi
        else:
            y_work[i] = np.nan
    if i_peak_prior >= 0 and i_peak_prior < n:
        i_peak = i_peak_prior
    else:
        ys = _smooth_uniform5_numba(y_work, n)
        Wprom = n // 10
        if Wprom < 80:
            Wprom = 80
        if Wprom > 2000:
            Wprom = 2000
        Wint = n // 12
        if Wint < 100:
            Wint = 100
        if Wint > 800:
            Wint = 800
        i_peak = _line_mask_peak_index_integral_and_prominence(ys, n, Wprom, Wint)
    if i_peak < 0 or i_peak >= n:
        return -1
    return int(i_peak)


@njit(cache=True)
def _rv_from_ccf_line_mask_numba_single(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    n_good: int,
    n_parabolic_samples: int,
    i_peak_prior: int,
) -> tuple[float, float, int]:
    """
    RV from mask-line (KPF / SPIROU-style) CCF.

    Convention is **fixed by algorithm**, not inferred from each CCF shape: the mask weights
    target **absorption** lines, so the weight-normalized statistic (``Σ w f·SN / Σ w``) is
    minimized near the
    correct Doppler shift. We therefore always work in ``-CCF`` (search maxima / parabolic
    vertex on ``-CCF``), and ``flipped`` is always 1 so returned/preview CCF rows are negated
    consistently across echelle orders.

    ``ccf_raw`` is the weight-normalized mask CCF row from :func:`_calc_mask_ccf_chunks_numba`.

    ``i_peak_prior`` : if ``0 <= i_peak_prior < n``, the LSTSQ parabolic window is centered on
    that velocity index (e.g. nearest grid point to the Gaussian mean from :func:`fit_ccf_rv`).
    If ``i_peak_prior < 0``, the peak index comes from a smoothed-CCF prominence search (per-chunk
    path).

    Sub-pixel RV uses a **quadratic least-squares** vertex on ``n_parabolic_samples``
    consecutive samples (odd, :math:`\\ge 3`, caller-validated; fewer near grid edges); otherwise
    a three-point analytic parabola (historic behaviour).

    **Returned** ``rv_err`` **(heuristic):** for an accepted LSTSQ fit, uses
    :math:`\\sigma_v^2 \\approx -\\mathrm{amp}/y''` with ``y'' = 2a`` from the LSTSQ quadratic
    on signed ``-CCF``, but ``amp`` is still the three-point peak contrast vs the CCF median
    baseline, then scales by :math:`1/\\sqrt{n_{\\rm good}}` (spectral pixels in the chunk).
    That can be **much larger** than the sub-m/s bias of the vertex on synthetic tests; the
    pipeline replaces ``RV_ERR`` on accepted parabolic windows with
    :func:`line_mask_parabolic_vertex_rv_err_levmar_m_s` when LM ``param_cov`` is available, else
    :func:`line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s`.
    """
    n = v_grid_m_s.size
    if ccf_raw.size != n:
        return np.nan, np.nan, 0

    baseline_raw = _nanmedian_numba_1d(ccf_raw)
    if not np.isfinite(baseline_raw):
        return np.nan, np.nan, 0

    sign = -1.0
    flipped = 1

    y_work = np.empty(n, dtype=np.float64)
    for i in range(n):
        yi = ccf_raw[i]
        if np.isfinite(yi):
            y_work[i] = sign * yi
        else:
            y_work[i] = np.nan

    if i_peak_prior >= 0 and i_peak_prior < n:
        i_peak = i_peak_prior
    else:
        # Low-pass filter, then choose among local maxima by integrated contrast × √prominence.
        ys = _smooth_uniform5_numba(y_work, n)
        Wprom = n // 10
        if Wprom < 80:
            Wprom = 80
        if Wprom > 2000:
            Wprom = 2000
        # Wide window for integrated line power (~20–40 km/s at 50 m/s sampling typical).
        Wint = n // 12
        if Wint < 100:
            Wint = 100
        if Wint > 800:
            Wint = 800
        i_peak = _line_mask_peak_index_integral_and_prominence(ys, n, Wprom, Wint)
    if i_peak < 0 or i_peak >= n:
        return np.nan, np.nan, flipped

    v_peak = float(v_grid_m_s[i_peak])
    v_step_edge = float(abs(v_grid_m_s[1] - v_grid_m_s[0])) if n > 1 else 1.0
    if i_peak <= 0 or i_peak >= (n - 1):
        fl = _ccf_rv_numba_err_floor(v_step_edge)
        rv_err_edge = max(float(v_step_edge * 2.0), fl)
        return v_peak, rv_err_edge, flipped

    v0 = float(v_grid_m_s[i_peak - 1])
    v1 = float(v_grid_m_s[i_peak])
    v2 = float(v_grid_m_s[i_peak + 1])
    h_m = v1 - v0
    h_p = v2 - v1
    v_step_eff = 0.5 * (h_m + h_p)
    if h_m <= 0.0 or h_p <= 0.0:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped

    y_m1 = float(sign * ccf_raw[i_peak - 1])
    y_0 = float(sign * ccf_raw[i_peak])
    y_p1 = float(sign * ccf_raw[i_peak + 1])
    baseline = float(sign * baseline_raw)
    amp = y_0 - baseline
    if not np.isfinite(amp) or amp <= 0.0:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped

    fl = _ccf_rv_numba_err_floor(v_step_eff)
    rv_ls, a_ls, ok_ls = _line_mask_parabolic_vertex_lstsq_npt_numba(
        v_grid_m_s, y_work, i_peak, int(n_parabolic_samples)
    )
    if ok_ls:
        rv = float(rv_ls)
        ypp = 2.0 * float(a_ls)
        if not np.isfinite(ypp) or ypp >= 0.0:
            sigma_v = float(abs(v_step_eff) * 2.0)
        else:
            sigma_v2 = max(float(-amp / ypp), fl * fl)
            sigma_v = float(np.sqrt(sigma_v2))
        n_scale = max(float(n_good), 1.0)
        rv_err = float(max(sigma_v / np.sqrt(n_scale), fl))
        return rv, rv_err, flipped

    # Parabola y = a*v^2 + b*v + c through (v0,y_m1), (v1,y_0), (v2,y_p1); vertex at -b/(2a).
    A = v1 * v1 - v0 * v0
    B = v1 - v0
    C = y_0 - y_m1
    D = v2 * v2 - v1 * v1
    E = v2 - v1
    F = y_p1 - y_0
    det = A * E - B * D
    if not np.isfinite(det) or abs(det) < 1e-30:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped
    a = (C * E - B * F) / det
    b = (A * F - C * D) / det
    if not np.isfinite(a) or abs(a) < 1e-30:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped
    rv = -b / (2.0 * a)
    if rv < min(v0, v2) or rv > max(v0, v2):
        rv = v_peak

    ypp = 2.0 * a
    if not np.isfinite(ypp) or ypp >= 0.0:
        sigma_v = float(abs(v_step_eff) * 2.0)
    else:
        sigma_v2 = max(float(-amp / ypp), fl * fl)
        sigma_v = float(np.sqrt(sigma_v2))
    n_scale = max(float(n_good), 1.0)
    rv_err = float(max(sigma_v / np.sqrt(n_scale), fl))
    return float(rv), float(rv_err), flipped


@njit(parallel=True, cache=True)
def _rv_from_ccf_line_mask_rows_numba(
    v_grid_m_s: np.ndarray,
    ccf_rows: np.ndarray,
    n_good_arr: np.ndarray,
    out_rv: np.ndarray,
    out_rv_err: np.ndarray,
    out_flip: np.ndarray,
    n_parabolic_samples: int,
    i_peak_prior: int,
) -> None:
    n_chunk = ccf_rows.shape[0]
    for i in prange(n_chunk):
        rv, rv_err, flipped = _rv_from_ccf_line_mask_numba_single(
            v_grid_m_s,
            ccf_rows[i],
            int(n_good_arr[i]),
            int(n_parabolic_samples),
            int(i_peak_prior),
        )
        out_rv[i] = rv
        out_rv_err[i] = rv_err
        out_flip[i] = flipped


def rv_per_chunks_line_mask(
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
    max_cores: int | None = None,
    return_ccf: bool = False,
    min_good_pixels: int = 10,
    ccf_parabolic_vertex_n_samples: int = 11,
    mask_ccf_pixel_interp_order: int = 3,
    window_line_by_line: bool = False,
    line_mask_line_chunk_v_center_m_s: float | None = None,
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
]:
    """
    Compute per-chunk RVs and optional CCF stacks for one echelle order.

    By default splits the order into equal-wavelength bins (count from
    :func:`ccf_order_n_chunks_for_span` given the order span and ``window_size_um``),
    then computes chunk CCF rows in one Numba batch via :func:`_calc_mask_ccf_chunks_numba` (parallel over chunks;
    delta-line interpolation at each Doppler-shifted rest wavelength), followed by a Numba chunk-parallel
    parabolic RV extraction. Chunks on an echelle row are iterated **longest λ first**
    (descending chunk-center wavelength, red → blue along the row), so downstream ``RVCCF`` rows
    and mask-overlay segments follow the same ordering as ascending echelle order in the exposure.
    RV extraction assumes a
    **fixed absorption-line convention** for all chunks and orders (fit on ``-CCF``, not
    data-driven emission/absorption guessing). Peak selection uses a 5-point smoothed CCF
    and a **combined score** (wide-window integrated excess above the global median times
    ``√prominence``) over strict local maxima—favoring the broad, physically coherent peak
    at the true RV over narrow spurious maxima. A fast Numba estimate is computed first;
    then each chunk is optionally updated with :func:`fit_ccf_rv` plus
    :func:`_refine_ccf_rv_after_gaussian_fit_ccf_line_mask` (LSTSQ parabolic vertex with
    ``ccf_parabolic_vertex_n_samples``, then Gaussian-smoothed coarse-grid fallback when
    needed), matching the single-chunk order-mean path in :func:`run_ccf_line_mask_for_fiber`.
    If that Gaussian+refine path does not return a finite RV, the Numba value is kept.
    Returns flattened lists of chunk centers and RVs.

    ``ccf_2d`` is the weight-normalized mask CCF used for RV extraction and stacking;
    ``ccf_preview_2d`` holds the same rows before the per-chunk sign flip applied for
    ``ccf_2d`` (both match the physical weighted CCF up to that convention).

    Parameters
    ----------
    wave_um, spec, spec_err : np.ndarray
        Wavelength (µm), flux, and error; same length.
    dq_bad_mask : np.ndarray
        True where pixel should be excluded (e.g. DO_NOT_USE).
    mask_lines : MaskLines
        Full-order mask; restricted per chunk via :func:`~hispecdrp.utils.line_mask_utils.restrict_mask_to_chunk`.
    window_size_um : float
        Target maximum segment width in µm; one full-order chunk if span ≤ this value.
        Ignored for chunk **count** when ``window_line_by_line`` is true (still used for the
        symmetric spectral band about each line’s predicted λ).
    v_min_m_s, v_max_m_s, v_step_m_s : float, optional
        Velocity search grid (m s⁻¹). Ignored when ``v_grid_m_s`` is provided.
    v_grid_m_s : np.ndarray or None, optional
        Explicit velocity samples (m s⁻¹), e.g. from :func:`build_hybrid_merged_velocity_grid`.
        When set, ``v_min_m_s`` / ``v_max_m_s`` / ``v_step_m_s`` are not used to build the grid,
        but together with ``v_grid_m_s`` they still define the velocity span used to restrict
        mask lines per chunk via :func:`~hispecdrp.utils.line_mask_utils.restrict_mask_to_chunk` (same coarse bounds passed in).
    zb : float, optional
        Barycentric factor passed to the CCF shift.
    max_cores : int or None, optional
        Numba thread count cap for chunk-parallel CCF evaluation.
    return_ccf : bool, optional
        If True, also return mean CCF rows per chunk (stacked for preview).
    min_good_pixels : int, optional
        Minimum unmasked finite pixels required per chunk.
    ccf_parabolic_vertex_n_samples : int, optional
        **Odd** integer :math:`\\ge 3`: number of coarse velocity samples in the LSTSQ
        parabolic vertex window. Per-chunk RVs center the window on the smoothed-CCF peak
        index. Raises :class:`ValueError` if the value is even or below ``3``. Fewer samples
        may be used automatically near the ends of the velocity grid.
    mask_ccf_pixel_interp_order : int, optional
        Polynomial **order** for sampling flux and the per-pixel gate array onto mask-line
        Doppler-shifted wavelengths in :func:`_calc_mask_ccf_chunks_numba` (Lagrange stencil
        on sorted pixel centers). ``1`` is linear (legacy); clamped to
        ``[1, MAX_MASK_CCF_PIXEL_INTERP_ORDER]``.
    window_line_by_line : bool, optional
        When true, use :func:`build_line_mask_chunk_wavelength_intervals_line_by_line` instead of
        equal-µm bins: chunks partition the order by observed mask-line wavelengths at
        ``line_mask_line_chunk_v_center_m_s`` (non-overlapping intervals, midpoint boundaries).
    line_mask_line_chunk_v_center_m_s : float or None, optional
        Required (finite) when ``window_line_by_line`` is true: pipeline velocity prior in m/s
        (typically ``ccf_input_velocity``), defining observed-frame mask-line positions.

    When ``window_line_by_line`` is true, spectral pixels per chunk are the intersection of the
    line-window Voronoi interval with a symmetric band
    :func:`mask_line_window_spectrum_band_um` about the chunk center (``window_size_um`` plus
    :data:`LINE_MASK_LINE_WINDOW_SPEC_REL_PAD`). If that intersection has fewer than
    ``min_good_pixels`` samples, the chunk falls back to the full Voronoi pixel set.

    Returns
    -------
    rvs : np.ndarray
        Per-chunk RV (m s⁻¹). After the Numba estimate, each chunk is replaced by the
        Gaussian + :func:`_refine_ccf_rv_after_gaussian_fit_ccf_line_mask` RV when that
        pipeline returns a finite value (``ccf_parabolic_vertex_n_samples`` controls the
        LSTSQ window there); otherwise the Numba RV is kept.
    rv_errs : np.ndarray
        Per-chunk RV uncertainty (m s\\ :sup:`−1`): when Gaussian+refine succeeds, its
        reported error; else :func:`line_mask_parabolic_vertex_rv_err_levmar_m_s` when LM
        covariance is available (else homoscedastic LSTSQ σ) on the Numba path; otherwise
        the Numba heuristic from :func:`_rv_from_ccf_line_mask_numba_single`.
    centers : np.ndarray
        Chunk center wavelength (µm).
    v_grid : np.ndarray
        Velocity grid used for the CCF.
    ccf_2d : np.ndarray or None
        If ``return_ccf``, shape ``(n_chunks, n_v)`` of weight-normalized CCF rows,
        **sign-flipped** per chunk to match the parabolic RV extraction convention; used for
        RVCCF supplements and chunk-level diagnostics. Else ``None``.
    wl_lo, wl_hi : np.ndarray
        Per-chunk wavelength bounds (µm), same length as ``rvs``.
    n_good : np.ndarray
        Per-chunk count of good pixels used in the mask CCF.
    ccf_line_std_2d : np.ndarray or None
        If ``return_ccf``, shape ``(n_chunks, n_v)``: per-velocity uncertainty combining
        (in quadrature) the **flux-error–propagated** standard error of the mask-weighted
        mean and the weighted line-to-line dispersion of :math:`x_j = F_j\\,\\mathrm{SN}_j`
        about that mean (see :func:`_calc_mask_ccf_chunks_numba`). Not flipped with the CCF
        row; use as a symmetric band around ``ccf_preview_2d``. If ``return_ccf`` is False,
        ``None``.
    ccf_preview_2d : np.ndarray or None
        If ``return_ccf``, same shape as ``ccf_2d``: **weight-normalized** mask CCF
        (matching :func:`_calc_mask_ccf_chunks_numba`) **before** the per-chunk sign flip
        applied for ``ccf_2d``. Else ``None``.
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    def _empty(v_grid_local: np.ndarray):
        z = np.array([], dtype=np.float64)
        zi = np.array([], dtype=np.int32)
        if not return_ccf:
            return (z, z, z, v_grid_local, None, z, z, zi, None, None)
        ccf_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        std_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        return (z, z, z, v_grid_local, ccf_out, z, z, zi, std_out, ccf_out)

    vr_lo = float(v_min_m_s)
    vr_hi = float(v_max_m_s)

    pio = int(mask_ccf_pixel_interp_order)
    if pio < 1:
        pio = 1
    elif pio > MAX_MASK_CCF_PIXEL_INTERP_ORDER:
        pio = MAX_MASK_CCF_PIXEL_INTERP_ORDER

    good_pix = ~d & np.isfinite(w) & np.isfinite(s) & np.isfinite(e)
    if v_grid_m_s is not None:
        v_grid = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    else:
        v_grid = np.arange(
            v_min_m_s,
            v_max_m_s + 0.5 * v_step_m_s,
            v_step_m_s,
            dtype=np.float64,
        )
    if np.sum(good_pix) < min_good_pixels:
        return _empty(v_grid)

    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return _empty(v_grid)

    span_um = float(w_max - w_min)
    use_line_chunks = bool(window_line_by_line)
    # Wavelength tolerance for treating a chunk's upper bound as the echelle-row edge (inclusive).
    _w_edge_eps = max(1e-15, abs(float(w_max)) * 1e-12)
    if use_line_chunks:
        if line_mask_line_chunk_v_center_m_s is None or not math.isfinite(
            float(line_mask_line_chunk_v_center_m_s)
        ):
            raise ValueError(
                "rv_per_chunks_line_mask: window_line_by_line requires finite "
                "line_mask_line_chunk_v_center_m_s (e.g. ccf_input_velocity in m/s)."
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
    all_n_good_out: list[int] = []
    all_ccf_rows: list[np.ndarray] = []
    all_ccf_preview_rows: list[np.ndarray] = []
    all_ccf_std_rows: list[np.ndarray] = []
    chunk_payloads: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
            int,
        ]
    ] = []

    for lam_lo, lam_hi, wl_center, right_closed in chunk_iter:
        if right_closed:
            in_voronoi = (w >= lam_lo) & (w <= lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        else:
            in_voronoi = (w >= lam_lo) & (w < lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        band_lo = float(lam_lo)
        band_hi = float(lam_hi)
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

        # Restrict mask lines so we don't waste time on lines that cannot land in this
        # chunk for any trial velocity in the search range.
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

        left_edges, right_edges = pixel_edges_from_wavelength(w_c)
        pixel_width = right_edges - left_edges
        pixel_width = np.where(pixel_width > 0, pixel_width, np.nan)
        finite = np.isfinite(s_c) & np.isfinite(e_c) & (e_c > 0) & np.isfinite(pixel_width)
        if int(np.sum(finite)) < min_good_pixels:
            continue
        s_use = np.where(finite, s_c, 0.0).astype(np.float64)
        sn = np.where(finite, 1.0, 0.0).astype(np.float64)
        e_use = np.where(finite, e_c, 0.0).astype(np.float64)
        w_lo_out = float(np.nanmin(w_c))
        w_hi_out = float(np.nanmax(w_c))
        chunk_payloads.append(
            (
                left_edges.astype(np.float64),
                right_edges.astype(np.float64),
                s_use,
                sn,
                e_use,
                mask_c.line_lambda_um.astype(np.float64),
                mask_c.line_left_edge.to_value(u.um).astype(np.float64),
                mask_c.line_right_edge.to_value(u.um).astype(np.float64),
                mask_c.weight.astype(np.float64),
                w_lo_out,
                w_hi_out,
                float(wl_center),
                int(np.sum(in_chunk)),
            )
        )

    if not chunk_payloads:
        return _empty(v_grid)

    n_parabolic_norm = _require_ccf_parabolic_vertex_n_samples(
        int(ccf_parabolic_vertex_n_samples)
    )

    n_chunk_valid = len(chunk_payloads)
    n_pix_max = max(len(t[0]) for t in chunk_payloads)
    n_line_max = max(len(t[5]) for t in chunk_payloads)
    left_2d = np.full((n_chunk_valid, n_pix_max), np.nan, dtype=np.float64)
    right_2d = np.full((n_chunk_valid, n_pix_max), np.nan, dtype=np.float64)
    s_use_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    sn_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    e_use_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    n_pix_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    line_lambda_rest_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_left_rest_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_right_rest_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_w_2d = np.zeros((n_chunk_valid, n_line_max), dtype=np.float64)
    n_line_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    centers = np.zeros(n_chunk_valid, dtype=np.float64)
    n_good_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    wl_lo_arr = np.zeros(n_chunk_valid, dtype=np.float64)
    wl_hi_arr = np.zeros(n_chunk_valid, dtype=np.float64)

    for i, payload in enumerate(chunk_payloads):
        left_i, right_i, s_i, sn_i, e_i, ll_i, lo_i, hi_i, lw_i, lam_lo_i, lam_hi_i, wl_i, ng_i = payload
        npix = len(left_i)
        nln = len(ll_i)
        left_2d[i, :npix] = left_i
        right_2d[i, :npix] = right_i
        s_use_2d[i, :npix] = s_i
        sn_2d[i, :npix] = sn_i
        e_use_2d[i, :npix] = e_i
        n_pix_arr[i] = npix
        line_lambda_rest_2d[i, :nln] = ll_i
        line_left_rest_2d[i, :nln] = lo_i
        line_right_rest_2d[i, :nln] = hi_i
        line_w_2d[i, :nln] = lw_i
        n_line_arr[i] = nln
        centers[i] = wl_i
        n_good_arr[i] = ng_i
        wl_lo_arr[i] = lam_lo_i
        wl_hi_arr[i] = lam_hi_i

    with numba_thread_scope(max_cores):
        ccf_rows, ccf_std_rows, _ = _calc_mask_ccf_chunks_numba(
            left_2d,
            right_2d,
            s_use_2d,
            e_use_2d,
            sn_2d,
            n_pix_arr,
            v_grid,
            float(zb),
            line_lambda_rest_2d,
            line_left_rest_2d,
            line_right_rest_2d,
            line_w_2d,
            n_line_arr,
            bool(return_ccf),
            int(pio),
            int(TOPHAT_MASK_CCF_SEGMENT_SAMPLES),
        )
        # σ_CCF: quadrature of flux-propagated SE of the mask mean and weighted line
        # dispersion of x_j = F_j·SN_j about that mean (Numba).
        if bool(return_ccf) and ccf_std_rows.size > 0:
            sf = np.asarray(ccf_std_rows, dtype=np.float64)
            ccf_std_rows = np.where(np.isfinite(sf), np.maximum(0.0, sf), np.nan)
        ccf_preview_unflipped: np.ndarray | None = (
            np.copy(ccf_rows) if bool(return_ccf) else None
        )
        rv_rows = np.empty(n_chunk_valid, dtype=np.float64)
        rv_err_rows = np.empty(n_chunk_valid, dtype=np.float64)
        flip_rows = np.zeros(n_chunk_valid, dtype=np.int64)
        _rv_from_ccf_line_mask_rows_numba(
            v_grid,
            ccf_rows,
            n_good_arr.astype(np.int64),
            rv_rows,
            rv_err_rows,
            flip_rows,
            int(n_parabolic_norm),
            -1,
        )

    vg_cc = np.ascontiguousarray(np.asarray(v_grid, dtype=np.float64).ravel(), dtype=np.float64)
    v_step_floor = float(effective_ccf_v_step_for_fit(vg_cc))
    for i in range(n_chunk_valid):
        row = np.ascontiguousarray(ccf_rows[i], dtype=np.float64)
        i_pk = int(_rv_from_ccf_line_mask_peak_index_numba(vg_cc, row, -1))
        row_err = None
        if ccf_std_rows.size > 0 and int(ccf_std_rows.shape[1]) == int(vg_cc.size):
            row_err = np.ascontiguousarray(ccf_std_rows[i], dtype=np.float64)
        sig_lm, _diag_lm = line_mask_parabolic_vertex_rv_err_levmar_m_s(
            vg_cc,
            row,
            i_peak=i_pk,
            n_pt_want=int(n_parabolic_norm),
            v_step_m_s_for_floor=v_step_floor,
            ccf_err=row_err,
        )
        if sig_lm is not None and math.isfinite(float(sig_lm)) and float(sig_lm) > 0.0:
            rv_err_rows[i] = float(sig_lm)
        else:
            sig_h, _diag_h = line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s(
                vg_cc,
                row,
                i_peak=i_pk,
                n_pt_want=int(n_parabolic_norm),
                v_step_m_s_for_floor=v_step_floor,
            )
            if sig_h is not None and math.isfinite(float(sig_h)) and float(sig_h) > 0.0:
                rv_err_rows[i] = float(sig_h)

    # Prefer Gaussian + bisector refine (same stack as order-mean RV when
    # ``n_chunks_order == 1`` and CCF preview slots). Keep Numba RV/σ when the fitter
    # path does not succeed.
    line_window_fit_ccf = bool(window_line_by_line)
    v_seed = (
        float(line_mask_line_chunk_v_center_m_s)
        if line_mask_line_chunk_v_center_m_s is not None
        and math.isfinite(float(line_mask_line_chunk_v_center_m_s))
        else None
    )
    for i in range(n_chunk_valid):
        rv_nb = float(rv_rows[i])
        if not np.isfinite(rv_nb):
            continue
        y_stored = np.ascontiguousarray(
            (
                -ccf_rows[i]
                if int(flip_rows[i]) == 1
                else np.asarray(ccf_rows[i], dtype=np.float64).copy()
            ),
            dtype=np.float64,
        )
        wl_i = float(centers[i])
        vsp_i = (
            dv_per_pixel_m_s(wl_i)
            if np.isfinite(wl_i) and wl_i > 0.0
            else float("nan")
        )
        e_row = None
        if ccf_std_rows.size > 0 and int(ccf_std_rows.shape[1]) == int(vg_cc.size):
            e_row = np.ascontiguousarray(ccf_std_rows[i], dtype=np.float64)
        try:
            rv_g, err_g, par_g, diag_g = fit_ccf_rv(
                vg_cc,
                y_stored,
                v_step_floor,
                n_good=int(n_good_arr[i]),
                return_params=True,
                vel_span_m_s_per_pixel=vsp_i if np.isfinite(vsp_i) else None,
                ccf_err=e_row,
                config=CCF_PREVIEW_SLOT_FIT_CONFIG,
                ccf_line_mask_stored_negated=LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
                line_mask_line_window_by_line=line_window_fit_ccf,
                gaussian_seed_mean_m_s=v_seed,
            )
        except Exception:
            continue
        if not (
            bool(diag_g.get("fit_ok"))
            and np.isfinite(rv_g)
            and np.isfinite(err_g)
            and float(err_g) > 0.0
        ):
            continue
        rv_bi, err_bi, _bmeta = _refine_ccf_rv_bisector_after_gaussian_fit(
            vg_cc,
            y_stored,
            v_step_floor,
            ccf_err=e_row,
        )
        if rv_bi is not None and math.isfinite(float(rv_bi)):
            rv_rows[i] = float(rv_bi)
            if err_bi is not None and math.isfinite(float(err_bi)) and float(err_bi) > 0.0:
                rv_err_rows[i] = float(err_bi)

    for i in range(n_chunk_valid):
        rv = float(rv_rows[i])
        rv_err = float(rv_err_rows[i])
        if not np.isfinite(rv):
            continue
        all_rvs.append(float(rv))
        all_rv_errs.append(float(rv_err) if np.isfinite(rv_err) and rv_err > 0 else np.nan)
        all_centers.append(float(centers[i]))
        all_wl_lo.append(float(wl_lo_arr[i]))
        all_wl_hi.append(float(wl_hi_arr[i]))
        all_n_good_out.append(int(n_good_arr[i]))
        if return_ccf:
            all_ccf_rows.append(-ccf_rows[i] if int(flip_rows[i]) == 1 else ccf_rows[i].copy())
            pu = ccf_preview_unflipped
            all_ccf_preview_rows.append(
                -pu[i] if int(flip_rows[i]) == 1 else pu[i].copy()
            )
            all_ccf_std_rows.append(ccf_std_rows[i].copy())
    if not all_rvs:
        return _empty(v_grid)

    rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    centers_arr = np.asarray(all_centers, dtype=np.float64)
    wl_lo_out = np.asarray(all_wl_lo, dtype=np.float64)
    wl_hi_out = np.asarray(all_wl_hi, dtype=np.float64)
    n_good_out = np.asarray(all_n_good_out, dtype=np.int32)
    if return_ccf:
        ccf_2d = np.vstack(all_ccf_rows).astype(np.float64) if all_ccf_rows else np.empty((0, len(v_grid)))
        ccf_preview_2d = (
            np.vstack(all_ccf_preview_rows).astype(np.float64)
            if all_ccf_preview_rows
            else np.empty((0, len(v_grid)))
        )
        ccf_std_2d = (
            np.vstack(all_ccf_std_rows).astype(np.float64) if all_ccf_std_rows else np.empty((0, len(v_grid)))
        )
    else:
        ccf_2d = None
        ccf_preview_2d = None
        ccf_std_2d = None
    return (
        rvs_arr,
        rv_errs_arr,
        centers_arr,
        v_grid,
        ccf_2d,
        wl_lo_out,
        wl_hi_out,
        n_good_out,
        ccf_std_2d,
        ccf_preview_2d,
    )


# =============================================================================
# Line-mask LSTSQ preview metadata, post-Gaussian refine, and per-fiber driver
# =============================================================================




def run_ccf_line_mask_for_fiber(
    *,
    input_model,
    fiber: str,
    do_orders: list[int],
    order_vec: list[int],
    window_size_um: float,
    mask_path: str | None,
    mask_wavelength=None,
    mask_weight=None,
    v_min_m_s: float = V_MIN_DEFAULT_M_S,
    v_max_m_s: float = V_MAX_DEFAULT_M_S,
    v_step_m_s: float = V_STEP_DEFAULT_M_S,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
    zb: float = 0.0,
    max_cores: int | None = None,
    want_preview: bool = False,
    debug_ccf: bool = False,
    mask_auto_hybrid_scout: bool = True,
    ccf_parabolic_vertex_n_samples: int = 11,
    min_good_pixels: int = 10,
    mask_ccf_pixel_interp_order: int = 3,
    window_line_by_line: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict | None,
    Table,
]:
    """
    Run line-mask CCF RV for one fiber (``StellarRVPrimitive(algorithm="ccf_line_mask")``).

    Reads ``SPEC_FIB*``, ``SPEC_ERR_FIB*``, ``SPEC_DQ_FIB*``, ``WAVE_FIB*`` from
    ``input_model``, resolves mask lines via :func:`~hispecdrp.utils.line_mask_utils.stellar_rv_ccf_mask_lines`, and processes
    each requested echelle order with :func:`rv_per_chunks_line_mask`. When an echelle
    row has **only one** line-mask wavelength chunk, all chunk rows in ``RVCCF`` (one row)
    use the Gaussian + parabolic path on the **order-mean** mask CCF (nanmean over chunk CCF
    rows on the velocity grid), same as before. When an order has **multiple** chunks
    (e.g. ``window_line_by_line`` Voronoi bins), **per-chunk** Numba RVs and
    uncertainties are kept on each ``RVCCF`` row. Fiber-level RV aggregation (median/MAD
    outlier rejection + inverse-variance mean) is performed by the stellar RV primitive
    (see :mod:`hispecdrp.primitives.stellar_rv.rv_combine_utils`) so that ``ccf_utils`` remains
    a dependency of stellar RV code and does not import from it.

    Parameters
    ----------
    input_model
        L2-like datamodel with fiber spectra and wavelengths.
    fiber : str
        Fiber label (e.g. ``"A"``) used in attribute names.
    do_orders : list of int
        Echelle orders to include.
    order_vec : list of int
        Full order indexing aligned with array axis 0.
    window_size_um : float
        Wavelength chunk width (µm).
    mask_path : str or None
        Path to mask file when not using in-memory ``mask_wavelength`` / ``mask_weight``;
        ``None`` selects the packaged default (see README). Loaded via :meth:`~hispecdrp.utils.line_mask_utils.MaskLines.load`.
    mask_wavelength, mask_weight : array-like or None, optional
        When **both** are set, build the mask in memory (rest wavelengths in µm, weights same
        length) and ignore ``mask_path`` for loading. See :func:`~hispecdrp.utils.line_mask_utils.stellar_rv_ccf_mask_lines`.
    v_min_m_s, v_max_m_s, v_step_m_s : float, optional
        **Broad** uniform velocity search (m s⁻¹). When ``ccf_hybrid_center_m_s`` is set,
        these define the coarse leg of a **merged** grid (see
        :func:`build_hybrid_merged_velocity_grid`); when unset, they define the sole CCF grid.
    ccf_hybrid_center_m_s, ccf_hybrid_half_width_m_s, ccf_hybrid_step_m_s : float or None
        When ``ccf_hybrid_center_m_s`` is finite, CCF is evaluated once on the merged velocity
        list (broad ``v_min_m_s`` … ``v_max_m_s`` at ``v_step_m_s`` plus dense samples around
        the center). When ``None`` / not finite, only the broad grid is used.
    zb : float, optional
        Barycentric factor for mask shift.
    max_cores : int or None, optional
        Numba thread count cap used for chunk-parallel mask CCF evaluation.
    want_preview : bool, optional
        If True, build order-averaged CCF data for Plotly preview.
    debug_ccf : bool, optional
        Extra logging per order.
    mask_auto_hybrid_scout : bool, optional
        When ``ccf_hybrid_center_m_s`` is unset and **exactly one** echelle order is processed,
        run a **full-order** coarse mask CCF (one chunk spanning the echelle row), take the
        sub-pixel peak on the order-mean CCF (same basis as the pipeline fit), build
        :func:`build_hybrid_merged_velocity_grid` at that center, and evaluate all wavelength
        sub-bins on the merged grid. Multi-order runs skip this so preview stacking keeps one
        velocity axis.
    ccf_parabolic_vertex_n_samples : int, optional
        **Odd** integer :math:`\\ge 3` (raises if even). For the **order-mean** path, the LSTSQ
        window is centered on the grid point nearest the Gaussian mean from :func:`fit_ccf_rv`;
        for per-chunk Numba RVs (when the order-mean Gaussian path does not apply), the window
        centers on the smoothed-CCF peak. Passed to :func:`rv_per_chunks_line_mask` and
        :func:`refine_ccf_rv_line_mask_lstsq_from_stored`.
    min_good_pixels : int, optional
        Minimum unmasked finite spectral pixels required per chunk (see ``rv_per_chunks_line_mask``).
    mask_ccf_pixel_interp_order : int, optional
        Lagrange interpolation **order** on the echelle wavelength axis when evaluating mask
        CCFs (``1`` = linear). Forwarded to :func:`rv_per_chunks_line_mask`.
    window_line_by_line : bool, optional
        When true, spectral chunks follow mask lines in the observed frame at
        ``ccf_hybrid_center_m_s`` (requires hybrid mode / finite ``ccf_hybrid_center_m_s``).
        Forwarded to :func:`rv_per_chunks_line_mask`.

    Returns
    -------
    all_wavelengths : np.ndarray
        Chunk center wavelengths (µm).
    all_rvs : np.ndarray
        Per-chunk RVs (m s⁻¹); within one echelle order, values are identical when the
        order-mean CCF Gaussian fit succeeds.
    all_rv_errs : np.ndarray
        Per-chunk uncertainties (same within each order when the Gaussian fit succeeds).
    all_orders : np.ndarray
        Echelle order index per chunk (-1 if length mismatch).
    preview_detail : dict or None
        Payload for stellar RV preview, or None.
    rvccf_chunk_supplement : astropy.table.Table
        Columns ``CHUNK_ID``, ``N_GOOD``, ``V_MIN``, ``V_MAX``, ``V_STEP``, ``N_VEL``, ``CCF``
        merged into ``RVCCF_FIB*`` via :func:`assemble_rvccf_fib_table`.

    Notes
    -----
    Mask and algorithm provenance: module docstring ``References``;
    ``hispecdrp/data/masks/README.md``.
    """

    logger.info(
        "CCF line_mask: starting fiber=%s n_do_orders=%d window_um=%.4f want_preview=%s "
        "debug_ccf=%s",
        fiber,
        len(do_orders),
        window_size_um,
        want_preview,
        debug_ccf,
    )
    if not np.isfinite(window_size_um) or float(window_size_um) <= 0:
        raise ValueError("window_size_um must be finite and > 0 for ccf_line_mask")
    hybrid = bool(
        ccf_hybrid_center_m_s is not None
        and np.isfinite(float(ccf_hybrid_center_m_s))
    )
    if hybrid:
        logger.info(
            "CCF line_mask: HYBRID mode center=%.2f half_width=%.2f refined_step=%.4f (m/s)",
            float(ccf_hybrid_center_m_s),
            float(ccf_hybrid_half_width_m_s),
            float(ccf_hybrid_step_m_s),
        )
    else:
        logger.info(
            "CCF line_mask: velocity grid (m/s) v_min=%.2f v_max=%.2f step=%.4f",
            float(v_min_m_s),
            float(v_max_m_s),
            float(v_step_m_s),
        )

    spec_attr = f"SPEC_FIB{fiber}"
    err_attr = f"SPEC_ERR_FIB{fiber}"
    dq_attr = f"SPEC_DQ_FIB{fiber}"
    wave_attr = f"WAVE_FIB{fiber}"

    spec = getattr(input_model, spec_attr)
    err = getattr(input_model, err_attr) if hasattr(input_model, err_attr) else np.ones_like(spec)
    dq = getattr(input_model, dq_attr) if hasattr(input_model, dq_attr) else np.zeros_like(spec, dtype=np.uint32)
    wave = getattr(input_model, wave_attr)

    if err is None:
        err = np.ones_like(spec, dtype=np.float64)
    if dq is None:
        dq = np.zeros_like(spec, dtype=np.uint32)

    # Load mask before building the velocity grid so line-window mode can tighten broad ``v``.
    mask_lines, mask_src_label = stellar_rv_ccf_mask_lines(
        mask_path=mask_path,
        mask_wavelength=mask_wavelength,
        mask_weight=mask_weight,
    )
    if mask_lines.line_lambda_um.size == 0:
        raise ValueError(f"Mask loaded no lines: {mask_src_label}")
    logger.info(
        "CCF line_mask: loaded mask source=%s n_line_segments=%d",
        mask_src_label,
        int(mask_lines.line_lambda_um.size),
    )

    # TODO: refine mask-CCF DQ filtering policy. Today we only gate on DO_NOT_USE;
    # review which spectral-quality bits are populated in L2 for this path and whether
    # additional flags (e.g. LOW_INST_TRANS, TELLURIC, OH_LINE) should be excluded.
    do_not_use = int(DQ_FLAGS.get("DO_NOT_USE", 1))
    # We treat any non-zero "do_not_use" bits as bad pixels.
    bad = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    order_tasks = [(oi, order) for oi, order in enumerate(order_vec) if order in do_orders]
    logger.info(
        "CCF line_mask: fiber=%s will process n_order_tasks=%d (echelle orders in do_orders)",
        fiber,
        len(order_tasks),
    )
    single_order_line_mask = len(order_tasks) == 1
    fiber_merged_v_grid: np.ndarray | None = None

    if window_line_by_line:
        if not hybrid or ccf_hybrid_center_m_s is None:
            raise ValueError(
                "run_ccf_line_mask_for_fiber: window_line_by_line requires finite "
                "ccf_hybrid_center_m_s (set ccf_input_velocity on the primitive)."
            )
        kw_rv_line_chunking: dict[str, float | bool] = {
            "window_line_by_line": True,
            "line_mask_line_chunk_v_center_m_s": float(ccf_hybrid_center_m_s),
        }
        logger.info(
            "CCF line_mask: line-by-line chunking v_center_m_s=%.2f",
            float(ccf_hybrid_center_m_s),
        )
    else:
        kw_rv_line_chunking = {}

    v_min_use = float(v_min_m_s)
    v_max_use = float(v_max_m_s)
    if window_line_by_line and hybrid:
        v_min_use, v_max_use, dv_chunk_max = line_mask_line_window_coarse_velocity_bounds_m_s(
            wave,
            order_tasks,
            mask_lines,
            v_center_m_s=float(ccf_hybrid_center_m_s),
            zb=float(zb),
            min_good_pixels=int(min_good_pixels),
            v_coarse_lo=float(v_min_m_s),
            v_coarse_hi=float(v_max_m_s),
            hybrid_half_width_m_s=float(ccf_hybrid_half_width_m_s),
            v_step_m_s=float(v_step_m_s),
        )
        logger.info(
            "CCF line_mask: line-window coarse V bounds tightened to [%.2f, %.2f] m/s "
            "(from [%.2f, %.2f]; max c·Δλ/λ over chunks ≈ %.2f m/s)",
            v_min_use,
            v_max_use,
            float(v_min_m_s),
            float(v_max_m_s),
            float(dv_chunk_max),
        )

    if hybrid:
        _h_vc = float(ccf_hybrid_center_m_s)
        _h_hw = float(ccf_hybrid_half_width_m_s)
        _h_vsf = float(ccf_hybrid_step_m_s)
        v_grid = build_hybrid_merged_velocity_grid(
            v_min_use,
            v_max_use,
            float(v_step_m_s),
            _h_vc,
            _h_hw,
            _h_vsf,
        )
    else:
        _h_vc = _h_hw = _h_vsf = float("nan")
        v_grid = np.arange(
            v_min_use,
            v_max_use + 0.5 * v_step_m_s,
            v_step_m_s,
            dtype=np.float64,
        )

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_wavelengths: list[float] = []
    all_orders: list[int] = []

    # Per-chunk RVs and RVCCF table CCF rows use ``ccf_2d`` from :func:`rv_per_chunks_line_mask`.
    # Order-mean Gaussian + parabolic refine applies only for **one** chunk per echelle row;
    # otherwise per-chunk Numba RVs are kept and preview still stacks chunk CCF rows per order.
    want_preview_effective = bool(want_preview)

    order_slots: list[dict] = []
    all_ccf_rows: list[np.ndarray] = []
    all_ccf_err_rows: list[np.ndarray] = []

    rvccf_rows_accum: list[np.ndarray] = []
    rvccf_n_good_accum: list[int] = []

    # Fiber-level auto-hybrid scout (multi-order): estimate CCF peak on the first valid
    # echelle row, then reuse the merged velocity grid for every order on this fiber.
    if bool(mask_auto_hybrid_scout) and not hybrid and fiber_merged_v_grid is None:
        for _oi_sc, _ord_sc in order_tasks:
            _w_sc = np.asarray(wave[_oi_sc], dtype=np.float64)
            _s_sc = np.asarray(spec[_oi_sc], dtype=np.float64)
            _e_sc = np.asarray(
                err[_oi_sc] if getattr(err, "ndim", 0) >= 2 else err, dtype=np.float64
            )
            _d_sc = np.asarray(
                bad[_oi_sc] if getattr(bad, "ndim", 0) >= 2 else bad, dtype=bool
            )
            _wmin, _wmax = float(np.nanmin(_w_sc)), float(np.nanmax(_w_sc))
            if not np.isfinite(_wmin) or not np.isfinite(_wmax) or _wmax <= _wmin:
                continue
            _span = float(_wmax - _wmin)
            _win_sc = max(float(window_size_um), _span * (1.0 + 1e-12))
            _kw_sc = {
                "ccf_parabolic_vertex_n_samples": int(ccf_parabolic_vertex_n_samples),
                "min_good_pixels": int(min_good_pixels),
                "mask_ccf_pixel_interp_order": int(mask_ccf_pixel_interp_order),
                **kw_rv_line_chunking,
            }
            try:
                _scout = rv_per_chunks_line_mask(
                    _w_sc,
                    _s_sc,
                    _e_sc,
                    _d_sc,
                    mask_lines,
                    window_size_um=_win_sc,
                    v_min_m_s=v_min_use,
                    v_max_m_s=v_max_use,
                    v_step_m_s=v_step_m_s,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                    **_kw_sc,
                )
            except Exception:
                continue
            (
                _rvs_sc,
                _errs_sc,
                _centers_sc,
                v_grid_sc,
                ccf_2d_sc,
                _wl_lo_sc,
                _wl_hi_sc,
                _ng_sc,
                _std_sc,
                _ccf_preview_sc,
            ) = _scout
            if int(_rvs_sc.size) <= 0:
                continue
            vg_sc = np.asarray(v_grid_sc, dtype=np.float64)
            m_dt = _nanmean_axis0_ignore_empty_columns(
                np.asarray(ccf_2d_sc, dtype=np.float64)
            )
            v_c = _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg_sc, m_dt)
            if math.isfinite(v_c):
                fiber_merged_v_grid = build_hybrid_merged_velocity_grid(
                    float(v_min_use),
                    float(v_max_use),
                    float(v_step_m_s),
                    float(v_c),
                    float(ccf_hybrid_half_width_m_s),
                    float(ccf_hybrid_step_m_s),
                )
                logger.info(
                    "CCF line_mask: auto-hybrid scout (fiber-level, order=%s) center=%.2f m/s "
                    "n_vel=%d fiber=%s",
                    _ord_sc,
                    float(v_c),
                    int(fiber_merged_v_grid.size),
                    fiber,
                )
            break

    # One inverse-variance weight per statistically independent RV estimate (see loop body).
    fiber_agg_rvs: list[float] = []
    fiber_agg_errs: list[float] = []
    chunk_fiber_agg_index: list[int] = []

    for oi, order in order_tasks:
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64)
        e = np.asarray(err[oi] if getattr(err, "ndim", 0) >= 2 else err, dtype=np.float64)
        d_bad = np.asarray(bad[oi] if getattr(bad, "ndim", 0) >= 2 else bad, dtype=bool)

        w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue

        order_ccf_rows: list[np.ndarray] = []
        order_v_grid: np.ndarray | None = None
        order_chunk_rvs: list[float] = []
        order_chunk_errs: list[float] = []
        order_chunk_wl: list[float] = []
        order_chunk_ng: list[int] = []
        v_auto_grid: np.ndarray | None = (
            np.asarray(fiber_merged_v_grid, dtype=np.float64)
            if fiber_merged_v_grid is not None and not hybrid
            else None
        )
        order_auto_hybrid_merged = fiber_merged_v_grid is not None and not hybrid

        kw_rv = {
            "ccf_parabolic_vertex_n_samples": int(ccf_parabolic_vertex_n_samples),
            "min_good_pixels": int(min_good_pixels),
            "mask_ccf_pixel_interp_order": int(mask_ccf_pixel_interp_order),
            **kw_rv_line_chunking,
        }
        # Legacy single-order auto-hybrid (when fiber-level scout did not run).
        span_um = float(w_max - w_min)
        window_um_scout = max(float(window_size_um), span_um * (1.0 + 1e-12))
        if (
            bool(mask_auto_hybrid_scout)
            and single_order_line_mask
            and not hybrid
            and v_auto_grid is None
        ):
            scout = rv_per_chunks_line_mask(
                w,
                s,
                e,
                d_bad,
                mask_lines,
                window_size_um=window_um_scout,
                v_min_m_s=v_min_use,
                v_max_m_s=v_max_use,
                v_step_m_s=v_step_m_s,
                zb=zb,
                max_cores=max_cores,
                return_ccf=True,
                **kw_rv,
            )
            (
                _rvs_sc,
                _errs_sc,
                _centers_sc,
                v_grid_sc,
                ccf_2d_sc,
                _wl_lo_sc,
                _wl_hi_sc,
                _ng_sc,
                _std_sc,
                _ccf_preview_sc,
            ) = scout
            if int(_rvs_sc.size) > 0:
                vg_sc = np.asarray(v_grid_sc, dtype=np.float64)
                m_dt = _nanmean_axis0_ignore_empty_columns(
                    np.asarray(ccf_2d_sc, dtype=np.float64)
                )
                v_c = _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg_sc, m_dt)

                if math.isfinite(v_c):
                    v_auto_grid = build_hybrid_merged_velocity_grid(
                        float(v_min_use),
                        float(v_max_use),
                        float(v_step_m_s),
                        float(v_c),
                        float(ccf_hybrid_half_width_m_s),
                        float(ccf_hybrid_step_m_s),
                    )
                    order_auto_hybrid_merged = True
                    fiber_merged_v_grid = v_auto_grid
                    logger.info(
                        "CCF line_mask: auto-hybrid scout (full-order mean CCF) center=%.2f m/s "
                        "n_vel=%d fiber=%s order=%s window_um_scout=%.6f",
                        float(v_c),
                        int(v_auto_grid.size),
                        fiber,
                        order,
                        float(window_um_scout),
                    )
                else:
                    v_auto_grid = np.asarray(v_grid_sc, dtype=np.float64)

        if hybrid:
            (
                rvs_o,
                errs_o,
                centers_o,
                v_grid_o,
                ccf_2d,
                _wl_lo_o,
                _wl_hi_o,
                ng_o,
                _ccf_std_2d,
                ccf_preview_2d,
            ) = rv_per_chunks_line_mask(
                w,
                s,
                e,
                d_bad,
                mask_lines,
                window_size_um=window_size_um,
                v_min_m_s=float(v_min_use),
                v_max_m_s=float(v_max_use),
                v_step_m_s=float(v_step_m_s),
                v_grid_m_s=v_grid,
                zb=zb,
                max_cores=max_cores,
                return_ccf=True,
                **kw_rv,
            )
        elif v_auto_grid is not None:
            (
                rvs_o,
                errs_o,
                centers_o,
                v_grid_o,
                ccf_2d,
                _wl_lo_o,
                _wl_hi_o,
                ng_o,
                _ccf_std_2d,
                ccf_preview_2d,
            ) = rv_per_chunks_line_mask(
                w,
                s,
                e,
                d_bad,
                mask_lines,
                window_size_um=window_size_um,
                v_min_m_s=float(v_min_use),
                v_max_m_s=float(v_max_use),
                v_step_m_s=float(v_step_m_s),
                v_grid_m_s=v_auto_grid,
                zb=zb,
                max_cores=max_cores,
                return_ccf=True,
                **kw_rv,
            )
        else:
            (
                rvs_o,
                errs_o,
                centers_o,
                v_grid_o,
                ccf_2d,
                _wl_lo_o,
                _wl_hi_o,
                ng_o,
                _ccf_std_2d,
                ccf_preview_2d,
            ) = rv_per_chunks_line_mask(
                w,
                s,
                e,
                d_bad,
                mask_lines,
                window_size_um=window_size_um,
                v_min_m_s=v_min_use,
                v_max_m_s=v_max_use,
                v_step_m_s=v_step_m_s,
                zb=zb,
                max_cores=max_cores,
                return_ccf=True,
                **kw_rv,
            )
        rvccf_order_start = len(rvccf_rows_accum)
        if rvs_o.size != 0:
            for ii in range(int(rvs_o.size)):
                order_chunk_rvs.append(float(rvs_o[ii]))
                ev = float(errs_o[ii])
                order_chunk_errs.append(ev if np.isfinite(ev) and ev > 0 else float("nan"))
                order_chunk_wl.append(float(centers_o[ii]))
                order_chunk_ng.append(int(ng_o[ii]))

            if ccf_2d is not None and ccf_2d.size > 0:
                for ii in range(int(rvs_o.size)):
                    rvccf_rows_accum.append(np.asarray(ccf_2d[ii], dtype=np.float64))
                    rvccf_n_good_accum.append(int(ng_o[ii]))

                v_now = np.asarray(v_grid_o, dtype=np.float64)
                # Stack chunk CCF rows for order-mean RV (same basis as chunk RV / RVCCF HDU).
                stack_for_order_mean = ccf_2d
                if order_v_grid is None:
                    order_v_grid = v_now
                    order_ccf_rows.append(stack_for_order_mean)
                elif order_v_grid.shape == v_now.shape and np.allclose(order_v_grid, v_now):
                    order_ccf_rows.append(stack_for_order_mean)

        n_chunks_order = len(order_chunk_rvs)
        if n_chunks_order == 0:
            continue

        order_used_gaussian_mean = False
        stacked_order: np.ndarray | None = None
        ccf_mean_order: np.ndarray | None = None
        ccf_err_mean_order: np.ndarray | None = None
        sigma_hp_order = float("nan")
        if order_ccf_rows and order_v_grid is not None:
            stacked_order = np.vstack(order_ccf_rows)
            ccf_mean_order = _nanmean_axis0_ignore_empty_columns(stacked_order)
            sigma_hp_order, _, _ = estimate_ccf_uncertainty_highpass_residual(ccf_mean_order)
            if np.isfinite(sigma_hp_order) and sigma_hp_order > 0.0:
                ccf_err_mean_order = np.full_like(
                    np.asarray(ccf_mean_order, dtype=np.float64),
                    float(sigma_hp_order),
                    dtype=np.float64,
                )
            else:
                ccf_err_mean_order = None

            v_fit = np.asarray(order_v_grid, dtype=np.float64)
            vstep_fit = (
                float(effective_ccf_v_step_for_fit(v_fit))
                if (hybrid or order_auto_hybrid_merged)
                else float(v_step_m_s)
            )
            wl_c = 0.5 * (w_min + w_max)
            vsp = dv_per_pixel_m_s(float(wl_c))
            n_good_fit = max(int(np.nansum(order_chunk_ng)), 1)
            # Duplicate the order-mean Gaussian RV onto every chunk row only when this echelle
            # row has a **single** line-mask chunk. Multiple chunks per order (e.g. line-window
            # Voronoi bins) keep per-chunk Numba RVs so RVCCF / outlier / RVCCFAVG aggregate over
            # all chunks; Plotly preview uses one CCF slot per chunk (see ``want_preview`` block).
            if n_chunks_order == 1:
                _gauss_seed = (
                    float(ccf_hybrid_center_m_s)
                    if hybrid
                    and ccf_hybrid_center_m_s is not None
                    and math.isfinite(float(ccf_hybrid_center_m_s))
                    else None
                )
                rv_g, err_g, _par_g, diag_g = fit_ccf_rv(
                    v_fit,
                    ccf_mean_order,
                    vstep_fit,
                    n_good=n_good_fit,
                    return_params=True,
                    vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
                    ccf_err=ccf_err_mean_order,
                    ccf_line_mask_stored_negated=LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
                    line_mask_line_window_by_line=bool(window_line_by_line),
                    gaussian_seed_mean_m_s=_gauss_seed,
                )
                if (
                    bool(diag_g.get("fit_ok"))
                    and np.isfinite(rv_g)
                    and np.isfinite(err_g)
                    and float(err_g) > 0.0
                ):
                    rv_gf = float(rv_g)
                    err_gf = float(err_g)
                    rv_bi, err_bi, _bmeta = _refine_ccf_rv_bisector_after_gaussian_fit(
                        v_fit,
                        ccf_mean_order,
                        vstep_fit,
                        ccf_err=ccf_err_mean_order,
                    )
                    if rv_bi is not None and math.isfinite(float(rv_bi)):
                        rv_gf = float(rv_bi)
                        if err_bi is not None and math.isfinite(float(err_bi)) and float(err_bi) > 0:
                            err_gf = float(err_bi)
                    order_chunk_rvs = [rv_gf] * n_chunks_order
                    order_chunk_errs = [err_gf] * n_chunks_order
                    order_used_gaussian_mean = True

        if order_used_gaussian_mean:
            fiber_agg_rvs.append(float(order_chunk_rvs[0]))
            fiber_agg_errs.append(float(order_chunk_errs[0]))
            ai = len(fiber_agg_rvs) - 1
            chunk_fiber_agg_index.extend([ai] * n_chunks_order)
        else:
            for rv_c, er_c in zip(order_chunk_rvs, order_chunk_errs):
                fiber_agg_rvs.append(float(rv_c))
                fiber_agg_errs.append(float(er_c))
                chunk_fiber_agg_index.append(len(fiber_agg_rvs) - 1)

        all_rvs.extend(order_chunk_rvs)
        all_rv_errs.extend(order_chunk_errs)
        all_wavelengths.extend(order_chunk_wl)
        all_orders.extend([int(order)] * n_chunks_order)

        # Preview slots: one CCF curve per line-mask chunk (dropdown in stellar RV preview).
        # When several chunks share an echelle order (line-window Voronoi bins), each chunk
        # gets its own slot + ``rvccf_chunk_id`` aligned with ``RVCCF`` ``CHUNK_ID``.
        if want_preview_effective and ccf_2d is not None and n_chunks_order > 0:
            c2 = np.asarray(ccf_2d, dtype=np.float64)
            if c2.ndim == 2 and c2.shape[0] == n_chunks_order:
                v_prev = np.asarray(v_grid_o, dtype=np.float64)
                std2 = (
                    np.asarray(_ccf_std_2d, dtype=np.float64)
                    if _ccf_std_2d is not None
                    and np.asarray(_ccf_std_2d).shape == c2.shape
                    else None
                )
                wlo_a = (
                    np.asarray(_wl_lo_o, dtype=np.float64)
                    if _wl_lo_o is not None
                    else None
                )
                whi_a = (
                    np.asarray(_wl_hi_o, dtype=np.float64)
                    if _wl_hi_o is not None
                    else None
                )
                ng_arr = np.asarray(ng_o, dtype=np.int64)
                n_rvccf_added = int(len(rvccf_rows_accum) - rvccf_order_start)
                have_rvccf_ids = n_rvccf_added == int(n_chunks_order) and int(n_chunks_order) > 0
                for ii in range(n_chunks_order):
                    row = np.asarray(c2[ii], dtype=np.float64)
                    slot_payload: dict = {
                        "kind": "order",
                        "order": int(order),
                        "chunk_index": int(ii),
                        "chunk_n": int(n_chunks_order),
                        "v": v_prev,
                        "ccf": row,
                        "wl_lo": float(wlo_a[ii])
                        if wlo_a is not None and ii < int(wlo_a.size)
                        else float(w_min),
                        "wl_hi": float(whi_a[ii])
                        if whi_a is not None and ii < int(whi_a.size)
                        else float(w_max),
                        "n_good": max(1, int(ng_arr[ii])) if ii < int(ng_arr.size) else 1,
                    }
                    if have_rvccf_ids:
                        slot_payload["rvccf_chunk_id"] = int(rvccf_order_start + ii)
                    ce_row: np.ndarray | None = None
                    if std2 is not None and np.any(np.isfinite(std2[ii])):
                        ce_row = np.asarray(std2[ii], dtype=np.float64)
                    elif (
                        n_chunks_order == 1
                        and ccf_err_mean_order is not None
                        and np.any(np.isfinite(ccf_err_mean_order))
                    ):
                        ce_row = np.asarray(ccf_err_mean_order, dtype=np.float64)
                    if ce_row is not None:
                        slot_payload["ccf_err"] = ce_row

                    order_slots.append(slot_payload)
                    all_ccf_rows.append(np.ascontiguousarray(c2[ii : ii + 1], dtype=np.float64))
                    if std2 is not None:
                        all_ccf_err_rows.append(
                            np.ascontiguousarray(std2[ii : ii + 1], dtype=np.float64)
                        )
                    elif (
                        n_chunks_order == 1
                        and ccf_err_mean_order is not None
                        and np.any(np.isfinite(ccf_err_mean_order))
                    ):
                        all_ccf_err_rows.append(
                            np.asarray(ccf_err_mean_order, dtype=np.float64, ndmin=2)
                        )
                    else:
                        all_ccf_err_rows.append(
                            np.full((1, int(c2.shape[1])), np.nan, dtype=np.float64)
                        )

        if debug_ccf and n_chunks_order > 0:
            if kw_rv_line_chunking:
                n_geom_dbg = line_mask_n_chunks_for_order_line_by_line(
                    w,
                    mask_lines,
                    v_center_m_s=float(kw_rv_line_chunking["line_mask_line_chunk_v_center_m_s"]),
                    zb=float(zb),
                    min_good_pixels=int(min_good_pixels),
                )
            else:
                n_geom_dbg = ccf_order_n_chunks_for_span(span_um, float(window_size_um))
            logger.info(
                "CCF line_mask fiber=%s order=%s: n_chunks_geom=%d n_chunks_out=%d rv_mean~%.1f m/s",
                fiber,
                order,
                int(n_geom_dbg),
                n_chunks_order,
                float(np.nanmedian(order_chunk_rvs)),
            )

    all_rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    all_rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    all_wavelengths_arr = np.asarray(all_wavelengths, dtype=np.float64)
    all_orders_arr = np.asarray(all_orders, dtype=np.int64)
    if len(all_orders_arr) != len(all_wavelengths_arr):
        all_orders_arr = np.full(len(all_wavelengths_arr), -1, dtype=np.int64)
    # Fiber-level RV aggregation (median/MAD + IV mean) is performed by the stellar RV primitive
    # so that ccf_utils remains a dependency of stellar_rv (and does not import from it).

    if hybrid:
        pv_min = float(v_grid[0])
        pv_max = float(v_grid[-1])
        pv_step = float(effective_ccf_v_step_for_fit(v_grid))
        preview_v_grid = v_grid
    elif fiber_merged_v_grid is not None:
        pv_min = float(fiber_merged_v_grid[0])
        pv_max = float(fiber_merged_v_grid[-1])
        pv_step = float(effective_ccf_v_step_for_fit(fiber_merged_v_grid))
        preview_v_grid = fiber_merged_v_grid
    else:
        pv_min = float(v_min_use)
        pv_max = float(v_max_use)
        pv_step = float(v_step_m_s)
        preview_v_grid = v_grid

    preview_detail = None
    if want_preview_effective and order_slots and all_ccf_rows:
        logger.info(
            "CCF line_mask: fiber=%s building preview n_preview_orders=%d",
            fiber,
            len(order_slots),
        )
        _preview_gauss_seed = (
            float(ccf_hybrid_center_m_s)
            if hybrid
            and ccf_hybrid_center_m_s is not None
            and math.isfinite(float(ccf_hybrid_center_m_s))
            else None
        )
        preview_detail = finalize_ccf_preview_dict(
            order_slots,
            all_ccf_rows,
            preview_v_grid,
            pv_min,
            pv_max,
            pv_step,
            all_ccf_err_rows if len(all_ccf_err_rows) == len(order_slots) else None,
            ccf_line_mask_stored_negated=LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
            ccf_parabolic_vertex_n_samples=int(ccf_parabolic_vertex_n_samples),
            line_mask_line_window_by_line=bool(window_line_by_line),
            line_mask_gaussian_seed_mean_m_s=_preview_gauss_seed,
        )

    logger.info(
        "CCF line_mask: fiber=%s finished n_chunks=%d preview=%s",
        fiber,
        len(all_wavelengths_arr),
        preview_detail is not None,
    )

    n_rvccf = len(rvccf_rows_accum)
    if n_rvccf > 0:
        rvccf_tbl = build_rvccf_chunk_supplement(
            chunk_id=np.arange(n_rvccf, dtype=np.int32),
            n_good=np.asarray(rvccf_n_good_accum, dtype=np.int32),
            v_min=pv_min,
            v_max=pv_max,
            v_step=pv_step,
            ccf_2d=np.stack(rvccf_rows_accum, axis=0),
        )
    else:
        rvccf_tbl = build_rvccf_chunk_supplement(
            chunk_id=np.array([], dtype=np.int32),
            n_good=np.array([], dtype=np.int32),
            v_min=pv_min,
            v_max=pv_max,
            v_step=pv_step,
            ccf_2d=np.empty((0, 0), dtype=np.float64),
        )

    return (
        all_wavelengths_arr,
        all_rvs_arr,
        all_rv_errs_arr,
        all_orders_arr,
        preview_detail,
        rvccf_tbl,
    )

