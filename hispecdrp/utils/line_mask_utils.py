"""
Build SPIROU-style ``.mas`` line masks from stellar template spectra.

Loads a template (BT-Settl HDF5 via :mod:`hispecdrp.utils.spectral_templates_utils`),
detects absorption centers with :func:`detect_template_line_centers_gradient_um`
(``numpy.gradient``: ``dF/dλ = 0``, ``d²F/dλ² > 0`` at interpolated roots), applies
depth / deblending / telluric filters, and writes a **two-column** mask (vacuum nm:
wavelength, weight). Lines have **infinitely narrow** extent on disk (no left/right
edges). Weights are proportional to **continuum-normalized absorption strength**
:math:`1 - F/F_c` at each line center (``F`` and ``F_c`` interpolated on the template
grid), then **renormalized so the weight column sums to 1**. Candidate lines whose
local minimum :math:`F/F_c` near the centroid falls below a small floor (default
:math:`10^{-4}`) are dropped as unreliable saturated cores.
Generation options are
serialized as JSON on a ``# HISPEC_DRP_MASK_META`` line for parsing
(:func:`read_mask_generation_meta`).

See ``hispecdrp/data/masks/README.md`` for mask format notes and
``hispecdrp/data/atmosphere/README.md`` for the transmission table used in telluric
filtering.

**Centroid vs** :mod:`hispecdrp.primitives.utils.ccf_utils`
-----------------------------------------------------------
:func:`~hispecdrp.primitives.utils.ccf_utils.load_mask` supplies each line as a rest
wavelength ``line_lambda_um`` (vacuum µm); the CCF never re-estimates line centers from
data. **Two-column** template masks store that wavelength directly; legacy three-column
SPIROU files list ``start_nm``, ``end_nm`` and loaders use ``(start+end)/2``.
:func:`fit_half_width_linear_from_mas` applies only to masks with measurable segment
width (delta rows are skipped).

:func:`detect_template_line_centers_parabolic_um` remains available to compare against
:func:`detect_template_line_centers_gradient_um` via :func:`compare_line_center_lists_um`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import argrelextrema

from hispecdrp.utils.spectral_templates_utils import load_template_from_path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WIDTH_CALIBRATION_K2_ALL",
    "MASK_LINE_HALF_WIDTH_SLOPE_NM_PER_NM",
    "LINE_MASK_META_PREFIX",
    "OPTIMAL_TEMPLATE_MASK_BUILD_KW",
    "LineMaskEntry",
    "build_line_mask_from_template",
    "compare_line_center_lists_um",
    "detect_template_line_centers_gradient_um",
    "detect_template_line_centers_parabolic_um",
    "fit_half_width_linear_from_mas",
    "load_atmospheric_transmission_um",
    "mean_transmission_over_velocity_interval_um",
    "parse_mask_generation_meta_line",
    "read_mask_generation_meta",
]

# Prefix for a single-line JSON blob of generation parameters (after ``# ``).
LINE_MASK_META_PREFIX = "# HISPEC_DRP_MASK_META "

# Half-width (nm) vs line-center vacuum wavelength (nm): half_width ≈ slope * λ_nm + intercept.
# Coefficients from a least-squares fit to bundled ``K2_all.mas`` (2797 segments); intercept
# is numerically ~0 because mask segments are an almost fixed Δln λ (see data/masks/README.md).
DEFAULT_WIDTH_CALIBRATION_K2_ALL: tuple[float, float] = (
    7.033348083502036e-06,
    0.0,
)

# Half-width law (nm): ``half_width_nm = slope * center_vac_nm + intercept`` for
# template-built SPIROU ``.mas`` files. This scale (nm per nm of line center) sets how
# wide each segment is before deblending / telluric filters.
MASK_LINE_HALF_WIDTH_SLOPE_NM_PER_NM: float = 1.0e-6

# Defaults for template-derived **delta** masks (gradient centroids, no segment width).
OPTIMAL_TEMPLATE_MASK_BUILD_KW: dict[str, Any] = {
    "min_continuum_normalized_depth": 0.1,
    "low_pass_width_multiplier": 7.283985083415576,
    "telluric_velocity_half_width_km_s": 288.2674944075802,
    "telluric_min_mean_transmission": 0.0,
    "continuum_filter_size": 181,
}

_DEFAULT_ATMOSPHERE_REL = os.path.join(
    os.path.dirname(__file__), "..", "data", "atmosphere", "mktrans_zm_10_10.dat"
)


def parse_mask_generation_meta_line(line: str) -> dict | None:
    """
    If ``line`` starts with :data:`LINE_MASK_META_PREFIX`, return the parsed JSON object.

    The remainder of the line after the prefix must be a single JSON object.
    """
    s = line.strip()
    if not s.startswith(LINE_MASK_META_PREFIX):
        return None
    payload = s[len(LINE_MASK_META_PREFIX) :].strip()
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def read_mask_generation_meta(path: str) -> dict | None:
    """Scan the start of a ``.mas`` file for :func:`parse_mask_generation_meta_line`."""
    with open(path, encoding="utf-8") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            meta = parse_mask_generation_meta_line(line)
            if meta is not None:
                return meta
    return None


@dataclass(frozen=True)
class LineMaskEntry:
    """One template-built mask row: vacuum wavelength (nm) and CCF weight."""

    wavelength_nm: float
    weight: float


def fit_half_width_linear_from_mas(mas_path: str) -> tuple[float, float]:
    """
    Fit ``half_width_nm = slope * center_nm + intercept`` from a SPIROU ``.mas`` file.

    Comment lines (``#``, ``;``, ``//``) are skipped; centers and half-widths are
    computed from the first two columns (start_nm, end_nm).
    """
    centers: list[float] = []
    halfw: list[float] = []
    with open(mas_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            centers.append(0.5 * (a + b))
            halfw.append(0.5 * (b - a))
    hw = np.asarray(halfw, dtype=np.float64)
    finite = np.asarray(centers, dtype=np.float64)
    m = hw > 1.0e-9
    if int(np.sum(m)) < 2:
        raise ValueError(
            f"Too few finite-width rows (need ≥2 with half-width > 1e-9 nm) to fit: {mas_path!r}"
        )
    x = finite[m]
    y = hw[m]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def load_atmospheric_transmission_um(
    path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Gemini ``mktrans`` two-column ASCII: wavelength (µm), transmission (0–1).

    Comment lines (``#``, ``;``, ``//``) are skipped. The bundled file under
    ``data/atmosphere/`` ships with a documented header; data rows are fixed-format
    decimals. External copies that omit a leading ``0`` on the first column
    (e.g. ``.900180``) are still accepted.
    """
    p = path if path is not None else os.path.normpath(_DEFAULT_ATMOSPHERE_REL)
    rows: list[tuple[float, float]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            if line[0] == ".":
                line = "0" + line
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                t = float(parts[1])
            except ValueError:
                continue
            rows.append((w, t))
    if len(rows) < 8:
        raise ValueError(f"Too few transmission samples in {p!r}")
    w = np.asarray([r[0] for r in rows], dtype=np.float64)
    t = np.asarray([r[1] for r in rows], dtype=np.float64)
    order = np.argsort(w)
    return w[order], t[order]


def mean_transmission_over_velocity_interval_um(
    line_wavelength_um: float,
    trans_wavelength_um: np.ndarray,
    transmission: np.ndarray,
    half_width_km_s: float,
) -> float:
    """
    Mean atmospheric transmission over ``line_wavelength_um * (1 ± v/c)`` (vacuum, µm).

    ``half_width_km_s`` is a **half-width** in km/s; the interval is
    :math:`\\pm` that value (total span :math:`2 v / c` in fractional wavelength).
    Sampling uses a fixed number of points on the interval; values are linearly
    interpolated in the transmission table and clipped to the table bounds.
    """
    lam = float(line_wavelength_um)
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    c_m_s = 299792458.0
    v_m_s = abs(float(half_width_km_s)) * 1000.0
    beta = min(v_m_s / c_m_s, 0.2)
    lo = lam * (1.0 - beta)
    hi = lam * (1.0 + beta)
    tw0 = float(trans_wavelength_um[0])
    tw1 = float(trans_wavelength_um[-1])
    lo = float(np.clip(lo, tw0, tw1))
    hi = float(np.clip(hi, tw0, tw1))
    if hi <= lo:
        return float(np.interp(lam, trans_wavelength_um, transmission))
    grid = np.linspace(lo, hi, num=32, dtype=np.float64)
    samp = np.interp(grid, trans_wavelength_um, transmission)
    return float(np.mean(samp))


def _continuum_envelope(flux: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    env = maximum_filter1d(flux, size=size, mode="nearest")
    return np.maximum(env, 1.0e-30)


def _local_minima_indices(flux: np.ndarray, order: int) -> np.ndarray:
    if order < 1:
        order = 1
    idx = argrelextrema(flux, np.less, order=order)[0]
    return np.asarray(idx, dtype=np.int64)


def _parabolic_minimum_wavelength_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    idx: int,
) -> float:
    """
    Sub-pixel wavelength of a local absorption minimum.

    Fits :math:`F(\\lambda) \\approx a\\lambda^2 + b\\lambda + c` to three consecutive
    samples bracketing ``idx``; returns the vertex if it is a **minimum** with
    :math:`a > 0` and lies inside the bracketed interval. Otherwise returns
    ``wave_um[idx]``.

    This aligns the stored segment midpoint ``(start_nm + end_nm)/2`` with the same
    convention :func:`~hispecdrp.primitives.utils.ccf_utils.load_mask_spirou_mas` uses for
    three-column SPIROU masks (``line_lambda_um``), while improving precision over using
    the grid sample at ``idx`` alone.
    """
    n = int(wave_um.size)
    if n < 3 or idx <= 0 or idx >= n - 1:
        return float(wave_um[idx])
    x = np.asarray(wave_um[idx - 1 : idx + 2], dtype=np.float64)
    yv = np.asarray(flux[idx - 1 : idx + 2], dtype=np.float64)
    if not np.all(np.isfinite(x) & np.isfinite(yv)):
        return float(wave_um[idx])
    # F = a x^2 + b x + c; minimum at x = -b/(2a) when a > 0
    a_mat = np.column_stack([x**2, x, np.ones(3, dtype=np.float64)])
    try:
        coef, _res, rank, _s = np.linalg.lstsq(a_mat, yv, rcond=None)
    except np.linalg.LinAlgError:
        return float(wave_um[idx])
    if rank < 3:
        return float(wave_um[idx])
    a, b, _c = (float(coef[0]), float(coef[1]), float(coef[2]))
    scale = max(np.max(np.abs(yv)), 1.0)
    if abs(a) < 1e-18 * scale:
        return float(wave_um[idx])
    if a <= 0.0:
        return float(wave_um[idx])
    xv = -b / (2.0 * a)
    lo, hi = float(np.min(x)), float(np.max(x))
    if xv < lo or xv > hi:
        return float(wave_um[idx])
    return float(xv)


def detect_template_line_centers_parabolic_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    *,
    local_min_order: int = 2,
    continuum_filter_size: int = 181,
    min_continuum_normalized_depth: float | None = None,
) -> np.ndarray:
    """
    Line centers (vacuum µm) from discrete local flux minima + parabolic refinement.

    This matches the **centroid** step in :func:`build_line_mask_from_template` before
    depth / deblend / telluric filters: :func:`_local_minima_indices` on ``flux`` and
    :func:`_parabolic_minimum_wavelength_um` at each index.

    Parameters
    ----------
    wave_um, flux
        Same sorted vacuum-µm grid as ``load_template_from_path`` (ascending λ).
    local_min_order
        ``order`` for :func:`scipy.signal.argrelextrema`.
    continuum_filter_size
        Used only if ``min_continuum_normalized_depth`` is set.
    min_continuum_normalized_depth
        If not ``None``, drop centers with ``1 - F/F_c`` below this threshold at the
        refined λ (``F``, ``F_c`` interpolated), matching the spirit of the mask depth
        cut in :func:`build_line_mask_from_template`.

    Returns
    -------
    np.ndarray
        1-D float64 centers in **vacuum microns**, sorted ascending.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    if w.size != f.size or w.size < 3:
        return np.array([], dtype=np.float64)
    idx = _local_minima_indices(f, int(local_min_order))
    idx = idx[(idx > 0) & (idx < w.size - 1)]
    if idx.size == 0:
        return np.array([], dtype=np.float64)
    centers = np.array(
        [_parabolic_minimum_wavelength_um(w, f, int(i)) for i in idx],
        dtype=np.float64,
    )
    centers.sort()
    if min_continuum_normalized_depth is None:
        return centers
    cont = _continuum_envelope(f, int(continuum_filter_size))
    fv = np.interp(centers, w, f)
    cv = np.interp(centers, w, cont)
    depths = 1.0 - fv / np.maximum(cv, 1.0e-30)
    m = depths >= float(min_continuum_normalized_depth)
    return np.sort(centers[m])


def _merge_sorted_centers_um(centers_um: np.ndarray, min_sep_um: float) -> np.ndarray:
    """Drop duplicates closer than ``min_sep_um`` (keep first)."""
    c = np.sort(np.asarray(centers_um, dtype=np.float64).ravel())
    c = c[np.isfinite(c)]
    if c.size == 0:
        return c
    sep = float(max(min_sep_um, 0.0))
    out: list[float] = [float(c[0])]
    for x in c[1:]:
        if x - out[-1] >= sep:
            out.append(float(x))
    return np.asarray(out, dtype=np.float64)


def _absorption_centers_from_flux_gradients_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    *,
    min_continuum_normalized_depth: float,
) -> np.ndarray:
    """
    Interpolated λ where ``dF/dλ = 0`` between samples, keeping minima (``d²F/dλ² > 0``).

    Uses :func:`numpy.gradient` with the wavelength coordinate (non-uniform spacing
    supported). Each bracket ``[i-1, i]`` where ``d1`` changes sign yields a linearly
    interpolated root; roots with non-positive second derivative (local maxima of
    flux) are discarded. Shallow points fail ``min_continuum_normalized_depth``.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.asarray(cont, dtype=np.float64).ravel()
    if w.size < 3 or w.size != f.size or f.size != c.size:
        return np.array([], dtype=np.float64)
    d1 = np.gradient(f, w, edge_order=1)
    d2 = np.gradient(d1, w, edge_order=1)
    centers: list[float] = []
    for i in range(1, w.size):
        a, b = float(d1[i - 1]), float(d1[i])
        wa, wb = float(w[i - 1]), float(w[i])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            lam = wa
        elif b == 0.0:
            lam = wb
        elif a * b < 0.0:
            t = a / (a - b)
            lam = (1.0 - t) * wa + t * wb
        else:
            continue
        d2v = float(np.interp(lam, w, d2))
        if d2v <= 0.0:
            continue
        fv = float(np.interp(lam, w, f))
        cv = float(np.interp(lam, w, c))
        depth = 1.0 - fv / max(cv, 1.0e-30)
        if depth < float(min_continuum_normalized_depth):
            continue
        centers.append(lam)
    return np.asarray(centers, dtype=np.float64)


def detect_template_line_centers_gradient_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    *,
    continuum_filter_size: int = 181,
    min_continuum_normalized_depth: float = 0.04,
    merge_min_sep_um: float | None = None,
) -> np.ndarray:
    """
    Absorption line centers (vacuum µm) from **gradient zeros** with positive curvature.

    Intervals where the first derivative of flux w.r.t. λ (``numpy.gradient``) crosses
    zero are refined to a wavelength; the point is kept only if the second derivative
    of flux w.r.t. λ is **positive** (flux minimum, not maximum) and the usual
    continuum-normalized depth exceeds ``min_continuum_normalized_depth``.

    This is an alternative to chaining :func:`_local_minima_indices` with
    :func:`_parabolic_minimum_wavelength_um`; lists can be compared with
    :func:`compare_line_center_lists_um`.

    Parameters
    ----------
    wave_um, flux
        Sorted vacuum-µm template grid (as from ``load_template_from_path``).
    continuum_filter_size
        Odd-ish size for :func:`_continuum_envelope` (depth definition).
    min_continuum_normalized_depth
        Minimum ``1 - F/F_c`` at the interpolated stationary point.
    merge_min_sep_um
        Merge roots closer than this (µm). Default: ``0.25 * median(Δλ)`` on the grid.

    Returns
    -------
    np.ndarray
        Centers in **vacuum microns**, sorted ascending.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    if w.size < 8:
        return np.array([], dtype=np.float64)
    cont = _continuum_envelope(f, int(continuum_filter_size))
    raw = _absorption_centers_from_flux_gradients_um(
        w,
        f,
        cont,
        min_continuum_normalized_depth=float(min_continuum_normalized_depth),
    )
    dw = np.diff(w)
    dw = dw[np.isfinite(dw) & (dw > 0)]
    med = float(np.median(dw)) if dw.size else 1.0e-6
    sep = float(0.25 * med) if merge_min_sep_um is None else float(merge_min_sep_um)
    return _merge_sorted_centers_um(raw, sep)


def compare_line_center_lists_um(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    *,
    match_tol_nm: float = 0.05,
) -> dict[str, Any]:
    """
    Match two line-center lists (vacuum µm) within ``match_tol_nm`` (nanometers).

    Returns counts and boolean masks over the **sorted** union of inputs for which
    centers fall in only A, only B, or both (matched pairs use greedy nearest-neighbor
    within tolerance on the merged sorted list — for diagnostics only).

    Notes
    -----
    Greedy matching: sort all centers from A and B with tags; sweep λ increasing,
    pairing unmatched A with next unmatched B if |Δλ| ≤ tol (in µm: ``tol_nm * 1e-3``).
    Simpler approach: for each a in A, match if any b in B with |a-b| <= tol_um.

    Returns
    -------
    dict
        Keys: ``n_a``, ``n_b``, ``n_match``, ``n_a_only``, ``n_b_only``,
        ``match_tol_nm``, ``matched_pairs`` (list of (a_um, b_um)), ``centers_a_only_um``,
        ``centers_b_only_um``.
    """
    tol_um = abs(float(match_tol_nm)) * 1.0e-3
    a = np.sort(np.asarray(centers_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(centers_b, dtype=np.float64).ravel())
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    used_b = np.zeros(b.size, dtype=bool)
    pairs: list[tuple[float, float]] = []
    a_only: list[float] = []
    for x in a:
        if b.size == 0:
            a_only.append(float(x))
            continue
        free = np.flatnonzero(~used_b)
        if free.size == 0:
            a_only.append(float(x))
            continue
        dist = np.abs(b[free] - x)
        k = int(np.argmin(dist))
        if float(dist[k]) <= tol_um:
            j = int(free[k])
            used_b[j] = True
            pairs.append((float(x), float(b[j])))
        else:
            a_only.append(float(x))
    b_only = [float(b[k]) for k in range(b.size) if not used_b[k]]
    n_match = len(pairs)
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_match": n_match,
        "n_a_only": len(a_only),
        "n_b_only": len(b_only),
        "match_tol_nm": float(match_tol_nm),
        "matched_pairs": pairs,
        "centers_a_only_um": np.asarray(a_only, dtype=np.float64),
        "centers_b_only_um": np.asarray(b_only, dtype=np.float64),
    }


def _cluster_max_depth(
    centers_nm: np.ndarray,
    merge_radii_nm: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    """Union lines with ``|Δλ| < r_i + r_j``; keep largest per-line metric per component (union-find).

    ``depths`` is the per-line scalar used to break ties (continuum-normalized depth
    or another positive strength measure).
    """
    n = int(centers_nm.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    max_r = float(np.max(merge_radii_nm))
    bin_w = max(2.0 * max_r, 1.0e-9)
    bins: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        lam = float(centers_nm[i])
        b0 = int(np.floor(lam / bin_w))
        for bb in (b0 - 1, b0, b0 + 1):
            for j in bins[bb]:
                if abs(centers_nm[i] - centers_nm[j]) < merge_radii_nm[i] + merge_radii_nm[j]:
                    union(i, j)
        bins[b0].append(i)

    roots: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)
    keep: list[int] = []
    for members in roots.values():
        best = max(members, key=lambda k: float(depths[k]))
        keep.append(best)
    return np.sort(np.asarray(keep, dtype=np.int64))


def _segment_mean_continuum_normalized_depth_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    lo_um: float,
    hi_um: float,
) -> float:
    """
    Mean :math:`1 - F/F_c` over discrete template samples inside ``[lo_um, hi_um]``.

    Helper for segment-averaged depth (e.g. comparisons); template-mask **weights** in
    :func:`build_line_mask_from_template` use line-center :math:`1 - F/F_c` only.
    """
    lo = float(lo_um)
    hi = float(hi_um)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    m = (wave_um >= lo) & (wave_um <= hi)
    if not np.any(m):
        return float("nan")
    c = np.maximum(cont[m], 1.0e-30)
    d = 1.0 - flux[m] / c
    v = float(np.mean(d))
    if not np.isfinite(v):
        return float("nan")
    return float(np.clip(v, 0.0, 1.0))


def _local_min_continuum_normalized_flux_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    centers_um: np.ndarray,
    *,
    half_width_um: float,
) -> np.ndarray:
    """
    Per centroid, minimum of :math:`F/F_c` over discrete samples in
    ``[λ - half_width, λ + half_width]``.

    Used to reject saturated / unreliable line cores (very small :math:`F/F_c`).
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.maximum(np.asarray(cont, dtype=np.float64).ravel(), 1.0e-30)
    lam = np.asarray(centers_um, dtype=np.float64).ravel()
    if w.size < 1 or lam.size == 0:
        return np.array([], dtype=np.float64)
    f_over_c = f / c
    hw = float(half_width_um)
    if not np.isfinite(hw) or hw <= 0.0:
        return np.full(lam.size, np.nan, dtype=np.float64)
    out = np.empty(lam.size, dtype=np.float64)
    for i in range(lam.size):
        lo = float(lam[i]) - hw
        hi = float(lam[i]) + hw
        m = (w >= lo) & (w <= hi)
        if np.any(m):
            out[i] = float(np.min(f_over_c[m]))
        else:
            out[i] = float(np.interp(float(lam[i]), w, f_over_c))
    return out


def _continuum_normalized_d2_flux_dlam2_at_um(
    wave_um: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    centers_um: np.ndarray,
) -> np.ndarray:
    """
    Second derivative w.r.t. λ of continuum-normalized flux ``F/F_c`` at ``centers_um``.

    Uses :func:`numpy.gradient` twice on ``F/F_c`` with coordinate ``wave_um``. At an
    absorption minimum, this is positive (upward curvature); larger values indicate
    sharper line cores for a given width.
    """
    w = np.asarray(wave_um, dtype=np.float64).ravel()
    f = np.asarray(flux, dtype=np.float64).ravel()
    c = np.maximum(np.asarray(cont, dtype=np.float64).ravel(), 1.0e-30)
    if w.size < 3 or w.size != f.size or f.size != c.size:
        return np.array([], dtype=np.float64)
    f_norm = f / c
    d1 = np.gradient(f_norm, w, edge_order=1)
    d2 = np.gradient(d1, w, edge_order=1)
    lam = np.asarray(centers_um, dtype=np.float64).ravel()
    return np.interp(lam, w, d2).astype(np.float64)


def build_line_mask_from_template(
    template_path: str,
    *,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    min_continuum_normalized_depth: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "min_continuum_normalized_depth"
    ],
    low_pass_width_multiplier: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "low_pass_width_multiplier"
    ],
    telluric_velocity_half_width_km_s: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "telluric_velocity_half_width_km_s"
    ],
    telluric_min_mean_transmission: float = OPTIMAL_TEMPLATE_MASK_BUILD_KW[
        "telluric_min_mean_transmission"
    ],
    transmission_file: str | None = None,
    continuum_filter_size: int = OPTIMAL_TEMPLATE_MASK_BUILD_KW["continuum_filter_size"],
    min_continuum_normalized_flux_floor: float | None = 1.0e-4,
    local_min_fn_window_half_width_nm: float | None = None,
    output_path: str | None = None,
    masks_directory: str | None = None,
) -> tuple[str, list[LineMaskEntry]]:
    """
    Detect lines in a template spectrum and write a filtered ``.mas`` mask.

    Centroids come from :func:`detect_template_line_centers_gradient_um`. Each output
    row is ``(wavelength_vacuum_nm, weight)`` with no segment edges. Raw weights are
    continuum-normalized depth :math:`1 - F/F_c` at each line center (interpolated),
    then are divided by their sum so **output weights sum to 1**.

    Deblending uses ``0.5 * low_pass_width_multiplier * median(Δλ_nm)`` as the merge
    radius per line (deepest line kept in each cluster), where ``median(Δλ_nm)`` is
    from the sorted template grid.

    Parameters
    ----------
    template_path
        Path to a supported template (BT-Settl ``.h5``).
    wl_min_um, wl_max_um
        Wavelength clip in **vacuum** microns before line search.
    min_continuum_normalized_depth
        Passed to :func:`detect_template_line_centers_gradient_um` and re-applied after
        interpolation to centers.
    low_pass_width_multiplier
        Scales the deblending radius vs. the template median wavelength step (nm).
    telluric_velocity_half_width_km_s
        Half-width in km/s for the telluric averaging interval (± this value).
    telluric_min_mean_transmission
        Lines with mean transmission below this value are removed.
    transmission_file
        Two-column ``mktrans``-style path (µm, transmission). Default: bundled
        ``data/atmosphere/mktrans_zm_10_10.dat``.
    continuum_filter_size
        Odd-ish pixel size for the maximum-filter continuum envelope.
    min_continuum_normalized_flux_floor
        If not ``None``, drop candidate lines whose **local minimum** of :math:`F/F_c`
        (over ``±local_min_fn_window_half_width_nm`` around the centroid) is **below**
        this value (default ``1e-4``). Saturated cores have :math:`F/F_c \\to 0` and are
        skipped.
    local_min_fn_window_half_width_nm
        Half-width of the wavelength interval (nm, vacuum) used for the local
        :math:`\\min(F/F_c)` test. If ``None``, uses ``5 ×`` the template median step (nm).
    output_path
        Full path for the output ``.mas``. If ``None``, writes under
        ``masks_directory`` as ``<template_basename>.mas``.
    masks_directory
        Default output directory (package ``data/masks`` if ``None``).

    Returns
    -------
    path_written, entries
        Output file path and in-memory rows (``wavelength_nm``, ``weight``).
    """
    trans_path_resolved = os.path.normpath(
        transmission_file if transmission_file is not None else _DEFAULT_ATMOSPHERE_REL
    )
    trans_path_abs = os.path.abspath(trans_path_resolved)

    wave_um, flux = load_template_from_path(
        template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    ok = np.isfinite(wave_um) & np.isfinite(flux) & (wave_um > 0)
    wave_um = wave_um[ok]
    flux = flux[ok]
    _ord = np.argsort(wave_um)
    wave_um = wave_um[_ord]
    flux = flux[_ord]
    if wave_um.size < 16:
        raise ValueError("Template wavelength/flux too short after clipping")

    cont = _continuum_envelope(flux, continuum_filter_size)

    dw_um = np.diff(wave_um)
    dw_um = dw_um[np.isfinite(dw_um) & (dw_um > 0)]
    med_dw_um = float(np.median(dw_um)) if dw_um.size else 1.0e-6
    median_dw_nm = med_dw_um * 1000.0

    centers_um = detect_template_line_centers_gradient_um(
        wave_um,
        flux,
        continuum_filter_size=int(continuum_filter_size),
        min_continuum_normalized_depth=float(min_continuum_normalized_depth),
        merge_min_sep_um=None,
    )
    if centers_um.size == 0:
        raise ValueError(
            "No gradient line centers; relax min_continuum_normalized_depth or continuum_filter_size"
        )

    flux_c = np.interp(centers_um, wave_um, flux)
    cont_c = np.interp(centers_um, wave_um, cont)
    depths = 1.0 - flux_c / np.maximum(cont_c, 1.0e-30)
    depths = np.clip(np.asarray(depths, dtype=np.float64), 0.0, 1.0)

    if local_min_fn_window_half_width_nm is not None:
        half_win_um = float(local_min_fn_window_half_width_nm) * 1.0e-3
    else:
        half_win_um = max(5.0 * med_dw_um, 1.0e-9)
    min_fn_local = _local_min_continuum_normalized_flux_um(
        wave_um, flux, cont, centers_um, half_width_um=half_win_um
    )

    centers_nm = centers_um * 1000.0

    m = depths >= float(min_continuum_normalized_depth)
    if min_continuum_normalized_flux_floor is not None:
        floor = float(min_continuum_normalized_flux_floor)
        m &= np.isfinite(min_fn_local) & (min_fn_local >= floor)
    centers_nm = centers_nm[m]
    depths = depths[m]

    if centers_nm.size == 0:
        raise ValueError("No lines left after depth threshold; relax min depth or continuum size")

    weights = np.asarray(depths, dtype=np.float64)
    if not np.any(weights > 0.0):
        raise ValueError(
            "No lines with positive continuum-normalized depth; check template sampling"
        )

    merge_r_nm = 0.5 * float(low_pass_width_multiplier) * float(median_dw_nm)
    merge_radii_nm = np.full(centers_nm.shape[0], merge_r_nm, dtype=np.float64)

    order = np.argsort(centers_nm)
    centers_nm = centers_nm[order]
    weights = weights[order]

    keep = _cluster_max_depth(centers_nm, merge_radii_nm, weights)
    centers_nm = centers_nm[keep]
    weights = weights[keep]

    tw, tt = load_atmospheric_transmission_um(transmission_file)
    vhalf = float(telluric_velocity_half_width_km_s)
    tmin = float(telluric_min_mean_transmission)
    keep_t: list[int] = []
    for i in range(centers_nm.size):
        lam_um = float(centers_nm[i] * 1.0e-3)
        mean_t = mean_transmission_over_velocity_interval_um(lam_um, tw, tt, vhalf)
        if np.isfinite(mean_t) and mean_t >= tmin:
            keep_t.append(i)
    if not keep_t:
        raise ValueError(
            "No lines left after telluric filter; raise telluric_min_mean_transmission "
            "or widen/change the transmission file"
        )
    keep_t_arr = np.asarray(keep_t, dtype=np.int64)
    centers_nm = centers_nm[keep_t_arr]
    weights = weights[keep_t_arr]

    w_sum = float(np.sum(weights))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Cannot normalize mask weights: non-positive or non-finite sum")
    weights = weights / w_sum

    entries: list[LineMaskEntry] = []
    for i in range(centers_nm.size):
        lam_nm = float(centers_nm[i])
        wt = float(weights[i])
        entries.append(LineMaskEntry(wavelength_nm=lam_nm, weight=wt))

    if masks_directory is None:
        masks_directory = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "masks")
        )
    os.makedirs(masks_directory, exist_ok=True)

    base = os.path.splitext(os.path.basename(template_path))[0]
    auto_name = f"{base}.mas"
    out = output_path if output_path is not None else os.path.join(masks_directory, auto_name)

    meta: dict = {
        "schema_version": 2,
        "generator": "hispecdrp.utils.line_mask_utils.build_line_mask_from_template",
        "template_path": os.path.abspath(template_path),
        "wl_min_um": float(wl_min_um),
        "wl_max_um": float(wl_max_um),
        "min_continuum_normalized_depth": float(min_continuum_normalized_depth),
        "low_pass_width_multiplier": float(low_pass_width_multiplier),
        "telluric_velocity_half_width_km_s": float(vhalf),
        "telluric_min_mean_transmission": float(tmin),
        "transmission_file": trans_path_abs,
        "continuum_filter_size": int(continuum_filter_size),
        "centroid_method": "gradient",
        "mask_line_geometry": "delta_two_column",
        "mask_weight_definition": "continuum_normalized_depth_1_minus_F_over_Fc",
        "weights_sum_normalized_to": 1.0,
        "deblend_merge_radius_nm": float(merge_r_nm),
        "median_template_step_nm": float(median_dw_nm),
        "local_min_fn_window_half_width_nm": float(half_win_um * 1000.0),
        "min_continuum_normalized_flux_floor": (
            None
            if min_continuum_normalized_flux_floor is None
            else float(min_continuum_normalized_flux_floor)
        ),
        "n_lines": int(len(entries)),
    }
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    header = (
        "# HISPEC DRP template-derived CCF line mask (columns: wavelength_vac_nm, weight).\n"
        f"# Template basename: {os.path.basename(template_path)}\n"
        "# Wavelength: vacuum nanometers; lines are infinitely narrow (no segment edges).\n"
        "# Weights: continuum-normalized depth at line center (see mask JSON); "
        "**normalized so the column sums to 1**.\n"
        "# Generation parameters: single-line JSON on the next line "
        "(parse with read_mask_generation_meta).\n"
        f"{LINE_MASK_META_PREFIX}{meta_json}\n"
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        for e in entries:
            f.write(f"  {e.wavelength_nm: .16e}       {e.weight: .16e}\n")

    logger.info("Wrote %d mask lines to %s", len(entries), out)
    return out, entries
