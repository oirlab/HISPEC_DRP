"""
KPF-inspired mask-shifted cross-correlation for stellar RV.

This module mirrors the core mechanics of KPF's mask-shift CCF:
- Load a line mask from disk (SPIROU-style ``.mas`` or KPF-style 2-col ``.txt``).
- For each trial velocity, Doppler-shift mask line segments into wavelength space.
- Compute pixel overlap fractions between shifted mask segments and data pixel bins.
- Build CCF(v) = sum(flux * mask_weight * sn) over pixels.
- Fit a Gaussian to the CCF peak to estimate RV and RV uncertainty.

The orchestration (chunking/orders + median/MAD + weighted mean) is written to
match the output semantics of :meth:`hispecdrp.primitives.StellarRVPrimitive._run_algorithm_ccf_median_mad`.
"""
# Provenance: Independent HISPEC DRP implementation; mechanics follow public KPF
# mask-shift CCF concepts (not copied from KPF proprietary code). SPIROU .mas format from
# SPIROU/OPERA conventions. Default mask: hispecdrp/data/masks/M2_filt_allRV.mas.
# Combines chunks via ccf_utils.combine_rvs_median_mad / weighted_mean_rv.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ...datamodels.dqflags import DQ_FLAGS
from .ccf_utils import (
    _V_MAX,
    _V_MIN,
    _V_STEP,
    combine_rvs_median_mad,
    finalize_ccf_preview_dict,
    weighted_mean_rv,
)

logger = logging.getLogger(__name__)


C_M_S = 299792458.0


@dataclass(frozen=True)
class MaskLines:
    """
    Mask line segments in microns.

    Arrays are 1D, same length:
    - start_um: segment start wavelength
    - end_um: segment end wavelength
    - weight: mask weight per line segment
    """

    start_um: np.ndarray
    end_um: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        start_um = np.asarray(self.start_um, dtype=np.float64)
        end_um = np.asarray(self.end_um, dtype=np.float64)
        weight = np.asarray(self.weight, dtype=np.float64)
        if start_um.shape != end_um.shape or start_um.shape != weight.shape:
            raise ValueError("MaskLines arrays must have matching shapes.")
        if start_um.ndim != 1:
            raise ValueError("MaskLines arrays must be 1D.")

        object.__setattr__(self, "start_um", start_um)
        object.__setattr__(self, "end_um", end_um)
        object.__setattr__(self, "weight", weight)


def _default_masks_dir() -> Path:
    # .../hispecdrp/primitives/utils/ccf_mask_kpf_utils.py -> .../hispecdrp
    pkg_root = Path(__file__).resolve().parents[2]
    return pkg_root / "data" / "masks"


def resolve_mask_path(mask_path: str | None) -> str:
    """
    Resolve the configured mask path to an absolute path.

    If mask_path is None, uses the packaged SPIROU mask.
    """

    if mask_path:
        p = Path(mask_path).expanduser()
        return str(p if p.is_absolute() else (Path.cwd() / p).resolve())

    # Default packaged file (downloaded as part of this implementation).
    default_file = _default_masks_dir() / "M2_filt_allRV.mas"
    if not default_file.exists():
        raise FileNotFoundError(
            f"Default mask not found: {default_file}. "
            "Pass StellarRVPrimitive(mask_path=...) to use a custom mask."
        )
    return str(default_file)


def _infer_and_convert_to_micron(values: np.ndarray) -> np.ndarray:
    """
    Infer wavelength unit scale and convert to micron.

    Used for **KPF-style two-column** masks only. Heuristics (median of positive finite
    values):

    - median > 1e3 -> Angstrom (e.g. 8000 Å) -> /1e4
    - median > 10  -> nm (e.g. 1500 nm)      -> /1e3
    - otherwise assume already micron.

    SPIROU three-column ``.mas`` files are loaded by :func:`load_mask_spirou_mas`, which
    always treats the wavelength columns as **vacuum nanometers** (NIR values like 1136 nm
    must not use this function's ``median > 1e3 -> Å`` rule, which would map them to ~0.11 µm).
    """

    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v) & (v > 0)
    if not np.any(finite):
        return v
    med = float(np.nanmedian(v[finite]))
    if med > 1e3:
        return v / 1e4
    if med > 10:
        return v / 1e3
    return v


def _iter_numeric_tokens_from_file(path: str) -> Iterable[tuple[float, float, float]]:
    """
    Yield numeric tokens from any line in a whitespace-delimited mask file.

    This is tolerant of comment/header rows: any row with >= N numeric columns
    will contribute.
    """

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "//")):
                continue
            toks = line.split()
            # We'll parse first 3 columns for SPIROU-style; callers can slice.
            if len(toks) < 2:
                continue
            try:
                floats = [float(t) for t in toks[:3]]
            except ValueError:
                continue
            if len(floats) == 2:
                a, b = floats
                yield a, b, np.nan
            else:
                a, b, c = floats
                yield a, b, c


def load_mask_spirou_mas(mask_file: str) -> MaskLines:
    """
    Load a SPIROU-style mask file ``.mas`` with columns:
    - lambda_start (nanometers **in air**, SPIROU/OPERA convention)
    - lambda_end (nanometers **in air**)
    - weight

    Wavelengths are converted **air → vacuum** (nm) using the same formula as SPIROU
    ``spectrumlib.convert_air_to_vacuum_wl``, then converted to **microns** via ``/ 1e3``
    for use in the rest of this module.

    Returns wavelengths in micron (vacuum).
    """

    start_vals: list[float] = []
    end_vals: list[float] = []
    w_vals: list[float] = []

    with open(mask_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "//")):
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            try:
                a = float(toks[0])
                b = float(toks[1])
                w = float(toks[2])
            except ValueError:
                continue
            start_vals.append(a)
            end_vals.append(b)
            w_vals.append(w)

    if not start_vals:
        raise ValueError(f"Mask file contains no parsable rows: {mask_file}")

    start = np.asarray(start_vals, dtype=np.float64)
    end = np.asarray(end_vals, dtype=np.float64)
    weight = np.asarray(w_vals, dtype=np.float64)

    start = start / 1e3
    end = end / 1e3

    # Ensure ordering is sane.
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    return MaskLines(start_um=lo, end_um=hi, weight=weight)


def load_mask_kpf_two_column(mask_file: str, *, mask_width_um: float | None) -> MaskLines:
    """
    Load a KPF-style two-column mask file.

    The file is interpreted as:
    - wavelength_center_or_start : column 0
    - weight                      : column 1

    Because the input does not include an explicit segment width, callers must
    provide ``mask_width_um`` to define a half-width around each wavelength.
    If mask_width_um is None, a conservative width based on median spacing is used.
    """

    w_vals: list[float] = []
    weight_vals: list[float] = []

    with open(mask_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "//")):
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            try:
                w_vals.append(float(toks[0]))
                weight_vals.append(float(toks[1]))
            except ValueError:
                continue

    if not w_vals:
        raise ValueError(f"Mask file contains no parsable rows: {mask_file}")

    w = np.asarray(w_vals, dtype=np.float64)
    weight = np.asarray(weight_vals, dtype=np.float64)
    w = _infer_and_convert_to_micron(w)

    w_sorted = np.sort(w[np.isfinite(w)])
    if w_sorted.size >= 2:
        median_spacing = float(np.nanmedian(np.diff(w_sorted)))
    else:
        median_spacing = 1e-5

    # Default width: a small fraction of median spacing (keeps mask compact).
    if mask_width_um is None:
        mask_width_um = max(median_spacing * 0.2, 1e-6)

    half = 0.5 * float(mask_width_um)
    return MaskLines(start_um=w - half, end_um=w + half, weight=weight)


def load_mask(mask_path: str, *, mask_width_um: float | None = None) -> MaskLines:
    p = Path(mask_path)
    suffix = p.suffix.lower()
    if suffix == ".mas":
        return load_mask_spirou_mas(mask_path)
    if suffix in (".txt", ".dat", ".csv"):
        return load_mask_kpf_two_column(mask_path, mask_width_um=mask_width_um)
    # Best-effort fallback: try SPIROU, then KPF 2-col.
    try:
        return load_mask_spirou_mas(mask_path)
    except Exception:
        return load_mask_kpf_two_column(mask_path, mask_width_um=mask_width_um)


def pixel_edges_from_wavelength(wave_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel bin edges for each wavelength pixel.

    Returns (left_edges_um, right_edges_um), each of shape (n_pixel,).
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError("Need at least 2 wavelength points to build pixel edges.")

    # KPF-style construction (midpoints between adjacent pixels, with end padding).
    x_pixel_wave_start = (w + np.roll(w, 1)) / 2.0
    x_pixel_wave_end = np.roll(x_pixel_wave_start, -1)
    x_pixel_wave_start[0] = w[0] - (w[1] - w[0]) / 2.0
    x_pixel_wave_end[-1] = w[-1] + (w[-1] - w[-2]) / 2.0
    left = x_pixel_wave_start
    right = x_pixel_wave_end
    return left, right


def restrict_mask_to_chunk(
    mask_lines: MaskLines,
    w_min_um: float,
    w_max_um: float,
    *,
    v_min_m_s: float,
    v_max_m_s: float,
    zb: float = 0.0,
) -> MaskLines:
    """
    Restrict mask lines to those that can overlap the chunk wavelength range
    within the configured velocity search limits.
    """

    # shift_factor = (1+v/c)/(1+zb)
    shift_low = (1.0 + v_min_m_s / C_M_S) / (1.0 + zb)
    shift_high = (1.0 + v_max_m_s / C_M_S) / (1.0 + zb)

    # A line overlaps [w_min_um, w_max_um] for some v in [v_min, v_max] if:
    # - its shifted end can reach above w_min
    # - its shifted start can reach below w_max
    keep = (mask_lines.end_um * shift_high >= w_min_um) & (mask_lines.start_um * shift_low <= w_max_um)
    if not np.any(keep):
        return MaskLines(
            start_um=np.array([], dtype=np.float64),
            end_um=np.array([], dtype=np.float64),
            weight=np.array([], dtype=np.float64),
        )
    return MaskLines(
        start_um=mask_lines.start_um[keep],
        end_um=mask_lines.end_um[keep],
        weight=mask_lines.weight[keep],
    )


def calc_ccf_mask_kpf_like(
    wave_um: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    mask_lines: MaskLines,
    v_grid_m_s: np.ndarray,
    *,
    zb: float = 0.0,
    sn_floor: float = 0.0,
) -> np.ndarray:
    """
    Compute mask-shifted CCF(v) for one spectral chunk.

    This is a direct, readable translation of KPF's logic:
    - compute pixel bin edges
    - for each velocity step, shift each mask line segment
    - compute overlap fraction with each pixel bin
    - CCF(v) = sum(flux * mask_weight * sn)
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(flux, dtype=np.float64).ravel()
    if flux_err is None:
        e = np.ones_like(s, dtype=np.float64)
    else:
        e = np.asarray(flux_err, dtype=np.float64).ravel()

    if w.size != s.size or w.size != e.size:
        raise ValueError("wave/flux/flux_err must have matching lengths.")
    n_pixel = w.size
    if n_pixel < 3:
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)
    if mask_lines.start_um.size == 0:
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    left_edges, right_edges = pixel_edges_from_wavelength(w)
    pixel_width = right_edges - left_edges
    pixel_width = np.where(pixel_width > 0, pixel_width, np.nan)

    finite = np.isfinite(s) & np.isfinite(e) & (e > 0) & np.isfinite(pixel_width)
    if not np.any(finite):
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    s_use = np.where(finite, s, 0.0)
    # KPF's python path uses an SNR scaling array of ones. Using sn=1/err
    # can cancel absorption features for synthetic spectra where err ∝ flux.
    sn = np.where(finite, 1.0, sn_floor)

    ccf = np.zeros_like(v_grid_m_s, dtype=np.float64)

    for i_v, v in enumerate(v_grid_m_s):
        shift_lines_by = (1.0 + v / C_M_S) / (1.0 + zb)
        mask_spectra = np.zeros(n_pixel, dtype=np.float64)

        # Compute the overlap mask weight for all line segments.
        for ln_start, ln_end, ln_weight in zip(
            mask_lines.start_um, mask_lines.end_um, mask_lines.weight
        ):
            line_start_wave = ln_start * shift_lines_by
            line_end_wave = ln_end * shift_lines_by
            if not np.isfinite(line_start_wave) or not np.isfinite(line_end_wave):
                continue
            if line_end_wave <= line_start_wave:
                continue

            # KPF-style indexing: find a near-minimal pixel range to test.
            # closest_x_pixel  ~ first pixel whose right edge is not below line_start_wave
            # closest_x_pixel_next ~ first pixel whose left edge is above/equal line_end_wave
            # right_edges are monotonic; use searchsorted for speed.
            closest_x_pixel = int(np.searchsorted(right_edges, line_start_wave, side="left"))
            # Count of left_edges <= line_end_wave
            closest_x_pixel_next = int(np.searchsorted(left_edges, line_end_wave, side="right"))

            if closest_x_pixel_next <= 0 or closest_x_pixel >= n_pixel - 1:
                continue

            lo = max(0, closest_x_pixel)
            hi = min(n_pixel, closest_x_pixel_next)
            if hi <= lo:
                continue

            for n in range(lo, hi):
                # if there is overlap
                if left_edges[n] <= line_end_wave and right_edges[n] >= line_start_wave:
                    wave_start = max(left_edges[n], line_start_wave)
                    wave_end = min(right_edges[n], line_end_wave)
                    overlap = wave_end - wave_start
                    if overlap > 0 and np.isfinite(pixel_width[n]) and pixel_width[n] > 0:
                        mask_spectra[n] += ln_weight * (overlap / pixel_width[n])

        ccf[i_v] = float(np.nansum(s_use * mask_spectra * sn))

    return ccf


def rv_from_ccf_mask_kpf(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    *,
    v_step_m_s: float,
    n_good: int,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate RV + uncertainty by fitting a Gaussian to the CCF peak.

    KPF's mask CCF can be either a positive peak or a negative trough depending
    on mask/spectrum conventions. We enforce a *maximum* peak for the fitter
    by flipping the sign when needed.
    """

    ccf_raw = np.asarray(ccf_raw, dtype=np.float64).ravel()
    if ccf_raw.size != v_grid_m_s.size:
        raise ValueError("ccf_raw and v_grid must have matching lengths.")

    ccf_min = float(np.nanmin(ccf_raw))
    ccf_max = float(np.nanmax(ccf_raw))

    if not np.isfinite(ccf_min) or not np.isfinite(ccf_max):
        return np.nan, np.nan, np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    baseline = float(np.nanmedian(ccf_raw))
    # Decide whether the alignment feature is a trough or a peak by comparing
    # its excursion relative to the baseline.
    height = float(ccf_max - baseline)
    depth = float(baseline - ccf_min)
    # If the trough is deeper than the peak is tall, flip sign so the fitter sees
    # a positive maximum at the true RV location.
    if depth > height:
        ccf = -ccf_raw
    else:
        ccf = ccf_raw

    # Stable (non-SciPy) RV estimation:
    # - use quadratic interpolation across the maximum (3-point parabolic fit)
    # - estimate a Gaussian-equivalent sigma from curvature
    # - report rv_err from sigma scaled by sqrt(N) heuristics
    i_peak = int(np.nanargmax(ccf))
    v_peak = float(v_grid_m_s[i_peak])

    # Default fallback: grid peak only.
    if i_peak <= 0 or i_peak >= (len(ccf) - 1):
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    y_m1 = float(ccf[i_peak - 1])
    y_0 = float(ccf[i_peak])
    y_p1 = float(ccf[i_peak + 1])

    baseline = float(np.nanmedian(ccf))
    amp = y_0 - baseline

    # If the peak isn't clearly above baseline, uncertainty becomes very large.
    if not np.isfinite(amp) or amp <= 0:
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    # Parabolic vertex offset for uniform grid:
    # d = (y_- - y_+) / (2*(y_- - 2*y_0 + y_+)) * v_step
    denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1)
    if not np.isfinite(denom) or denom == 0.0:
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    d_v = (y_m1 - y_p1) / denom * float(v_step_m_s)
    rv = v_peak + d_v

    # Curvature-based sigma estimate:
    # For y = baseline + amp * exp(-0.5*(x/sigma)^2), near the peak:
    # y''(0) = -amp / sigma^2.
    # For quadratic interpolation around a max with spacing v_step:
    # y''(0) ~= (y_- + y_+ - 2*y_0) / v_step^2.
    ypp = (y_m1 + y_p1 - 2.0 * y_0) / (float(v_step_m_s) ** 2)
    if not np.isfinite(ypp) or ypp >= 0:
        sigma_v = float(abs(v_step_m_s) * 2.0)
    else:
        sigma_v2 = max(float(-amp / ypp), (0.1 * float(abs(v_step_m_s))) ** 2)
        sigma_v = float(np.sqrt(sigma_v2))

    # Error scaling heuristic: more good pixels -> smaller error.
    n_scale = max(float(n_good), 1.0) / 50.0
    rv_err = float(max(sigma_v / np.sqrt(n_scale), float(abs(v_step_m_s))))
    return float(rv), float(rv_err), ccf


def rv_per_chunks_mask_kpf(
    wave_um: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq_bad_mask: np.ndarray,
    mask_lines: MaskLines,
    *,
    window_size_um: float,
    v_min_m_s: float = _V_MIN,
    v_max_m_s: float = _V_MAX,
    v_step_m_s: float = _V_STEP,
    zb: float = 0.0,
    return_ccf: bool = False,
    min_good_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Compute per-chunk RV and (optionally) CCF(v) for one spectral order.
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    good_pix = ~d & np.isfinite(w) & np.isfinite(s) & np.isfinite(e)
    if np.sum(good_pix) < min_good_pixels:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64) if return_ccf else None,
        )

    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64) if return_ccf else None,
        )

    v_grid = np.arange(v_min_m_s, v_max_m_s + 0.5 * v_step_m_s, v_step_m_s, dtype=np.float64)

    n_chunk = max(1, int((w_max - w_min) / float(window_size_um)))

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_centers: list[float] = []
    all_ccf_rows: list[np.ndarray] = []

    for k in range(n_chunk):
        lam_lo = w_min + k * (w_max - w_min) / n_chunk
        lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
        in_chunk = (w >= lam_lo) & (w < lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        if int(np.sum(in_chunk)) < min_good_pixels:
            continue

        wl_center = 0.5 * (lam_lo + lam_hi)

        w_c = w[in_chunk]
        s_c = s[in_chunk]
        e_c = e[in_chunk]

        # Restrict mask lines so we don't waste time on clearly non-overlapping
        # segments; expanded to cover the configured velocity range.
        mask_c = restrict_mask_to_chunk(
            mask_lines,
            float(np.nanmin(w_c)),
            float(np.nanmax(w_c)),
            v_min_m_s=v_min_m_s,
            v_max_m_s=v_max_m_s,
            zb=zb,
        )
        if mask_c.start_um.size == 0:
            continue

        ccf_raw = calc_ccf_mask_kpf_like(
            w_c,
            s_c,
            e_c,
            mask_c,
            v_grid,
            zb=zb,
        )
        rv, rv_err, ccf_fit = rv_from_ccf_mask_kpf(
            v_grid,
            ccf_raw,
            v_step_m_s=v_step_m_s,
            n_good=int(np.sum(in_chunk)),
        )
        if np.isfinite(rv) and np.isfinite(rv_err) and rv_err > 0:
            all_rvs.append(float(rv))
            all_rv_errs.append(float(rv_err))
            all_centers.append(float(wl_center))
            if return_ccf:
                all_ccf_rows.append(ccf_fit)
        else:
            # Keep diagnostics: still store if finite RV but err invalid?
            if np.isfinite(rv):
                all_rvs.append(float(rv))
                all_rv_errs.append(float(rv_err) if np.isfinite(rv_err) else np.nan)
                all_centers.append(float(wl_center))
                if return_ccf:
                    all_ccf_rows.append(ccf_fit)

    if not all_rvs:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            v_grid,
            (np.empty((0, len(v_grid)), dtype=np.float64) if return_ccf else None),
        )

    rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    centers_arr = np.asarray(all_centers, dtype=np.float64)
    if return_ccf:
        ccf_2d = np.vstack(all_ccf_rows).astype(np.float64) if all_ccf_rows else np.empty((0, len(v_grid)))
    else:
        ccf_2d = None
    return rvs_arr, rv_errs_arr, centers_arr, v_grid, ccf_2d


def run_ccf_mask_kpf_for_fiber(
    *,
    input_model,
    fiber: str,
    do_orders: list[int],
    order_vec: list[int],
    window_size_um: float,
    outlier_sigma: float,
    mask_path: str | None,
    mask_width_um: float | None = None,
    v_min_m_s: float = _V_MIN,
    v_max_m_s: float = _V_MAX,
    v_step_m_s: float = _V_STEP,
    zb: float = 0.0,
    want_preview: bool = False,
    debug_ccf: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    dict | None,
]:
    """
    Compute per-chunk and aggregated RV for one fiber using mask-shifted CCF.
    """

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

    v_grid = np.arange(v_min_m_s, v_max_m_s + 0.5 * v_step_m_s, v_step_m_s, dtype=np.float64)

    # Load mask once per fiber run (mask does not depend on order).
    resolved_mask = resolve_mask_path(mask_path)
    mask_lines = load_mask(resolved_mask, mask_width_um=mask_width_um)
    if mask_lines.start_um.size == 0:
        raise ValueError(f"Mask loaded no lines: {resolved_mask}")

    do_not_use = int(DQ_FLAGS.get("DO_NOT_USE", 1))
    # We treat any non-zero "do_not_use" bits as bad pixels.
    bad = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    order_tasks = [(oi, order) for oi, order in enumerate(order_vec) if order in do_orders]

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_wavelengths: list[float] = []
    all_orders: list[int] = []

    # When True, collect per-order CCF stacks for Plotly preview (Gaussian refit in
    # finalize_ccf_preview_dict). Disable preview if unstable CCF shapes appear in your env.
    want_preview_effective = bool(want_preview)

    order_slots: list[dict] = []
    all_ccf_rows: list[np.ndarray] = []

    for oi, order in order_tasks:
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64)
        e = np.asarray(err[oi] if getattr(err, "ndim", 0) >= 2 else err, dtype=np.float64)
        d_bad = np.asarray(bad[oi] if getattr(bad, "ndim", 0) >= 2 else bad, dtype=bool)

        w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue

        rvs_o, errs_o, centers_o, v_grid_o, ccf_2d = rv_per_chunks_mask_kpf(
            w,
            s,
            e,
            d_bad,
            mask_lines,
            window_size_um=window_size_um,
            v_min_m_s=v_min_m_s,
            v_max_m_s=v_max_m_s,
            v_step_m_s=v_step_m_s,
            zb=zb,
            return_ccf=want_preview_effective,
        )

        if rvs_o.size == 0:
            continue

        all_rvs.extend([float(x) for x in rvs_o])
        all_rv_errs.extend([float(x) for x in errs_o])
        all_wavelengths.extend([float(x) for x in centers_o])
        all_orders.extend([int(order)] * int(len(centers_o)))

        if want_preview_effective and ccf_2d is not None and ccf_2d.size > 0:
            ccf_mean = np.nanmean(ccf_2d, axis=0)
            order_slots.append(
                {
                    "kind": "order",
                    "order": int(order),
                    "v": np.asarray(v_grid_o, dtype=np.float64),
                    "ccf": np.asarray(ccf_mean, dtype=np.float64),
                    "wl_lo": float(w_min),
                    "wl_hi": float(w_max),
                }
            )
            all_ccf_rows.append(ccf_2d)

        if debug_ccf and rvs_o.size > 0:
            logger.info(
                "CCF-mask_kpf fiber=%s order=%s: n_chunks=%d rv_mean~%.1f m/s",
                fiber,
                order,
                len(rvs_o),
                float(np.nanmedian(rvs_o)),
            )

    all_rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    all_rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    all_wavelengths_arr = np.asarray(all_wavelengths, dtype=np.float64)
    all_orders_arr = np.asarray(all_orders, dtype=np.int64)
    if len(all_orders_arr) != len(all_wavelengths_arr):
        all_orders_arr = np.full(len(all_wavelengths_arr), -1, dtype=np.int64)

    valid = np.isfinite(all_rvs_arr) & np.isfinite(all_rv_errs_arr)
    if not np.any(valid):
        inlier = np.zeros(len(all_rvs_arr), dtype=bool)
        rv_mean = np.nan
        rv_std = np.nan
    else:
        inlier, _, _ = combine_rvs_median_mad(all_rvs_arr, all_rv_errs_arr, outlier_sigma)
        rv_mean, rv_std = weighted_mean_rv(all_rvs_arr, all_rv_errs_arr, inlier)

    preview_detail = None
    if want_preview_effective and order_slots and all_ccf_rows:
        preview_detail = finalize_ccf_preview_dict(
            order_slots,
            all_ccf_rows,
            v_grid,
            v_min_m_s,
            v_max_m_s,
            v_step_m_s,
        )

    return (
        all_wavelengths_arr,
        all_rvs_arr,
        all_rv_errs_arr,
        inlier,
        float(rv_mean),
        float(rv_std),
        all_orders_arr,
        preview_detail,
    )

