"""Trace-profile moments for spectroperfection extraction.

Spatial profile (0th/2nd spatial moment) comes from a fiberflat rate map.
Iso-λ shear and spectral 2nd moment come from LFC spots: each comb peak is a
tilted 2-D PSF whose covariance encodes ``∂row/∂col`` and the line-spread
width along the dispersion axis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

from ...datamodels import hispec_config
from ...datamodels.tracepos import TracePosModel
from ...datamodels.traceprof import TraceProfModel

logger = logging.getLogger(__name__)

__all__ = [
    "spot_window_moments",
    "measure_lfc_spot_moments_along_trace",
    "measure_fiberflat_spatial_sigma",
    "interpolate_moments_to_pixels",
    "fill_traceprof_spectroperfection_moments",
]


def _n_orders_n_pix(
    spectrograph: str | None,
    *,
    n_pix: int = 4096,
    n_orders: int | None = None,
) -> tuple[int, int]:
    if n_orders is None:
        sp = str(spectrograph or "BSPEC").upper()
        n_orders = len(list(hispec_config.HISPECConfig.get_order_list(sp)))
    return int(n_orders), int(n_pix)


def spot_window_moments(
    rate: np.ndarray,
    row: float,
    col: float,
    *,
    half_win: int = 6,
) -> dict[str, float]:
    """
    Flux-weighted 1st/2nd moments of a 2-D window about ``(row, col)``.

    Oriented detector: spectral axis is ``row``, cross-dispersion is ``col``.
    Shear is ``cov(row, col) / var(col)`` = ``∂row/∂col`` of the iso-λ ridge.
    """
    rate = np.asarray(rate, dtype=np.float64)
    ny, nx = rate.shape
    r0 = int(np.rint(row))
    c0 = int(np.rint(col))
    hw = int(half_win)
    rlo, rhi = max(0, r0 - hw), min(ny, r0 + hw + 1)
    clo, chi = max(0, c0 - hw), min(nx, c0 + hw + 1)
    stamp = np.maximum(rate[rlo:rhi, clo:chi], 0.0)
    total = float(np.sum(stamp))
    nan = {
        "row": float("nan"),
        "col": float("nan"),
        "shear": float("nan"),
        "sigma_spec": float("nan"),
        "sigma_spat": float("nan"),
        "mom2_cross": float("nan"),
        "flux": total,
        "n_pix": int(stamp.size),
    }
    if not np.isfinite(total) or total <= 0.0 or stamp.size < 9:
        return nan
    rr = np.arange(rlo, rhi, dtype=np.float64)[:, None]
    cc = np.arange(clo, chi, dtype=np.float64)[None, :]
    mu_r = float(np.sum(stamp * rr) / total)
    mu_c = float(np.sum(stamp * cc) / total)
    var_r = float(np.sum(stamp * (rr - mu_r) ** 2) / total)
    var_c = float(np.sum(stamp * (cc - mu_c) ** 2) / total)
    cov_rc = float(np.sum(stamp * (rr - mu_r) * (cc - mu_c)) / total)
    if not (np.isfinite(var_c) and var_c > 1.0e-8):
        shear = float("nan")
        sig_spec = float(np.sqrt(max(var_r, 0.0)))
    else:
        shear = cov_rc / var_c
        # Spectral width at fixed cross-disp (residual after the tilt).
        sig_spec = float(np.sqrt(max(var_r - shear * shear * var_c, 0.0)))
    sig_spat = float(np.sqrt(max(var_c, 0.0)))
    return {
        "row": mu_r,
        "col": mu_c,
        "shear": float(shear),
        "sigma_spec": sig_spec,
        "sigma_spat": sig_spat,
        "mom2_cross": cov_rc,
        "flux": total,
        "n_pix": int(stamp.size),
    }


def measure_lfc_spot_moments_along_trace(
    rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float = 8.0,
    half_win: int = 6,
    min_prominence_frac: float = 0.05,
    min_separation_pix: int = 8,
    expected_xy: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Measure LFC spot moments along one order's trace.

    If ``expected_xy`` is given (deposit ``(row, col)`` pairs), those centers
    are used. Otherwise peaks are found on the aperture-summed spectral profile.
    """
    rate = np.asarray(rate, dtype=np.float64)
    ty = np.asarray(trace_y, dtype=np.float64)
    ny, nx = rate.shape
    spots: list[tuple[float, float]] = []
    if expected_xy:
        spots = [(float(r), float(c)) for r, c in expected_xy]
    else:
        collapsed = np.zeros(ny, dtype=np.float64)
        half = 0.5 * float(extraction_aperture)
        for x in range(ny):
            yc = ty[x]
            if not np.isfinite(yc):
                continue
            y0 = max(0, int(np.floor(yc - half)))
            y1 = min(nx, int(np.ceil(yc + half)) + 1)
            collapsed[x] = float(np.sum(np.maximum(rate[x, y0:y1], 0.0)))
        peak = float(np.nanmax(collapsed)) if collapsed.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            return []
        idx, _props = find_peaks(
            collapsed,
            prominence=float(min_prominence_frac) * peak,
            distance=max(1, int(min_separation_pix)),
        )
        for x in idx:
            yc = ty[int(x)]
            if np.isfinite(yc):
                spots.append((float(x), float(yc)))

    out: list[dict[str, Any]] = []
    for row, col in spots:
        mom = spot_window_moments(rate, row, col, half_win=half_win)
        mom["row_seed"] = float(row)
        mom["col_seed"] = float(col)
        out.append(mom)
    return out


def measure_fiberflat_spatial_sigma(
    rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float = 8.0,
) -> np.ndarray:
    """Per-spectral-row spatial Gaussian σ from a fiberflat rate map."""
    rate = np.asarray(rate, dtype=np.float64)
    ty = np.asarray(trace_y, dtype=np.float64)
    ny, nx = rate.shape
    sig = np.full(ny, np.nan, dtype=np.float64)
    half = 0.5 * float(extraction_aperture)
    for x in range(ny):
        yc = ty[x]
        if not np.isfinite(yc):
            continue
        y0 = max(0, int(np.floor(yc - half)))
        y1 = min(nx, int(np.ceil(yc + half)) + 1)
        if y1 - y0 < 3:
            continue
        yy = np.arange(y0, y1, dtype=np.float64)
        w = np.maximum(rate[x, y0:y1], 0.0)
        tot = float(np.sum(w))
        if tot <= 0.0:
            continue
        mu = float(np.sum(w * yy) / tot)
        var = float(np.sum(w * (yy - mu) ** 2) / tot)
        sig[x] = float(np.sqrt(max(var, 0.0)))
    return sig


def interpolate_moments_to_pixels(
    spot_rows: np.ndarray,
    values: np.ndarray,
    n_pix: int,
) -> np.ndarray:
    """PCHIP-interpolate finite spot samples onto ``0 .. n_pix-1``."""
    x = np.asarray(spot_rows, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    out = np.full(int(n_pix), np.nan, dtype=np.float64)
    if int(np.sum(ok)) < 1:
        return out
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.maximum(np.bincount(inv), 1)
    pix = np.arange(int(n_pix), dtype=np.float64)
    if ux.size == 1:
        out[:] = float(uy[0])
        return out
    f = PchipInterpolator(ux, uy, extrapolate=True)
    out[:] = np.asarray(f(pix), dtype=np.float64)
    med = float(np.nanmedian(out))
    if np.isfinite(med):
        out[~np.isfinite(out)] = med
    return out


def fill_traceprof_spectroperfection_moments(
    traceprof: TraceProfModel,
    *,
    fiberflat_rate: np.ndarray,
    lfc_rate: np.ndarray,
    tracepos: TracePosModel,
    spectrograph: str,
    fiber: int,
    orders: list[int] | None = None,
    extraction_aperture: float = 8.0,
    half_win: int = 6,
    expected_xy_by_order: dict[int, list[tuple[float, float]]] | None = None,
) -> TraceProfModel:
    """
    Write shear / σ_spec (LFC) and σ_spat (fiberflat) into ``traceprof``.

    ``PROF_FIB`` is left unchanged (fiberflat image). Moment HDUs are
    ``(n_orders, n_pix)`` on the same spectral-row grid as ``TracePos``.
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    order_list = list(
        orders
        if orders is not None
        else hispec_config.HISPECConfig.get_order_list(spectrograph)
    )
    ty_all = np.asarray(getattr(tracepos, f"pos_fib{fiber}"), dtype=np.float64)
    n_orders, n_pix = ty_all.shape
    ff = np.asarray(fiberflat_rate, dtype=np.float64)
    lfc = np.asarray(lfc_rate, dtype=np.float64)

    shear = np.full((n_orders, n_pix), np.nan, dtype=np.float64)
    sig_spec = np.full((n_orders, n_pix), np.nan, dtype=np.float64)
    sig_spat = np.full((n_orders, n_pix), np.nan, dtype=np.float64)
    mom2 = np.full((n_orders, n_pix), np.nan, dtype=np.float64)

    ost = int(hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"])
    for od in order_list:
        j = int(od) - ost
        if j < 0 or j >= n_orders:
            continue
        ty = ty_all[j]
        expected = None
        if expected_xy_by_order is not None:
            expected = expected_xy_by_order.get(int(od))
        spots = measure_lfc_spot_moments_along_trace(
            lfc,
            ty,
            extraction_aperture=extraction_aperture,
            half_win=half_win,
            expected_xy=expected,
        )
        if spots:
            rows = np.array([s["row"] for s in spots], dtype=np.float64)
            shear[j] = interpolate_moments_to_pixels(
                rows, np.array([s["shear"] for s in spots]), n_pix
            )
            sig_spec[j] = interpolate_moments_to_pixels(
                rows, np.array([s["sigma_spec"] for s in spots]), n_pix
            )
            mom2[j] = interpolate_moments_to_pixels(
                rows, np.array([s["mom2_cross"] for s in spots]), n_pix
            )
            logger.info(
                "LFC moments fiber %s order %s: n_spots=%s median shear=%.4f "
                "sigma_spec=%.3f",
                fiber,
                od,
                len(spots),
                float(np.nanmedian(shear[j])),
                float(np.nanmedian(sig_spec[j])),
            )
        sig_spat[j] = measure_fiberflat_spatial_sigma(
            ff, ty, extraction_aperture=extraction_aperture
        )

    setattr(traceprof, f"shear_fib{fiber}", shear)
    setattr(traceprof, f"sigma_spec_fib{fiber}", sig_spec)
    setattr(traceprof, f"sigma_spat_fib{fiber}", sig_spat)
    setattr(traceprof, f"mom2_cross_fib{fiber}", mom2)
    return traceprof
