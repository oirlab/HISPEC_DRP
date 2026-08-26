"""
Numba-parallel Gaussian (+ fixed baseline) CCF centroid for line-mask chunk RVs.

Replaces the serial Astropy :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`
loop in :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask` with a
``prange`` over chunks. Model matches the line-mask stored (peak-up) convention:
fixed baseline + positive-amplitude Gaussian; two velocity passes; optional
inverse-variance weights from ``ccf_err``.
"""
from __future__ import annotations

import math

import numpy as np

from ....utils.parallelization_utils import njit, numba_thread_scope, prange

__all__ = [
    "fit_ccf_rv_rows_numba",
]


@njit(cache=True, inline="always")
def _nanmedian_1d(a: np.ndarray) -> float:
    n = a.size
    tmp = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        v = a[i]
        if np.isfinite(v):
            tmp[m] = v
            m += 1
    if m == 0:
        return np.nan
    # insertion sort of the finite prefix
    for i in range(1, m):
        key = tmp[i]
        j = i - 1
        while j >= 0 and tmp[j] > key:
            tmp[j + 1] = tmp[j]
            j -= 1
        tmp[j + 1] = key
    if m % 2 == 1:
        return float(tmp[m // 2])
    return 0.5 * (float(tmp[m // 2 - 1]) + float(tmp[m // 2]))


@njit(cache=True, inline="always")
def _mad_1d(a: np.ndarray, med: float) -> float:
    n = a.size
    tmp = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        v = a[i]
        if np.isfinite(v):
            tmp[m] = abs(v - med)
            m += 1
    if m == 0:
        return np.nan
    for i in range(1, m):
        key = tmp[i]
        j = i - 1
        while j >= 0 and tmp[j] > key:
            tmp[j + 1] = tmp[j]
            j -= 1
        tmp[j + 1] = key
    if m % 2 == 1:
        mad = float(tmp[m // 2])
    else:
        mad = 0.5 * (float(tmp[m // 2 - 1]) + float(tmp[m // 2]))
    return 1.4826 * mad


@njit(cache=True)
def _sigma_clipped_median(y: np.ndarray, n_sigma: float, max_iter: int) -> float:
    n = y.size
    keep = np.ones(n, dtype=np.bool_)
    med = _nanmedian_1d(y)
    if not np.isfinite(med):
        return np.nan
    for _ in range(max_iter):
        mad = _mad_1d(y, med)
        if not np.isfinite(mad) or mad <= 0.0:
            break
        thr = n_sigma * mad
        changed = False
        for i in range(n):
            if keep[i] and np.isfinite(y[i]) and abs(y[i] - med) > thr:
                keep[i] = False
                changed = True
        if not changed:
            break
        tmp = np.empty(n, dtype=np.float64)
        m = 0
        for i in range(n):
            if keep[i] and np.isfinite(y[i]):
                tmp[m] = y[i]
                m += 1
        if m < 3:
            break
        med = _nanmedian_1d(tmp[:m])
        if not np.isfinite(med):
            break
    return med


@njit(cache=True, inline="always")
def _gauss(v: float, amp: float, mu: float, sig: float, baseline: float) -> float:
    if sig <= 0.0 or not np.isfinite(sig):
        return baseline
    x = (v - mu) / sig
    return amp * math.exp(-0.5 * x * x) + baseline


@njit(cache=True)
def _lm_gauss_fixed_baseline(
    v: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    use_w: bool,
    amp0: float,
    mu0: float,
    sig0: float,
    baseline: float,
    maxiter: int,
) -> tuple[float, float, float, float, bool]:
    """
    Weighted LM for ``y = A·exp(-½((v-μ)/σ)²) + B`` with **fixed** ``B``.

    Returns ``(mu, mu_err, amp, sig, ok)``. ``mu_err`` from ``sqrt(H^{-1}_{μμ})`` when
    the approximate Hessian is positive-definite; else NaN.
    """
    n = v.size
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, False

    amp = float(amp0)
    mu = float(mu0)
    sig = float(abs(sig0))
    if not (np.isfinite(amp) and np.isfinite(mu) and np.isfinite(sig) and sig > 0.0):
        return np.nan, np.nan, np.nan, np.nan, False
    if amp < 0.0:
        amp = abs(amp)

    # Scale y for numerical stability (same idea as Astropy path).
    ymax = 0.0
    for i in range(n):
        ai = abs(y[i] - baseline)
        if np.isfinite(ai) and ai > ymax:
            ymax = ai
    ys = 1.0 / ymax if ymax > 0.0 else 1.0
    amp *= ys
    b_s = baseline * ys

    lam = 1e-3
    prev_chi2 = 1e300
    ok = False

    for _it in range(maxiter):
        # Accumulate J^T W J (3x3) and J^T W r
        H00 = H01 = H02 = H11 = H12 = H22 = 0.0
        g0 = g1 = g2 = 0.0
        chi2 = 0.0
        n_used = 0
        for i in range(n):
            vi = v[i]
            yi = y[i] * ys
            if not (np.isfinite(vi) and np.isfinite(yi)):
                continue
            wi = 1.0
            if use_w:
                wi = w[i]
                if not (np.isfinite(wi) and wi > 0.0):
                    continue
            if sig <= 1e-30:
                continue
            x = (vi - mu) / sig
            e = math.exp(-0.5 * x * x)
            model = amp * e + b_s
            r = yi - model
            # d model / d amp, mu, sig
            ja = e
            jm = amp * e * x / sig
            js = amp * e * x * x / sig
            chi2 += wi * r * r
            n_used += 1
            H00 += wi * ja * ja
            H01 += wi * ja * jm
            H02 += wi * ja * js
            H11 += wi * jm * jm
            H12 += wi * jm * js
            H22 += wi * js * js
            g0 += wi * ja * r
            g1 += wi * jm * r
            g2 += wi * js * r

        if n_used < 5:
            return np.nan, np.nan, np.nan, np.nan, False

        # Already at (or extremely near) a minimum: accept seed / current params.
        if chi2 <= 1e-24 * max(1.0, float(n_used)) and np.isfinite(mu):
            ok = True
            break

        # (H + λ diag(H)) δ = g
        a00 = H00 * (1.0 + lam)
        a11 = H11 * (1.0 + lam)
        a22 = H22 * (1.0 + lam)
        a01 = H01
        a02 = H02
        a12 = H12

        # Solve 3x3 via Cramer's rule / explicit inverse
        det = (
            a00 * (a11 * a22 - a12 * a12)
            - a01 * (a01 * a22 - a12 * a02)
            + a02 * (a01 * a12 - a11 * a02)
        )
        if not np.isfinite(det) or abs(det) < 1e-30 * (abs(a00) + abs(a11) + abs(a22) + 1.0):
            lam *= 10.0
            if lam > 1e12:
                break
            continue

        inv00 = (a11 * a22 - a12 * a12) / det
        inv01 = (a02 * a12 - a01 * a22) / det
        inv02 = (a01 * a12 - a02 * a11) / det
        inv11 = (a00 * a22 - a02 * a02) / det
        inv12 = (a01 * a02 - a00 * a12) / det
        inv22 = (a00 * a11 - a01 * a01) / det

        d0 = inv00 * g0 + inv01 * g1 + inv02 * g2
        d1 = inv01 * g0 + inv11 * g1 + inv12 * g2
        d2 = inv02 * g0 + inv12 * g1 + inv22 * g2

        amp_try = amp + d0
        mu_try = mu + d1
        sig_try = sig + d2
        if amp_try < 0.0:
            amp_try = 0.5 * amp
        if sig_try <= 1e-30:
            sig_try = 0.5 * sig

        chi2_try = 0.0
        for i in range(n):
            vi = v[i]
            yi = y[i] * ys
            if not (np.isfinite(vi) and np.isfinite(yi)):
                continue
            wi = 1.0
            if use_w:
                wi = w[i]
                if not (np.isfinite(wi) and wi > 0.0):
                    continue
            x = (vi - mu_try) / sig_try
            e = math.exp(-0.5 * x * x)
            r = yi - (amp_try * e + b_s)
            chi2_try += wi * r * r

        # Accept non-increasing χ² (equality covers an already-optimal seed).
        if chi2_try <= chi2 * (1.0 + 1e-15) + 1e-30:
            amp = amp_try
            mu = mu_try
            sig = sig_try
            lam = max(lam * 0.3, 1e-7)
            ok = True
            if abs(prev_chi2 - chi2_try) < 1e-12 * (1.0 + chi2_try):
                prev_chi2 = chi2_try
                break
            prev_chi2 = chi2_try
        else:
            lam *= 10.0
            if lam > 1e12:
                break

    # Finite params are usable even if LM never improved (perfect seed).
    if not (np.isfinite(mu) and np.isfinite(amp) and np.isfinite(sig) and sig > 0.0):
        return np.nan, np.nan, np.nan, np.nan, False
    ok = True

    # Final Hessian (un-damped) for μ uncertainty; rescale amp back
    H00 = H01 = H02 = H11 = H12 = H22 = 0.0
    for i in range(n):
        vi = v[i]
        yi = y[i] * ys
        if not (np.isfinite(vi) and np.isfinite(yi)):
            continue
        wi = 1.0
        if use_w:
            wi = w[i]
            if not (np.isfinite(wi) and wi > 0.0):
                continue
        x = (vi - mu) / sig
        e = math.exp(-0.5 * x * x)
        ja = e
        jm = amp * e * x / sig
        js = amp * e * x * x / sig
        H00 += wi * ja * ja
        H01 += wi * ja * jm
        H02 += wi * ja * js
        H11 += wi * jm * jm
        H12 += wi * jm * js
        H22 += wi * js * js

    det = (
        H00 * (H11 * H22 - H12 * H12)
        - H01 * (H01 * H22 - H12 * H02)
        + H02 * (H01 * H12 - H11 * H02)
    )
    mu_err = np.nan
    if np.isfinite(det) and abs(det) > 1e-30:
        # cov_μμ = (H^{-1})_{11}
        c11 = (H00 * H22 - H02 * H02) / det
        if np.isfinite(c11) and c11 > 0.0:
            mu_err = math.sqrt(c11)

    amp_out = amp / ys
    return float(mu), float(mu_err), float(amp_out), float(sig), True


@njit(cache=True)
def _window_gauss_fit(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray,
    use_err: bool,
    center: float,
    hw: float,
    sig_guess: float,
    maxiter: int,
    min_points: int,
) -> tuple[float, float, float, float, bool]:
    """Fit Gaussian in ``|v-center|<=hw``. Returns ``(mu, mu_err, amp, sig, ok)``."""
    n = v_grid.size
    m = 0
    for i in range(n):
        vi = v_grid[i]
        if np.isfinite(vi) and np.isfinite(ccf[i]) and abs(vi - center) <= hw:
            m += 1
    if m < min_points:
        return np.nan, np.nan, np.nan, np.nan, False
    vv = np.empty(m, dtype=np.float64)
    yy = np.empty(m, dtype=np.float64)
    ww = np.empty(m, dtype=np.float64)
    j = 0
    for i in range(n):
        vi = v_grid[i]
        yi = ccf[i]
        if not (np.isfinite(vi) and np.isfinite(yi)):
            continue
        if abs(vi - center) > hw:
            continue
        vv[j] = vi
        yy[j] = yi
        if use_err and ccf_err.size == n and np.isfinite(ccf_err[i]) and ccf_err[i] > 0.0:
            ww[j] = 1.0 / (ccf_err[i] * ccf_err[i])
        else:
            ww[j] = 1.0
        j += 1
    n_w = 0
    for k in range(m):
        if ww[k] != 1.0:
            n_w += 1
    use_w = use_err and n_w >= min_points

    baseline = _sigma_clipped_median(yy, 3.0, 5)
    if not np.isfinite(baseline):
        baseline = _nanmedian_1d(yy)
    amp0 = yy[0] - baseline
    for k in range(m):
        a = yy[k] - baseline
        if a > amp0:
            amp0 = a
    if amp0 <= 0.0:
        amp0 = abs(_nanmedian_1d(yy) - baseline)
        if amp0 <= 0.0:
            amp0 = 1e-30

    mu_s = center
    sig_s = abs(sig_guess)
    mu = np.nan
    mu_err = np.nan
    amp = np.nan
    sig = np.nan
    ok = False
    for _clip in range(4):
        mu, mu_err, amp, sig, ok = _lm_gauss_fixed_baseline(
            vv, yy, ww, use_w, amp0, mu_s, sig_s, baseline, maxiter
        )
        if not ok:
            return np.nan, np.nan, np.nan, np.nan, False
        res = np.empty(m, dtype=np.float64)
        for k in range(m):
            res[k] = yy[k] - _gauss(vv[k], amp, mu, sig, baseline)
        rmed = _nanmedian_1d(res)
        rmad = _mad_1d(res, rmed)
        if not np.isfinite(rmad) or rmad <= 0.0:
            return mu, mu_err, amp, sig, True
        thr = 4.0 * rmad
        m2 = 0
        for k in range(m):
            if abs(res[k] - rmed) <= thr:
                m2 += 1
        if m2 < min_points or m2 == m:
            return mu, mu_err, amp, sig, True
        vv2 = np.empty(m2, dtype=np.float64)
        yy2 = np.empty(m2, dtype=np.float64)
        ww2 = np.empty(m2, dtype=np.float64)
        j2 = 0
        for k in range(m):
            if abs(res[k] - rmed) <= thr:
                vv2[j2] = vv[k]
                yy2[j2] = yy[k]
                ww2[j2] = ww[k]
                j2 += 1
        vv = vv2
        yy = yy2
        ww = ww2
        m = m2
        amp0 = amp
        mu_s = mu
        sig_s = sig
        baseline = _sigma_clipped_median(yy, 3.0, 3)
        if not np.isfinite(baseline):
            baseline = _nanmedian_1d(yy)
    return mu, mu_err, amp, sig, True


@njit(cache=True)
def _fit_one_ccf_row(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray,
    use_err: bool,
    seed_m_s: float,
    pass1_hw_m_s: float,
    pass2_hw_m_s: float,
    init_sig_m_s: float,
    maxiter: int,
    min_points: int,
) -> tuple[float, float, bool]:
    """Peak-up CCF row → ``(rv_m_s, rv_err_m_s, ok)``.

    Internal LM runs in **km s⁻¹** (Astropy ``fit_ccf_rv`` convention) for Hessian
    conditioning on broad ±500 km/s grids.
    """
    n = v_grid.size
    if ccf.size != n or n < min_points:
        return np.nan, np.nan, False

    n_fin = 0
    for i in range(n):
        if np.isfinite(v_grid[i]) and np.isfinite(ccf[i]):
            n_fin += 1
    if n_fin < min_points:
        return np.nan, np.nan, False

    # Work in km/s for the fitter.
    v_k = np.empty(n, dtype=np.float64)
    for i in range(n):
        v_k[i] = v_grid[i] * 0.001
    seed_k = seed_m_s * 0.001 if np.isfinite(seed_m_s) else np.nan
    pass1_hw_k = pass1_hw_m_s * 0.001
    pass2_hw_k = pass2_hw_m_s * 0.001
    init_sig_k = init_sig_m_s * 0.001

    v_lo = 1e300
    v_hi = -1e300
    for i in range(n):
        if np.isfinite(v_k[i]) and np.isfinite(ccf[i]):
            vi = v_k[i]
            if vi < v_lo:
                v_lo = vi
            if vi > v_hi:
                v_hi = vi
    span = v_hi - v_lo
    if not (np.isfinite(span) and span > 0.0):
        return np.nan, np.nan, False

    i_peak = -1
    best = -1e300
    if np.isfinite(seed_k) and v_lo <= seed_k <= v_hi:
        half_w = max(3.5, 0.38 * span)
        for i in range(n):
            vi = v_k[i]
            yi = ccf[i]
            if not (np.isfinite(vi) and np.isfinite(yi)):
                continue
            if vi < seed_k - half_w or vi > seed_k + half_w:
                continue
            if yi > best:
                best = yi
                i_peak = i
    if i_peak < 0:
        for i in range(n):
            yi = ccf[i]
            if np.isfinite(yi) and np.isfinite(v_k[i]) and yi > best:
                best = yi
                i_peak = i
    if i_peak < 0:
        return np.nan, np.nan, False
    mu0 = float(v_k[i_peak])

    eps = max(1e-3, 0.002 * span)
    hw1 = min(pass1_hw_k, max(eps, 0.48 * span))
    hw2 = min(pass2_hw_k, max(eps, 0.42 * span))

    mu1, err1, _amp1, sig1, ok1 = _window_gauss_fit(
        v_k, ccf, ccf_err, use_err, mu0, hw1, init_sig_k, maxiter, min_points
    )
    if not ok1:
        return np.nan, np.nan, False
    sig2 = sig1 if np.isfinite(sig1) and sig1 > 0.0 else init_sig_k
    mu2, err2, _a2, _s2, ok2 = _window_gauss_fit(
        v_k, ccf, ccf_err, use_err, mu1, hw2, sig2, maxiter, min_points
    )
    if ok2 and np.isfinite(mu2):
        mu_f, err_f = mu2, err2
    else:
        mu_f, err_f = mu1, err1

    if not np.isfinite(mu_f):
        return np.nan, np.nan, False
    # Convert km/s → m/s. Floor formal σ so a near-singular Hessian cannot report ~0.
    mu_m = float(mu_f) * 1000.0
    dv_m = (span * 1000.0) / max(n_fin - 1, 1)
    err_floor = max(abs(dv_m) * 0.5, 1.0)
    if np.isfinite(err_f) and err_f > 0.0:
        err_m = max(float(err_f) * 1000.0, err_floor)
    else:
        err_m = max(abs(dv_m) * 2.0 / math.sqrt(float(n_fin)), err_floor)
    return mu_m, float(err_m), True


@njit(parallel=True, cache=True)
def _fit_ccf_rv_rows_numba_impl(
    v_grid: np.ndarray,
    ccf_rows: np.ndarray,
    ccf_err_rows: np.ndarray,
    use_err: bool,
    seed_m_s: float,
    pass1_hw_m_s: float,
    pass2_hw_m_s: float,
    init_sig_m_s: float,
    maxiter: int,
    min_points: int,
    out_rv: np.ndarray,
    out_rv_err: np.ndarray,
    out_ok: np.ndarray,
) -> None:
    n_chunk = ccf_rows.shape[0]
    for i in prange(n_chunk):
        rv, rv_err, ok = _fit_one_ccf_row(
            v_grid,
            ccf_rows[i],
            ccf_err_rows[i],
            use_err,
            seed_m_s,
            pass1_hw_m_s,
            pass2_hw_m_s,
            init_sig_m_s,
            maxiter,
            min_points,
        )
        out_rv[i] = rv
        out_rv_err[i] = rv_err
        out_ok[i] = 1 if ok else 0


def fit_ccf_rv_rows_numba(
    v_grid_m_s: np.ndarray,
    ccf_rows: np.ndarray,
    *,
    ccf_err_rows: np.ndarray | None = None,
    gaussian_seed_mean_m_s: float | None = None,
    pass1_half_width_m_s: float = 800_000.0,
    pass2_half_width_m_s: float = 120_000.0,
    initial_stddev_m_s: float = 5_000.0,
    levmar_maxiter: int = 450,
    min_points: int = 5,
    max_cores: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Gaussian centroids for peak-up CCF rows.

    Parameters
    ----------
    v_grid_m_s :
        Shared velocity grid (m s⁻¹), shape ``(n_v,)``.
    ccf_rows :
        CCF values, shape ``(n_chunk, n_v)``, **peak-up** (absorption as positive lobe).
    ccf_err_rows :
        Optional per-sample σ, same shape as ``ccf_rows``.
    gaussian_seed_mean_m_s :
        Optional discrete-peak seed center (m s⁻¹).
    max_cores :
        Numba thread cap for ``prange``.

    Returns
    -------
    rv, rv_err, ok :
        Arrays length ``n_chunk`` (``ok`` is int64 0/1).
    """
    vg = np.ascontiguousarray(np.asarray(v_grid_m_s, dtype=np.float64).ravel())
    rows = np.ascontiguousarray(np.asarray(ccf_rows, dtype=np.float64))
    if rows.ndim != 2:
        raise ValueError("ccf_rows must be 2-D (n_chunk, n_v)")
    n_chunk, n_v = rows.shape
    if int(vg.size) != int(n_v):
        raise ValueError("v_grid_m_s length must match ccf_rows.shape[1]")

    use_err = False
    if ccf_err_rows is not None:
        er = np.ascontiguousarray(np.asarray(ccf_err_rows, dtype=np.float64))
        if er.shape == rows.shape:
            use_err = True
        else:
            er = np.ones_like(rows)
    else:
        er = np.ones_like(rows)

    seed = (
        float(gaussian_seed_mean_m_s)
        if gaussian_seed_mean_m_s is not None and math.isfinite(float(gaussian_seed_mean_m_s))
        else float("nan")
    )
    out_rv = np.full(n_chunk, np.nan, dtype=np.float64)
    out_err = np.full(n_chunk, np.nan, dtype=np.float64)
    out_ok = np.zeros(n_chunk, dtype=np.int64)

    with numba_thread_scope(max_cores):
        _fit_ccf_rv_rows_numba_impl(
            vg,
            rows,
            er,
            bool(use_err),
            float(seed),
            float(pass1_half_width_m_s),
            float(pass2_half_width_m_s),
            float(initial_stddev_m_s),
            int(levmar_maxiter),
            int(min_points),
            out_rv,
            out_err,
            out_ok,
        )
    return out_rv, out_err, out_ok
