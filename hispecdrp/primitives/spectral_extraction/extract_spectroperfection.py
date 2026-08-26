"""Spectroperfection extract from TraceProf shear + LSF moments.

Forward operator: each spectral sample deposits a sheared spatial profile
(from the fiberflat) along iso-λ. Default ``spectral_kernel='bilinear'``
inverts that pixel-grid operator (LSF geometry via the measured shear).
``spectral_kernel='gaussian'`` also inverts an LFC-measured spectral LSF.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .extraction_utils import get_extraction_window
from .extract_sheared_sp import _profile_on_cut, iter_sheared_scatter_weights

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _gauss_pdf(dx: np.ndarray | float, sigma: float) -> np.ndarray | float:
    sig = max(float(sigma), 1.0e-3)
    z = np.asarray(dx, dtype=np.float64) / sig
    return np.exp(-0.5 * z * z) / (sig * _SQRT_2PI)


def forward_spectroperfection(
    spec: np.ndarray,
    profile_img: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    sigma_spec: np.ndarray | None = None,
    lsf_n_sigma: float = 5.0,
    pix_range: tuple[int, int] | None = None,
    spectral_kernel: str = "bilinear",
) -> np.ndarray:
    """Deposit ``spec`` with the spectroperfection operator (``d = A f``)."""
    spec = np.asarray(spec, dtype=np.float64)
    profile_img = np.asarray(profile_img, dtype=np.float64)
    ny, nx = profile_img.shape
    model = np.zeros((ny, nx), dtype=np.float64)
    kernel = str(spectral_kernel).lower()
    if kernel in ("bilinear", "pixel", "sheared"):
        x0, x1 = _pix_lo_hi(ny, pix_range)
        for x, y, r0, r1, w0, w1, _pix_w in iter_sheared_scatter_weights(
            trace_y,
            profile_img,
            extraction_aperture=extraction_aperture,
            shear=shear,
        ):
            if x < x0 or x >= x1:
                continue
            fx = float(spec[x])
            if not np.isfinite(fx):
                continue
            model[r0, y] += fx * w0
            if r1 != r0 and w1 != 0.0:
                model[r1, y] += fx * w1
        return model
    for j, y, x, w in iter_spectroperfection_weights(
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        shear=shear,
        sigma_spec=sigma_spec,
        lsf_n_sigma=lsf_n_sigma,
        pix_range=pix_range,
    ):
        model[j, y] += w * float(spec[x])
    return model


def _pix_lo_hi(ny: int, pix_range: tuple[int, int] | None) -> tuple[int, int]:
    if pix_range is None:
        return 0, int(ny)
    return max(0, int(pix_range[0])), min(int(ny), int(pix_range[1]) + 1)


def iter_spectroperfection_weights(
    trace_y: np.ndarray,
    profile_img: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None,
    sigma_spec: np.ndarray | None,
    lsf_n_sigma: float = 5.0,
    pix_range: tuple[int, int] | None = None,
):
    """
    Yield ``(j, y, x, weight)``: detector spectral row ``j``, cross-disp ``y``,
    source spectral sample ``x``.
    """
    profile_img = np.asarray(profile_img, dtype=np.float64)
    trace_y = np.asarray(trace_y, dtype=np.float64)
    ny, nx = profile_img.shape
    shear_arr = (
        np.zeros(ny, dtype=np.float64)
        if shear is None
        else np.asarray(shear, dtype=np.float64)
    )
    sig_arr = (
        np.full(ny, 0.55, dtype=np.float64)
        if sigma_spec is None
        else np.asarray(sigma_spec, dtype=np.float64)
    )
    nsig = float(lsf_n_sigma)
    x0, x1 = _pix_lo_hi(ny, pix_range)
    for x in range(x0, x1):
        yc = float(trace_y[x])
        if not np.isfinite(yc):
            continue
        s = float(shear_arr[x]) if np.isfinite(shear_arr[x]) else 0.0
        sig = float(sig_arr[x]) if np.isfinite(sig_arr[x]) else 0.55
        sig = max(sig, 1.0e-3)
        window_inds, W = get_extraction_window(yc, float(extraction_aperture), nx)
        if window_inds.size == 0:
            continue
        if abs(s) > 1.0e-12:
            P = _profile_on_cut(profile_img, x, window_inds, yc, s)
        else:
            P = np.maximum(profile_img[x, window_inds], 0.0)
        spat = P * np.asarray(W, dtype=np.float64)
        spat_sum = float(np.sum(spat))
        if spat_sum <= 0.0 or not np.isfinite(spat_sum):
            continue
        spat = spat / spat_sum
        for k, y in enumerate(window_inds):
            w_spat = float(spat[k])
            if w_spat <= 0.0:
                continue
            x_iso = float(x) + s * (float(y) - yc)
            j0 = int(np.floor(x_iso - nsig * sig))
            j1 = int(np.ceil(x_iso + nsig * sig))
            j0 = max(0, j0)
            j1 = min(ny - 1, j1)
            if j1 < j0:
                continue
            js = np.arange(j0, j1 + 1, dtype=np.float64)
            w_lsf = np.asarray(_gauss_pdf(js - x_iso, sig), dtype=np.float64)
            lsf_sum = float(np.sum(w_lsf))
            if lsf_sum <= 0.0:
                continue
            w_lsf = w_lsf / lsf_sum
            for t, j in enumerate(range(j0, j1 + 1)):
                w = w_spat * float(w_lsf[t])
                if w > 0.0:
                    yield int(j), int(y), int(x), w


def extract_spectroperfection(
    rate: np.ndarray,
    rate_var: np.ndarray,
    profile_img: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    sigma_spec: np.ndarray | None = None,
    lsf_n_sigma: float = 5.0,
    rate_dq: np.ndarray | None = None,
    do_not_use: int = 1,
    pix_range: tuple[int, int] | None = None,
    spectral_kernel: str = "bilinear",
    ridge_frac: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Banded weighted LS spectroperfection extract.

    ``profile_img`` is the 2-D fiberflat spatial profile. ``shear`` is the
    iso-λ tilt from LFC spots. The default ``spectral_kernel='bilinear'``
    inverts the sheared pixel-grid deposit (same 1-D basis as box/optimal);
    ``'gaussian'`` additionally inverts an LFC-measured spectral LSF and
    needs ``ridge_frac`` to stay stable on real PSFs.
    """
    rate = np.asarray(rate, dtype=np.float64)
    rate_var = np.asarray(rate_var, dtype=np.float64)
    profile_img = np.asarray(profile_img, dtype=np.float64)
    ny, nx = rate.shape
    kernel = str(spectral_kernel).lower()
    if ridge_frac is None:
        ridge_frac = 1.0e-3 if kernel in ("gaussian", "lsf") else 0.0

    pix_key_to_i: dict[tuple[int, int], int] = {}
    obs_data: list[float] = []
    obs_ivar: list[float] = []

    def _pix_index(r: int, c: int) -> int | None:
        key = (r, c)
        if key in pix_key_to_i:
            return pix_key_to_i[key]
        val = float(rate[r, c])
        var = float(rate_var[r, c])
        if rate_dq is not None and (int(rate_dq[r, c]) & int(do_not_use)) != 0:
            return None
        if not np.isfinite(val) or not np.isfinite(var) or var <= 0.0:
            return None
        idx = len(obs_data)
        pix_key_to_i[key] = idx
        obs_data.append(val)
        obs_ivar.append(1.0 / var)
        return idx

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    x0, x1 = _pix_lo_hi(ny, pix_range)
    if kernel in ("bilinear", "pixel", "sheared"):
        for x, y, r0, r1, w0, w1, _pix_w in iter_sheared_scatter_weights(
            trace_y,
            profile_img,
            extraction_aperture=extraction_aperture,
            shear=shear,
        ):
            if x < x0 or x >= x1:
                continue
            if w0 != 0.0:
                iobs = _pix_index(r0, y)
                if iobs is not None:
                    rows.append(iobs)
                    cols.append(int(x))
                    vals.append(float(w0))
            if r1 != r0 and w1 != 0.0:
                iobs = _pix_index(r1, y)
                if iobs is not None:
                    rows.append(iobs)
                    cols.append(int(x))
                    vals.append(float(w1))
    elif kernel in ("gaussian", "lsf"):
        for j, y, x, w in iter_spectroperfection_weights(
            trace_y,
            profile_img,
            extraction_aperture=extraction_aperture,
            shear=shear,
            sigma_spec=sigma_spec,
            lsf_n_sigma=lsf_n_sigma,
            pix_range=pix_range,
        ):
            iobs = _pix_index(j, y)
            if iobs is None:
                continue
            rows.append(iobs)
            cols.append(int(x))
            vals.append(float(w))
    else:
        raise ValueError(
            f"Unknown spectral_kernel {spectral_kernel!r}; "
            "use 'bilinear' or 'gaussian'."
        )

    spec = np.full(ny, np.nan, dtype=np.float64)
    spec_err = np.full(ny, np.nan, dtype=np.float64)
    if not rows:
        return {"spec": spec, "err": spec_err}

    n_obs = len(obs_data)
    A = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(n_obs, ny),
    ).tocsr()
    ivar = np.asarray(obs_ivar, dtype=np.float64)
    d = np.asarray(obs_data, dtype=np.float64)
    sqrt_ivar = np.sqrt(ivar)
    A_w = A.multiply(sqrt_ivar[:, None])
    AtA = (A_w.T @ A_w).tocsc()
    Atb = A.T @ (ivar * d)
    diag = np.asarray(AtA.diagonal(), dtype=np.float64)
    active = diag > 0.0
    if not np.any(active):
        return {"spec": spec, "err": spec_err}
    idx = np.where(active)[0]
    AtA_a = AtA[idx][:, idx]
    Atb_a = Atb[idx]
    if float(ridge_frac) > 0.0:
        ridge = float(ridge_frac) * float(np.median(diag[idx]))
        if np.isfinite(ridge) and ridge > 0.0:
            AtA_a = AtA_a + ridge * sparse.eye(idx.size, format="csc")
    solve = factorized(AtA_a.tocsc())
    f_a = np.asarray(solve(Atb_a), dtype=np.float64)
    spec[idx] = f_a
    spec_err[idx] = 1.0 / np.sqrt(np.maximum(diag[idx], 1.0e-30))
    return {"spec": spec, "err": spec_err}
