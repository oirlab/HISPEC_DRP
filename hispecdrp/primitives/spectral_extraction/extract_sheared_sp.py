"""Iso-λ sheared separable forward model + banded LS extract (spectroperfectionism-lite).

Horne gather/scatter are **not** inverses when neighboring iso-λ cuts overlap.
For a controlled self-check the world must be the scatter operator ``A``, and
extraction must solve the normal equations ``(A^T N^{-1} A) f = A^T N^{-1} d``.
Noiseless ``d = A f`` then round-trips to machine precision.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .extraction_utils import get_extraction_window


def _profile_on_cut(
    profile_img: np.ndarray,
    x: int,
    window_inds: np.ndarray,
    yc: float,
    shear: float,
) -> np.ndarray:
    """Sample continuum profile along the iso-λ cut (linear in spectral axis)."""
    ny = profile_img.shape[0]
    P = np.empty(window_inds.size, dtype=np.float64)
    s = float(shear)
    for k, y in enumerate(window_inds):
        x_frac = float(x) + s * (float(y) - float(yc))
        if x_frac <= 0.0:
            P[k] = max(float(profile_img[0, y]), 0.0)
        elif x_frac >= ny - 1:
            P[k] = max(float(profile_img[ny - 1, y]), 0.0)
        else:
            i0 = int(np.floor(x_frac))
            t = x_frac - i0
            P[k] = max(
                (1.0 - t) * float(profile_img[i0, y])
                + t * float(profile_img[i0 + 1, y]),
                0.0,
            )
    return P


def iter_sheared_scatter_weights(
    trace_y: np.ndarray,
    profile_img: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None,
):
    """
    Yield ``(x, y, row0, row1, w0, w1, pix_weight)`` scatter terms.

    ``pix_weight`` is ``PP * W`` for the aperture sample; ``w0/w1`` split that
    weight across spectral rows ``row0/row1`` (bilinear).
    """
    profile_img = np.asarray(profile_img, dtype=np.float64)
    trace_y = np.asarray(trace_y, dtype=np.float64)
    ny, nx = profile_img.shape
    shear_arr = (
        np.zeros(ny, dtype=np.float64)
        if shear is None
        else np.asarray(shear, dtype=np.float64)
    )
    for x in range(ny):
        yc = float(trace_y[x])
        if not np.isfinite(yc):
            continue
        s = float(shear_arr[x]) if np.isfinite(shear_arr[x]) else 0.0
        window_inds, W = get_extraction_window(yc, float(extraction_aperture), nx)
        if window_inds.size == 0:
            continue
        if abs(s) > 1.0e-12:
            P = _profile_on_cut(profile_img, x, window_inds, yc, s)
        else:
            P = np.maximum(profile_img[x, window_inds], 0.0)
        denom = float(np.sum(P * W))
        if denom <= 0.0 or not np.isfinite(denom):
            continue
        PP = P / denom
        for k, y in enumerate(window_inds):
            pix_w = float(PP[k] * W[k])
            if pix_w <= 0.0:
                continue
            x_frac = float(x) + s * (float(y) - yc)
            if x_frac <= 0.0:
                yield x, int(y), 0, 0, pix_w, 0.0, pix_w
            elif x_frac >= ny - 1:
                yield x, int(y), ny - 1, ny - 1, pix_w, 0.0, pix_w
            else:
                i0 = int(np.floor(x_frac))
                t = x_frac - i0
                yield x, int(y), i0, i0 + 1, pix_w * (1.0 - t), pix_w * t, pix_w


def forward_sheared_separable(
    spec: np.ndarray,
    profile_img: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separable forward model ``M = A f`` matching the sheared scatter operator.

    Returns ``(model, aperture_mask)``.
    """
    spec = np.asarray(spec, dtype=np.float64)
    profile_img = np.asarray(profile_img, dtype=np.float64)
    ny, nx = profile_img.shape
    model = np.zeros((ny, nx), dtype=np.float64)
    mask = np.zeros((ny, nx), dtype=bool)
    for x, y, r0, r1, w0, w1, _pix_w in iter_sheared_scatter_weights(
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        shear=shear,
    ):
        fx = float(spec[x])
        if not np.isfinite(fx):
            continue
        model[r0, y] += fx * w0
        mask[r0, y] = True
        if r1 != r0 and w1 != 0.0:
            model[r1, y] += fx * w1
            mask[r1, y] = True
    return model, mask


def extract_sheared_banded_ls(
    rate: np.ndarray,
    rate_var: np.ndarray,
    profile_img: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    rate_dq: np.ndarray | None = None,
    do_not_use: int = 1,
) -> dict[str, np.ndarray]:
    """
    Banded weighted LS extract: ``f = argmin ||N^{-1/2}(d - A f)||``.

    ``rate_var`` is per-pixel variance (``rate_err**2``). Pixels with
    non-finite data, non-positive variance, or ``DO_NOT_USE`` are dropped.
    """
    rate = np.asarray(rate, dtype=np.float64)
    rate_var = np.asarray(rate_var, dtype=np.float64)
    profile_img = np.asarray(profile_img, dtype=np.float64)
    ny, nx = rate.shape

    # Collect COO contributions and pixel observations.
    # Map (row, col) -> local observation index.
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

    for x, y, r0, r1, w0, w1, _pix_w in iter_sheared_scatter_weights(
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        shear=shear,
    ):
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

    # Normal equations with sqrt(ivar) weighting via diagonal.
    # AtA = A^T diag(ivar) A,  Atb = A^T diag(ivar) d
    sqrt_ivar = np.sqrt(ivar)
    A_w = A.multiply(sqrt_ivar[:, None])
    AtA = (A_w.T @ A_w).tocsc()
    Atb = A.T @ (ivar * d)

    # Drop empty spectral columns (no support).
    diag = np.asarray(AtA.diagonal(), dtype=np.float64)
    active = diag > 0.0
    if not np.any(active):
        return {"spec": spec, "err": spec_err}

    idx = np.where(active)[0]
    AtA_a = AtA[idx][:, idx]
    Atb_a = Atb[idx]
    solve = factorized(AtA_a.tocsc())
    f_a = solve(Atb_a)
    # Diagonal covariance approx from inverse diagonal of AtA (cheap bound).
    # Use 1/diag for err when off-diagonals exist (slightly optimistic).
    err_a = np.sqrt(1.0 / np.maximum(diag[idx], 1.0e-30))

    spec[idx] = f_a
    spec_err[idx] = err_a
    return {"spec": spec, "err": spec_err}
