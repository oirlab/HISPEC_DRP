"""Full spectroperfectionism: sparse banded LS with HDF PSF deposit operator.

The forward operator matches :func:`_deposit_oversampled_psf_numba` (native
detector) followed by the sim orientation map used by Stage-1
(``orders_are_vertical=True`` / :func:`_orient_hdf_pixel_xy`). Extraction
solves ``(Aᵀ N⁻¹ A) f = Aᵀ N⁻¹ d`` with SciPy sparse factorization.

Memory notes
------------
HDF PSF stamps are large (``~256×256`` oversampled). Full-order assembly of
every stamp weight for all ``4096`` spectral samples can exceed 16 GB. Prefer:

- ``pix_range=(lo, hi)`` for windowed extracts (readiness / fidelity),
- ``psf_rel_threshold`` to crop near-zero stamp wings,
- a single process-wide HDF cache (see ``e2e_pipeline.clear_hdf_model_cache``).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import factorized


@njit(fastmath=True)
def _emit_oriented_deposit_weights(
    out_row: np.ndarray,
    out_col: np.ndarray,
    out_val: np.ndarray,
    psf: np.ndarray,
    xc: float,
    yc: float,
    oversample: float,
    nx_native: int,
    abs_threshold: float,
) -> int:
    """
    Write oriented ``(row, col, weight)`` samples for one normalized PSF deposit.

    Skips ``|psf| < abs_threshold``. Returns the number of written entries.
    Capacity must be ``>=`` number of above-threshold samples × 4.
    """
    H, W = psf.shape
    y0 = yc * oversample - (H * 0.5 + 1.0)
    x0 = xc * oversample - (W * 0.5 + 1.0)
    n = 0
    n_max = out_row.size
    for j in range(H):
        y_s = (j + y0) / oversample
        i0 = int(np.floor(y_s))
        dy = y_s - i0
        for i in range(W):
            val = psf[j, i]
            if val < abs_threshold and val > -abs_threshold:
                continue
            x_s = (i + x0) / oversample
            j0 = int(np.floor(x_s))
            dx = x_s - j0
            for di in range(2):
                iy = i0 + di
                wy = (1.0 - dy) if di == 0 else dy
                if wy == 0.0 or iy < 0 or iy >= nx_native:
                    continue
                for dj in range(2):
                    ix = j0 + dj
                    wx = (1.0 - dx) if dj == 0 else dx
                    if wx == 0.0 or ix < 0 or ix >= nx_native:
                        continue
                    w = val * wy * wx
                    if w == 0.0:
                        continue
                    if n >= n_max:
                        return n
                    out_row[n] = (nx_native - 1) - ix
                    out_col[n] = (nx_native - 1) - iy
                    out_val[n] = w
                    n += 1
    return n


def crop_psf_stamp(
    stamp: np.ndarray,
    *,
    rel_threshold: float = 1.0e-2,
) -> tuple[np.ndarray, float, float]:
    """
    Crop a PSF stamp to the bounding box of significant flux.

    Returns ``(cropped, dx_det, dy_det)`` where the detector-pixel offsets
    restore the original stamp-center → ``(xc, yc)`` deposit convention after
    the array shrinks (add to ``xc`` / ``yc`` before deposit).
    """
    psf = np.asarray(stamp, dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError(f"stamp must be 2-D; got {psf.shape}")
    h, w = psf.shape
    peak = float(np.max(np.abs(psf)))
    if not np.isfinite(peak) or peak <= 0.0:
        return psf, 0.0, 0.0
    msk = np.abs(psf) >= float(rel_threshold) * peak
    if not np.any(msk):
        return psf, 0.0, 0.0
    ys, xs = np.where(msk)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = np.ascontiguousarray(psf[y0:y1, x0:x1])
    # Original geometric center in full-stamp index space.
    cy0 = 0.5 * (h - 1)
    cx0 = 0.5 * (w - 1)
    hc, wc = cropped.shape
    cy1 = cy0 - y0
    cx1 = cx0 - x0
    # Deposit centers on 0.5*(H_new-1); shift (xc, yc) to compensate.
    dy_os = cy1 - 0.5 * (hc - 1)
    dx_os = cx1 - 0.5 * (wc - 1)
    return cropped, float(dx_os), float(dy_os)


@njit(fastmath=True)
def deposit_psf_to_oriented(
    det_oriented: np.ndarray,
    psf: np.ndarray,
    xc: float,
    yc: float,
    oversample: float,
    scale: float,
    nx_native: int,
) -> None:
    """Deposit one scaled PSF into an oriented rate map (in-place)."""
    H, W = psf.shape
    ny_o, nx_o = det_oriented.shape
    y0 = yc * oversample - (H * 0.5 + 1.0)
    x0 = xc * oversample - (W * 0.5 + 1.0)
    for j in range(H):
        y_s = (j + y0) / oversample
        i0 = int(np.floor(y_s))
        dy = y_s - i0
        for i in range(W):
            x_s = (i + x0) / oversample
            j0 = int(np.floor(x_s))
            dx = x_s - j0
            val = psf[j, i] * scale
            if val == 0.0:
                continue
            for di in range(2):
                iy = i0 + di
                wy = (1.0 - dy) if di == 0 else dy
                if wy == 0.0 or iy < 0 or iy >= nx_native:
                    continue
                for dj in range(2):
                    ix = j0 + dj
                    wx = (1.0 - dx) if dj == 0 else dx
                    if wx == 0.0 or ix < 0 or ix >= nx_native:
                        continue
                    orow = (nx_native - 1) - ix
                    ocol = (nx_native - 1) - iy
                    if 0 <= orow < ny_o and 0 <= ocol < nx_o:
                        det_oriented[orow, ocol] += val * wy * wx


def _psf_oversample(block: dict, pixelsize: float) -> float:
    spacing = float(block.get("psf_data_spacing", pixelsize))
    if spacing <= 0.0:
        return 1.0
    return float(pixelsize) / spacing


def _nearest_psf_index(psfw: np.ndarray, lam: float) -> int:
    return int(np.argmin(np.abs(psfw - float(lam))))


def forward_psf_deposit_oriented(
    spec: np.ndarray,
    wave: np.ndarray,
    *,
    fx_native,
    fy_native,
    psfw: np.ndarray,
    psf_stamps,
    oversample: float,
    nx_native: int,
    det_shape: tuple[int, int] | None = None,
    psf_rel_threshold: float = 1.0e-2,
) -> np.ndarray:
    """Forward model ``M = A f`` via direct oriented deposits."""
    spec = np.asarray(spec, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    if det_shape is None:
        det = np.zeros((int(nx_native), int(nx_native)), dtype=np.float64)
    else:
        det = np.zeros(det_shape, dtype=np.float64)
    psfw = np.asarray(psfw, dtype=np.float64)
    cropped: dict[int, tuple[np.ndarray, float, float]] = {}
    for s in range(wave.size):
        fx = float(spec[s])
        if not np.isfinite(fx) or fx == 0.0:
            continue
        lam = float(wave[s])
        if not np.isfinite(lam):
            continue
        xc = float(fx_native(lam))
        yc = float(fy_native(lam))
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        idx = _nearest_psf_index(psfw, lam)
        if idx not in cropped:
            psf = np.asarray(psf_stamps[idx], dtype=np.float64)
            psf_sum = float(np.sum(psf))
            if psf_sum <= 0.0:
                continue
            c, dx, dy = crop_psf_stamp(
                psf / psf_sum, rel_threshold=psf_rel_threshold
            )
            cropped[idx] = (c, dx / float(oversample), dy / float(oversample))
        psf_n, dx_det, dy_det = cropped[idx]
        deposit_psf_to_oriented(
            det,
            psf_n,
            xc + dx_det,
            yc + dy_det,
            float(oversample),
            fx,
            int(nx_native),
        )
    return det


def extract_spectrum_spectroperfectionism(
    rate: np.ndarray,
    rate_var: np.ndarray,
    wave: np.ndarray,
    *,
    fx_native,
    fy_native,
    psfw: np.ndarray,
    psf_stamps,
    oversample: float,
    nx_native: int,
    spectral_bandwidth: int | None = None,
    pix_range: tuple[int, int] | None = None,
    psf_rel_threshold: float = 1.0e-2,
    rate_dq: np.ndarray | None = None,
    do_not_use: int = 1,
) -> dict[str, np.ndarray]:
    """
    Banded weighted LS extract with HDF PSF deposit operator.

    ``rate`` / ``rate_var`` are oriented (spectral rows × cross-disp cols).

    Parameters
    ----------
    pix_range
        Optional ``(i0, i1)`` half-open spectral-row window to solve. Use this
        on 16 GB machines; full ``4096``-pixel orders with ``256×256`` stamps
        can exhaust RAM.
    psf_rel_threshold
        Crop each stamp to flux ≥ this fraction of the stamp peak before
        depositing (default ``1e-2``).
    """
    rate = np.asarray(rate, dtype=np.float64)
    rate_var = np.asarray(rate_var, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    psfw = np.asarray(psfw, dtype=np.float64)
    ny, nx = rate.shape
    n_spec = int(wave.size)
    nx_native = int(nx_native)

    if len(psf_stamps) == 0:
        raise ValueError("No PSF stamps provided")

    if pix_range is None:
        i0, i1 = 0, n_spec
    else:
        i0 = max(0, int(pix_range[0]))
        i1 = min(n_spec, int(pix_range[1]))
    if i1 <= i0:
        raise ValueError(f"empty pix_range={pix_range}")

    # Crop only stamps needed for this spectral window (full-order pre-crop of
    # every 256×256 HDF stamp can exhaust RAM on 16 GB machines).
    cropped: dict[int, tuple[np.ndarray, float, float]] = {}

    def _crop_stamp(idx: int) -> tuple[np.ndarray, float, float] | None:
        hit = cropped.get(idx)
        if hit is not None:
            return hit
        psf = np.asarray(psf_stamps[idx], dtype=np.float64)
        psf_sum = float(np.sum(psf))
        if psf_sum <= 0.0 or not np.isfinite(psf_sum):
            return None
        c, dx, dy = crop_psf_stamp(
            psf / psf_sum, rel_threshold=float(psf_rel_threshold)
        )
        item = (c, dx / float(oversample), dy / float(oversample))
        cropped[idx] = item
        return item

    needed: set[int] = set()
    for s in range(i0, i1):
        lam = float(wave[s])
        if not np.isfinite(lam):
            continue
        needed.add(_nearest_psf_index(psfw, lam))
    max_h = max_w = 1
    for idx in sorted(needed):
        item = _crop_stamp(idx)
        if item is None:
            continue
        max_h = max(max_h, item[0].shape[0])
        max_w = max(max_w, item[0].shape[1])

    max_per_col = int(max_h * max_w * 4)
    if spectral_bandwidth is None:
        spectral_bandwidth = max(
            8, int(np.ceil(0.5 * max_w / float(oversample))) + 4
        )
    half_b = int(spectral_bandwidth)

    n_win = i1 - i0
    est_nnz = int(n_win) * max_per_col
    # ~40M float64 COO triples ≈ 1 GB; keep headroom for AtA + HDF cache.
    if est_nnz > 20_000_000:
        raise MemoryError(
            f"SP extract window too large for safe memory use: "
            f"pix_range=[{i0},{i1}) ≈ {est_nnz:.2e} COO weights "
            f"({len(cropped)} stamps). Pass a shorter pix_range "
            f"(e.g. 64–128 pixels) or raise psf_rel_threshold."
        )

    pix_key_to_i: dict[tuple[int, int], int] = {}
    obs_data: list[float] = []
    obs_ivar: list[float] = []

    def _pix_index(r: int, c: int) -> int | None:
        key = (r, c)
        if key in pix_key_to_i:
            return pix_key_to_i[key]
        if r < 0 or c < 0 or r >= ny or c >= nx:
            return None
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

    buf_r = np.empty(max_per_col, dtype=np.int64)
    buf_c = np.empty(max_per_col, dtype=np.int64)
    buf_v = np.empty(max_per_col, dtype=np.float64)

    cap = min(est_nnz, max(65536, n_win * 4096))
    coo_r = np.empty(cap, dtype=np.int32)
    coo_c = np.empty(cap, dtype=np.int32)
    coo_v = np.empty(cap, dtype=np.float64)
    n_coo = 0

    def _append_coo(iobs: int, scol: int, w: float) -> None:
        nonlocal n_coo, coo_r, coo_c, coo_v, cap
        if n_coo >= cap:
            cap = int(cap * 1.5) + 1
            coo_r = np.resize(coo_r, cap)
            coo_c = np.resize(coo_c, cap)
            coo_v = np.resize(coo_v, cap)
        coo_r[n_coo] = iobs
        coo_c[n_coo] = scol
        coo_v[n_coo] = w
        n_coo += 1

    abs_thr = 0.0
    for s in range(i0, i1):
        lam = float(wave[s])
        if not np.isfinite(lam):
            continue
        xc = float(fx_native(lam))
        yc = float(fy_native(lam))
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        idx = _nearest_psf_index(psfw, lam)
        item = _crop_stamp(idx)
        if item is None:
            continue
        psf_n, dx_det, dy_det = item
        n_w = _emit_oriented_deposit_weights(
            buf_r,
            buf_c,
            buf_v,
            psf_n,
            xc + dx_det,
            yc + dy_det,
            float(oversample),
            nx_native,
            abs_thr,
        )
        local_col = s - i0
        for k in range(n_w):
            orow = int(buf_r[k])
            ocol = int(buf_c[k])
            if abs(orow - s) > half_b:
                continue
            iobs = _pix_index(orow, ocol)
            if iobs is None:
                continue
            _append_coo(iobs, local_col, float(buf_v[k]))

    # Drop cropped stamps before allocating AtA (can be tens–hundreds of MB).
    del cropped, buf_r, buf_c, buf_v

    spec = np.full(n_spec, np.nan, dtype=np.float64)
    spec_err = np.full(n_spec, np.nan, dtype=np.float64)
    if n_coo == 0:
        return {
            "spec": spec,
            "err": spec_err,
            "pix_range": (i0, i1),
            "n_stamps": len(needed),
        }

    A = sparse.coo_matrix(
        (coo_v[:n_coo], (coo_r[:n_coo], coo_c[:n_coo])),
        shape=(len(obs_data), n_win),
    ).tocsr()
    del coo_r, coo_c, coo_v, pix_key_to_i

    ivar = np.asarray(obs_ivar, dtype=np.float64)
    d = np.asarray(obs_data, dtype=np.float64)
    del obs_data, obs_ivar
    sqrt_ivar = np.sqrt(ivar)
    A_w = A.multiply(sqrt_ivar[:, None])
    AtA = (A_w.T @ A_w).tocsc()
    Atb = A.T @ (ivar * d)

    try:
        solve = factorized(AtA)
        f = solve(Atb)
    except Exception:
        f = sparse.linalg.lsqr(A_w, sqrt_ivar * d, atol=1e-12, btol=1e-12)[0]

    spec[i0:i1] = f
    diag = np.asarray(AtA.diagonal(), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_win = np.where(diag > 0.0, 1.0 / np.sqrt(diag), np.nan)
    spec_err[i0:i1] = err_win
    return {
        "spec": spec,
        "err": spec_err,
        "pix_range": (i0, i1),
        "n_stamps": int(len(needed)),
        "n_coo": int(n_coo),
    }


def extract_order_spectroperfectionism_from_hdf(
    rate: np.ndarray,
    rate_var: np.ndarray,
    *,
    spectrograph: str,
    fiber: int,
    order: int,
    wave: np.ndarray | None = None,
    spectral_bandwidth: int | None = None,
    pix_range: tuple[int, int] | None = None,
    psf_rel_threshold: float = 1.0e-2,
    rate_dq: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """SP extract one echelle order using HDF PSF stamps + deposit-inverse WAVE."""
    from ...simulations.e2e_pipeline import (
        _load_hdf_model_data,
        get_sim_hdf_trace_grids,
    )

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    order = int(order)
    _, model_data = _load_hdf_model_data(spectrograph)
    pixelsize = float(model_data.get("pixelsize", 1.0))
    nx = int(model_data.get("nx", rate.shape[0]))
    block = model_data[f"fiber_{fiber}"][order]
    ow = np.asarray(block["xy_wavelengths"], dtype=np.float64)
    ox = np.asarray(block["x"], dtype=np.float64)
    oy = np.asarray(block["y"], dtype=np.float64)
    psfw = np.asarray(block["psf_wavelengths"], dtype=np.float64)
    psf_stamps = block["psf_stamps"]
    oversample = _psf_oversample(block, pixelsize)
    fx = PchipInterpolator(ow, ox, extrapolate=True)
    fy = PchipInterpolator(ow, oy, extrapolate=True)

    if wave is None:
        grids = get_sim_hdf_trace_grids(spectrograph)
        wave = np.asarray(grids[fiber][order]["wavelength"], dtype=np.float64)

    return extract_spectrum_spectroperfectionism(
        rate,
        rate_var,
        wave,
        fx_native=fx,
        fy_native=fy,
        psfw=psfw,
        psf_stamps=psf_stamps,
        oversample=oversample,
        nx_native=nx,
        spectral_bandwidth=spectral_bandwidth,
        pix_range=pix_range,
        psf_rel_threshold=psf_rel_threshold,
        rate_dq=rate_dq,
    )
