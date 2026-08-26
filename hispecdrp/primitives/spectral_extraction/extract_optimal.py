"""Optimal spectral extraction (Horne), with optional iso-λ shear."""

from __future__ import annotations

from numba import njit
import numpy as np

from ...datamodels.dqflags import DQ_FLAGS
from .extraction_utils import get_extraction_window

DO_NOT_USE = DQ_FLAGS["DO_NOT_USE"]


@njit(nogil=True, parallel=False, cache=True)
def _interp_spectral_sample(
    arr: np.ndarray,
    x_frac: float,
    y: int,
    orders_are_vertical: bool,
) -> float:
    """Linear interpolate ``arr`` along the spectral axis at cross-disp ``y``."""
    if orders_are_vertical:
        n_spec = arr.shape[0]
        if x_frac <= 0.0:
            return float(arr[0, y])
        if x_frac >= n_spec - 1:
            return float(arr[n_spec - 1, y])
        i0 = int(np.floor(x_frac))
        t = x_frac - i0
        return float((1.0 - t) * arr[i0, y] + t * arr[i0 + 1, y])
    n_spec = arr.shape[1]
    if x_frac <= 0.0:
        return float(arr[y, 0])
    if x_frac >= n_spec - 1:
        return float(arr[y, n_spec - 1])
    i0 = int(np.floor(x_frac))
    t = x_frac - i0
    return float((1.0 - t) * arr[y, i0] + t * arr[y, i0 + 1])


@njit(nogil=True, parallel=False, cache=True)
def _nearest_dq_sample(
    dq: np.ndarray,
    x_frac: float,
    y: int,
    orders_are_vertical: bool,
) -> np.uint32:
    if orders_are_vertical:
        n_spec = dq.shape[0]
        i0 = int(np.floor(x_frac + 0.5))
        if i0 < 0:
            i0 = 0
        if i0 > n_spec - 1:
            i0 = n_spec - 1
        return np.uint32(dq[i0, y])
    n_spec = dq.shape[1]
    i0 = int(np.floor(x_frac + 0.5))
    if i0 < 0:
        i0 = 0
    if i0 > n_spec - 1:
        i0 = n_spec - 1
    return np.uint32(dq[y, i0])


def extract_spectrum_optimal(
    input_data: np.ndarray,
    input_err: np.ndarray,
    input_dq: np.ndarray,
    input_var_rnoise: np.ndarray,
    input_var_poisson: np.ndarray,
    input_mid_time: np.ndarray,
    trace_positions: np.ndarray,
    profile: np.ndarray,
    extraction_aperture: float,
    pix_range: tuple | None = None,
    spec_output: np.ndarray | None = None,
    err_output: np.ndarray | None = None,
    var_rnoise_output: np.ndarray | None = None,
    var_poisson_output: np.ndarray | None = None,
    dq_output: np.ndarray | None = None,
    mid_time_output: np.ndarray | None = None,
    orders_are_vertical: bool = True,
    shear: np.ndarray | None = None,
) -> dict:
    """
    Optimal spectral extraction.

    Parameters
    ----------
    shear : np.ndarray | None
        Optional iso-λ shear ``∂(spectral)/∂(cross-disp)`` per spectral sample
        (same length as ``trace_positions``). When set, data/profile are sampled
        along ``x' = x + shear[x] * (y - y_c)`` before the Horne sum.
    """
    ny, nx = input_data.shape
    if pix_range is None:
        if orders_are_vertical:
            pix_range = (0, ny - 1)
            npix = ny
        else:
            pix_range = (0, nx - 1)
            npix = nx
    else:
        npix = (ny if orders_are_vertical else nx)

    data_dtype = np.asarray(input_data).dtype
    if input_err is None:
        input_err = np.zeros((ny, nx), dtype=data_dtype)
    if input_var_rnoise is None:
        input_var_rnoise = np.zeros((ny, nx), dtype=data_dtype)
    if input_var_poisson is None:
        input_var_poisson = np.zeros((ny, nx), dtype=data_dtype)
    if input_dq is None:
        input_dq = np.zeros((ny, nx), dtype=np.uint32)
    if input_mid_time is None:
        input_mid_time = np.zeros((ny, nx), dtype=np.float64)

    if spec_output is None:
        spec_output = np.full(npix, np.nan, dtype=data_dtype)
    if err_output is None:
        err_output = np.full(npix, np.nan, dtype=input_err.dtype)
    if var_rnoise_output is None:
        var_rnoise_output = np.full(npix, np.nan, dtype=input_var_rnoise.dtype)
    if var_poisson_output is None:
        var_poisson_output = np.full(npix, np.nan, dtype=input_var_poisson.dtype)
    if dq_output is None:
        dq_output = np.full(npix, 0, dtype=input_dq.dtype)
    if mid_time_output is None:
        mid_time_output = np.full(npix, np.nan, dtype=input_mid_time.dtype)

    trace_positions = np.asarray(trace_positions, dtype=np.float64)
    if shear is None:
        shear_arr = np.zeros(trace_positions.shape[0], dtype=np.float64)
        use_shear = False
    else:
        shear_arr = np.asarray(shear, dtype=np.float64)
        if shear_arr.shape[0] != trace_positions.shape[0]:
            raise ValueError(
                "shear must match trace_positions length "
                f"({shear_arr.shape[0]} != {trace_positions.shape[0]})"
            )
        use_shear = bool(np.any(np.isfinite(shear_arr) & (np.abs(shear_arr) > 1.0e-12)))

    if use_shear:
        _extract_optimal_sheared_numba(
            input_data=np.asarray(input_data, dtype=np.float64),
            input_err=np.asarray(input_err, dtype=np.float64),
            input_dq=np.asarray(input_dq, dtype=np.uint32),
            input_var_rnoise=np.asarray(input_var_rnoise, dtype=np.float64),
            input_var_poisson=np.asarray(input_var_poisson, dtype=np.float64),
            input_mid_time=np.asarray(input_mid_time, dtype=np.float64),
            trace_positions=trace_positions,
            profile=np.asarray(profile, dtype=np.float64),
            shear=shear_arr,
            extraction_aperture=float(extraction_aperture),
            pix_range=pix_range,
            spec_output=spec_output.astype(np.float64, copy=False),
            err_output=err_output.astype(np.float64, copy=False),
            var_rnoise_output=var_rnoise_output.astype(np.float64, copy=False),
            var_poisson_output=var_poisson_output.astype(np.float64, copy=False),
            dq_output=dq_output,
            mid_time_output=mid_time_output.astype(np.float64, copy=False),
            orders_are_vertical=orders_are_vertical,
        )
    else:
        _extract_optimal_numba(
            input_data=input_data,
            input_err=input_err,
            input_dq=input_dq,
            input_var_rnoise=input_var_rnoise,
            input_var_poisson=input_var_poisson,
            input_mid_time=input_mid_time,
            trace_positions=trace_positions,
            profile=profile,
            extraction_aperture=extraction_aperture,
            pix_range=pix_range,
            spec_output=spec_output,
            err_output=err_output,
            var_rnoise_output=var_rnoise_output,
            var_poisson_output=var_poisson_output,
            dq_output=dq_output,
            mid_time_output=mid_time_output,
            orders_are_vertical=orders_are_vertical,
        )

    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output,
        mid_time=mid_time_output,
    )


@njit(nogil=True, parallel=False, cache=True)
def _horne_accumulate(
    D: np.ndarray,
    D_var: np.ndarray,
    D_var_poisson: np.ndarray,
    D_var_rnoise: np.ndarray,
    D_dq: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    M_in: np.ndarray,
):
    """Shared Horne weighted sums for one spectral sample (in-place masks)."""
    for i in range(D.size):
        if np.isfinite(D[i]) and not np.isfinite(D_var[i]):
            D_var[i] = 0.0
        if np.isfinite(D[i]) and not np.isfinite(D_var_poisson[i]):
            D_var_poisson[i] = 0.0
        if np.isfinite(D[i]) and not np.isfinite(D_var_rnoise[i]):
            D_var_rnoise[i] = 0.0

    M = ((D_dq & DO_NOT_USE) == 0) & np.isfinite(D) & M_in

    if np.any(~np.isfinite(P)):
        M[:] = False

    bad = np.where(~M)[0]
    n_good = len(M) - len(bad)
    if n_good == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.uint32(DO_NOT_USE),
            np.nan,
        )

    D[bad] = 0.0
    D_var[bad] = np.inf
    D_var_poisson[bad] = np.inf
    D_var_rnoise[bad] = np.inf

    psum = np.sum(P * W)
    if psum <= 0.0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.uint32(DO_NOT_USE),
            np.nan,
        )
    PP = P / psum

    w_tot = np.zeros_like(D_var)
    w_rn = np.zeros_like(D_var_rnoise)
    w_p = np.zeros_like(D_var_poisson)

    good_tot = D_var > 0
    good_rn = D_var_rnoise > 0
    good_p = D_var_poisson > 0

    w_tot[good_tot] = 1.0 / D_var[good_tot]
    w_rn[good_rn] = 1.0 / D_var_rnoise[good_rn]
    w_p[good_p] = 1.0 / D_var_poisson[good_p]

    zero_tot = D_var == 0
    zero_rn = D_var_rnoise == 0
    zero_p = D_var_poisson == 0

    if np.any(zero_tot):
        w_tot[:] = 0.0
        w_tot[zero_tot] = 1.0
    if np.any(zero_rn):
        w_rn[:] = 0.0
        w_rn[zero_rn] = 1.0
    if np.any(zero_p):
        w_p[:] = 0.0
        w_p[zero_p] = 1.0

    denom_tot = np.sum(M * PP**2 * W * w_tot)
    denom_rn = np.sum(M * PP**2 * W * w_rn)
    denom_p = np.sum(M * PP**2 * W * w_p)

    if denom_tot == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.uint32(DO_NOT_USE),
            np.nan,
        )

    spec = np.sum(D * M * PP * W * w_tot) / denom_tot
    err = np.sqrt(np.sum(PP * W) / denom_tot)
    var_rn = np.sum(PP * W) / denom_rn if denom_rn > 0 else 0.0
    var_p = np.sum(PP * W) / denom_p if denom_p > 0 else 0.0
    wsum = np.sum(w_tot * M)
    mid = np.sum(T * M * w_tot) / wsum if wsum > 0 else np.nan
    return spec, err, var_rn, var_p, np.uint32(0), mid


@njit(nogil=True, parallel=False, cache=True)
def _extract_optimal_numba(
    input_data: np.ndarray,
    input_err: np.ndarray,
    input_dq: np.ndarray,
    input_var_rnoise: np.ndarray,
    input_var_poisson: np.ndarray,
    input_mid_time: np.ndarray,
    trace_positions: np.ndarray,
    profile: np.ndarray,
    extraction_aperture: float,
    pix_range: tuple,
    spec_output: np.ndarray,
    err_output: np.ndarray,
    var_rnoise_output: np.ndarray,
    var_poisson_output: np.ndarray,
    dq_output: np.ndarray,
    mid_time_output: np.ndarray,
    orders_are_vertical: bool = True,
):
    n_cross = input_data.shape[1] if orders_are_vertical else input_data.shape[0]

    for x in range(pix_range[0], pix_range[1] + 1):
        window_inds, W = get_extraction_window(
            trace_positions[x], extraction_aperture, n_cross
        )
        if len(window_inds) == 0 or np.sum(W) <= 0.0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            mid_time_output[x] = np.nan
            continue

        if orders_are_vertical:
            D = input_data[x, window_inds].copy()
            D_var = input_err[x, window_inds] ** 2
            D_var_poisson = input_var_poisson[x, window_inds].copy()
            D_var_rnoise = input_var_rnoise[x, window_inds].copy()
            D_dq = input_dq[x, window_inds]
            P = profile[x, window_inds].copy()
            T = input_mid_time[x, window_inds].copy()
        else:
            D = input_data[window_inds, x].copy()
            D_var = input_err[window_inds, x] ** 2
            D_var_poisson = input_var_poisson[window_inds, x].copy()
            D_var_rnoise = input_var_rnoise[window_inds, x].copy()
            D_dq = input_dq[window_inds, x].copy()
            P = profile[window_inds, x].copy()
            T = input_mid_time[window_inds, x].copy()

        M_in = np.ones(D.size, dtype=np.bool_)
        spec, err, var_rn, var_p, dq, mid = _horne_accumulate(
            D, D_var, D_var_poisson, D_var_rnoise, D_dq, P, T, W, M_in
        )
        spec_output[x] = spec
        err_output[x] = err
        var_rnoise_output[x] = var_rn
        var_poisson_output[x] = var_p
        dq_output[x] = dq
        mid_time_output[x] = mid


@njit(nogil=True, parallel=False, cache=True)
def _extract_optimal_sheared_numba(
    input_data: np.ndarray,
    input_err: np.ndarray,
    input_dq: np.ndarray,
    input_var_rnoise: np.ndarray,
    input_var_poisson: np.ndarray,
    input_mid_time: np.ndarray,
    trace_positions: np.ndarray,
    profile: np.ndarray,
    shear: np.ndarray,
    extraction_aperture: float,
    pix_range: tuple,
    spec_output: np.ndarray,
    err_output: np.ndarray,
    var_rnoise_output: np.ndarray,
    var_poisson_output: np.ndarray,
    dq_output: np.ndarray,
    mid_time_output: np.ndarray,
    orders_are_vertical: bool = True,
):
    n_cross = input_data.shape[1] if orders_are_vertical else input_data.shape[0]

    for x in range(pix_range[0], pix_range[1] + 1):
        yc = trace_positions[x]
        window_inds, W = get_extraction_window(yc, extraction_aperture, n_cross)
        if len(window_inds) == 0 or np.sum(W) <= 0.0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            mid_time_output[x] = np.nan
            continue

        s = shear[x]
        if not np.isfinite(s):
            s = 0.0

        nwin = len(window_inds)
        D = np.empty(nwin, dtype=np.float64)
        D_err = np.empty(nwin, dtype=np.float64)
        D_var_poisson = np.empty(nwin, dtype=np.float64)
        D_var_rnoise = np.empty(nwin, dtype=np.float64)
        D_dq = np.empty(nwin, dtype=np.uint32)
        P = np.empty(nwin, dtype=np.float64)
        T = np.empty(nwin, dtype=np.float64)
        M_in = np.ones(nwin, dtype=np.bool_)

        for k in range(nwin):
            y = int(window_inds[k])
            x_frac = float(x) + s * (float(y) - float(yc))
            D[k] = _interp_spectral_sample(
                input_data, x_frac, y, orders_are_vertical
            )
            D_err[k] = _interp_spectral_sample(
                input_err, x_frac, y, orders_are_vertical
            )
            D_var_poisson[k] = _interp_spectral_sample(
                input_var_poisson, x_frac, y, orders_are_vertical
            )
            D_var_rnoise[k] = _interp_spectral_sample(
                input_var_rnoise, x_frac, y, orders_are_vertical
            )
            P[k] = _interp_spectral_sample(
                profile, x_frac, y, orders_are_vertical
            )
            T[k] = _interp_spectral_sample(
                input_mid_time, x_frac, y, orders_are_vertical
            )
            D_dq[k] = _nearest_dq_sample(
                input_dq, x_frac, y, orders_are_vertical
            )

        D_var = D_err * D_err
        spec, err, var_rn, var_p, dq, mid = _horne_accumulate(
            D, D_var, D_var_poisson, D_var_rnoise, D_dq, P, T, W, M_in
        )
        spec_output[x] = spec
        err_output[x] = err
        var_rnoise_output[x] = var_rn
        var_poisson_output[x] = var_p
        dq_output[x] = dq
        mid_time_output[x] = mid
