from __future__ import annotations
import numpy as np
from numba import njit, prange
from ...utils.math import biweight_location, biweight_midvariance
from ... import datamodels


@njit(nogil=True)
def sigma_clip(
    x: np.ndarray,
    mask: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    maxiters: int = 50,
    min_keep: int = 3,
) -> tuple[np.ndarray, float, float, int]:
    """
    Perform sigma clipping on an input array using the biweight location and midvariance.

    Parameters
    ----------
    x : np.ndarray
        Input array of data values to sigma clip.
    mask : np.ndarray
        Boolean array of the same shape as x where True indicates "good" and False indicates "bad".
    sigma_thresh_low : float | None, optional
        Low sigma threshold for sigma clipping. If None, no low sigma clipping is performed.
    sigma_thresh_high : float | None, optional
        High sigma threshold for sigma clipping. If None, no high sigma clipping is performed.
    maxiters : int, optional
        Maximum number of iterations for sigma clipping. Default is 50.
    min_keep : int, optional
        Minimum number of good values to keep after sigma clipping. Default is 3.

    Returns
    -------
    tuple[np.ndarray, float, float, int]
        Tuple of (mask_out, center, stddev, n_iters) where:
            - mask_out : boolean array of the same shape as input mask with outliers masked as False.
            - center : float, the biweight location of the input data.
            - stddev : float, the biweight midvariance (standard deviation) of the input data.
            - n_iters : int, number of iterations performed.
    """

    center = np.nan
    stddev = np.nan

    min_batch_size = 3

    for i in range(maxiters):
        x_masked = x[mask]  # select only unmasked values
        if len(x_masked) <= min_batch_size:
            break

        center = biweight_location(x_masked)
        var = biweight_midvariance(x_masked, M=center)
        stddev = np.sqrt(var)

        n_good = np.sum(mask)
        _mask_outliers(
            x, center, stddev, mask,
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=min_keep,
        )
        n_good_new = np.sum(mask)
        if n_good_new == n_good:
            break

    # Total number of iterations used
    iters_out = i + 1

    return mask, center, stddev, iters_out


@njit(nogil=True)
def _mask_outliers(
    x: np.ndarray,
    center: float,
    stddev: float,
    mask: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    min_keep: int = 3,
) -> np.ndarray:
    """
    Mask outliers in a dataset based on sigma thresholds, respecting a
    minimum number of good values to retain.

    Parameters
    ----------
    x : np.ndarray
        1D array of data values.
    center : float
        Central location (e.g. mean or median) of the data.
    stddev : float
        Standard deviation of the data.
    mask : np.ndarray
        1D data mask where False is bad, True is good. Updated in-place.
    sigma_thresh_low : float | None, optional
        Low sigma threshold for outlier rejection. If None, no low-side
        clipping is performed.
    sigma_thresh_high : float | None, optional
        High sigma threshold for outlier rejection. If None, no high-side
        clipping is performed.
    min_keep : int, optional
        Minimum number of unmasked values to retain after clipping.
        Candidates are ranked from most to least extreme; the least extreme
        are preserved when the clip budget would otherwise be exceeded.
        Default is 3.

    Returns
    -------
    mask : np.ndarray
        Mask updated in-place and returned.
    """
    do_low  = sigma_thresh_low  is not None
    do_high = sigma_thresh_high is not None

    if do_low:
        low_thresh  = -sigma_thresh_low  * stddev
    if do_high:
        high_thresh =  sigma_thresh_high * stddev

    # --- Pass 1: count good values and collect clipping candidates -------
    # Each candidate records its absolute residual and flat index so we can
    # rank by extremeness and honour min_keep.
    n_good = 0
    cand_resid = np.empty(x.size, dtype=np.float64)
    cand_idx   = np.empty(x.size, dtype=np.int64)
    n_cand = 0

    flat = 0
    for i in np.ndindex(x.shape):
        if mask[i]:
            n_good += 1
            res = x[i] - center
            if (do_low and res < low_thresh) or (do_high and res > high_thresh):
                cand_resid[n_cand] = abs(res)
                cand_idx[n_cand]   = flat
                n_cand += 1
        flat += 1

    if n_cand == 0:
        return mask

    # --- Determine how many we can actually clip -------------------------
    max_clip = n_good - min_keep
    if max_clip <= 0:
        return mask  # Already at or below min_keep; clip nothing.

    n_clip = min(n_cand, max_clip)

    # --- Pass 2: apply the mask ------------------------------------------
    if n_clip == n_cand:
        # Clip all candidates — no ranking needed.
        for k in range(n_cand):
            mask.flat[cand_idx[k]] = False
    else:
        # Keep the (n_cand - n_clip) least extreme candidates.
        # np.partition places the n_clip-th largest at index [n_cand - n_clip]
        # with everything to its right being >= that value.
        resids = cand_resid[:n_cand].copy()
        clip_threshold = np.partition(resids, n_cand - n_clip)[n_cand - n_clip]

        clipped = 0
        for k in range(n_cand):
            if clipped >= n_clip:
                break
            if cand_resid[k] >= clip_threshold:
                mask.flat[cand_idx[k]] = False
                clipped += 1

    return mask


@njit(nogil=True, parallel=True, cache=True)
def sigma_clip_frames(
    data_cube: np.ndarray,
    mask_cube: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    min_keep: int = 3,
):
    """Per-pixel sigma clipping for a (n_frames, ny, nx) frame cube."""
    _, ny, nx = data_cube.shape
    for i in prange(ny):
        for j in range(nx):
            sigma_clip(
                data_cube[:, i, j],
                mask_cube[:, i, j],
                sigma_thresh_low=sigma_thresh_low,
                sigma_thresh_high=sigma_thresh_high,
                min_keep=min_keep,
            )


@njit(nogil=True, parallel=True, cache=True)
def sigma_clip_spectra(
    data_2d: np.ndarray,
    mask_2d: np.ndarray,
    sigma_thresh_low: float | None = None,
    sigma_thresh_high: float | None = None,
    min_keep: int = 3,
):
    """Per-pixel sigma clipping for a (n_frames, n_pix) spectra stack."""
    _, n_pix = data_2d.shape
    for j in prange(n_pix):
        sigma_clip(
            data_2d[:, j],
            mask_2d[:, j],
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=min_keep,
        )