import numpy as np
from numba import njit


@njit(nogil=True, parallel=False, cache=True)
def get_extraction_window(yc: float, window_size: float, n_pixels: int = -1):
    """
    Computes the fractional overlap of pixels with the extraction window.

    Parameters
    ----------
    yc : float
        Center of the extraction window (in pixel coordinates).
    window_size : float
        Size of the extraction window (in pixels) centered in yc.
    n_pixels : int, optional
        When ``>= 0``, clip candidate indices to ``[0, n_pixels - 1]`` so callers
        never receive out-of-bounds cross-dispersion indices (avoids Numba OOB
        segfaults when trace positions are extrapolated or non-finite).

    Returns
    -------
    idx : np.ndarray
        Indices of the pixels that might have overlap with the extraction window.
    W : np.ndarray
        Fractional overlap of each pixel in idx with the extraction window.
    """
    if not np.isfinite(yc):
        idx = np.empty(0, dtype=np.int64)
        W = np.empty(0, dtype=np.float32)
        return idx, W

    half = window_size / 2.0
    window_start = yc - half
    window_end = yc + half

    # Find first and last pixels that overlap with the window
    i_first = int(np.floor(window_start + 0.5))
    i_last = int(np.ceil(window_end - 0.5))

    if n_pixels >= 0:
        if n_pixels <= 0:
            idx = np.empty(0, dtype=np.int64)
            W = np.empty(0, dtype=np.float32)
            return idx, W
        i_first = max(0, min(i_first, n_pixels - 1))
        i_last = max(0, min(i_last, n_pixels - 1))
        if i_last < i_first:
            idx = np.empty(0, dtype=np.int64)
            W = np.empty(0, dtype=np.float32)
            return idx, W

    idx = np.arange(i_first, i_last + 1)
    W = np.empty(len(idx), dtype=np.float32)

    # Calculate fractional overlap for each pixel
    for j in range(len(idx)):
        i = idx[j]
        pixel_start = i - 0.5
        pixel_end = i + 0.5
        overlap_start = max(window_start, pixel_start)
        overlap_end = min(window_end, pixel_end)
        W[j] = max(0.0, overlap_end - overlap_start)

    return idx, W