import numpy as np
from numba import njit


@njit(nogil=True, parallel=False, cache=True)
def get_extraction_window(yc: float, window_size: float):
    """
    Computes the fractional overlap of pixels with the extraction window.

    Parameters
    ----------
    yc : float
        Center of the extraction window (in pixel coordinates).
    window_size : float
        Size of the extraction window (in pixels) centered in yc.

    Returns
    -------
    idx : np.ndarray
        Indices of the pixels that might have overlap with the extraction window.
    W : np.ndarray
        Fractional overlap of each pixel in idx with the extraction window.
    """

    half = window_size / 2.0
    window_start = yc - half
    window_end = yc + half
    
    # Find first and last pixels that overlap with the window
    i_first = int(np.floor(window_start + 0.5))
    i_last = int(np.ceil(window_end - 0.5))
    
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