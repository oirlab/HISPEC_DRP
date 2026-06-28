import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks, savgol_filter

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "find_emission_lines",
]

def _estimate_emission_spectrum_background(
    spec : np.ndarray,
    min_line_spacing : int = 12,
    n_lines : int = 10,
    deg : int = 1
) -> np.ndarray:
    filter_width = int(min_line_spacing / 3 * n_lines)
    if filter_width % 2 == 0:
        filter_width += 1  # Ensure filter width is odd
    background = _nanmin_filter1d(spec, filter_width)
    background = savgol_filter(background, filter_width, polyorder=deg, mode='nearest')
    return background


def _nanmin_filter1d(data: np.ndarray, width: int) -> np.ndarray:
    if width % 2 == 0:
        width += 1  # ensure odd
    pad = width // 2
    padded = np.pad(data, pad_width=pad, mode='edge')
    windows = sliding_window_view(padded, width)
    return np.nanmin(windows, axis=1)


def find_emission_lines(
    spec : np.ndarray,
    spec_dq : np.ndarray,
    min_line_spacing : int, # Pixels
    num_lines_background : float | None = 3.0,
    centroid_num_pixels : int = 3,
    refine_with_centroid : bool = True,
    drop_first_last : bool = True
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """
    Find the initial line positions using a peak finding algorithm followed by a centroid refinement (optional).

    Parameters
    ----------
    spec : np.ndarray
        The input spectrum.
    spec_dq : np.ndarray
        The data quality mask for the spectrum (0=good).
    min_line_spacing : int
        Minimum spacing between lines in pixels.
    num_lines_background : float | None
        Number of lines to use for background estimation. If None, no background subtraction is performed.
    centroid_num_pixels : int
        Number of pixels to use for centroid refinement.
    refine_with_centroid : bool
        Whether to refine the line positions using centroid calculation.
    drop_first_last : bool
        Whether to drop the first and last detected peaks (to avoid edge effects).
    """

    n_pix = len(spec)

    good = np.where(spec_dq == 0)[0]
    if len(good) == 0:
        logger.warning("No good pixels found in spectrum.")
        return np.array([]), None
    
    # Estimate and remove background
    if num_lines_background is not None:
        background = _estimate_emission_spectrum_background(
            spec,
            min_line_spacing=min_line_spacing,
            n_lines=3
        )
        spec_nobkg = spec - background
    else:
        spec_nobkg = spec.copy()

    # Set bad pixels to zero
    spec_nobkg[spec_dq > 0] = 0

    # Get integer locations
    peaks, props = find_peaks(
        spec_nobkg,
        distance=min_line_spacing,
        width=1.5,
        rel_height=0.5,
    )

    # Optionally drop edges (might be noise)
    # Safest for LFC/etalon spectra that might have a partial line at the edge.
    if drop_first_last:
        peaks = peaks[1:-1]  # Drop first and last peaks, possibly noise
    n_peaks = len(peaks)

    # NOTE: Keep plots here for debugging.
    # import matplotlib.pyplot as plt
    #plt.plot(spec); [plt.axvline(l, color='r', linestyle='--') for m in lines]
    #plt.plot(spec); plt.plot(background, color='orange', label='Background'); plt.show()

    # Refine lines based on centroid
    if refine_with_centroid:

        # Init output array
        lines_out = np.full(n_peaks, np.nan, dtype=np.float64)

        # Refine each peak with a centroid calculation
        for i in range(n_peaks):

            # Start/stop indices for each peak
            ki = int(max(peaks[i] - centroid_num_pixels//2, 0))
            kf = int(min(peaks[i] + centroid_num_pixels//2, n_pix - 1))
            _good = np.where(spec_dq[ki:kf+1] == 0)[0]
            _spec = spec[ki:kf+1][_good]
            if len(_good) > 0:
                lc = np.sum(_good * _spec) / np.sum(_spec) + ki
                lines_out[i] = lc
            else:
                lines_out[i] = peaks[i]

    # If not refining, just return the integer peak locations
    else:
        lines_out = np.array(lines_out, dtype=np.float64)

    return lines_out, props