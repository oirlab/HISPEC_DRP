import numpy as np
import astropy.table

__all__ = ['create_ramp']

# NOTE: Find a new home for this function

def create_ramp(
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float = 1.7, # seconds between reads
    n_reads : int = 10,
    gain : np.ndarray | float = 2.0, # e-/ADU
    bias : np.ndarray | float = 1000, # e-
    read_noise : float | None = 3, # e- standard deviation
    kTC_noise : float | None = 50, # e- standard deviation, None to skip
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0, # MJD start time, not first read
) -> dict[str, np.ndarray]:
    """
    Simulate a detector up-the-ramp (UTR) readout sequence for a given source image.

    Parameters
    ----------
    source : np.ndarray
        2D array of source signal (e-/s/pixel) including all sources.
    readtime : float
        Time between reads (seconds).
    n_reads : int
        Number of reads in the ramp sequence (default: 10).
    gain : float or np.ndarray
        Gain of the detector (e-/ADU).
    bias : np.ndarray or float, optional
        Bias level to add to the first read (default: 0).
    read_noise : float or None, optional
        Standard deviation of read noise (e-) for each read (default: 3 e-).
    kTC_noise : float or None, optional
        kTC noise parameter (e-) to add to the first read (default: 50 e-).
    nonlin_coeffs : np.ndarray or None, optional
        Nonlinearity correction coefficients (default: None). Applies as data = np.polyval(nonlin_coeffs, data).
    poisson_noise : bool, optional
        If True, apply Poisson noise to the signal (default: True).
    mjd_start : float, optional
        Modified Julian Date for the start time (default: 60577.0).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
            'times': astropy.table.Table of read numbers, times, and exptimes
            'data': 3D array (n_reads, ny, nx) of ramp data
            'dq': 3D array (n_reads, ny, nx) of data quality flags

    Raises
    ------
    ValueError
        If any value in the output 'data' array is negative.
    """

    times = astropy.table.Table(
        names=['READNUM', 'TIMES', 'EXPTIME'],
        dtype=[np.int16, np.float64, np.float64]
    )
    times.add_row((0, mjd_start, 0))

    data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    dq = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.int16)

    # First, compute photon counts only (no read noise or bias)
    for i in np.arange(1,n_reads):
        t = i * readtime

        if poisson_noise:
            data[i, :, :] = data[i-1, :, :] + np.random.poisson(lam=source * readtime, size=source.shape)# Add new read
        else:
            data[i, :, :] = data[i-1, :, :] + source * readtime

        mjd = mjd_start + t / 86400.0
        times.add_row((i, mjd, t))

    # apply non-linear behavior
    if nonlin_coeffs is not None:
        data = np.polyval(nonlin_coeffs, data)

    # Add read noise
    if read_noise is not None:
        data += np.random.normal(loc=0, scale=read_noise, size=data.shape)

    # Add bias (and kTC noise if desired)
    data += bias
    if kTC_noise is not None:
        data += np.random.normal(loc=0, scale=kTC_noise, size=source.shape)

    # Apply gain
    if gain:
        data /= gain

    # Check for negative values before returning
    if np.any(data < 0):
        raise ValueError("Negative value found in 'data' array before returning from create_ramp. "
                         "Data needs to be all positive to be converted to np.uint16."
                         "Suggestion: Raise bias value.")

    # Clip data to maximum value of uint16
    data = np.clip(data, 0, np.iinfo(np.uint16).max)

    # Round and convert to uint16
    data = np.round(data).astype(np.uint16)

    return dict(times=times, data=data, dq=dq)
