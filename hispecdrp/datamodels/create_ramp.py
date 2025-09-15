import numpy as np
import astropy.table

__all__ = ['create_ramp']

# NOTE: Find a new home for this function

def create_ramp(
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float = 1.7, # seconds between reads
    n_reads : int = 10,
    gain : np.ndarray | float = 2.0, # e-/ADU
    flat : np.ndarray | None = None,
    bias : np.ndarray | float = 1000, # e-
    dark : np.ndarray | float = 0, # e-/s/pixel
    read_noise : float | None = 3, # e- standard deviation
    kTC_noise : float | None = 50, # e- standard deviation, None to skip
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0, # MJD start time of first read of first pixel
    convert_to_uint16 : bool = True,
    clip_ramps: bool = True,
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
    flat : np.ndarray or None, optional
        Flat field array (default: None).
        We assume that flats have been corrected for gain variations already. In this function the gain is applied after
         the flat, but in the data reduction pipeline the gain correction should be applied first.
    bias : np.ndarray or float, optional
        Bias level to add to the first read (default: 0).
    dark : np.ndarray or float, optional
        Dark current level to add to each read (e-/s/pixel) (default: 0)
        The flat field is applied BFEORE adding the dark current. This is because dark subtraction is typically done
        before the flat fielding step.
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
    convert_to_uint16 : bool, optional
        If True, round and convert the output data to np.uint16 (default: True).
    clip_ramps: bool, optional
        If True, clip_ramps the output data to the range of np.uint16 (0 to 65535) (default: True).
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

    data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    _dark_UTR = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    dq_raw = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.int16)

    if not isinstance(dark, np.ndarray):
        _dark = np.full(source.shape, dark, dtype=np.float32)
    else:
        _dark = dark.astype(np.float32)

    # First, compute photon counts only (no read noise or bias)
    for i in np.arange(1,n_reads):
        if poisson_noise:
            data[i, :, :] = data[i-1, :, :] + np.random.poisson(lam=source * readtime, size=source.shape)# Add new read
            _dark_UTR[i, :, :] = _dark_UTR[i-1, :, :] + np.random.poisson(lam=_dark * readtime, size=source.shape)
        else:
            data[i, :, :] = data[i-1, :, :] + source * readtime
            _dark_UTR[i, :, :] = _dark_UTR[i-1, :, :] + _dark * readtime


    if flat is not None:
        data *= flat[None, :, :]

    data += _dark_UTR

    # Apply gain
    data /= gain

    # apply non-linear behavior
    if nonlin_coeffs is not None:
        data = np.polyval(nonlin_coeffs, data)

    # Add read noise
    if read_noise is not None:
        data += np.random.normal(loc=0, scale=read_noise/gain, size=data.shape)

    # Add bias (and kTC noise if desired)
    data += bias/gain
    if kTC_noise is not None:
        data += np.random.normal(loc=0, scale=kTC_noise/gain, size=source.shape)[None, :, :]


    # Check for negative values before returning
    if np.any(data < 0):
        raise ValueError("Negative value found in 'data' array before returning from create_ramp. "
                         "Data needs to be all positive to be converted to np.uint16."
                         "Suggestion: Raise bias value.")

    if clip_ramps:
        # Clip data to maximum value of uint16
        data = np.clip(data, 0, float(np.iinfo(np.uint16).max))

    if convert_to_uint16:
        # Round and convert to uint16
        data = np.round(data).astype(np.uint16)

    _meta = dict(clock_rate=1/(readtime/(source.shape[0]*source.shape[1])),# in Hz
                 mjd_start=mjd_start)

    return dict(data=data, dq_raw=dq_raw, meta=_meta)
