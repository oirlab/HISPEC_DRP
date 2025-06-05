import numpy as np

__all__ = ['create_ramp']

# NOTE: Find a new home for this function

def create_ramp(
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float, # e- RMS
    n_reads : int,
    bias : np.ndarray | float = 0, # e-
    read_noise : float = 0, first_read_noise : float | None = None,
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True
) -> dict[str, np.ndarray]:
    times = np.zeros(n_reads, dtype=float)
    data = np.zeros(shape=(source.shape[0], source.shape[1], n_reads), dtype=np.int16)
    dq = np.zeros(shape=(source.shape[0], source.shape[1], n_reads), dtype=np.int16)
    data_poisson = np.zeros(shape=(source.shape[0], source.shape[1], n_reads), dtype=float)
    if first_read_noise is None:
        first_read_noise = read_noise
    data[:, :, 0] += bias  # Add bias to first read
    for i in range(n_reads):
        t = (i+ 1) * readtime
        image_prev = data_poisson[:, :, i-1] if i > 0 else np.zeros_like(source) # Previous read without read noise
        image_new = image_prev + source * readtime # Add new read
        if poisson_noise:
            image_new = np.random.poisson(lam=image_new, size=source.shape)
        data_poisson[:, :, i] = image_new
        if i == 0 and first_read_noise > 0:
            image_new += np.random.normal(loc=0, scale=first_read_noise, size=source.shape)
        elif i > 0 and read_noise > 0:
            image_new += np.random.normal(loc=0, scale=read_noise, size=source.shape)
        times[i] = t
        if nonlin_coeffs is not None:
            image_new *= np.round(np.polyval(nonlin_coeffs[::-1], t).astype(np.int16))
        data[:, :, i] = image_new.astype(np.int16)
    return dict(times=times, data=data, dq=dq)