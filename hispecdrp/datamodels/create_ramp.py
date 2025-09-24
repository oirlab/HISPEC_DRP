import numpy as np
import astropy.table
import multiprocessing as mp
import ctypes

__all__ = ['create_ramp']

# NOTE: Find a new home for this function

def _create_ramp(
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float = 1.7, # seconds between reads
    n_reads : int = 10,
    gain : np.ndarray | float = 1.0, # e-/ADU
    flat : np.ndarray | None = None,
    bias : np.ndarray | float = 1000, # e-
    dark : np.ndarray | float = 0, # e-/s/pixel
    read_noise : np.ndarray | float | None = 3, # e- standard deviation
    kTC_noise : np.ndarray | float | None = 50, # e- standard deviation, None to skip
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0, # MJD start time of first read of first pixel
    convert_to_uint16 : bool = True,
    clip_ramps: bool = True,
) -> dict[str, np.ndarray]:

    data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    _dark_UTR = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    dq_raw = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.uint32)

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
    # if poisson_noise:
    #     # Vectorized: shape (n_reads-1, ny, nx)
    #     signal_poisson = np.random.poisson(lam=source * readtime, size=(n_reads - 1, *source.shape)).astype(np.float32)
    #     dark_poisson = np.random.poisson(lam=_dark * readtime, size=(n_reads - 1, *source.shape)).astype(np.float32)
    #     # Cumulative sum along reads, prepend zeros for first read
    #     data = np.concatenate([np.zeros((1, *source.shape), dtype=np.float32), np.cumsum(signal_poisson, axis=0)],
    #                           axis=0)
    #     _dark_UTR = np.concatenate([np.zeros((1, *source.shape), dtype=np.float32), np.cumsum(dark_poisson, axis=0)],
    #                                axis=0)
    # else:
    #     # Vectorized: just multiply and broadcast
    #     data = (np.arange(n_reads,dtype=np.float32)[:, None, None]* readtime) * source.astype(np.float32)[None, :, :]
    #     _dark_UTR = (np.arange(n_reads,dtype=np.float32)[:, None, None]* readtime) * _dark.astype(np.float32)[None, :, :]

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



    if clip_ramps:
        # Clip data to maximum value of uint16
        data = np.clip(data, 0, float(np.iinfo(np.uint16).max))

    if convert_to_uint16:
        # Check for negative values before returning
        if np.any(data < 0):
            raise ValueError("Negative value found in 'data' array before returning from create_ramp. "
                             "Data needs to be all positive to be converted to np.uint16."
                             "Suggestion: Raise bias value.")

        # Round and convert to uint16
        data = np.round(data).astype(np.uint16)



    return dict(data=data, dq_raw=dq_raw)

def _create_ramp_worker(args):
    """
    Worker function for multiprocessing. Runs create_ramp on a block of rows.
    Args:
        args (tuple): (source_block, readtime, n_reads, gain_block, flat_block, bias_block, dark_block, read_noise, kTC_noise, nonlin_coeffs, poisson_noise, mjd_start, convert_to_uint16, clip_ramps)
    Returns:
        dict[str, np.ndarray]: Output from create_ramp for the block
    """
    block_start, block_end, readtime, n_reads, nonlin_coeffs, poisson_noise, mjd_start, convert_to_uint16, clip_ramps, types_tuple = args
    mp_data_type, mp_float_type, mp_dq_type = types_tuple

    data_np = _arraytonumpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = _arraytonumpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_type)
    source_np = _arraytonumpy(shared_source, shared_source_shape, dtype=mp_float_type)
    gain_np = _arraytonumpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    flat_np = _arraytonumpy(shared_flat, shared_flat_shape, dtype=mp_float_type)
    bias_np = _arraytonumpy(shared_bias, shared_bias_shape, dtype=mp_float_type)
    dark_np = _arraytonumpy(shared_dark, shared_dark_shape, dtype=mp_float_type)
    read_noise_np = _arraytonumpy(shared_read_noise, shared_read_noise_shape, dtype=mp_float_type)
    kTC_noise_np = _arraytonumpy(shared_kTC_noise, shared_kTC_noise_shape, dtype=mp_float_type)

    outputs = _create_ramp(
        source=source_np[block_start:block_end, :],
        readtime=readtime,
        n_reads=n_reads,
        gain=gain_np[block_start:block_end, :],
        flat=flat_np[block_start:block_end, :],
        bias = bias_np[block_start:block_end, :],
        dark = dark_np[block_start:block_end, :],
        read_noise = read_noise_np[block_start:block_end, :],
        kTC_noise = kTC_noise_np[block_start:block_end, :],
        nonlin_coeffs = nonlin_coeffs,
        poisson_noise = poisson_noise,
        mjd_start = mjd_start,
        convert_to_uint16 = convert_to_uint16,
        clip_ramps = clip_ramps
    )

    data_np[:,block_start:block_end, :] = outputs['data']
    dq_raw_np[:,block_start:block_end, :] = outputs['dq_raw']

    return (block_start, block_end)


def create_ramp(
    source : np.ndarray,
    readtime : float = 1.7,
    n_reads : int = 10,
    gain : np.ndarray | float = 1.0,
    flat : np.ndarray | None = None,
    bias : np.ndarray | float = 1000,
    dark : np.ndarray | float = 0,
    read_noise : float | None = 3,
    kTC_noise : float | None = 50,
    nonlin_coeffs : np.ndarray | None = None,
    poisson_noise : bool = True,
    mjd_start : float = 60577.0,
    convert_to_uint16 : bool = True,
    clip_ramps: bool = True,
    max_cores: int = 1,
    block_size: int | None = None,
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
        Bias level to add to the first read (default: 1000 e-).
    dark : np.ndarray or float, optional
        Dark current level to add to each read (e-/s/pixel) (default: 0)
        The flat field is applied BFEORE adding the dark current. This is because dark subtraction is typically done
        before the flat fielding step.
    read_noise : np.ndarray or float or None, optional
        Standard deviation of read noise (e-) for each read (default: 3 e-).
    kTC_noise : np.ndarray or float or None, optional
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
    ny, nx = source.shape


    # Run in parallel
    if max_cores > 1:

        if convert_to_uint16:
            mp_data_type = ctypes.c_ushort
        else:
            mp_data_type = ctypes.c_float
        mp_float_type = ctypes.c_float
        mp_dq_type = ctypes.c_uint

        types_tuple = (mp_data_type, mp_float_type, mp_dq_type)

        data_mp = mp.Array(mp_data_type, nx * ny * n_reads)
        data_shape = (n_reads, ny, nx)
        data_np = _arraytonumpy(data_mp, data_shape, dtype=mp_data_type)

        dq_raw_mp = mp.Array(mp_dq_type, nx * ny * n_reads)
        dq_raw_shape = (n_reads, ny, nx)
        dq_raw_np = _arraytonumpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_type)

        source_mp = mp.Array(mp_float_type, nx * ny)
        source_shape = (ny, nx)
        source_np = _arraytonumpy(source_mp, source_shape, dtype=mp_float_type)
        source_np[:] = source

        if not isinstance(gain, np.ndarray):
            gain = np.full(source.shape, gain, dtype=np.float32)
        gain_mp = mp.Array(mp_float_type, nx * ny)
        gain_shape = (ny, nx)
        gain_np = _arraytonumpy(gain_mp, gain_shape, dtype=mp_float_type)
        gain_np[:] = gain

        if flat is not None:
            if not isinstance(flat, np.ndarray):
                flat = np.full(source.shape, flat, dtype=np.float32)
        else:
            flat = np.ones(source.shape, dtype=np.float32)
        flat_mp = mp.Array(mp_float_type, nx * ny)
        flat_shape = (ny, nx)
        flat_np = _arraytonumpy(flat_mp, flat_shape, dtype=mp_float_type)
        flat_np[:] = flat

        if not isinstance(bias, np.ndarray):
            bias = np.full(source.shape, bias, dtype=np.float32)
        bias_mp = mp.Array(mp_float_type, nx * ny)
        bias_shape = (ny, nx)
        bias_np = _arraytonumpy(bias_mp, bias_shape, dtype=mp_float_type)
        bias_np[:] = bias

        if not isinstance(dark, np.ndarray):
            dark = np.full(source.shape, dark, dtype=np.float32)
        dark_mp = mp.Array(mp_float_type, nx * ny)
        dark_shape = (ny, nx)
        dark_np = _arraytonumpy(dark_mp, dark_shape, dtype=mp_float_type)
        dark_np[:] = dark

        if read_noise is not None:
            if not isinstance(read_noise, np.ndarray):
                read_noise = np.full(source.shape, read_noise, dtype=np.float32)
        else:
            read_noise = np.zeros(source.shape, dtype=np.float32)
        read_noise_mp = mp.Array(mp_float_type, nx * ny)
        read_noise_shape = (ny, nx)
        read_noise_np = _arraytonumpy(read_noise_mp, read_noise_shape, dtype=mp_float_type)
        read_noise_np[:] = read_noise

        if kTC_noise is not None:
            if not isinstance(kTC_noise, np.ndarray):
                kTC_noise = np.full(source.shape, kTC_noise, dtype=np.float32)
        else:
            kTC_noise = np.zeros(source.shape, dtype=np.float32)
        kTC_noise_mp = mp.Array(mp_float_type, nx * ny)
        kTC_noise_shape = (ny, nx)
        kTC_noise_np = _arraytonumpy(kTC_noise_mp, kTC_noise_shape, dtype=mp_float_type)
        kTC_noise_np[:] = kTC_noise

        if block_size is None:
            block_size = max(1, ny // (max_cores)) if max_cores > 1 else ny

        # Prepare blocks
        args_list = []
        for block_start in range(0, ny, block_size):
            block_end = min(block_start + block_size, ny)

            args = (block_start, block_end,
                    readtime,
                    n_reads,
                    nonlin_coeffs,
                    poisson_noise,
                    mjd_start,
                    convert_to_uint16,
                    clip_ramps,
                    types_tuple
                    )
            args_list.append(args)

        _init_args = (data_mp, data_shape,
                dq_raw_mp, dq_raw_shape,
                source_mp, source_shape,
                gain_mp, gain_shape,
                flat_mp, flat_shape,
                bias_mp, bias_shape,
                dark_mp, dark_shape,
                read_noise_mp, read_noise_shape,
                kTC_noise_mp, kTC_noise_shape
                      )
        tpool = mp.Pool(processes=max_cores, initializer=_tpool_init,
                        initargs=_init_args, maxtasksperchild=50)
        tasks = [tpool.apply_async(_create_ramp_worker, args=(_args,))
                 for _args in args_list]

        # save it to shared memory
        for task_index, task in enumerate(tasks):
            print("Waiting for task {0}/{1}".format(task_index + 1, len(tasks)))
            task.wait()

        print("Closing threadpool")
        tpool.close()
        tpool.join()

        _outputs = dict(data=data_np, dq_raw=dq_raw_np)
    else:
        _outputs = _create_ramp(
            source=source,
            readtime=readtime,
            n_reads=n_reads,
            gain=gain,
            flat=flat,
            bias=bias,
            dark=dark,
            read_noise=read_noise,
            kTC_noise=kTC_noise,
            nonlin_coeffs=nonlin_coeffs,
            poisson_noise=poisson_noise,
            mjd_start=mjd_start,
            convert_to_uint16=convert_to_uint16,
            clip_ramps=clip_ramps
        )

    _meta = dict(clock_rate=1/(readtime/(source.shape[0]*source.shape[1])),# in Hz
                 mjd_start=mjd_start)
    _outputs['meta'] = _meta
    return _outputs

def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array
    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def _tpool_init(data, data_shape,
                dq_raw, dq_raw_shape,
                source, source_shape,
                gain, gain_shape,
                flat, flat_shape,
                bias, bias_shape,
                dark, dark_shape,
                read_noise, read_noise_shape,
                kTC_noise, kTC_noise_shape
                ):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).
    """
    global shared_data, shared_data_shape, \
        shared_dq_raw, shared_dq_raw_shape, \
        shared_source, shared_source_shape, \
        shared_gain, shared_gain_shape, \
        shared_flat, shared_flat_shape, \
        shared_bias, shared_bias_shape, \
        shared_dark, shared_dark_shape, \
        shared_read_noise, shared_read_noise_shape, \
        shared_kTC_noise, shared_kTC_noise_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_source = source
    shared_source_shape = source_shape

    shared_gain = gain
    shared_gain_shape = gain_shape

    shared_flat = flat
    shared_flat_shape = flat_shape

    shared_bias = bias
    shared_bias_shape = bias_shape

    shared_dark = dark
    shared_dark_shape = dark_shape

    shared_read_noise = read_noise
    shared_read_noise_shape = read_noise_shape

    shared_kTC_noise = kTC_noise
    shared_kTC_noise_shape = kTC_noise_shape
