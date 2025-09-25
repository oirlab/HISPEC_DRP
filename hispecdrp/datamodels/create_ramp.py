import numpy as np
import multiprocessing as mp
import ctypes

from ..utils.parallelization_utils import _arraytonumpy

__all__ = ['create_ramp']

# NOTE: Find a new home for this function

def _create_ramp(
        data,dq_raw,
    source : np.ndarray, # e- / s / pixel including all sources
    readtime : float = 1.7, # seconds between reads
    n_reads : int = 10,
    gain : np.ndarray | float = 1.0, # e-/ADU
    flat : np.ndarray | float = 1.0,
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

    if convert_to_uint16:
        _data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
    else:
        _data = data

    # First, compute photon counts only (no read noise or bias)
    for i in np.arange(1,n_reads):
        if poisson_noise:
            _data[i, :, :] = _data[i-1, :, :] + \
                            (np.random.poisson(lam=source * readtime, size=source.shape)*flat[None, :, :] +\
                            np.random.poisson(lam=dark * readtime, size=source.shape)) / gain
        else:
            _data[i, :, :] = _data[i-1, :, :] + (source * readtime *flat[None, :, :] + dark * readtime) / gain

    # apply non-linear behavior
    if nonlin_coeffs is not None:
        _data[:] = np.polyval(nonlin_coeffs, _data)

    # Add read noise
    if read_noise is not None:
        _data += np.random.normal(loc=0, scale=read_noise/gain, size=_data.shape)

    # Add bias (and kTC noise if desired)
    _data += bias/gain
    if kTC_noise is not None:
        _data += np.random.normal(loc=0, scale=kTC_noise/gain, size=source.shape)[None, :, :]



    if clip_ramps:
        # Clip _data to maximum value of uint16
        _data[:] = np.clip(_data, 0, float(np.iinfo(np.uint16).max))

    if convert_to_uint16:
        # Check for negative values before returning
        if np.any(_data < 0):
            raise ValueError("Negative value found in 'data' array before returning from create_ramp. "
                             "Data needs to be all positive to be converted to np.uint16."
                             "Suggestion: Raise bias value.")

        # Round and convert to uint16
        # todo fix this
        data[:] = np.round(_data).astype(np.uint16)

    return

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
        data_np[:,block_start:block_end, :],
        dq_raw_np[:,block_start:block_end, :],
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

        gain_mp = mp.Array(mp_float_type, nx * ny)
        gain_shape = (ny, nx)
        gain_np = _arraytonumpy(gain_mp, gain_shape, dtype=mp_float_type)
        if not isinstance(gain, np.ndarray):
            gain_np[:]  = np.full(source.shape, gain, dtype=np.float32)
        else:
            gain_np[:] = gain.astype(np.float32)

        flat_mp = mp.Array(mp_float_type, nx * ny)
        flat_shape = (ny, nx)
        flat_np = _arraytonumpy(flat_mp, flat_shape, dtype=mp_float_type)
        if flat is not None:
            if not isinstance(flat, np.ndarray):
                flat_np[:] = np.full(source.shape, flat, dtype=np.float32)
            else:
                flat_np[:] = flat
        else:
            flat_np[:] = np.ones(source.shape, dtype=np.float32)


        bias_mp = mp.Array(mp_float_type, nx * ny)
        bias_shape = (ny, nx)
        bias_np = _arraytonumpy(bias_mp, bias_shape, dtype=mp_float_type)
        if not isinstance(bias, np.ndarray):
            bias_np[:] = np.full(source.shape, bias, dtype=np.float32)
        else:
            bias_np[:] = bias

        dark_mp = mp.Array(mp_float_type, nx * ny)
        dark_shape = (ny, nx)
        dark_np = _arraytonumpy(dark_mp, dark_shape, dtype=mp_float_type)
        if not isinstance(dark, np.ndarray):
            dark_np[:] = np.full(source.shape, dark, dtype=np.float32)
        else:
            dark_np[:] = dark

        read_noise_mp = mp.Array(mp_float_type, nx * ny)
        read_noise_shape = (ny, nx)
        read_noise_np = _arraytonumpy(read_noise_mp, read_noise_shape, dtype=mp_float_type)
        if read_noise is not None:
            if not isinstance(read_noise, np.ndarray):
                read_noise_np[:] = np.full(source.shape, read_noise, dtype=np.float32)
            else:
                read_noise_np[:] = read_noise
        else:
            read_noise_np[:] = np.zeros(source.shape, dtype=np.float32)


        kTC_noise_mp = mp.Array(mp_float_type, nx * ny)
        kTC_noise_shape = (ny, nx)
        kTC_noise_np = _arraytonumpy(kTC_noise_mp, kTC_noise_shape, dtype=mp_float_type)
        if kTC_noise is not None:
            if not isinstance(kTC_noise, np.ndarray):
                kTC_noise_np[:] = np.full(source.shape, kTC_noise, dtype=np.float32)
            else:
                kTC_noise_np[:] = kTC_noise
        else:
            kTC_noise_np[:] = np.zeros(source.shape, dtype=np.float32)

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
        data = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.float32)
        dq_raw = np.zeros(shape=(n_reads, source.shape[0], source.shape[1]), dtype=np.uint32)
        _outputs = _create_ramp(
            data, dq_raw,
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
        _outputs = dict(data=data, dq_raw=dq_raw)

    _meta = dict(clock_rate=1/(readtime/(source.shape[0]*source.shape[1])),# in Hz
                 mjd_start=mjd_start)
    _outputs['meta'] = _meta
    return _outputs



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
