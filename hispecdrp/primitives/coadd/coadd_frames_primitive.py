from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from numba import njit, prange
from ...utils.math import weighted_mean, bitwise_or_reduce_uint32
import logging

logger = logging.getLogger(__name__)

__all__ = ['CoaddFramesPrimitive','coadd_frames']

from ...datamodels.dqflags import DQ_FLAGS
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


class CoaddFramesPrimitive(BasePrimitive):
    """
    Coadd multiple level 0 (rate maps) frames of the same kind to increase signal-to-noise.
    Works on darks and flats calibration models too.

    An ancestor FITS extension is created to include the list of filenames that were combined.

    This primitive combines input data products using a weighted average (if mean_calc = 'wmean'),
    where the weights are given by the inverse variances
    (1 / model.err^2). The output error and variance arrays are computed
    using the corresponding weighted standard error. Data quality (DQ)
    flags from all inputs are propagated to the output using a bitwise OR.

    Parameters
    ----------
    input : list[str or DataModel] or dict[str, str or DataModel] or L0Model
        Input files or data models to coadd. Inputs may be provided as a list
        or as a dictionary; when a dictionary is given, each group of inputs
        is coadded separately.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'coadded'``.
    mean_calc : str, optional
        Method for calculating the output data values. Currently, only
        ``'wmean'`` (weighted mean) is implemented.
    err_calc : str, optional
        Method for calculating the output error and variance. Currently,
        only ``'wse'`` (weighted standard error) is implemented.

    Returns
    -------
    output : DataModel or dict[str, DataModel]
        Coadded data model(s) with propagated errors and data quality flags.
        If the input was provided as a dictionary, a dictionary of coadded
        models is returned.
    """

    ARGUMENTS = {
        'suffix' : 'coadded',
        'mean_calc' : 'wmean',
        'err_calc' : 'wse',
        'output_model' : None,
    }

    def _perform(self):

        input_type = type(self.input)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            return dict(output=self.input)
        if input_type in (list, tuple):
            input_grouped = datamodels.group_by_fiber_combination(self.input)
            output = []
            for key, group in input_grouped.items():
                logger.info(f"Coadding group with objects {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))
                output.append(coadd_group(group, primitive=self))
        elif input_type is dict:
            output = {}
            for key, group in self.input.items():
                logger.info(f"Coadding group {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))
                output[key] = coadd_group(group, primitive=self)

        return dict(output=output)


# Several functions below are used by other primitives
def coadd_group(
    models : list[datamodels.DataModel],
    model_class_out : type[datamodels.DataModel] | None = None,
    mean_calc : str = 'wmean', # TBD
    err_calc : str = 'wse', # TBD
    primitive : BasePrimitive | None = None,
) -> datamodels.DataModel:
    """
    Coadd multiple level 0 (rate maps) frames of the same kind to increase signal-to-noise.
    Works on darks and flats calibration models too.

    Parameters
    ----------
    models : list[str or DataModel]
        Input files or data models to coadd.
        Method for calculating the output data values. Currently, only
        ``'wmean'`` (weighted mean) is implemented.
    err_calc : str, optional
        Method for calculating the output error and variance. Currently,
        only ``'wse'`` (weighted standard error) is implemented.

    Returns
    -------
    model_out : DataModel or dict[str, DataModel]
        Coadded data model(s) with propagated errors and data quality flags.
        If the input was provided as a dictionary, a dictionary of coadded
        models is returned.
    """

    input_models = [datamodels.open(f) for f in models]
    attrs = get_data_attrs(input_models[0])
    frame_cubes = make_frame_cube(input_models, tuple(attrs.values()), copy=False)
    result = coadd_frames(
        data_cube=frame_cubes[attrs['data_ext']],
        err_cube=frame_cubes[attrs['err_ext']],
        var_poisson_cube=frame_cubes[attrs['var_poisson_ext']],
        var_rnoise_cube=frame_cubes[attrs['var_rnoise_ext']],
        dq_cube=frame_cubes[attrs['dq_ext']],
        mid_time_cube=frame_cubes[attrs['mid_time_ext']] if 'mid_time_ext' in attrs else None,
        mean_calc=mean_calc,
        err_calc=err_calc
    )
    data = {
        attrs['data_ext']: result['data'],
        attrs['err_ext']: result['err'],
        attrs['var_poisson_ext']: result['var_poisson'],
        attrs['var_rnoise_ext']: result['var_rnoise'],
        attrs['dq_ext']: result['dq']
    }
    if 'mid_time_ext' in attrs:
        data[attrs['mid_time_ext']] = result['mid_time']

    if model_class_out is None:
        model_class_out = input_models[0].__class__
    model_class_out = input_models[0].__class__
    model_out = model_class_out.from_datamodels(
        input=input_models,
        data=data,
        append_ancestors=False,
    )
    if primitive is not None:
        primitive.add_data_ancestors(model_out, input_models)

    return model_out

def make_frame_cube(
    input : list[str | datamodels.DataModel],
    attrs : tuple[str, ...],
    copy : bool = False
) -> dict[str, np.ndarray]:
    """
    Create a cube from a list of input frames.

    Parameters
    ----------
    input : list[str | datamodels.DataModel]
        List of input frames to combine.
    attrs : tuple[str, ...]
        List of attributes to extract from the input frames.
    copy : bool, optional
        If True, copy the data from the input frames.

    Returns
    -------
    output : dict[str, np.ndarray]
        Dictionary of cubes with shape (Nframes, Ny, Nx).
    """
    out = {}
    n_frames = len(input)
    for i in range(n_frames):
        model = datamodels.open(input[i])
        ny, nx = getattr(model, attrs[0]).shape
        for attr in attrs:
            arr = getattr(model, attr)
            if i == 0:
                out[attr] = np.zeros((n_frames, ny, nx), dtype=arr.dtype)
            if copy and isinstance(input[i], datamodels.DataModel):
                out[attr][i, :, :] = arr.copy()
            else:
                out[attr][i, :, :] = arr
    return out

def get_data_attrs(input_model : datamodels.DataModel) -> dict[str, str]:
    if isinstance(input_model, datamodels.L0Model):
        return dict(
            data_ext='rate',
            err_ext='rate_err',
            var_poisson_ext='rate_var_poisson',
            var_rnoise_ext='rate_var_rnoise',
            dq_ext='dq',
            mid_time_ext='mid_time'
        )
    elif isinstance(input_model, datamodels.DarkModel):
        return dict(
            data_ext='rate',
            err_ext='rate_err',
            var_poisson_ext='rate_var_poisson',
            var_rnoise_ext='rate_var_rnoise',
            dq_ext='dq'
        )
    elif isinstance(input_model, datamodels.DetectorFlatModel):
        return dict(
            data_ext='flat',
            err_ext='flat_err',
            var_poisson_ext='flat_var_poisson',
            var_rnoise_ext='flat_var_rnoise',
            dq_ext='dq'
        )

def coadd_frames(
    data_cube : np.ndarray,
    err_cube : np.ndarray,
    var_poisson_cube : np.ndarray,
    var_rnoise_cube : np.ndarray,
    dq_cube : np.ndarray,
    mid_time_cube : np.ndarray = None,
    mean_calc : str = 'wmean', # TBD
    err_calc : str = 'wse', # TBD
) -> dict:
    """
    Coadd a cube of frames using a specified method.

    Parameters
    ----------
    data_cube : np.ndarray
        Cube of data to coadd with shape (Nframes, Ny, Nx).
    err_cube : np.ndarray
        Cube of errors with shape (Nframes, Ny, Nx).
    var_poisson_cube : np.ndarray
        Cube of Poisson variances with shape (Nframes, Ny, Nx).
    var_rnoise_cube : np.ndarray
        Cube of read noise variances with shape (Nframes, Ny, Nx).
    dq_cube : np.ndarray
        Cube of data quality flags with shape (Nframes, Ny, Nx).
    mid_time_cube : np.ndarray
        Cube of mid-times with shape (Nframes, Ny, Nx).
    mean_calc : str, optional
        Method to use for calculating the mean. Currently only 'wmean' (weighted mean) is implemented.
    err_calc : str, optional
        Method to use for calculating the error. Currently only 'wse' (Weighted Standard Error) is implemented.
    """
    if mean_calc != 'wmean' and err_calc != 'wse':
        raise ValueError("Only 'wmean' is currently implemented for mean_calc and 'wse' for err_calc.")

    # Shape
    n_frames, ny, nx = data_cube.shape

    # Create output arrays
    data_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    err_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    var_poisson_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    var_rnoise_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    dq_out = np.full((ny, nx), np.nan, dtype=dq_cube.dtype)
    if mid_time_cube is not None:
        mid_times_out = np.full((ny, nx), np.nan, dtype=mid_time_cube.dtype)
    else:
        mid_time_cube = np.full((n_frames,ny, nx), np.nan, dtype=data_cube.dtype)
        mid_times_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)

    _coadd_frames_numba(
        data_cube=data_cube, err_cube=err_cube,
        var_poisson_cube=var_poisson_cube, var_rnoise_cube=var_rnoise_cube,
        dq_cube=dq_cube,
        mid_time_cube=mid_time_cube,
        data_out=data_out, err_out=err_out,
        var_poisson_out=var_poisson_out, var_rnoise_out=var_rnoise_out,
        dq_out=dq_out,
        mid_times_out=mid_times_out,
    )

    return dict(
        data=data_out,
        err=err_out, var_poisson=var_poisson_out, var_rnoise=var_rnoise_out,
        dq=dq_out,
        mid_time=mid_times_out
    )

@njit(nogil=True, parallel=True, cache=True)
def _coadd_frames_numba(
    data_cube : np.ndarray,
    err_cube : np.ndarray,
    var_poisson_cube : np.ndarray,
    var_rnoise_cube : np.ndarray,
    dq_cube : np.ndarray,
    mid_time_cube : np.ndarray,
    data_out : np.ndarray,
    err_out : np.ndarray,
    var_poisson_out : np.ndarray,
    var_rnoise_out : np.ndarray,
    dq_out : np.ndarray,
    mid_times_out : np.ndarray,
):
    n_frames, ny, nx = data_cube.shape
    for i in prange(ny):
        for j in range(nx):
            good = np.where(dq_cube[:, i, j] == 0)[0]
            n_good = len(good)
            if n_good == 0:
                data_out[i, j] = np.nan
                err_out[i, j] = np.nan
                var_poisson_out[i, j] = np.nan
                var_rnoise_out[i, j] = np.nan
                dq_out[i, j] = DO_NOT_USE
                mid_times_out[i, j] = np.nan
            elif n_good == 1:
                k = good[0]
                data_out[i, j] = data_cube[k, i, j]
                err_out[i, j] = err_cube[k, i, j]
                var_poisson_out[i, j] = var_poisson_cube[k, i, j]
                var_rnoise_out[i, j] = var_rnoise_cube[k, i, j]
                dq_out[i, j] = dq_cube[k, i, j]
                mid_times_out[i, j] = mid_time_cube[k, i, j]
            else:
                w = 1 / err_cube[good, i, j]**2
                data_out[i, j] = weighted_mean(data_cube[good, i, j], w)
                err_out[i, j] = np.sum(1 / err_cube[good, i, j]**2)**-0.5
                var_poisson_out[i, j] = np.sum(1 / var_poisson_cube[good, i, j])**-1
                var_rnoise_out[i, j] = np.sum(1 / var_rnoise_cube[good, i, j])**-1
                dq_out[i, j] = bitwise_or_reduce_uint32(dq_cube[good, i, j])
                mid_times_out[i, j] = np.mean(mid_time_cube[good, i, j])

