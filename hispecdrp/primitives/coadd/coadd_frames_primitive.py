from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from numba import njit, prange
from ...utils.math import weighted_mean, bitwise_or_reduce_uint32

__all__ = ['CoaddFramesPrimitive']

from ...datamodels.dqflags import DQ_FLAGS
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


class CoaddFramesPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'coadded',
        'mean_calc' : 'wmean',
        'err_calc' : 'wmean',
        'output_model' : None,
    }

    def _perform(self):

        input_type = type(self.input)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            self.log.warning("Input is a single model or path, returning without coadding.")
            return dict(output=self.input)
        if input_type in (list, tuple):
            input_grouped = datamodels.group_input_by_fiber_object(self.input)
            output = []
            for key, group in input_grouped.items():
                self.log.info(f"Coadding group with objects {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))
                output.append(coadd_group(group, self.__class__.__name__))
        elif input_type is dict:
            output = {}
            for key, group in self.input.items():
                self.log.info(f"Coadding group {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))
                output[key] = coadd_group(group, self.__class__.__name__)
        return dict(output=output)


def coadd_group(group : list[datamodels.DataModel],
                primitive_name : str = 'None',
                mean_calc : str = 'wmean', # TBD
                err_calc : str = 'wmean', # TBD
                ) -> datamodels.DataModel:

    input_models = [datamodels.open(f) for f in group]
    attrs = get_data_attrs(input_models[0])
    frame_cubes = make_frame_cube(input_models, tuple(attrs.values()), copy=False)
    result = coadd_frames(
        data_cube=frame_cubes[attrs['data_ext']],
        err_cube=frame_cubes[attrs['err_ext']],
        var_poisson_cube=frame_cubes[attrs['var_poisson_ext']],
        var_rnoise_cube=frame_cubes[attrs['var_rnoise_ext']],
        dq_cube=frame_cubes[attrs['dq_ext']],
        mid_time_cube=frame_cubes[attrs['mid_time_ext']],
        mean_calc=mean_calc,
        err_calc=err_calc
    )
    data = {
        attrs['data_ext']: result['data'],
        attrs['err_ext']: result['err'],
        attrs['var_poisson_ext']: result['var_poisson'],
        attrs['var_rnoise_ext']: result['var_rnoise'],
        attrs['dq_ext']: result['dq'],
        attrs['mid_time_ext']: result['mid_time']
    }
    meta_merged = datamodels.merge_meta(input_models)
    model_class_out = input_models[0].__class__
    model_out = model_class_out(data=data, meta=meta_merged, config=input_models[0].config)
    model_out.add_data_ancestors(input_models,primitive_name=primitive_name)
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
            data_ext='dark',
            err_ext='dark_err',
            var_poisson_ext='dark_var_poisson',
            var_rnoise_ext='dark_var_rnoise',
            dq_ext='dq',
            mid_time_ext='mid_time'
        )
    elif isinstance(input_model, datamodels.DetectorFlatModel):
        return dict(
            data_ext='flat',
            err_ext='flat_err',
            var_poisson_ext='flat_var_poisson',
            var_rnoise_ext='flat_var_rnoise',
            dq_ext='dq',
            mid_time_ext='mid_time'
        )

def coadd_frames(
    data_cube : np.ndarray,
    err_cube : np.ndarray,
    var_poisson_cube : np.ndarray,
    var_rnoise_cube : np.ndarray,
    dq_cube : np.ndarray,
    mid_time_cube : np.ndarray,
    mean_calc : str = 'wmean', # TBD
    err_calc : str = 'wmean', # TBD
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
        Method to use for calculating the error. Currently only 'wmean' is implemented.
    """
    if mean_calc != 'wmean' and err_calc != 'wmean':
        raise ValueError("Only 'wmean' is currently implemented for mean_calc and err_calc.")

    # Shape
    n_frames, ny, nx = data_cube.shape

    # Create output arrays
    data_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    err_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    var_poisson_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    var_rnoise_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    dq_out = np.full((ny, nx), np.nan, dtype=dq_cube.dtype)
    mid_times_out = np.full((ny, nx), np.nan, dtype=mid_time_cube.dtype)
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

