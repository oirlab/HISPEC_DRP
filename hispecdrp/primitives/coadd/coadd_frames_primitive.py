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
        'err_calc' : 'measure',
        'data_ext' : 'rate',
        'err_ext' : 'rate_err',
        'var_poisson_ext' : 'rate_var_poisson',
        'var_rnoise_ext' : 'rate_var_rnoise',
        'dq_ext' : 'dq',
        'mid_time_ext' : 'mid_time'
    }

    def _perform(self):

        input_type = type(self.input)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            self.log.warning("Input is a single model or path, returning without coadding.")
            return dict(output=self.input)
        if input_type in (list, tuple):
            self.log.info(f"Coadding {len(self.input)} frames:\n" + "\n".join([str(i) for i in self.input]))
            output = self.coadd_group(self.input)
        elif input_type is dict:
            output = {}
            for key, group in self.input.items():
                self.log.info(f"Coadding group {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))
                output[key] = self.coadd_group(group)
        return dict(output=output)

    def coadd_group(self, group : list[datamodels.DataModel]) -> datamodels.DataModel:
        attrs = (
            self.data_ext,
            self.err_ext, self.var_poisson_ext, self.var_rnoise_ext,
            self.dq_ext,
            self.mid_time_ext
        )
        input_models = [datamodels.open(f) for f in group]
        frame_cubes = self.make_frame_cube(input_models, attrs, copy=False)
        result = coadd_frames(
            data_cube=frame_cubes[self.data_ext],
            err_cube=frame_cubes[self.err_ext],
            var_poisson_cube=frame_cubes[self.var_poisson_ext],
            var_rnoise_cube=frame_cubes[self.var_rnoise_ext],
            dq_cube=frame_cubes[self.dq_ext],
            mid_time_cube=frame_cubes[self.mid_time_ext],
            mean_calc=self.mean_calc,
            err_calc=self.err_calc
        )
        model_class_out = input_models[0].__class__
        model_out = model_class_out(
            data=dict(
                rate=result['data'],
                rate_err=result['err'],
                rate_var_poisson=result['var_poisson'],
                rate_var_rnoise=result['var_rnoise'],
                dq=result['dq'],
                mid_time=result['mid_time']
            ),
            meta=input_models[0].meta.model_copy()
        )
        model_out.add_coadd_sources(input_models)
        return model_out

    @staticmethod
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


def coadd_frames(
    data_cube : np.ndarray,
    err_cube : np.ndarray,
    var_poisson_cube : np.ndarray,
    var_rnoise_cube : np.ndarray,
    dq_cube : np.ndarray,
    mid_time_cube : np.ndarray,
    mean_calc : str = 'wmean', # TBD
    err_calc : str = 'measure', # TBD
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
        Method to use for calculating the error. Currently only 'propagate' is implemented.
    """
    # Shape
    n_frames, ny, nx = data_cube.shape

    # Create output arrays
    data_out = np.zeros((ny, nx), dtype=data_cube.dtype)
    err_out = np.zeros((ny, nx), dtype=data_cube.dtype)
    var_poisson_out = np.zeros((ny, nx), dtype=data_cube.dtype)
    var_rnoise_out = np.zeros((ny, nx), dtype=data_cube.dtype)
    dq_out = np.zeros((ny, nx), dtype=dq_cube.dtype)
    mid_times_out = np.zeros((ny, nx), dtype=mid_time_cube.dtype)
    _coadd_frames_numba(
        data_cube, err_cube, var_poisson_cube, var_rnoise_cube, dq_cube, mid_time_cube,
        data_out, err_out, var_poisson_out, var_rnoise_out, dq_out, mid_times_out,
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