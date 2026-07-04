from __future__ import annotations
from ..base_primitive import BasePrimitive
from ...datamodels import DataModel
from ... import datamodels
import numpy as np
from numba import njit, prange
from ...utils.math import bitwise_or_reduce_uint32
from .coadd_utils import sigma_clip_frames

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
        ``'wse'`` (weighted standard error, default) or ``'wstddev'``
        (weighted standard deviation).

    Returns
    -------
    output : DataModel or dict[str, DataModel]
        Coadded data model(s) with propagated errors and data quality flags.
        If the input was provided as a dictionary, a dictionary of coadded
        models is returned.
    """

    ARGUMENTS = {
        'suffix' : 'coadded',
        'output_model' : None,
        'do_sigma_clip' : False,
        'sigma_thresh_low' : 5.0,
        'sigma_thresh_high' : 5.0,
        'min_keep' : 3,
        'mean_calc' : 'wmean',
        'err_calc' : 'wse',
    }

    def _perform(self):

        input_type = type(self.input)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            self.run = False
            return dict(output=self.input)
        
        # Group by fiber illumination pattern
        if input_type in (list, tuple):
            input_grouped = datamodels.group_by_fiber_combination(self.input)
        elif input_type is dict:
            input_grouped = self.input

        # Coadd each group separately
        outputs = []
        for key, group in input_grouped.items():

            logger.info(f"Coadding group {key} ({len(group)} frames):\n" + "\n".join([str(i) for i in group]))

            output = coadd_frames(
                group,
                do_sigma_clip=self.do_sigma_clip,
                sigma_thresh_low=self.sigma_thresh_low,
                sigma_thresh_high=self.sigma_thresh_high,
                min_keep=self.min_keep,
                mean_calc=self.mean_calc,
                err_calc=self.err_calc,
                primitive=self,
            )
            if isinstance(key, tuple):
                illum_idx = next(
                    (i for i, val in enumerate(key) if str(val).upper() != "BLOCKED"),
                    0,
                )
                output.generate_filename(suffix=f"coadded-fib{illum_idx + 1}")
            outputs.append(output)

        return dict(output=outputs)


def coadd_frames(
    input : list[str | 'DataModel'],
    do_sigma_clip : bool = False,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int | float = 3,
    mean_calc : str = 'wmean',
    err_calc : str = 'wse',
    attrs : dict[str, str] | None = None,
    primitive : BasePrimitive | None = None,
):
    # If only one model provided, just return it as is.
    if len(input) == 1:
        only = input[0]
        return datamodels.open(only) if isinstance(only, str) else only

    # Open the models
    input_models = [datamodels.open(f) for f in input]

    # Get the attr names for data, err, etc.
    if attrs is None:
        attrs = _get_frame_data_attrs(input_models[0])
    
    # Create cubes
    input_frames = _make_frame_batch(input_models, attrs)

    # Coadd group
    result = _coadd_frames(
        input_frames,
        do_sigma_clip=do_sigma_clip,
        sigma_thresh_low=sigma_thresh_low,
        sigma_thresh_high=sigma_thresh_high,
        mean_calc=mean_calc,
        err_calc=err_calc,
        min_keep=min_keep,
    )

    # Output model class
    model_class_out = datamodels.open(input_models[0]).__class__
    model_out = model_class_out.from_datamodels(
        input_models,
        primitive=primitive
    )

    for attr_common, attr_unique in attrs.items():
        setattr(model_out, attr_unique, result[attr_common])

    return model_out


def _get_frame_data_attrs(input_model : datamodels.DataModel) -> dict[str, str]:
    attrs = {}
    
    # Main data attribute
    if input_model.has_extension('rate'):
        attrs['data'] = 'rate'
    elif input_model.has_extension('bias'):
        attrs['data'] = 'bias'
    elif input_model.has_extension('flat'):
        attrs['data'] = 'flat'
    else:
        raise ValueError("Could not determine data attribute for input model.")

    # Error attribute
    err_ext = next((ext for ext in input_model.EXTENSIONS if 'ERR' in ext), None)
    if err_ext is not None:
        attrs['err'] = err_ext

    # Variance attributes
    var_poisson_ext = next((ext for ext in input_model.EXTENSIONS if 'VAR_POISSON' in ext), None)
    if var_poisson_ext is not None:
        attrs['var_poisson'] = var_poisson_ext
    var_rnoise_ext = next((ext for ext in input_model.EXTENSIONS if 'VAR_RNOISE' in ext), None)
    if var_rnoise_ext is not None:
        attrs['var_rnoise'] = var_rnoise_ext

    # DQ attribute
    dq_ext = next((ext for ext in input_model.EXTENSIONS if 'DQ' in ext), None)
    if dq_ext is not None:
        attrs['dq'] = dq_ext

    # Mid-time attribute
    mid_time_ext = next((ext for ext in input_model.EXTENSIONS if 'MID_TIME' in ext), None)
    if mid_time_ext is not None:
        attrs['mid_time'] = mid_time_ext

    return attrs


def _make_frame_batch(
    input : list[str | DataModel],
    attrs : tuple[str],
) -> dict[str, np.ndarray]:
    out = {}
    n_frames = len(input)
    for i in range(n_frames):
        model = datamodels.open(input[i])
        ny, nx = getattr(model, attrs['data']).shape
        for attr_common, attr_unique in attrs.items():
            arr = getattr(model, attr_unique)
            if i == 0:
                out[attr_common] = np.zeros((n_frames, ny, nx), dtype=arr.dtype)
            out[attr_common][i, :, :] = arr
    return out


def _coadd_frames(
    input_frames : dict[str, np.ndarray],
    do_sigma_clip : bool = False,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int | float = 3,
    mean_calc : str = 'wmean',
    err_calc : str = 'wse'
) -> dict:
    """
    Coadd a cube of frames using a specified method.

    Parameters
    ----------
    input_frames : dict[str, np.ndarray]
        Dictionary containing the data cube and associated error, variance, DQ, and mid-time cubes.
    do_sigma_clip : bool, optional
        Whether to perform sigma clipping to reject outliers before coadding.
        Default is False.
    sigma_thresh_low : float | None, optional
        Lower threshold for sigma clipping. Default is None (no sigma clipping).
    sigma_thresh_high : float | None, optional
        Upper threshold for sigma clipping. Default is None (no sigma clipping).
    min_keep : int or float, optional
        Minimum number of good frames to keep after sigma clipping.
        If a float between 0 and 1, it is interpreted as a fraction of the total number of frames.
        If an integer greater than or equal to 1, it is interpreted as the absolute number of frames.
        Default is 3 (int).
    mean_calc : str, optional
        Method for calculating the mean. Currently only 'wmean' (weighted mean) is implemented.
        Default is 'wmean'.
    err_calc : str, optional
        Method for calculating the error. Currently only 'wse' (weighted standard error) is implemented.
        Default is 'wse'.
    """

    # Data
    data_cube = input_frames['data']
    dq_cube = input_frames['dq']
    err_cube = input_frames.get('err') # Optional err input, but always output

    # Shape
    n_frames, ny, nx = data_cube.shape

    # Mask for sigma clipping
    mask = np.isfinite(data_cube)
    if 'mask' in input_frames:
        mask &= input_frames['mask'].astype(bool)

    # Weights from error
    if 'err' in input_frames:
        error_frames = input_frames['err']
        weights = 1 / error_frames**2
    else:
        weights = np.ones_like(data_cube, dtype=np.float32)

    # Set weights to 0 where mask is True
    weights[~mask] = 0

    # Sigma clipping if requested
    if do_sigma_clip:
        if n_frames > 3:
            if min_keep < 1:
                min_keep = int(np.ceil(min_keep * n_frames))
            min_keep = int(min_keep)
            sigma_clip_frames(
                data_cube, mask,
                sigma_thresh_low=sigma_thresh_low,
                sigma_thresh_high=sigma_thresh_high,
                min_keep=min_keep,
            )

            bad = ~mask
            weights[bad] = 0
            dq_cube[bad] |= DO_NOT_USE
        else:
            logger.warning("Not enough frames for sigma clipping; skipping.")

    # Create output arrays for data, err, and dq
    data_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    err_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    dq_out = np.zeros((ny, nx), dtype=dq_cube.dtype)
    
    # Init output variance arrays if inputs are provided
    var_poisson_cube = input_frames.get('var_poisson')
    if var_poisson_cube is not None:
        var_poisson_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    else:
        var_poisson_out = None

    var_rnoise_cube = input_frames.get('var_rnoise')
    if var_rnoise_cube is not None:
        var_rnoise_out = np.full((ny, nx), np.nan, dtype=data_cube.dtype)
    else:
        var_rnoise_out = None

    # Init output mid time array if input is provided
    mid_time_cube = input_frames.get('mid_time')
    if mid_time_cube is not None:
        mid_times_out = np.full((ny, nx), np.nan, dtype=mid_time_cube.dtype)
    else:
        mid_times_out = None

    if err_calc == 'wse':
        err_calc_flag = 0
    elif err_calc == 'wstddev':
        err_calc_flag = 1
    else:
        raise ValueError(f"err_calc must be 'wse' or 'wstddev', got {err_calc!r}")

    _coadd_frames_numba(
        data_cube=data_cube,
        err_cube=err_cube,
        var_poisson_cube=var_poisson_cube, var_rnoise_cube=var_rnoise_cube,
        dq_cube=dq_cube,
        mid_time_cube=mid_time_cube,
        data_out=data_out,
        err_out=err_out,
        var_poisson_out=var_poisson_out, var_rnoise_out=var_rnoise_out,
        dq_out=dq_out,
        mid_times_out=mid_times_out,
        err_calc=err_calc_flag,
    )

    return dict(
        data=data_out,
        err=err_out,
        var_poisson=var_poisson_out, var_rnoise=var_rnoise_out,
        dq=dq_out,
        mid_time=mid_times_out
    )


@njit(nogil=True, parallel=False, cache=True)
def _coadd_frames_numba(
    data_cube : np.ndarray,
    data_out : np.ndarray,
    dq_cube : np.ndarray,
    err_cube : np.ndarray | None = None,
    var_poisson_cube : np.ndarray | None = None,
    var_rnoise_cube : np.ndarray | None = None,
    mid_time_cube : np.ndarray | None = None,
    err_out : np.ndarray | None = None,
    var_poisson_out : np.ndarray | None = None,
    var_rnoise_out : np.ndarray | None = None,
    dq_out : np.ndarray | None = None,
    mid_times_out : np.ndarray | None = None,
    err_calc : int = 0,  # 0=wse, 1=wstddev
):
    n_frames, ny, nx = data_cube.shape
    for i in prange(ny):
        for j in range(nx):

            # Good indices where dq == 0
            good = np.where(dq_cube[:, i, j] == 0)[0]
            n_good = len(good)

            # Case 1: No good pixels, nans
            if n_good == 0:
                data_out[i, j] = np.nan
                if err_out is not None:
                    err_out[i, j] = np.nan
                if var_poisson_out is not None:
                    var_poisson_out[i, j] = np.nan
                if var_rnoise_out is not None:
                    var_rnoise_out[i, j] = np.nan
                if dq_out is not None:
                    dq_out[i, j] = DO_NOT_USE
                if mid_times_out is not None:
                    mid_times_out[i, j] = np.nan
            # Case 2: One good pixel, just take that value
            elif n_good == 1:
                k = good[0]
                data_out[i, j] = data_cube[k, i, j]
                if err_out is not None:
                    err_out[i, j] = err_cube[k, i, j]
                if var_poisson_out is not None:
                    var_poisson_out[i, j] = var_poisson_cube[k, i, j]
                if var_rnoise_out is not None:
                    var_rnoise_out[i, j] = var_rnoise_cube[k, i, j]
                if dq_out is not None:
                    dq_out[i, j] = dq_cube[k, i, j]
                if mid_times_out is not None:
                    mid_times_out[i, j] = mid_time_cube[k, i, j]
            # Case 3: Multiple good pixels, weighted mean
            else:
                if err_cube is None:
                    w = np.ones(n_good, dtype=err_cube.dtype)
                else:
                    w = 1 / err_cube[good, i, j]**2
                data_out[i, j] = np.sum(w * data_cube[good, i, j]) / np.sum(w)

                if err_out is not None:
                    if err_calc == 0:  # wse
                        err_out[i, j] = 1 / np.sqrt(np.sum(w))
                    else:              # wstddev
                        sum_w = np.sum(w)
                        wmean = data_out[i, j]
                        wvar = np.sum(w * (data_cube[good, i, j] - wmean) ** 2) / sum_w
                        err_out[i, j] = np.sqrt(wvar)
                
                if var_poisson_out is not None:
                    var_poisson_out[i, j] = 1 / np.sum(1 / var_poisson_cube[good, i, j])
                if var_rnoise_out is not None:
                    var_rnoise_out[i, j] = 1 / np.sum(1 / var_rnoise_cube[good, i, j])
                if dq_out is not None:
                    dq_out[i, j] = bitwise_or_reduce_uint32(dq_cube[good, i, j])
                if mid_times_out is not None:
                    mid_times_out[i, j] = np.mean(mid_time_cube[good, i, j])
