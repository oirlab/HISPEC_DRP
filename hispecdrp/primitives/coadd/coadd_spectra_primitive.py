from __future__ import annotations
from ..base_primitive import BasePrimitive
from ...datamodels import DataModel
from ... import datamodels
import numpy as np
from numba import njit, prange
from ..utils.array_utils import bitwise_or_reduce_uint32
from .coadd_frames_primitive import sigma_clip
import logging

logger = logging.getLogger(__name__)

__all__ = ['CoaddSpectraPrimitive', 'coadd_spectra']

from ...datamodels.dqflags import DQ_FLAGS
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


class CoaddSpectraPrimitive(BasePrimitive):
    """
    Coadd multiple spectral-like models (e.g. Level 2 spectra) of the same kind to increase signal-to-noise.

    An ancestor FITS extension is created to include the list of filenames that were combined.

    This primitive combines input data products using a weighted average (if mean_calc = 'wmean'),
    where the weights are given by the inverse variances
    (1 / model.err^2). The output error and variance arrays are computed
    using the corresponding weighted standard error. Data quality (DQ)
    flags from all inputs are propagated to the output using a bitwise OR.

    Parameters
    ----------
    input : list[str or DataModel]
        Input files or data models to coadd. All inputs are combined into a
        single output model (each fiber/order trace is coadded independently).
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'coadded'``.
    mean_calc : str, optional
        Method for calculating the output data values. Currently, only
        ``'wmean'`` (weighted mean) is implemented.
    err_calc : str, optional
        Method for calculating the output error and variance. Currently,
        only ``'wse'`` (weighted standard error) is implemented.
    do_fibers : list of int, optional
        Fibers to coadd. If None, all fibers on the input model are used.
    do_orders : list of int, optional
        Orders to coadd. If None, all orders on the input model are used.

    Returns
    -------
    output : DataModel
        Coadded data model with propagated errors and data quality flags.
    """

    PARAMETERS = {
        'suffix' : 'coadded',
        'output_model' : None,
        'do_sigma_clip' : False,
        'sigma_thresh_low' : 5.0,
        'sigma_thresh_high' : 5.0,
        'min_keep' : 3,
        'mean_calc' : 'wmean',
        'err_calc' : 'wse',
        'do_fibers' : None,
        'do_orders' : None,
    }

    def _perform(self):

        input_type = type(self.dataset)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            self.run = False
            return dict(output=self.dataset)

        # Determine which fibers/orders to coadd
        input_models = [datamodels.open(f) for f in self.dataset]
        do_fibers, do_orders = self.get_do_traces(input_models[0], self.do_fibers, self.do_orders)

        logger.info(f"Coadding {len(self.dataset)} spectra:\n" + "\n".join([str(m) for m in input_models]))

        # Coadd
        output_model = coadd_spectra(
            input_models,
            do_fibers=do_fibers,
            do_orders=do_orders,
            do_sigma_clip=self.do_sigma_clip,
            sigma_thresh_low=self.sigma_thresh_low,
            sigma_thresh_high=self.sigma_thresh_high,
            min_keep=self.min_keep,
            mean_calc=self.mean_calc,
            err_calc=self.err_calc,
            primitive=self,
        )

        return dict(output=output_model)


def coadd_spectra(
    input : list[str | 'DataModel'],
    do_fibers : list[str | int],
    do_orders : list[int],
    do_sigma_clip : bool = False,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int | float = 3,
    mean_calc : str = 'wmean',
    err_calc : str = 'wse',
    primitive : BasePrimitive | None = None,
):
    # If only one input is provided, return it as is.
    if len(input) == 1:
        only = input[0]
        return datamodels.open(only) if isinstance(only, str) else only

    # Open input models
    input_models = [datamodels.open(f) for f in input]

    # Init output model
    model_class_out = input_models[0].__class__
    model_out = model_class_out.from_datamodels(input_models, primitive=primitive)
    orders = model_out.make_order_vec().tolist()

    # Loop over fibers/orders, coadding each spectral trace separately
    for fiber in do_fibers:

        attrs = _get_spec_data_attrs(input_models[0], fiber=fiber)

        for order in do_orders:

            # Make the batch for this fiber/order
            spec_batch = _make_spec_batch(input_models, attrs, order=order)

            # Do coadding
            result = _coadd_spectra(
                spec_batch,
                do_sigma_clip=do_sigma_clip,
                sigma_thresh_low=sigma_thresh_low,
                sigma_thresh_high=sigma_thresh_high,
                min_keep=min_keep,
                mean_calc=mean_calc,
                err_calc=err_calc,
            )

            # Store results in the output model
            idx_order = orders.index(order)
            for attr_common, attr_unique in attrs.items():
                if not model_out.has_extension_value(attr_unique):
                    setattr(
                        model_out, attr_unique,
                        np.zeros(input_models[0].shape, dtype=result[attr_common].dtype)
                    )
                getattr(model_out, attr_unique)[idx_order] = result[attr_common]

    return model_out


def _make_spec_batch(
    input_models : list[str | DataModel],
    attrs : dict[str, str],
    order : int
) -> dict[str, np.ndarray]:
    """Build (n_frames, n_pix) arrays for one order from a list of spectral models."""
    out = {}
    n_frames = len(input_models)
    for i in range(n_frames):
        model = datamodels.open(input_models[i])
        n_pix = model.shape[1]
        orders = model.make_order_vec().tolist()
        idx_order = orders.index(order)
        for attr_common, attr_unique in attrs.items():
            arr = getattr(model, attr_unique)
            if i == 0:
                out[attr_common] = np.zeros((n_frames, n_pix), dtype=arr.dtype)
            out[attr_common][i, :] = arr[idx_order]
    return out


def _get_spec_data_attrs(input_model : datamodels.DataModel, fiber : str | int) -> dict[str, str]:
    attrs = {}

    # Data and error attributes; fall back to the uncalibrated spectrum if the
    # calibrated one hasn't been populated yet.
    if input_model.has_extension_value(f'spec_fib{fiber}'):
        attrs['data'] = f'spec_fib{fiber}'
        attrs['err'] = f'spec_err_fib{fiber}'
        attrs['var_poisson'] = f'spec_var_poisson_fib{fiber}'
        attrs['var_rnoise'] = f'spec_var_rnoise_fib{fiber}'
    else:
        attrs['data'] = f'spec_uncal_fib{fiber}'
        attrs['err'] = f'spec_uncal_err_fib{fiber}'
        attrs['var_poisson'] = f'spec_uncal_var_poisson_fib{fiber}'
        attrs['var_rnoise'] = f'spec_uncal_var_rnoise_fib{fiber}'

    # DQ
    attrs['dq'] = f'spec_dq_fib{fiber}'

    # Mid-time
    attrs['mid_time'] = f'mid_time_fib{fiber}'

    return attrs


def _coadd_spectra(
    input_frames : dict[str, np.ndarray],
    do_sigma_clip : bool = False,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int | float = 3,
    mean_calc : str = 'wmean',
    err_calc : str = 'wse',
) -> dict:
    """
    Coadd a (n_frames, n_pix) stack of 1D spectra using a specified method.

    Parameters
    ----------
    input_frames : dict[str, np.ndarray]
        Dictionary containing the data array and associated error, variance, DQ, and mid-time arrays.
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
    data_2d = input_frames['data']
    dq_2d = input_frames['dq']
    err_2d = input_frames.get('err')  # Optional err input, but always output

    # Shape
    n_frames, n_pix = data_2d.shape

    # Mask
    mask = np.isfinite(data_2d)
    if 'dq' in input_frames:
        mask &= dq_2d == 0

    # Weights from error
    if 'err' in input_frames:
        weights = 1 / err_2d**2
    else:
        weights = np.ones_like(data_2d, dtype=np.float32)

    # Set weights to 0 where mask is False
    weights[~mask] = 0

    # Sigma clipping if requested
    if do_sigma_clip and n_frames > 3:
        if min_keep < 1:
            min_keep = int(np.ceil(min_keep * n_frames))
        min_keep = int(min_keep)
        sigma_clip_spectra(
            data_2d, mask,
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=min_keep,
        )

        bad = ~mask
        weights[bad] = 0
        dq_2d[bad] |= DO_NOT_USE

    # Create output arrays
    data_out = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    err_out = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    dq_out = np.zeros(n_pix, dtype=dq_2d.dtype)

    var_poisson_2d = input_frames.get('var_poisson')
    var_rnoise_2d = input_frames.get('var_rnoise')

    if var_poisson_2d is not None:
        var_poisson_out = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    else:
        var_poisson_out = None
    if var_rnoise_2d is not None:
        var_rnoise_out = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    else:
        var_rnoise_out = None

    mid_time_2d = input_frames.get('mid_time')

    if mid_time_2d is not None:
        mid_times_out = np.full(n_pix, np.nan, dtype=mid_time_2d.dtype)
    else:
        mid_times_out = None

    _coadd_spectra_numba(
        spec_2d=data_2d,
        err_2d=err_2d,
        var_poisson_2d=var_poisson_2d, var_rnoise_2d=var_rnoise_2d,
        dq_2d=dq_2d,
        mid_time_2d=mid_time_2d,
        spec_out=data_out,
        err_out=err_out,
        var_poisson_out=var_poisson_out, var_rnoise_out=var_rnoise_out,
        dq_out=dq_out,
        mid_times_out=mid_times_out,
    )

    return dict(
        data=data_out,
        err=err_out,
        var_poisson=var_poisson_out, var_rnoise=var_rnoise_out,
        dq=dq_out,
        mid_time=mid_times_out
    )


@njit(nogil=True, parallel=False, cache=True)
def _coadd_spectra_numba(
    spec_2d : np.ndarray,
    spec_out : np.ndarray,
    dq_2d : np.ndarray,
    err_2d : np.ndarray | None = None,
    var_poisson_2d : np.ndarray | None = None,
    var_rnoise_2d : np.ndarray | None = None,
    mid_time_2d : np.ndarray | None = None,
    err_out : np.ndarray | None = None,
    var_poisson_out : np.ndarray | None = None,
    var_rnoise_out : np.ndarray | None = None,
    dq_out : np.ndarray | None = None,
    mid_times_out : np.ndarray | None = None,
):
    n_frames, n_pix = spec_2d.shape
    for i in range(n_pix):

        # Good indices where dq == 0
        good = np.where(dq_2d[:, i] == 0)[0]
        n_good = len(good)

        # Case 1: No good pixels, nans
        if n_good == 0:
            spec_out[i] = np.nan
            if err_out is not None:
                err_out[i] = np.nan
            if var_poisson_out is not None:
                var_poisson_out[i] = np.nan
            if var_rnoise_out is not None:
                var_rnoise_out[i] = np.nan
            if dq_out is not None:
                dq_out[i] = DO_NOT_USE
            if mid_times_out is not None:
                mid_times_out[i] = np.nan
        # Case 2: One good pixel, just take that value
        elif n_good == 1:
            k = good[0]
            spec_out[i] = spec_2d[k, i]
            if err_out is not None:
                err_out[i] = err_2d[k, i]
            if var_poisson_out is not None:
                var_poisson_out[i] = var_poisson_2d[k, i]
            if var_rnoise_out is not None:
                var_rnoise_out[i] = var_rnoise_2d[k, i]
            if dq_out is not None:
                dq_out[i] = dq_2d[k, i]
            if mid_times_out is not None:
                mid_times_out[i] = mid_time_2d[k, i]
        # Case 3: Multiple good pixels, weighted mean
        else:
            if err_2d is None:
                w = np.ones(n_good, dtype=np.float32)
            else:
                w = 1 / err_2d[good, i]**2
            spec_out[i] = np.sum(w * spec_2d[good, i]) / np.sum(w)

            if err_out is not None:
                err_out[i] = 1 / np.sqrt(np.sum(w))

            if var_poisson_out is not None:
                var_poisson_out[i] = 1 / np.sum(1 / var_poisson_2d[good, i])
            if var_rnoise_out is not None:
                var_rnoise_out[i] = 1 / np.sum(1 / var_rnoise_2d[good, i])
            if dq_out is not None:
                dq_out[i] = bitwise_or_reduce_uint32(dq_2d[good, i])
            if mid_times_out is not None:
                mid_times_out[i] = np.mean(mid_time_2d[good, i])


@njit(nogil=True, parallel=True, cache=True)
def sigma_clip_spectra(
    data_2d : np.ndarray,
    mask_2d : np.ndarray,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int = 3,
):
    _, n_pix = data_2d.shape
    for j in prange(n_pix):
        sigma_clip(
            data_2d[:, j],
            mask_2d[:, j],
            sigma_thresh_low=sigma_thresh_low,
            sigma_thresh_high=sigma_thresh_high,
            min_keep=min_keep,
        )
