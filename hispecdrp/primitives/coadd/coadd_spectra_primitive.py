from __future__ import annotations
from ..base_primitive import BasePrimitive
from ...datamodels import DataModel
from ... import datamodels
import numpy as np
from numba import njit, prange
from ...utils.math import bitwise_or_reduce_uint32
from .coadd_utils import sigma_clip_spectra
import logging

logger = logging.getLogger(__name__)

__all__ = ['CoaddSpectraPrimitive','coadd_spectra']

from ...datamodels.dqflags import DQ_FLAGS
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


class CoaddSpectraPrimitive(BasePrimitive):
    """
    Coadd multiple spectral-like models of the same kind to increase signal-to-noise.

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

        input_type = type(self.input)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            self.run = False
            return dict(output=self.input)

        # Determine which fibers to coadd
        input_models = [datamodels.open(f) for f in self.input]
        do_fibers, do_orders = self.get_do_traces(input_models[0], self.do_fibers, self.do_orders)

        logger.info(f"Coadding {len(self.input)} spectra):\n" + "\n".join([str(m) for m in input_models]))

        # Coadd
        output_model = coadd_spectra(
            input_models,
            do_sigma_clip=self.do_sigma_clip,
            sigma_thresh_low=self.sigma_thresh_low,
            sigma_thresh_high=self.sigma_thresh_high,
            min_keep=self.min_keep,
            mean_calc=self.mean_calc,
            err_calc=self.err_calc,
            do_orders=do_orders,
            do_fibers=do_fibers,
            primitive=self,
        )

        return dict(output=output_model)

def coadd_spectra(
    input : list[str | 'DataModel'],
    do_fibers : list[str | int],
    do_orders : list[str | int],
    do_sigma_clip : bool = False,
    sigma_thresh_low : float | None = None,
    sigma_thresh_high : float | None = None,
    min_keep : int | float = 3,
    mean_calc : str = 'wmean',
    err_calc : str = 'wse',
    attrs : dict[str, str] | None = None,
    primitive : BasePrimitive | None = None,
):

    # If only one input is provided, return it without coadding
    if len(input) == 1:
        return datamodels.open(input[0])

    # Open input models
    input_models = [datamodels.open(f) for f in input]

    # Init output model
    model_class_out = input_models[0].__class__
    model_out = model_class_out.from_datamodels(input_models, primitive=primitive)
    orders = model_out.make_order_vec().tolist()

    # Loop over fibers/orders
    for fiber in do_fibers:
        for order in do_orders:

            # Make the batch for this fiber/order
            attrs = _get_spec_data_attrs(input_models[0], fiber=fiber)
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
    """Build (n_frames, n_pix) arrays from a list of spectral models."""
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


def _get_spec_data_attrs(input_model : datamodels.DataModel, fiber : str | int | None) -> dict[str, str]:

    attrs = {}

    # Data and error attributes
    is_cal = input_model.has_extension(f'spec_fib{fiber}')
    if is_cal:
        attrs['data'] = f'spec_fib{fiber}'
        attrs['err'] = f'spec_err_fib{fiber}'
        attrs['var_poisson'] = f'spec_var_poisson_fib{fiber}'
        attrs['var_rnoise'] = f'spec_var_rnoise_fib{fiber}'
    else:
        attrs['data'] = f'spec_uncal_fib{fiber}'
        attrs['err'] = f'spec_err_uncal_fib{fiber}'
        attrs['var_poisson'] = f'spec_var_poisson_uncal_fib{fiber}'
        attrs['var_rnoise'] = f'spec_var_rnoise_uncal_fib{fiber}'

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
    data_2d = input_frames['data']
    dq_2d = input_frames['dq']
    err_2d = input_frames.get('err')

    n_frames, n_pix = data_2d.shape

    mask = np.isfinite(data_2d) & (dq_2d == 0)

    if do_sigma_clip:
        if n_frames > 3:
            if min_keep < 1:
                min_keep = int(np.ceil(min_keep * n_frames))
            min_keep = int(min_keep)
            sigma_clip_spectra(
                data_2d, mask,
                sigma_thresh_low=sigma_thresh_low,
                sigma_thresh_high=sigma_thresh_high,
                min_keep=min_keep,
            )
            dq_2d[~mask] |= DO_NOT_USE
        else:
            logger.warning("Not enough frames for sigma clipping; skipping.")

    data_out = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    err_out  = np.full(n_pix, np.nan, dtype=data_2d.dtype)
    dq_out   = np.zeros(n_pix, dtype=dq_2d.dtype)

    var_poisson_2d = input_frames.get('var_poisson')
    var_rnoise_2d  = input_frames.get('var_rnoise')
    mid_time_2d    = input_frames.get('mid_time')

    var_poisson_out = np.full(n_pix, np.nan, dtype=data_2d.dtype) if var_poisson_2d is not None else None
    var_rnoise_out  = np.full(n_pix, np.nan, dtype=data_2d.dtype) if var_rnoise_2d  is not None else None
    mid_times_out   = np.full(n_pix, np.nan, dtype=mid_time_2d.dtype) if mid_time_2d is not None else None

    if err_calc == 'wse':
        err_calc_flag = 0
    elif err_calc == 'wstddev':
        err_calc_flag = 1
    else:
        raise ValueError(f"err_calc must be 'wse' or 'wstddev', got {err_calc!r}")

    _coadd_spectra_numba(
        spec_2d=data_2d,
        dq_2d=dq_2d,
        spec_out=data_out,
        err_2d=err_2d,
        var_poisson_2d=var_poisson_2d,
        var_rnoise_2d=var_rnoise_2d,
        mid_time_2d=mid_time_2d,
        err_out=err_out,
        var_poisson_out=var_poisson_out,
        var_rnoise_out=var_rnoise_out,
        dq_out=dq_out,
        mid_times_out=mid_times_out,
        err_calc=err_calc_flag,
    )

    return dict(
        data=data_out,
        err=err_out,
        var_poisson=var_poisson_out,
        var_rnoise=var_rnoise_out,
        dq=dq_out,
        mid_time=mid_times_out,
    )


@njit(nogil=True, parallel=False, cache=True)
def _coadd_spectra_numba(
    spec_2d : np.ndarray,
    dq_2d : np.ndarray,
    spec_out : np.ndarray,
    err_2d : np.ndarray | None = None,
    var_poisson_2d : np.ndarray | None = None,
    var_rnoise_2d : np.ndarray | None = None,
    mid_time_2d : np.ndarray | None = None,
    err_out : np.ndarray | None = None,
    var_poisson_out : np.ndarray | None = None,
    var_rnoise_out : np.ndarray | None = None,
    dq_out : np.ndarray | None = None,
    mid_times_out : np.ndarray | None = None,
    err_calc : int = 0,  # 0=wse, 1=wstddev
):
    _, n_pix = spec_2d.shape
    for i in prange(n_pix):
        good = np.where(dq_2d[:, i] == 0)[0]
        n_good = len(good)

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
        elif n_good == 1:
            k = good[0]
            spec_out[i] = spec_2d[k, i]
            if err_out is not None:
                err_out[i] = err_2d[k, i] if err_2d is not None else np.nan
            if var_poisson_out is not None:
                var_poisson_out[i] = var_poisson_2d[k, i]
            if var_rnoise_out is not None:
                var_rnoise_out[i] = var_rnoise_2d[k, i]
            if dq_out is not None:
                dq_out[i] = dq_2d[k, i]
            if mid_times_out is not None:
                mid_times_out[i] = mid_time_2d[k, i]
        else:
            if err_2d is None:
                w = np.ones(n_good, dtype=np.float32)
            else:
                w = 1 / err_2d[good, i]**2
            sum_w = np.sum(w)
            wmean = np.sum(w * spec_2d[good, i]) / sum_w
            spec_out[i] = wmean

            if err_out is not None:
                if err_calc == 0:  # wse
                    err_out[i] = 1 / np.sqrt(sum_w)
                else:              # wstddev
                    wvar = np.sum(w * (spec_2d[good, i] - wmean) ** 2) / sum_w
                    err_out[i] = np.sqrt(wvar)

            if var_poisson_out is not None:
                var_poisson_out[i] = 1 / np.sum(1 / var_poisson_2d[good, i])
            if var_rnoise_out is not None:
                var_rnoise_out[i] = 1 / np.sum(1 / var_rnoise_2d[good, i])
            if dq_out is not None:
                dq_out[i] = bitwise_or_reduce_uint32(dq_2d[good, i])
            if mid_times_out is not None:
                mid_times_out[i] = np.mean(mid_time_2d[good, i])
