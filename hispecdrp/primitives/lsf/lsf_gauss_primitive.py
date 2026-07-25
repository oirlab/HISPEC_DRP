from ..base_primitive import BasePrimitive
from ... import datamodels
import logging

import numpy as np
from ...utils.math import solve_chebyshev2d_ols

logger = logging.getLogger(__name__)

__all__ = ['CalculateGaussianLSFPrimitive']


class CalculateGaussianLSFPrimitive(BasePrimitive):

    """
    NOTE: *Under development*

    Calculate a Gaussian LSF model by fitting a 2D Chebyshev polynomial to the Gaussian FWHMs as a function of pixel and echelle order.

    The unstructured set of points {x_i, m_i, sigma_i} is fit with a 2D Chebyshev polynomial and then evaluated at each detecor pixel to determine the appropriate Gaussian LSF width at that pixel and wavelength. The LSF width is never updated according to any drifts. The LSF can then be calculated at any sub-pixel or wavelength by interpolating the resulting sigmas at each pixel.

    Parameters
    ----------
    input : str or DataModel
        Input data model or path to input data model. The input data model must contain the attribute ``lfc_fitparams_fib#`` for each fiber, where # is the fiber number.
    do_fibers : list[int] | None
        List of fibers to fit. If None, fit all fibers.
    fit_orders : list[int] | None
        List of echelle orders to fit. If None, fit all orders.
    deg_intra_order : int
        Degree of the polynomial within an order.
    deg_inter_order : int | 'auto'
        Degree of the polynomial between orders. If 'auto', the degree is determined based on the number of orders being fit.
    opt_weights : bool
        Whether to use the inverse variance of the FWHM measurements as weights in the fit. Default is True.
    max_iters : int
        Maximum number of iterations for the fitting process.
    sigma_cut : float
        Cutoff for the residuals in terms of standard deviation for iterative outlier rejection. If None, no outlier rejection is performed.

    Returns
    -------
    output_model : GaussianLSFModel
        Output data model containing the fitted Gaussian LSF parameters.

    """

    ARGUMENTS = {
        'do_fibers' : None,
        'fit_orders' : None,
        'deg_intra_order' : 3,
        'deg_inter_order' : 'auto',
        'opt_weights' : True,
        'max_iters' : 100,
        'sigma_cut' : 4.0,
        'output_model': 'GaussianLSFModel',
    }

    def _perform(self):

        input_model = datamodels.open(self.input)

        # Get fibers and orders to fit
        do_fibers, fit_orders = self.get_do_traces(input_model, self.do_fibers, self.fit_orders)

        # Default inter order deg
        if self.deg_inter_order == 'auto':
            if len(fit_orders) == 1:
                deg_inter_order = 0
            elif len(fit_orders) == 2:
                deg_inter_order = 1
            else:
                anchor_degree = 4
                anchor_orders = 10
                max_degree = 6
                deg_inter_order = int(np.minimum(np.round(anchor_degree * np.log(len(fit_orders)) / np.log(anchor_orders)), max_degree))
            logger.info(f"deg_inter_order='auto', using {deg_inter_order=}.")
        else:
            deg_inter_order = self.deg_inter_order

        fit_results = []

        # Get the target output model
        output_model = datamodels.GaussianLSFModel.from_datamodels(
            input_model, include_data=True,
            primitive=self
        )

        for fiber in do_fibers:
            params_table = getattr(input_model, f'lfc_fitparams_fib{fiber}')
            lsfnorm_table = getattr(output_model, f'cheb_norm_fib{fiber}')
            waves = params_table['WAVE']
            pixels = params_table['MU_PIX']
            orders = params_table['ORDER']
            fwhms = params_table['FWHM_WAVE']
            fwhm_errs = params_table['FWHM_ERR_WAVE']
            mask = np.nonzero(np.isin(orders, fit_orders))[0]
            pixels = pixels[mask].copy()
            orders = orders[mask].copy()
            waves = waves[mask].copy()
            fwhms = fwhms[mask].copy()
            if self.opt_weights:
                fwhm_errs = fwhm_errs[mask].copy()
                weights = 1 / fwhm_errs**2
            else:
                weights = None
            fit_result = fit_gausslsf_chebyshev2d(
                pixels, orders, fwhms,
                deg_intra_order=self.deg_intra_order, deg_inter_order=deg_inter_order,
                max_iters=self.max_iters, sigma_cut=self.sigma_cut,
                weights=weights,
            )
            fit_results.append(fit_result)

            nx = getattr(input_model, f'spec_fib{fiber}').shape[1]
            pixel_vec = np.arange(nx)
            order_vec = input_model.make_order_vec()
            lsfnorm_table.add_row({
                'FIBER': fiber,
                'PIXEL_NORM_MIN': fit_result['pixel_bounds'][0],
                'PIXEL_NORM_MAX': fit_result['pixel_bounds'][1],
                'ORDER_NORM_MIN': fit_result['order_bounds'][0],
                'ORDER_NORM_MAX': fit_result['order_bounds'][1],
            })
            lsf_fwhms_full = _build_lsf_poly2d_structured(
                fit_result['coeffs'],
                pixel_vec,
                order_vec,
                fit_result['pixel_bounds'], fit_result['order_bounds']
            )
            setattr(output_model, f'lsf_params_fib{fiber}', lsf_fwhms_full)

        return dict(output=output_model, fit_results=fit_results)

def fit_gausslsf_chebyshev2d(
    pixels : np.ndarray,
    orders : np.ndarray,
    fwhms : np.ndarray,
    deg_intra_order : int,
    deg_inter_order : int,
    pixel_bounds : tuple[float] | None = None,
    order_bounds : tuple[float] | None = None,
    max_iters : float | None = None,
    sigma_cut : float | None = None,
    weights : np.ndarray | None = None,
) -> dict:
    """
    Fit a 2D Chebyshev polynomial to the Gaussian sigmas (units microns) as a function of pixel and echelle order.

    Parameters
    ----------
    pixels : np.ndarray
        Pixel values.
    orders : np.ndarray
        Echelle order values.
    fwhms : np.ndarray
        Gaussian FWHMs in units of wavelength.
    deg_intra_order : int
        Degree of the polynomial within an order.
    deg_inter_order : int
        Degree of the polynomial between orders.
    pixel_bounds : tuple[float] or None
        Bounds for pixel values for normalization of the data.
    order_bounds : tuple[float] or None
        Bounds for order values for normalization of the data.
    max_iters : int or None
        Maximum number of iterations for the fitting process.
    sigma_cut : float or None
        Cutoff for the residuals in terms of standard deviation.
    weights : np.ndarray or None
        Fit weights. Defaults to uniform weights.

    Returns
    -------
    dict
        Dictionary containing the results of the fit.
    """

    # Copy to not modify original data
    pixels = np.copy(pixels)
    orders = np.copy(orders)
    if weights is None:
        weights = np.ones(pixels.shape)
    else:
        weights = np.copy(weights)

    if pixel_bounds is None:
        pixel_bounds = (np.nanmin(pixels) - 10, np.nanmax(pixels) + 10)
    if order_bounds is None:
        order_bounds = (np.nanmin(orders) - 10, np.nanmax(orders) + 10)

    # Get normalized vals
    pixels_norm = _normalize_val(pixels, *pixel_bounds)
    orders_norm = _normalize_val(orders, *order_bounds)

    # Good vals
    good = np.where(np.isfinite(weights) & (weights > 0))[0]
    coeffs = None

    # Iteratively fit
    for i in range(max_iters):

        # Fit
        x = pixels_norm[good]
        y = orders_norm[good]
        z = fwhms[good]
        w = weights[good]
        coeffs = solve_chebyshev2d_ols(x, y, z, deg_x=deg_intra_order, deg_y=deg_inter_order, weights=w)

        # Model and residuals
        model_flat = _build_lsf_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds)
        residuals = fwhms - model_flat
        if sigma_cut is not None:
            stddev = np.nanstd(residuals)
            good_new = np.where(
                (weights > 0)
                & np.isfinite(weights)
                & (np.abs(residuals) < sigma_cut * stddev)
            )[0]
            if len(good) == len(good_new):
                break
            else:
                good = good_new
        else:
            break

    return dict(
        coeffs=coeffs, good=good, iterations=i+1,
        model=model_flat, residuals=residuals,
        order_bounds=order_bounds, pixel_bounds=pixel_bounds
    )

def _normalize_val(x, lo, hi):
    mu = (lo + hi) / 2
    s = hi - lo
    return 2 * (x - mu) / s

def _build_lsf_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    pixels_norm = _normalize_val(pixels, *pixel_bounds)
    orders_norm = _normalize_val(orders, *order_bounds)
    return np.polynomial.chebyshev.chebval2d(pixels_norm, orders_norm, coeffs)

def _build_lsf_poly2d_structured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    nx, n_orders = len(pixels), len(orders)
    out = np.full((n_orders, nx), np.nan)
    for i in range(n_orders):
        out[i, :] = _build_lsf_poly2d_unstructured(coeffs, pixels, np.full(nx, orders[i]), pixel_bounds, order_bounds)
    return out