from ... import datamodels
from ..base_primitive import BasePrimitive

from ...utils.math import delta_wave_to_delta_v, solve_chebyshev2d_ols
from keckdrpframework.models.arguments import Arguments
import numpy as np


__all__ = ['WLSChebyshev2DPrimitive']


class WLSChebyshev2DPrimitive(BasePrimitive):
    """
    Finds and fits 1D LFC modes with a Gaussian + background profile.
    """

    ARGUMENTS = {
        'deg_intra_order' : 'auto',
        'deg_inter_order' : 9,
        'max_iters' : 100,
        'sigma_cut' : 4,
        'max_vel_cut' : 1E3,
        'opt_weights' : False,
        'fit_orders' : None,
        'do_fibers' : None,
    }

    def _perform(self):

        # Open the input model
        input_model = datamodels.open(self.args.name)

        # Default inter order deg
        if self.args.deg_intra_order == 'auto':
            anchor_degree = 6
            anchor_orders = 10
            max_degree = 10
            deg_inter_order = int(np.min(np.round(anchor_degree * np.log(len(self.args.fit_orders)) / np.log(anchor_orders)), max_degree))
            self.logger.info(f"deg_inter_order='auto', using {deg_inter_order=}.")
        else:
            deg_inter_order = self.args.deg_inter_order

        fit_results = []

        # Get the output model type
        output_model = self.get_output_model(input_model, model_suffix='LFCWavesolModel')

        for fiber in self.args.do_fibers:

            # Get the table for the fiber
            params_table = getattr(input_model, f'lfc_fitparams_fib{fiber}')

            # Setup input vectors
            pixels = params_table['MU_PIX']
            orders = params_table['ORDER']
            waves = params_table['WAVE']
            mask = np.nonzero(np.isin(orders, self.args.fit_orders))[0]
            pixels = pixels[mask]
            orders = orders[mask]
            waves = waves[mask]
            if self.args.opt_weights:
                weights = 1 / params_table['MU_ERR_WAVE'][mask]**2
            else:
                weights = None

            # 2D Fit
            fit_result = fit_wavesol_chebyshev2d(
                pixels, orders, waves,
                deg_intra_order=self.args.deg_intra_order, deg_inter_order=deg_inter_order,
                max_iters=self.args.max_iters, sigma_cut=self.args.sigma_cut, max_vel_cut=self.args.max_vel_cut,
                weights=weights,
            )
            fit_results.append(fit_result)

            # Set results in model
            setattr(output_model, f'cc_coeff_fib{fiber}', fit_result['coeffs'])
            nx = getattr(output_model, f'spec_fib{fiber}').shape[1]
            pixel_vec = np.arange(nx)
            order_vec = self.make_order_vec(output_model)
            output_model.meta.wavesol.pixel_min_norm = fit_result['pixel_bounds'][0]
            output_model.meta.wavesol.pixel_max_norm = fit_result['pixel_bounds'][1]
            output_model.meta.wavesol.order_min_norm = fit_result['order_bounds'][0]
            output_model.meta.wavesol.order_max_norm = fit_result['order_bounds'][1]
            wavesol_full = _build_wavesol_poly2d_structured(
                fit_result['coeffs'],
                pixel_vec,
                order_vec,
                fit_result['pixel_bounds'], fit_result['order_bounds']
            )
            setattr(output_model, f'wave_fib{fiber}', wavesol_full)
        
        return Arguments(name=output_model, fit_results=fit_results)

    @staticmethod
    def make_order_vec(model : datamodels.DataModel) -> np.ndarray:
        mi = model.meta.order_start
        mf = model.meta.order_end
        step = 1 if mf >= mi else -1
        return np.arange(mi, mf + step, step).astype(np.int16)


def _normalize_val(x, lo, hi):
    mu = (lo + hi) / 2
    s = hi - lo
    return 2 * (x - mu) / s

def fit_wavesol_chebyshev2d(
    pixels : np.ndarray,
    orders : np.ndarray,
    waves : np.ndarray,
    deg_intra_order : int,
    deg_inter_order : int,
    pixel_bounds : tuple[float] | None = None,
    order_bounds : tuple[float] | None = None,
    max_iters : float | None = None,
    sigma_cut : float | None = None,
    max_vel_cut : float | None = None,
    weights : np.ndarray | None = None,
) -> dict:
    """
    Fit a 2D Chebyshev polynomial to the wavelength solution data.

    Args:
        pixels (np.ndarray): Pixel values.
        orders (np.ndarray): Echelle order values.
        waves (np.ndarray): Wavelength values.
        deg_intra_order (int): Degree of the polynomial within an order.
        deg_inter_order (int): Degree of the polynomial between orders.
        pixel_bounds (tuple[float] | None): Bounds for pixel values for normalization of the data.
        order_bounds (tuple[float] | None): Bounds for order values for normalization of the data.
        max_iters (int | None): Maximum number of iterations for the fitting process.
        sigma_cut (float | None): Cutoff for the residuals in terms of standard deviation.
        max_vel_cut (float | None): Maximum velocity cutoff for the residuals.
        weights (np.ndarray | None): Fit weights. Defaults to uniform weights.

    Returns:
        dict : Dictionary containing the results of the fit.
    """

    # Copy to not modify original data
    pixels = np.copy(pixels)
    orders = np.copy(orders)
    waves = np.copy(waves)
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

    # Actual vals to fit
    orders_times_waves = orders * waves

    # Good vals
    good = np.where(np.isfinite(weights) & (weights > 0))[0]
    coeffs = None

    # Iteratively fit
    for i in range(max_iters):

        # Fit
        x = pixels_norm[good]
        y = orders_norm[good]
        z = orders_times_waves[good]
        w = weights[good]
        coeffs = solve_chebyshev2d_ols(x, y, z, deg_x=deg_intra_order, deg_y=deg_inter_order, weights=w)

        # Model and residuals
        model_flat = _build_wavesol_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds)
        residuals = waves - model_flat
        if sigma_cut is not None or max_vel_cut is not None:
            sigma = np.nanstd(residuals)
            residuals_vel = delta_wave_to_delta_v(residuals, waves)
            good_new = np.where(
                (weights > 0)
                & np.isfinite(weights)
                & (np.abs(residuals_vel) < max_vel_cut)
                & (np.abs(residuals) < sigma_cut * sigma)
            )[0]
            if len(good) == len(good_new):
                break
            else:
                good = good_new
        else:
            break

    return dict(
        coeffs=coeffs, good=good, iterations=i+1,
        model=model_flat, residuals=residuals, residuals_vel=residuals_vel,
        order_bounds=order_bounds, pixel_bounds=pixel_bounds
    )

def _build_wavesol_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    pixels_norm = _normalize_val(pixels, *pixel_bounds)
    orders_norm = _normalize_val(orders, *order_bounds)
    return np.polynomial.chebyshev.chebval2d(pixels_norm, orders_norm, coeffs) / orders

def _build_wavesol_poly2d_structured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    nx, n_orders = len(pixels), len(orders)
    out = np.full((n_orders, nx), np.nan)
    for i in range(n_orders):
        out[i, :] = _build_wavesol_poly2d_unstructured(coeffs, pixels, np.full(nx, orders[i]), pixel_bounds, order_bounds)
    return out