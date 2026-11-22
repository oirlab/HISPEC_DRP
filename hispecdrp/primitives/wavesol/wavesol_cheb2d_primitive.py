from ... import datamodels
from ..base_primitive import BasePrimitive
import logging

from ..utils.spectral_utils import delta_wave_to_delta_vel
from ..utils.array_utils import auto_deg_inter_order, solve_chebyshev2d_ols
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['CalculateChebyshev2DWLSPrimitive', 'wavesol_norm_bounds']

WAVE_MODEL_CHEBYSHEV2D = "chebyshev2d"
WAVE_MODEL_CUBIC_SPLINE_1D = "cubic_spline_1d"
WAVE_MODELS = (WAVE_MODEL_CHEBYSHEV2D, WAVE_MODEL_CUBIC_SPLINE_1D)
DEFAULT_DEG_INTRA_ORDER = 6


def wavesol_norm_bounds(Ncols : int, order_vec) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Normalization bounds of the 2D Chebyshev wavelength solution, fixed by the detector and the model's orders.

    The pixel range (0, Ncols - 1) is padded by 10 and the full order range of the model (``make_order_vec``)
    by 1 on each side, so the bounds do not depend on which orders or lines are fit and are not stored.

    Returns
    -------
    pixel_bounds, order_bounds : tuple[float, float]
    """
    return (-10.0, Ncols - 1 + 10.0), (float(np.min(order_vec)) - 1.0, float(np.max(order_vec)) + 1.0)


class CalculateChebyshev2DWLSPrimitive(BasePrimitive):

    """
    Fit a wavelength solution from LFC line centroids.

    The default ``wave_model='chebyshev2d'`` fits a 2D Chebyshev polynomial
    to ``{pixel, order, m*λ}`` (a grating-equation expansion).
    ``wave_model='cubic_spline_1d'`` instead interpolates each order with a
    cubic spline through ``(MU_PIX, WAVE)``. The spline is optional and is
    not the default.

    Parameters
    ----------
    wave_model : {'chebyshev2d', 'cubic_spline_1d'}
        Wavelength model used to fill ``wave_fib*``. Default ``chebyshev2d``.
    deg_intra_order : int or 'auto'
        Degree of the polynomial within an order (Chebyshev model only).
        ``'auto'`` uses :data:`DEFAULT_DEG_INTRA_ORDER`.
    deg_inter_order : int or 'auto'
        Degree of the polynomial between orders. If 'auto', it will be set based on the number of orders being fit.
    max_iters : int
        Maximum number of iterations for the 2D fitting process.
    sigma_cut : float
        Sigma clipping cutoff in units of standard deviation of the residuals.
    max_vel_cut : float
        Maximum velocity cutoff for the residuals. Points with velocity residuals larger than this cutoff will be excluded from the fit in the next iteration. Units = m/s.
    opt_weights : bool
        Whether to use optimized weights based on the input model's wavelength errors. If False, uniform weights will be used. Default is True.
    fit_orders : list of int or None
        List of orders to fit. If None, all orders will be fit.
    do_fibers : list of int or None
        List of fibers to fit. If None, all fibers will be fit (separately in series).

    Returns
    -------
    output : LFCWavesolModel
        The wavelength solution model.
    
    """

    PARAMETERS = {
        'wave_model' : WAVE_MODEL_CHEBYSHEV2D,
        'deg_intra_order' : DEFAULT_DEG_INTRA_ORDER,
        'deg_inter_order' : 'auto',
        'max_iters' : 100,
        'sigma_cut' : 4,
        'max_vel_cut' : 1E3,
        'opt_weights' : True,
        'fit_orders' : None,
        'do_fibers' : None,
    }

    def _perform(self):

        # Open the input model
        input_model = datamodels.open(self.dataset)

        # Get fibers and orders
        do_fibers, fit_orders = self.get_do_traces(input_model, self.do_fibers, self.fit_orders)

        wave_model = str(self.wave_model).lower()
        if wave_model not in WAVE_MODELS:
            raise ValueError(
                f"wave_model={self.wave_model!r} is invalid; expected one of {WAVE_MODELS}"
            )

        if self.deg_intra_order == 'auto':
            deg_intra_order = DEFAULT_DEG_INTRA_ORDER
            logger.info(f"deg_intra_order='auto', using {deg_intra_order=}.")
        else:
            deg_intra_order = self.deg_intra_order

        # Default inter order deg (Chebyshev model only)
        if wave_model == WAVE_MODEL_CHEBYSHEV2D and self.deg_inter_order == 'auto':
            deg_inter_order = auto_deg_inter_order(len(fit_orders))
            logger.info(f"deg_inter_order='auto', using {deg_inter_order=}.")
        else:
            deg_inter_order = self.deg_inter_order

        fit_results = []

        # Get the output model type
        output_model = datamodels.LFCWavesolModel.from_datamodels(
            input_model, include_data=True, primitive=self
        )
        if hasattr(output_model.meta, "wave_model"):
            output_model.meta.wave_model = wave_model

        for fiber in do_fibers:

            # Get the table for the fiber
            params_table = getattr(input_model, f'lfc_fitparams_fib{fiber}')

            # Setup input vectors
            pixels = params_table['MU_PIX']
            orders = params_table['ORDER']
            waves = params_table['WAVE']
            mask = np.nonzero(np.isin(orders, fit_orders))[0]
            pixels = pixels[mask]
            orders = orders[mask]
            waves = waves[mask]
            if self.opt_weights:
                weights = 1 / params_table['MU_ERR_WAVE'][mask]**2
            else:
                weights = None

            nx = getattr(output_model, f'spec_fib{fiber}').shape[1]
            pixel_vec = np.arange(nx)
            order_vec = output_model.make_order_vec()

            if wave_model == WAVE_MODEL_CUBIC_SPLINE_1D:
                fit_result = fit_wavesol_spline_1d(
                    pixels, orders, waves, pixel_vec, order_vec, weights=weights
                )
                wavesol_full = fit_result["wave_grid"]
            else:
                pixel_bounds, order_bounds = wavesol_norm_bounds(nx, order_vec)
                fit_result = fit_wavesol_chebyshev2d(
                    pixels, orders, waves,
                    deg_intra_order=deg_intra_order, deg_inter_order=deg_inter_order,
                    pixel_bounds=pixel_bounds, order_bounds=order_bounds,
                    max_iters=self.max_iters, sigma_cut=self.sigma_cut, max_vel_cut=self.max_vel_cut,
                    weights=weights,
                )
                if fit_result["coeffs"] is not None:
                    setattr(output_model, f'cheb_coeffs_fib{fiber}', fit_result['coeffs'])
                wavesol_full = _build_wavesol_poly2d_structured(
                    fit_result['coeffs'],
                    pixel_vec,
                    order_vec,
                    fit_result['pixel_bounds'], fit_result['order_bounds']
                )
            fit_results.append(fit_result)
            setattr(output_model, f'wave_fib{fiber}', wavesol_full)

            # Set unreliable WLS in DQ
            bad_orders = np.setdiff1d(order_vec, fit_orders)
            # Copy: flagging must not mutate the input model's DQ plane.
            dq = np.array(getattr(output_model, f'spec_dq_fib{fiber}'), copy=True)
            flag_value = datamodels.DQ_FLAGS['UNRELIABLE_WS']
            for order in bad_orders:
                oi = np.abs(order - output_model.meta.order_start)
                dq[oi, :] |= flag_value
            
            params_table_good = params_table[fit_result['good']]
            for order in fit_orders:
                order_mask = params_table_good['ORDER'] == order
                if np.sum(order_mask) > 0:
                    oi = np.abs(order - output_model.meta.order_start)
                    line_centers_pix = params_table_good['MU_PIX'][order_mask]
                    k1 = int(np.floor(np.min(line_centers_pix)))
                    k2 = int(np.ceil(np.max(line_centers_pix)))
                    if k1 > 0:
                        dq[oi, :k1] |= flag_value
                    if k2 < nx - 1:
                        dq[oi, k2+1:] |= flag_value
                    if wave_model == WAVE_MODEL_CUBIC_SPLINE_1D:
                        # Spline WAVE in the first / last knot interval
                        # rests on a boundary segment of the spline and
                        # shows the largest interpolation error; flag it
                        # as unreliable like the out-of-range pixels.
                        centers = np.sort(np.asarray(line_centers_pix, dtype=float))
                        if centers.size >= 2:
                            b1 = int(np.floor(centers[1])) + 1
                            dq[oi, k1:b1] |= flag_value
                            b2 = int(np.ceil(centers[-2]))
                            dq[oi, b2:k2 + 1] |= flag_value
            setattr(output_model, f'spec_dq_fib{fiber}', dq)

        return dict(output=output_model, fit_results=fit_results)


def _normalize_val(x, lo, hi):
    mu = (lo + hi) / 2
    s = hi - lo
    if s == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return 2 * (x - mu) / s

def fit_wavesol_spline_1d(
    pixels: np.ndarray,
    orders: np.ndarray,
    waves: np.ndarray,
    pixel_vec: np.ndarray,
    order_vec: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict:
    """Per-order cubic spline through LFC ``(MU_PIX, WAVE)``."""
    from scipy.interpolate import CubicSpline

    pixels = np.asarray(pixels, dtype=float)
    orders = np.asarray(orders, dtype=int)
    waves = np.asarray(waves, dtype=float)
    pixel_vec = np.asarray(pixel_vec, dtype=float)
    order_vec = np.asarray(order_vec, dtype=int)
    if weights is None:
        weights = np.ones(pixels.shape, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    wave_grid = np.full((order_vec.size, pixel_vec.size), np.nan, dtype=float)
    model_at_lines = np.full(pixels.shape, np.nan, dtype=float)
    good_mask = np.zeros(pixels.size, dtype=bool)

    for i, od in enumerate(order_vec):
        sel = (
            (orders == int(od))
            & np.isfinite(pixels)
            & np.isfinite(waves)
            & np.isfinite(weights)
            & (weights > 0)
        )
        n_sel = int(sel.sum())
        if n_sel < 4:
            logger.warning(
                "Order %s has %s usable LFC lines; need 4 for a cubic spline.",
                od,
                n_sel,
            )
            continue
        x = pixels[sel]
        y = waves[sel]
        order = np.argsort(x)
        x, y = x[order], y[order]
        _, uniq = np.unique(np.round(x, 6), return_index=True)
        x, y = x[uniq], y[uniq]
        if x.size < 4:
            logger.warning(
                "Order %s has %s unique LFC pixels; need 4 for a cubic spline.",
                od,
                x.size,
            )
            continue
        fn = CubicSpline(x, y, extrapolate=False)
        inside = (pixel_vec >= x[0]) & (pixel_vec <= x[-1])
        wave_grid[i, inside] = fn(pixel_vec[inside])
        at_x = fn(pixels[sel])
        model_at_lines[sel] = at_x
        good_mask[sel] = np.isfinite(at_x)

    residuals = waves - model_at_lines
    residuals_vel = delta_wave_to_delta_vel(residuals, waves)
    if pixels.size == 0:
        pixel_bounds = (0.0, 1.0)
        order_bounds = (0.0, 1.0)
    else:
        pixel_bounds = (float(np.nanmin(pixels)) - 10.0, float(np.nanmax(pixels)) + 10.0)
        order_bounds = (float(np.nanmin(orders)) - 1.0, float(np.nanmax(orders)) + 1.0)
    return dict(
        coeffs=None,
        good=np.where(good_mask)[0],
        iterations=1,
        pixels=pixels,
        wave=waves,
        model=model_at_lines,
        residuals=residuals,
        residuals_vel=residuals_vel,
        order_bounds=order_bounds,
        pixel_bounds=pixel_bounds,
        wave_grid=wave_grid,
        wave_model=WAVE_MODEL_CUBIC_SPLINE_1D,
    )


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
        order_bounds = (np.nanmin(orders) - 1, np.nanmax(orders) + 1)

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
            residuals_vel = delta_wave_to_delta_vel(residuals, waves)
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
        pixels=pixels, wave=waves, model=model_flat, residuals=residuals, residuals_vel=residuals_vel,
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