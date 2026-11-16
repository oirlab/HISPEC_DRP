from ... import datamodels
from ..base_primitive import BasePrimitive
import logging

from ..utils.spectral_utils import delta_wave_to_delta_vel
from ..utils.array_utils import solve_chebyshev2d_ols
from ..line_fitting.fit_lines import LINE_DQ_FLAGS
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['CalculateChebyshev2DWLSPrimitive']

WAVE_MODEL_CHEBYSHEV2D = "chebyshev2d"
WAVE_MODEL_CUBIC_SPLINE_1D = "cubic_spline_1d"
WAVE_MODELS = (WAVE_MODEL_CHEBYSHEV2D, WAVE_MODEL_CUBIC_SPLINE_1D)
DEFAULT_DEG_INTRA_ORDER = 9


# ---------------------------------------------------------------------------
# Per-source configuration: the only place source-specific names/defaults live.
# ---------------------------------------------------------------------------
# input_model_class : name of the input datamodel class in `datamodels` (used for source detection)
# table_prefix      : input table attribute is f'{table_prefix}_fitparams_fib{fiber}'
# model_class       : name of the output datamodel class in `datamodels`
# defaults          : fit settings used when the user leaves a parameter as None.
#                     LFC values match the previous primitive defaults. Etalon and
#                     uranium values are PLACEHOLDERS: tune on real data.
SOURCE_CONFIG = {
    'lfc': dict(
        input_model_class='LFCFitParamsModel',  
        table_prefix='lfc',
        model_class='LFCWavesolModel',
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=DEFAULT_DEG_INTRA_ORDER,
                      deg_inter_order='auto', max_iters=100, sigma_cut=4, max_vel_cut=1E3,
                      opt_weights=True),
    ),
    'etalon': dict(
        input_model_class='EtalonRefModel',  
        table_prefix='etalon',
        model_class='EtalonWavesolModel',
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=DEFAULT_DEG_INTRA_ORDER,
                      deg_inter_order='auto', max_iters=100, sigma_cut=4, max_vel_cut=1E3,
                      opt_weights=True),
    ),
    'uranium': dict(
        input_model_class='UraniumFitParamsModel', 
        table_prefix='uranium',
        model_class='UraniumWavesolModel',
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=4,  # far fewer lines per order
                      deg_inter_order='auto', max_iters=100, sigma_cut=3, max_vel_cut=300,
                      opt_weights=True),
    ),
}


def auto_deg_inter_order(
    n_orders,
    anchor_degree=6,
    anchor_orders=10,
    max_degree=10,
):
    """Choose a Chebyshev inter-order degree from the number of fit orders.

    Uses ``np.minimum`` (element-wise cap), not ``np.min(..., max_degree)``.
    The latter is parsed as ``axis=max_degree`` and raises ``AxisError``.
    """
    n_orders = int(n_orders)
    if n_orders <= 1:
        return 0
    if n_orders == 2:
        return 1
    return int(
        np.minimum(
            np.round(anchor_degree * np.log(n_orders) / np.log(anchor_orders)),
            max_degree,
        )
    )


class CalculateChebyshev2DWLSPrimitive(BasePrimitive):

    """
    Fit a wavelength solution from fitted line centroids (LFC, etalon, or uranium lamp).

    The calibration source is autodetected from the class of the input datamodel. The source sets the input table, the output datamodel, and the default fit settings.

    The default ``wave_model='chebyshev2d'`` fits a 2D Chebyshev polynomial
    to ``{pixel, order, m*λ}`` (a grating-equation expansion).
    ``wave_model='cubic_spline_1d'`` instead interpolates each order with a
    cubic spline through ``(MU_PIX, WAVE)``. The spline is optional and is
    not the default.

    Any parameter left as None uses the default for the detected source (see SOURCE_CONFIG).

    Parameters
    ----------
    source : {'lfc', 'etalon', 'uranium'} or None
        Calibration source. If None, autodetected from the input model class. Set explicitly to override.
    wave_model : {'chebyshev2d', 'cubic_spline_1d'} or None
        Wavelength model used to fill ``wave_fib*``.
    deg_intra_order : int, 'auto' or None
        Degree of the polynomial within an order (Chebyshev model only).
        ``'auto'`` uses :data:`DEFAULT_DEG_INTRA_ORDER`.
    deg_inter_order : int, 'auto' or None
        Degree of the polynomial between orders. If 'auto', it will be set based on the number of orders being fit.
    max_iters : int or None
        Maximum number of iterations for the 2D fitting process.
    sigma_cut : float or None
        Sigma clipping cutoff in units of standard deviation of the residuals. Use float('inf') to disable.
    max_vel_cut : float or None
        Maximum velocity cutoff for the residuals. Points with velocity residuals larger than this cutoff will be excluded from the fit in the next iteration. Units = m/s. Use float('inf') to disable.
    opt_weights : bool or None
        Whether to use optimized weights based on the input model's wavelength errors. If False, uniform weights will be used.
    fit_orders : list of int or None
        List of orders to fit. If None, all orders will be fit.
    do_fibers : list of int or None
        List of fibers to fit. If None, all fibers will be fit (separately in series).

    Returns
    -------
    output : datamodel (type depends on source)
        The wavelength solution model.

    """

    # None means "use the per-source default from SOURCE_CONFIG"
    PARAMETERS = {
        'source' : None,
        'wave_model' : None,
        'deg_intra_order' : None,
        'deg_inter_order' : None,
        'max_iters' : None,
        'sigma_cut' : None,
        'max_vel_cut' : None,
        'opt_weights' : None,
        'fit_orders' : None,
        'do_fibers' : None,
        'reject_flags': None,
    }

    def _detect_source(self, input_model):
        """Return the source name, from the user override or the input model's class."""
        if self.source is not None:
            if self.source not in SOURCE_CONFIG:
                raise ValueError(f"Unknown source {self.source!r}; options: {list(SOURCE_CONFIG)}")
            return self.source

        found = []
        for name, cfg in SOURCE_CONFIG.items():
            cls = getattr(datamodels, cfg['input_model_class'], None)
            if cls is not None and isinstance(input_model, cls):
                found.append(name)

        if len(found) == 1:
            return found[0]
        if not found:
            raise ValueError(
                f"Could not detect a calibration source from input model type "
                f"{type(input_model).__name__}. Expected one of "
                f"{[c['input_model_class'] for c in SOURCE_CONFIG.values()]}, or set the 'source' parameter."
            )
        raise ValueError(f"Input model matches several sources {found}. Set the 'source' parameter.")

    def _perform(self):

        # Open the input model
        input_model = datamodels.open(self.dataset)

        # Get fibers and orders
        do_fibers, fit_orders = self.get_do_traces(input_model, self.do_fibers, self.fit_orders)

        # Detect source and resolve settings (user value wins, else source default)
        source = self._detect_source(input_model)
        cfg = SOURCE_CONFIG[source]
        s = {k: (getattr(self, k) if getattr(self, k) is not None else v) for k, v in cfg['defaults'].items()}

        wave_model = str(s['wave_model']).lower()
        if wave_model not in WAVE_MODELS:
            raise ValueError(
                f"wave_model={s['wave_model']!r} is invalid; expected one of {WAVE_MODELS}"
            )

        if s['deg_intra_order'] == 'auto':
            deg_intra_order = DEFAULT_DEG_INTRA_ORDER
            logger.info(f"deg_intra_order='auto', using {deg_intra_order=}.")
        else:
            deg_intra_order = int(s['deg_intra_order'])

        # Inter order deg (Chebyshev model only)
        deg_inter_order = s['deg_inter_order']
        if wave_model == WAVE_MODEL_CHEBYSHEV2D:
            if deg_inter_order == 'auto':
                deg_inter_order = auto_deg_inter_order(len(fit_orders))
                logger.info(f"deg_inter_order='auto', using {deg_inter_order=}.")
            deg_inter_order = int(deg_inter_order)
            max_allowed = max(len(fit_orders) - 1, 0)
            if deg_inter_order > max_allowed:
                logger.warning(
                    f"deg_inter_order={deg_inter_order} exceeds what {len(fit_orders)} order(s) "
                    f"can constrain; using {max_allowed}."
                )
                deg_inter_order = max_allowed

        logger.info(f"Detected source '{source}', {wave_model=}, {deg_intra_order=}, {deg_inter_order=}, "
                    f"max_iters={s['max_iters']}, sigma_cut={s['sigma_cut']}, max_vel_cut={s['max_vel_cut']}.")

        fit_results = []

        # lines to flag out
        reject_mask = np.uint8(0)
        if self.reject_flags is None:
            reject_flags = [name for name in LINE_DQ_FLAGS if name != 'GOOD']
            self.reject_flags = reject_flags
        for flag in self.reject_flags:
            reject_mask |= LINE_DQ_FLAGS[flag]


        # Output model type depends on the source
        model_cls = getattr(datamodels, cfg['model_class'])
        output_model = model_cls.from_datamodels(
            input_model, include_data=True, primitive=self
        )
        if hasattr(output_model.meta, "wave_model"):
            output_model.meta.wave_model = wave_model

        for fiber in do_fibers:

            # Get the table for the fiber
            params_table = getattr(input_model, f"{cfg['table_prefix']}_fitparams_fib{fiber}")

            # Get the wavesol norm table for the fiber
            chebnorm_table = getattr(output_model, f'cheb_norm_fib{fiber}')

            # Setup input vectors
            pixels = params_table['MU_PIX']
            orders = params_table['ORDER']
            waves = params_table['WAVE']
            try:
                line_dq = np.asarray(params_table['LINE_DQ'], dtype=np.uint8)
            except (KeyError, ValueError):
                logger.warning("LINE_DQ column not found; no lines will be rejected by flag.")
                line_dq = np.zeros(len(params_table), dtype=np.uint8)
            mask = np.isin(orders, fit_orders) & ((line_dq & reject_mask) == 0)
            pixels = pixels[mask]
            orders = orders[mask]
            waves = waves[mask]
            if s['opt_weights']:
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
                fit_result = fit_wavesol_chebyshev2d(
                    pixels, orders, waves,
                    deg_intra_order=deg_intra_order, deg_inter_order=deg_inter_order,
                    max_iters=s['max_iters'], sigma_cut=s['sigma_cut'], max_vel_cut=s['max_vel_cut'],
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

            chebnorm_table.add_row({
                'PIXEL_NORM_MIN': fit_result['pixel_bounds'][0],
                'PIXEL_NORM_MAX': fit_result['pixel_bounds'][1],
                'ORDER_NORM_MIN': fit_result['order_bounds'][0],
                'ORDER_NORM_MAX': fit_result['order_bounds'][1],
            })
            setattr(output_model, f'wave_fib{fiber}', wavesol_full)

            # Set unreliable WLS in DQ
            bad_orders = np.setdiff1d(order_vec, fit_orders)
            # Copy: flagging must not mutate the input model's DQ plane.
            dq = np.array(getattr(output_model, f'spec_dq_fib{fiber}'), copy=True)
            flag_value = datamodels.DQ_FLAGS['UNRELIABLE_WS']
            for order in bad_orders:
                oi = np.abs(order - output_model.meta.order_start)
                dq[oi, :] |= flag_value

            # fit_result['good'] indexes the masked (fit_orders only) arrays, so apply the same mask first
            params_table_good = params_table[mask][fit_result['good']]
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

        return dict(output=output_model, fit_results=fit_results, source=source)


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
    """Per-order cubic spline through fitted line ``(MU_PIX, WAVE)``."""
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
                "Order %s has %s usable lines; need 4 for a cubic spline.",
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
                "Order %s has %s unique line pixels; need 4 for a cubic spline.",
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
        order_bounds = (float(np.nanmin(orders)) - 10.0, float(np.nanmax(orders)) + 10.0)
    return dict(
        coeffs=None,
        good=np.where(good_mask)[0],
        iterations=1,
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
    max_iters : int | None = None,
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
        max_iters (int | None): Maximum number of iterations (None means a single fit).
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
    if max_iters is None:
        max_iters = 1

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
    n_coeffs = (deg_intra_order + 1) * (deg_inter_order + 1)

    # Iteratively fit
    for i in range(max_iters):

        if len(good) < n_coeffs:
            raise RuntimeError(
                f"Only {len(good)} usable lines for {n_coeffs} coefficients "
                f"({deg_intra_order=}, {deg_inter_order=}). Lower the degrees or loosen the clipping."
            )

        # Fit
        x = pixels_norm[good]
        y = orders_norm[good]
        z = orders_times_waves[good]
        w = weights[good]
        coeffs = solve_chebyshev2d_ols(x, y, z, deg_x=deg_intra_order, deg_y=deg_inter_order, weights=w)

        # Model and residuals
        model_flat = _build_wavesol_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds)
        residuals = waves - model_flat
        residuals_vel = delta_wave_to_delta_vel(residuals, waves)
        if sigma_cut is not None or max_vel_cut is not None:
            sigma = np.nanstd(residuals)
            keep = (weights > 0) & np.isfinite(weights)
            if max_vel_cut is not None:
                keep &= np.abs(residuals_vel) < max_vel_cut
            if sigma_cut is not None:
                keep &= np.abs(residuals) < sigma_cut * sigma
            good_new = np.where(keep)[0]
            if len(good) == len(good_new):
                break
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