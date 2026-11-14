from ... import datamodels
from ..base_primitive import BasePrimitive
from ...calibrations import LSFSelector
from ...utils.parallelization_utils import numba_thread_scope
import logging

from ..lsf.lsf_gauss_primitive import _build_lsf_poly2d_unstructured
from .fit_gascell import (
    make_fit_context, coeffs_from_wavelength_grid, fit_wavelength_solution, evaluate_forward_model,
    build_continuum_1d,
)
from .utils import read_gascell_fts
from ..utils.array_utils import auto_deg_inter_order
from .wavesol_cheb2d_primitive import _build_wavesol_poly2d_structured
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['FitGasCellWavesolPrimitive']

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


class FitGasCellWavesolPrimitive(BasePrimitive):

    """
    Fit the wavelength solution of a gas cell spectrum by forward modeling it with an FTS reference transmission.

    The wavelength solution uses the same 2D Chebyshev convention as :class:`CalculateChebyshev2DWLSPrimitive`: m * lambda is a 2D Chebyshev series in the normalized pixel and normalized echelle order. For each fiber, the forward model is the FTS transmission evaluated at the current wavelength solution on an oversampled pixel grid, convolved with the Gaussian LSF, times a per-order Chebyshev continuum. All orders are fit simultaneously with Levenberg-Marquardt, starting from the input wavelength solution.

    Parameters
    ----------
    input : str or L2Model
        Input gas cell spectrum with an initial wavelength solution (``wave_fib#``), e.g. from :class:`SetInitialWLSPrimitive`.
    initial_wavesol : str
        Wavelength solution the fit starts from. 'input' uses the input model's ``wave_fib#``. 'simulated' uses the simulated (pyechelle HDF) wavelength solution for the input model's spectrograph and each fiber, from :meth:`~hispecdrp.simulations.SimulateRateMap.get_wavelength_solution`, and ignores ``wave_fib#``. Default is 'input'.
    wave_deg_intra_order : int
        Degree of the polynomial within an order.
    wave_deg_inter_order : int or 'auto'
        Degree of the polynomial between orders. If 'auto', it will be set based on the number of orders being fit.
    continuum_deg_intra_order : int or None
        Degree in pixel of the Chebyshev continuum, fit independently for each order. If None, no continuum is fit. Default is 2.
    osamp : int
        Oversampling factor (odd) of the pixel grid the FTS reference is sampled and convolved on.
    kernel_size_sigma : float
        Half-width of the LSF kernels in units of the largest Gaussian sigma.
    fts_resolution : float
        Resolving power R of the FTS scans. The FTS reference is assumed to be the true transmission convolved with
        a Gaussian of FWHM lambda / R, so the LSF kernels are narrowed to FWHM = sqrt(FWHM_lsf^2 - (lambda / R)^2).
        Use ``np.inf`` (or None) for no correction, e.g. for data simulated from the FTS reference itself. Default is 300,000.
    max_iters : int
        Maximum number of residual evaluations of the Levenberg-Marquardt fit (``max_nfev`` of ``scipy.optimize.least_squares``).
    ftol : float
        Relative decrease in chi^2 below which the fit is considered converged (``ftol`` of ``scipy.optimize.least_squares``).
    use_opt_weights : bool
        Whether to weight each pixel by the inverse variance from ``spec_err``. If False, all unmasked pixels are weighted equally. Default is True.
    fts_scans : list of str or None
        FTS scans to include, e.g. ``['B0216.1a', 'B0216.3b']``. The reference transmission is the product of the selected scans, and each gas cell may appear only once. If None, the 'a' scan of every cell is used.
    fit_orders : list of int or None
        List of orders to fit. If None, all orders will be fit. Orders with no pixels inside the FTS wavelength range are skipped.
    do_fibers : list of int or None
        List of fibers to fit. If None, all fibers will be fit (separately in series).

    Calibrations
    ------------
    lsf : str or GaussianLSFModel
        - selector: :class:`~hispecdrp.calibrations.selectors.LSFSelector`

    Returns
    -------
    output : GasCellWavesolModel
        The wavelength solution model. Besides the wavelength solution (``wave_fib#``, ``cheb_coeffs_fib#``, ``cheb_norm_fib#``), it holds the best-fit forward model (``model_fib#``) and its residuals (``model_resid_fib#``), the best-fit continuum (``cont_fib#``), its Chebyshev coefficients (``cont_coeffs_fib#``, normalized with ``cheb_norm_fib#``) and its model type (``cont_info_fib#``, '1d' or 'none'). The arrays are on the pixel grid and NaN on orders not fit. The continuum coefficients have one row per order (NaN for orders not fit); evaluate them with :func:`~hispecdrp.primitives.wavesol.fit_gascell.build_continuum_1d`.

    """

    PARAMETERS = {
        'initial_wavesol' : 'input',
        'wave_deg_intra_order' : 4,
        'wave_deg_inter_order' : 'auto',
        'continuum_deg_intra_order' : 2,
        'osamp' : 5,
        'kernel_size_sigma' : 5.0,
        'fts_resolution' : 300_000,
        'max_iters' : 50,
        'ftol' : 1E-9,
        'use_opt_weights' : True,
        'fts_scans' : None,
        'fit_orders' : None,
        'do_fibers' : None,
    }

    CALIBRATIONS = {
        'lsf': {
            'selector': LSFSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        if self.initial_wavesol not in ('input', 'simulated'):
            raise ValueError(f"initial_wavesol must be 'input' or 'simulated'; got {self.initial_wavesol!r}.")

        # Open the input model
        input_model = datamodels.open(self.dataset)

        # Get calibrations
        lsf, _ = self.get_calibration(input_model, 'lsf')
        lsf_model = datamodels.open(lsf)

        # Get fibers and orders
        do_fibers, fit_orders = self.get_do_traces(input_model, self.do_fibers, self.fit_orders)
        order_vec = input_model.make_order_vec()

        # FTS reference transmission on an increasing wavelength grid (microns)
        lam_ref, F_ref, fts_scans = _load_fts_reference(self.fts_scans)
        logger.info(f"Using FTS scans {fts_scans}.")

        # Simulated wavelength solution for every fiber, read once
        if self.initial_wavesol == 'simulated':
            from ...simulations.simulate_rate_map import SimulateRateMap
            from ...simulations.simulate_up_the_ramp import (
                stack_wavelength_solution_rows,
            )

            spectrograph = input_model.meta.spectrograph
            sim_wavesol = SimulateRateMap().get_wavelength_solution(spectrograph=spectrograph)
            logger.info(f"Starting from the simulated {spectrograph} wavelength solution.")

        fit_results = []

        # Get the output model type
        output_model = datamodels.GasCellWavesolModel.from_datamodels(
            input_model, include_data=True, primitive=self
        )

        for fiber in do_fibers:

            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            nx = spec.shape[1]
            if self.initial_wavesol == 'simulated':
                wave0 = stack_wavelength_solution_rows(sim_wavesol[fiber], order_vec, n_pix=nx)
            else:
                wave0 = getattr(input_model, f'wave_fib{fiber}')

            # Only fit pixels inside the FTS wavelength range
            in_ref = (wave0 >= lam_ref[0]) & (wave0 <= lam_ref[-1])
            oi_fit = np.array([oi for oi, order in enumerate(order_vec) if order in fit_orders and np.any(in_ref[oi])], dtype=int)
            orders = order_vec[oi_fit]
            skipped = np.setdiff1d(fit_orders, orders)
            if len(skipped) > 0:
                logger.warning(f"Fiber {fiber}: skipping orders {list(skipped)} with no pixels inside the FTS wavelength range.")
            if len(orders) == 0:
                logger.warning(f"Fiber {fiber}: no orders to fit.")
                continue
            deg_inter_order = _resolve_deg_inter_order(self.wave_deg_inter_order, len(orders), name='wave_deg_inter_order')

            # Forward model setup
            kernels = _make_gaussian_lsf_kernels(
                lsf_model, fiber, orders, nx, self.osamp, self.kernel_size_sigma, fts_resolution=self.fts_resolution,
            )
            sigma = spec_err[oi_fit].astype(np.float64)
            if not self.use_opt_weights:
                # Uniform weights, but keep masking pixels with an invalid error
                sigma = np.where(np.isfinite(sigma) & (sigma > 0), 1.0, np.nan)
            ctx = make_fit_context(
                orders, spec[oi_fit].astype(np.float64), sigma,
                lam_ref, F_ref, kernels, self.osamp,
                self.wave_deg_intra_order, deg_inter_order,
                dq=np.where(in_ref[oi_fit], spec_dq[oi_fit], 1),
                continuum_deg_intra_order=self.continuum_deg_intra_order,
            )

            # Fit, starting from the starting wavelength solution (wave0)
            c0 = coeffs_from_wavelength_grid(wave0[oi_fit], ctx)
            with numba_thread_scope(self.max_cores):
                cmat, amat, cov, chi2_red, info = fit_wavelength_solution(ctx, c0, max_nfev=self.max_iters, ftol=self.ftol)
                model = evaluate_forward_model(cmat, amat, ctx)
            ctx['pool'].shutdown()
            logger.info(f"Fiber {fiber}: {info['nfev']} evaluations, converged={info['converged']} ({info['message']}), chi2_red={chi2_red:.3f}.")
            if amat is None:
                continuum_model = 'none'
                continuum = np.ones_like(model)
            else:
                continuum_model = '1d'
                continuum = build_continuum_1d(amat, np.arange(nx), ctx['pixel_bounds'])
            fit_results.append(dict(
                fiber=fiber, orders=orders, coeffs=cmat, continuum_coeffs=amat, continuum_model=continuum_model,
                model=model, continuum=continuum,  # (Norders fit, Ncols) on the pixel grid
                cov=cov, chi2_red=chi2_red, info=info,
                pixel_bounds=ctx['pixel_bounds'], order_bounds=ctx['order_bounds'],
            ))

            # Set results in model
            setattr(output_model, f'cheb_coeffs_fib{fiber}', cmat)
            getattr(output_model, f'cheb_norm_fib{fiber}').add_row({
                'PIXEL_NORM_MIN': ctx['pixel_bounds'][0],
                'PIXEL_NORM_MAX': ctx['pixel_bounds'][1],
                'ORDER_NORM_MIN': ctx['order_bounds'][0],
                'ORDER_NORM_MAX': ctx['order_bounds'][1],
            })
            wavesol_full = _build_wavesol_poly2d_structured(
                cmat, np.arange(nx), order_vec,
                ctx['pixel_bounds'], ctx['order_bounds'],
            )
            setattr(output_model, f'wave_fib{fiber}', wavesol_full)
            resid = np.full(spec.shape, np.nan, dtype=np.float32)
            resid[oi_fit] = spec[oi_fit] - model
            setattr(output_model, f'model_resid_fib{fiber}', resid)
            model_full = np.full(spec.shape, np.nan, dtype=np.float32)
            model_full[oi_fit] = model
            setattr(output_model, f'model_fib{fiber}', model_full)

            # Continuum: arrays on the pixel grid (NaN on orders not fit), coefficients and model type
            cont_full = np.full(spec.shape, np.nan, dtype=np.float32)
            cont_full[oi_fit] = continuum
            setattr(output_model, f'cont_fib{fiber}', cont_full)
            if continuum_model == '1d':
                # One row per order of the model, NaN rows for orders not fit
                amat_full = np.full((len(order_vec), amat.shape[1]), np.nan)
                amat_full[oi_fit] = amat
                setattr(output_model, f'cont_coeffs_fib{fiber}', amat_full)
            getattr(output_model, f'cont_info_fib{fiber}').add_row({'MODEL': continuum_model})

            # Set unreliable WLS in DQ: orders not fit, and pixels outside the FTS range in fit orders
            dq = getattr(output_model, f'spec_dq_fib{fiber}')
            flag_value = datamodels.DQ_FLAGS['UNRELIABLE_WS']
            not_fit = np.ones(len(order_vec), dtype=bool)
            not_fit[oi_fit] = False
            dq[not_fit, :] |= flag_value
            dq[oi_fit] |= np.where(in_ref[oi_fit], 0, flag_value).astype(dq.dtype)

        return dict(output=output_model, fit_results=fit_results)


def _resolve_deg_inter_order(deg_inter_order, n_orders, name='deg_inter_order'):
    """Same 'auto' rule as CalculateChebyshev2DWLSPrimitive. name is the parameter name used in the log message."""
    if deg_inter_order != 'auto':
        return deg_inter_order
    deg = auto_deg_inter_order(n_orders)
    logger.info(f"{name}='auto', using {name}={deg}.")
    return deg


def _load_fts_reference(fts_scans=None):
    """
    Product of the selected FTS transmission scans on an increasing wavelength grid.

    Returns (wave in microns, transmission, scan names). Scan names look like 'B0216.3a',
    where the number is the gas cell and the letter the series; each cell may be used once.
    """
    fts = read_gascell_fts()
    if fts_scans is None:
        fts_scans = [name for name in fts['scans'] if name.endswith('a')]
    unknown = set(fts_scans) - set(fts['scans'])
    if unknown:
        raise ValueError(f"Unknown FTS scans {sorted(unknown)}; available scans are {list(fts['scans'])}.")
    cells = [name.rsplit('.', 1)[1][:-1] for name in fts_scans]
    if len(set(cells)) != len(cells):
        raise ValueError(f"Each gas cell may be used once; got scans {fts_scans}.")
    transmission = np.prod([fts['scans'][name]['transmission'] for name in fts_scans], axis=0)
    wave = 1E4 / fts['wavenumber']  # cm-1 -> microns
    return wave[::-1].copy(), transmission[::-1].copy(), list(fts_scans)


def _make_gaussian_lsf_kernels(lsf_model, fiber, orders, nx, osamp, kernel_size_sigma, fts_resolution=np.inf):
    """
    Gaussian LSF kernels (Norders, nx*osamp, Ks) on the oversampled pixel grid, in the layout of make_fit_context.

    The LSF FWHM (microns) is evaluated at each oversampled pixel from the GaussianLSFModel coefficients
    and converted to pixels with the dispersion of the LSF model's own wavelength grid. If fts_resolution
    is finite, the FTS broadening lambda / fts_resolution (FWHM) is removed from it in quadrature.
    """
    correct_fts = fts_resolution is not None and np.isfinite(fts_resolution)
    coeffs = getattr(lsf_model, f'cheb_coeffs_fib{fiber}')
    norm_table = getattr(lsf_model, f'cheb_norm_fib{fiber}')
    pixel_bounds = (norm_table['PIXEL_NORM_MIN'][0], norm_table['PIXEL_NORM_MAX'][0])
    order_bounds = (norm_table['ORDER_NORM_MIN'][0], norm_table['ORDER_NORM_MAX'][0])
    lsf_wave = getattr(lsf_model, f'wave_fib{fiber}')

    pix = np.arange(nx)
    pix_s = (np.arange(nx * osamp) - osamp // 2) / osamp
    sigma_pix = np.empty((len(orders), len(pix_s)))
    for o, order in enumerate(orders):
        fwhm_wave = _build_lsf_poly2d_unstructured(coeffs, pix_s, np.full(len(pix_s), order), pixel_bounds, order_bounds)
        wave_o = lsf_wave[lsf_model.order_index(order)]
        if correct_fts:
            fwhm2 = fwhm_wave**2 - (np.interp(pix_s, pix, wave_o) / fts_resolution)**2
            if not np.all(fwhm2 > 0):
                raise ValueError(
                    f"Fiber {fiber} order {order}: the LSF is not wider than the FTS resolution "
                    f"R={fts_resolution:g} everywhere, so the FTS broadening can't be removed."
                )
            fwhm_wave = np.sqrt(fwhm2)
        dispersion = np.interp(pix_s, pix, np.abs(np.gradient(wave_o)))
        sigma_pix[o] = fwhm_wave / dispersion / SIGMA_TO_FWHM

    half_width = int(np.ceil(kernel_size_sigma * np.max(sigma_pix) * osamp))
    taps = np.arange(-half_width, half_width + 1) / osamp  # tap offsets in pixels
    kernels = np.exp(-0.5 * (taps[None, None, :] / sigma_pix[:, :, None])**2).astype(np.float32)
    kernels /= kernels.sum(axis=-1, keepdims=True)
    return kernels
