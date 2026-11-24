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
from .gascell_utils import read_gascell_fts
from ..utils.array_utils import auto_deg_inter_order
from .wavesol_cheb2d_primitive import DEFAULT_DEG_INTRA_ORDER, _build_wavesol_poly2d_structured, wavesol_norm_bounds
import numpy as np

logger = logging.getLogger(__name__)


__all__ = ['FitGasCellWavesolPrimitive']

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


class FitGasCellWavesolPrimitive(BasePrimitive):

    """
    Fit the wavelength solution of a gas cell spectrum by forward modeling it with an FTS reference transmission.

    The wavelength solution uses the same 2D Chebyshev convention as :class:`CalculateChebyshev2DWLSPrimitive`: m * lambda is a 2D Chebyshev series in the normalized pixel and normalized echelle order. For each fiber, the forward model is the FTS transmission evaluated at the current wavelength solution on an oversampled pixel grid, convolved with the Gaussian LSF, times a per-order Chebyshev continuum. All orders are fit simultaneously with Levenberg-Marquardt, starting from the input wavelength solution.

    The FTS reference is the total transmission of all gas cells (``transmission_tot`` from :func:`~hispecdrp.primitives.wavesol.gascell_utils.read_gascell_fts`, one scan per cell) on its wavelength grid in microns. The FTS scan is a flux density (per unit wavelength), not a flux integrated over each sample, so it is sampled at the model wavelengths by plain (linear) interpolation; no flux-conserving rebinning is needed.

    The FTS scans are already broadened by the FTS resolution, which is fixed in wavenumber (``fts_resolution``, by default ``meta['resolution']`` in cm-1 from the FTS file). It is treated as the FWHM of a Gaussian, so the LSF kernels are narrowed to FWHM = sqrt(FWHM_lsf^2 - FWHM_fts^2), where FWHM_fts = lambda^2 * resolution / 1E4 (microns) is approximated as linear in pixel across each order.

    Parameters
    ----------
    input : str or L2Model
        Input gas cell spectrum with an initial wavelength solution (``wave_fib#``), e.g. from :class:`SetInitialWLSPrimitive`.
    initial_wavesol : str
        Wavelength solution the fit starts from. 'input' uses the input model's ``wave_fib#``. 'simulated' uses the simulated (pyechelle HDF) wavelength solution for the input model's spectrograph and each fiber, from :meth:`~hispecdrp.simulations.SimulateRateMap.get_wavelength_solution`, and ignores ``wave_fib#``. Default is 'input'.
    wave_deg_intra_order : int or 'auto'
        Degree of the polynomial within an order. ``'auto'`` uses :data:`~hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive.DEFAULT_DEG_INTRA_ORDER`, as for the LFC. Default is 'auto'.
    wave_deg_inter_order : int or 'auto'
        Degree of the polynomial between orders. If 'auto', it is set from the number of requested orders (``fit_orders``) with the same rule as the LFC (:func:`~hispecdrp.primitives.utils.array_utils.auto_deg_inter_order`). Default is 'auto'.
    continuum_deg_intra_order : int or None
        Degree in pixel of the Chebyshev continuum, fit independently for each order. If None, no continuum is fit. Default is 2.
    osamp : int
        Oversampling factor (odd) of the pixel grid the FTS reference is sampled and convolved on.
    kernel_size_sigma : float
        Half-width of the LSF kernels in units of the largest Gaussian sigma.
    fts_resolution : float, None or bool
        FTS resolution in wavenumber (cm-1), removed from the LSF kernels in quadrature. If None or True, it is read from the FTS file (``meta['resolution']``). If False, the FTS broadening is not removed, e.g. for data simulated from the FTS reference itself. Default is None.
    max_iters : int
        Maximum number of residual evaluations of the Levenberg-Marquardt fit (``max_nfev`` of ``scipy.optimize.least_squares``).
    ftol : float
        Relative decrease in chi^2 below which the fit is considered converged (``ftol`` of ``scipy.optimize.least_squares``).
    use_opt_weights : bool
        Whether to weight each pixel by the inverse variance from ``spec_err``. If False, all unmasked pixels are weighted equally. Default is True.
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
        The wavelength solution model. Besides the wavelength solution (``wave_fib#``, ``cheb_coeffs_fib#``, normalized with :func:`~hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive.wavesol_norm_bounds`), it holds the best-fit forward model (``model_fib#``) and its residuals (``model_resid_fib#``), the best-fit continuum (``cont_fib#``), its Chebyshev coefficients (``cont_coeffs_fib#``, with the same pixel normalization) and its model type (``cont_info_fib#``, '1d' or 'none'). The arrays are on the pixel grid and NaN on orders not fit. The continuum coefficients have one row per order (NaN for orders not fit); evaluate them with :func:`~hispecdrp.primitives.wavesol.fit_gascell.build_continuum_1d`.

    """

    PARAMETERS = {
        'initial_wavesol' : 'input',
        'wave_deg_intra_order' : 'auto',
        'wave_deg_inter_order' : 'auto',
        'continuum_deg_intra_order' : 2,
        'osamp' : 5,
        'kernel_size_sigma' : 5.0,
        'fts_resolution' : None,
        'max_iters' : 50,
        'ftol' : 1E-9,
        'use_opt_weights' : True,
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

        # Polynomial degrees, with the same 'auto' rules as CalculateChebyshev2DWLSPrimitive
        if self.wave_deg_intra_order == 'auto':
            deg_intra_order = DEFAULT_DEG_INTRA_ORDER
            logger.info(f"wave_deg_intra_order='auto', using {deg_intra_order=}.")
        else:
            deg_intra_order = self.wave_deg_intra_order
        if self.wave_deg_inter_order == 'auto':
            deg_inter_order = auto_deg_inter_order(len(fit_orders))
            logger.info(f"wave_deg_inter_order='auto', using {deg_inter_order=}.")
        else:
            deg_inter_order = self.wave_deg_inter_order

        # FTS reference: total transmission of all gas cells on an increasing wavelength grid (microns)
        fts = read_gascell_fts()
        lam_ref, F_ref = fts['wavelength'], fts['transmission_tot']
        logger.info(f"Using FTS gas cells {fts['meta']['cells']}.")
        if self.fts_resolution is None or self.fts_resolution is True:
            fts_resolution = fts['meta']['resolution']
        elif self.fts_resolution is False:
            fts_resolution = 0.0
        else:
            fts_resolution = float(self.fts_resolution)
        logger.info(f"FTS resolution removed from the LSF: {fts_resolution:g} cm-1.")

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

            # Forward model setup
            kernels = _make_gaussian_lsf_kernels(
                lsf_model, fiber, orders, nx, self.osamp, self.kernel_size_sigma,
                fts_resolution=fts_resolution,
            )
            sigma = spec_err[oi_fit].astype(np.float64)
            if not self.use_opt_weights:
                # Uniform weights, but keep masking pixels with an invalid error
                sigma = np.where(np.isfinite(sigma) & (sigma > 0), 1.0, np.nan)
            ctx = make_fit_context(
                orders, spec[oi_fit].astype(np.float64), sigma,
                lam_ref, F_ref, kernels, self.osamp,
                deg_intra_order, deg_inter_order,
                dq=np.where(in_ref[oi_fit], spec_dq[oi_fit], 1),
                continuum_deg_intra_order=self.continuum_deg_intra_order,
                order_bounds=wavesol_norm_bounds(nx, order_vec)[1],
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
            residuals = spec[oi_fit] - model
            residuals_norm = residuals / spec_err[oi_fit]
            # Reduced chi2 per order with the fit's weights and mask. Each order has its own
            # continuum coefficients and an equal share of the shared wavelength coefficients,
            # so the per-order degrees of freedom sum to the fiber's.
            isig = ctx['isig']
            chi2_orders = np.sum(np.where(isig > 0, residuals * isig, 0.0)**2, axis=1)
            dof_orders = np.count_nonzero(isig, axis=1) - ctx['Q'] - ctx['nc'] / len(orders)
            chi2_red_orders = np.where(dof_orders > 0, chi2_orders / np.where(dof_orders > 0, dof_orders, 1), np.nan)
            # Mean of the starting wavelength grid per fit order (NaN where it has no finite values)
            wave0_fit = np.asarray(wave0[oi_fit], dtype=np.float64)
            n_finite = np.count_nonzero(np.isfinite(wave0_fit), axis=1)
            mean_wave_start = np.where(
                n_finite > 0, np.nansum(wave0_fit, axis=1) / np.maximum(n_finite, 1), np.nan,
            )
            fit_results.append(dict(
                mean_wave_start=mean_wave_start,
                fiber=fiber, orders=orders, coeffs=cmat, continuum_coeffs=amat, continuum_model=continuum_model,
                # (Norders fit, Ncols) on the pixel grid
                model=model, continuum=continuum, residuals=residuals, residuals_norm=residuals_norm,
                cov=cov, chi2_red=chi2_red, chi2_red_orders=chi2_red_orders, info=info,
                pixel_bounds=ctx['pixel_bounds'], order_bounds=ctx['order_bounds'],
            ))

            # Set results in model
            setattr(output_model, f'cheb_coeffs_fib{fiber}', cmat)
            wavesol_full = _build_wavesol_poly2d_structured(
                cmat, np.arange(nx), order_vec,
                ctx['pixel_bounds'], ctx['order_bounds'],
            )
            setattr(output_model, f'wave_fib{fiber}', wavesol_full)
            resid = np.full(spec.shape, np.nan, dtype=np.float32)
            resid[oi_fit] = residuals
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

    def _create_preview(self):
        """
        Interactive preview of the gas cell fit: the data and the best-fit forward model (top) and
        their residuals (bottom) against the fitted wavelength, one fit order at a time.

        A slider steps through the fitted orders in increasing wavelength (decreasing order number).
        Each tick shows the order number above the rail and the order's mean wavelength below it
        (from the starting wavelength grid where available, otherwise from the fit). With more than
        one fitted fiber, a dropdown left of the slider selects the fiber and rebuilds the slider
        for it. Each (fiber, order) combination has its own three traces, and the controls only
        toggle their visibility. Built by :func:`.gascell_preview.create_gascell_preview`.
        """
        from .gascell_preview import create_gascell_preview
        create_gascell_preview(self)


def _make_gaussian_lsf_kernels(lsf_model, fiber, orders, nx, osamp, kernel_size_sigma, fts_resolution=0.0):
    """
    Gaussian LSF kernels (Norders, nx*osamp, Ks) on the oversampled pixel grid, in the layout of make_fit_context.

    The LSF FWHM (microns) is evaluated at each oversampled pixel from the GaussianLSFModel coefficients
    and converted to pixels with the dispersion of the LSF model's own wavelength grid. If fts_resolution
    (the FTS resolution in cm-1, fixed in wavenumber) is > 0, the FTS broadening is removed from it in
    quadrature. The FTS FWHM in microns is lambda^2 * fts_resolution / 1E4, approximated as linear in pixel
    across each order.
    """
    correct_fts = fts_resolution is not None and fts_resolution > 0
    coeffs = getattr(lsf_model, f'cheb_coeffs_fib{fiber}')
    lsf_wave = getattr(lsf_model, f'wave_fib{fiber}')
    pixel_bounds, order_bounds = wavesol_norm_bounds(lsf_wave.shape[1], lsf_model.make_order_vec())

    pix = np.arange(nx)
    pix_s = (np.arange(nx * osamp) - osamp // 2) / osamp
    sigma_pix = np.empty((len(orders), len(pix_s)))
    for o, order in enumerate(orders):
        fwhm_wave = _build_lsf_poly2d_unstructured(coeffs, pix_s, np.full(len(pix_s), order), pixel_bounds, order_bounds)
        wave_o = lsf_wave[lsf_model.order_index(order)]
        if correct_fts:
            # FTS FWHM in microns, linear in pixel across the order
            fts_fwhm_wave = np.polyval(np.polyfit(pix, wave_o**2 * fts_resolution / 1E4, 1), pix_s)
            fwhm2 = fwhm_wave**2 - fts_fwhm_wave**2
            if not np.all(fwhm2 > 0):
                raise ValueError(
                    f"Fiber {fiber} order {order}: the LSF is not wider than the FTS resolution "
                    f"({fts_resolution:g} cm-1) everywhere, so the FTS broadening can't be removed."
                )
            fwhm_wave = np.sqrt(fwhm2)
        dispersion = np.interp(pix_s, pix, np.abs(np.gradient(wave_o)))
        sigma_pix[o] = fwhm_wave / dispersion / SIGMA_TO_FWHM

    half_width = int(np.ceil(kernel_size_sigma * np.max(sigma_pix) * osamp))
    taps = np.arange(-half_width, half_width + 1) / osamp  # tap offsets in pixels
    kernels = np.exp(-0.5 * (taps[None, None, :] / sigma_pix[:, :, None])**2).astype(np.float32)
    kernels /= kernels.sum(axis=-1, keepdims=True)
    return kernels
