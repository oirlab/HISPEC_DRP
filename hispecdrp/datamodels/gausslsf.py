import numpy as np
from numba import njit

from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_lfcfitparam_extensions

from ..primitives.lsf.lsf_convolve import convolve_numba_1d_chromatic

__all__ = ['GaussianLSFModel']


class GaussianLSFModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The effective line spread function (LSF) is modeled with a Gaussian where the width is described with a 2D polynomial (pixel, order, width).
    """

    EXTENSIONS = {
        **create_dataspec_extensions(title='LFC spectrum.', units='e-/s'),
        **create_specdq_extensions(title='LFC spectrum data quality.'),
        **create_wave_extensions(title='LFC spectrum wavelength grid.'),
        # No default given -> not auto-initialized (behavior unchanged), though
        # the shape (Nord x 4096 x 3: mu, sigma, offset) is now fully resolvable.
        'LSF_PARAMS': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y', 3), enumerate='FIB',
            title='LFC spectrum Gaussian LSF parameters (mu, sigma, offset).'
        ),
        **create_chebyshev_extensions(),
        **create_lfcfitparam_extensions(),
    }

    CAL_TYPE = 'lsf'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None

    def make_kernels(
        self,
        wave : np.ndarray,
        fiber : int, order : int,
        kernel_size_sigma : float = 5.0
    ) -> np.ndarray:
        """
        Generate the Gaussian LSFs for the input wavelength array
        using the LSF FWHM polynomial coeffs.

        Parameters
        ----------
        wave : np.ndarray
            1-D array of wavelengths (in microns) for which to generate the LSF
        fiber : int
            Fiber number (1, 2, or 3)
        order : int
            Echelle order number for which to generate the LSF
        kernel_size_sigma : float, optional
            Number of sigma to include in the LSF grid (default: 5.0)
        
        Returns
        -------
        lsfs : np.ndarray
            2-D array of LSFs with shape (len(wave), n_lsf), where n_lsf is the number of pixels in the LSF grid.
        """
        from ..primitives.lsf.lsf_gauss_primitive import _build_lsf_poly2d_unstructured
        coeffs = self.cheb_coeffs[fiber]
        pixel_bounds = (
            self.cheb_norm[fiber]['PIXEL_NORM_MIN'],
            self.cheb_norm[fiber]['PIXEL_NORM_MAX']
        )
        order_bounds = (
            self.cheb_norm[fiber]['ORDER_NORM_MIN'],
            self.cheb_norm[fiber]['ORDER_NORM_MAX']
        )
        idx_ord = self.order_index(order)
        orders = np.full_like(wave, order, dtype=int)
        pixarr = np.arange(len(self.wave[fiber][idx_ord]))
        pixels = np.interp(wave, self.wave[fiber][idx_ord], pixarr, left=pixarr[0], right=pixarr[-1])
        fwhms = _build_lsf_poly2d_unstructured(
            coeffs, pixels, orders, pixel_bounds, order_bounds
        )
        dw = abs(wave[2] - wave[1])
        kernel_grid = _make_kernel_grid(
            dw=dw,
            fwhms=fwhms,
            kernel_size_sigma=kernel_size_sigma
        )
        kernels = _make_gaussians_chromatic(
            kernel_grid, fwhms,
        )
        return kernel_grid, kernels

    def convolve_spectrum(
        self,
        wave : np.ndarray, spec : np.ndarray,
        fiber : int, order : int,
        kernel_size_sigma : float = 5.0,
    ) -> np.ndarray:
        _, kernels = self.make_kernels(
            wave=wave,
            fiber=fiber, order=order,
            kernel_size_sigma=kernel_size_sigma
        )
        return convolve_numba_1d_chromatic(spec, kernels)


@njit(nogil=True)
def _make_kernel_grid(
    dw : float,
    fwhms : np.ndarray,
    kernel_size_sigma : float = 5.0
) -> np.ndarray:
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    nx_lsf = int(2 * np.ceil(np.max(fwhms) / SIGMA_TO_FWHM * kernel_size_sigma / dw))
    if nx_lsf % 2 == 0:
        nx_lsf += 1
    kernel_grid = dw * (np.arange(nx_lsf) - nx_lsf // 2)
    return kernel_grid

@njit(nogil=True)
def _make_gaussians_chromatic(
    kernel_grid : np.ndarray,
    fwhms : np.ndarray,
) -> np.ndarray:
    """
    Generate a set of Gaussian LSFs for the input x array and fwhms.

    Parameters
    ----------
    kernel_grid : np.ndarray
        1-D array of x values (e.g., wavelength offsets) for the LSF grid.
    fwhms : np.ndarray
        1-D array of FWHM values for each x value
    kernel_size_sigma : float, optional
        Number of sigma to include in the LSF grid (default: 5.0)

    Returns
    -------
    kernels : np.ndarray
        2-D array of Gaussian LSFs with shape (len(fwhms), len(x)).
    """
    nx_lsf = len(kernel_grid)
    n_lsfs = len(fwhms)
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    kernels = np.zeros((n_lsfs, nx_lsf), dtype=np.float32)
    for i in range(n_lsfs):
        kernels[i, :] = np.exp(-0.5 * (kernel_grid / (fwhms[i] / SIGMA_TO_FWHM))**2)
        kernels[i, :] /= np.sum(kernels[i, :])
    return kernels