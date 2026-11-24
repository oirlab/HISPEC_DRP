import numpy as np
from numba import njit

from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_lfcfitparam_extensions

__all__ = ['GaussianLSFModel']


def _interp_pixel_from_wave(wave_query: np.ndarray, wave_ord: np.ndarray) -> np.ndarray:
    """Map wavelengths onto detector pixel index, allowing a decreasing WAVE row."""
    wave_query = np.asarray(wave_query, dtype=np.float64)
    wave_ord = np.asarray(wave_ord, dtype=np.float64)
    pixarr = np.arange(wave_ord.size, dtype=np.float64)
    if wave_ord.size < 2:
        return np.full(wave_query.shape, 0.0 if wave_ord.size else np.nan)
    if wave_ord[0] > wave_ord[-1]:
        return np.interp(
            wave_query, wave_ord[::-1], pixarr[::-1], left=pixarr[-1], right=pixarr[0]
        )
    return np.interp(wave_query, wave_ord, pixarr, left=pixarr[0], right=pixarr[-1])


class GaussianLSFModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The effective line spread function (LSF) is modeled with a Gaussian where the width is described with a 2D polynomial (pixel, order, width).
    """

    EXTENSIONS = {
        **create_dataspec_extensions(title='LFC spectrum.', units='e-/s'),
        **create_specdq_extensions(title='LFC spectrum data quality.'),
        **create_wave_extensions(title='LFC spectrum wavelength grid.'),
        # Evaluated FWHM (micron) on the instrument grid (2-D); Chebyshev
        # coeffs are the source of truth for :meth:`make_kernels`.
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
        if self.meta.target_name is None:
            spectrograph = self.meta.spectrograph
            if spectrograph is not None:
                self.meta.target_name = f'LFC-{str(spectrograph)[0].upper()}'

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None

    def make_kernels(
        self,
        wave: np.ndarray,
        fiber: int,
        order: int,
        kernel_size_sigma: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gaussian LSF kernels on a uniform wavelength grid.

        Parameters
        ----------
        wave
            1-D uniform wavelength grid (micron).
        fiber
            Fiber number (1, 2, or 3).
        order
            Echelle order.
        kernel_size_sigma
            Half-width of each kernel in units of Gaussian σ.

        Returns
        -------
        kernel_grid, kernels
            ``kernel_grid`` is the shared wavelength-offset axis; ``kernels``
            has shape ``(len(wave), n_kernel)`` and is row-normalized.
        """
        from ..primitives.lsf.lsf_gauss_primitive import _build_lsf_poly2d_unstructured
        from ..primitives.wavesol.wavesol_cheb2d_primitive import wavesol_norm_bounds

        wave = np.asarray(wave, dtype=np.float64).ravel()
        if wave.size < 2:
            raise ValueError("wave must have at least two samples")
        if not np.all(np.isfinite(wave)):
            raise ValueError("wave must be finite")

        coeffs = self.cheb_coeffs[fiber]
        if coeffs is None:
            raise ValueError(
                f"GaussianLSFModel has no cheb_coeffs_fib{fiber}; "
                "run CalculateGaussianLSFPrimitive or load an LFC LSF cal."
            )
        idx_ord = self.order_index(order)
        w_ord = np.asarray(self.wave[fiber][idx_ord], dtype=np.float64)
        pixel_bounds, order_bounds = wavesol_norm_bounds(w_ord.size, self.make_order_vec())
        pixels = _interp_pixel_from_wave(wave, w_ord)
        orders = np.full(wave.shape, int(order), dtype=np.float64)
        fwhms = np.asarray(
            _build_lsf_poly2d_unstructured(
                coeffs, pixels, orders, pixel_bounds, order_bounds
            ),
            dtype=np.float64,
        )
        dw = float(np.median(np.abs(np.diff(wave))))
        if not np.isfinite(dw) or dw <= 0.0:
            raise ValueError("wave must be a uniform grid with nonzero spacing")
        kernel_grid = _make_kernel_grid(
            dw=dw, fwhms=fwhms, kernel_size_sigma=float(kernel_size_sigma)
        )
        kernels = _make_gaussians_chromatic(kernel_grid, fwhms)
        return kernel_grid, kernels

    def convolve_spectrum(
        self,
        wave: np.ndarray,
        spec: np.ndarray,
        fiber: int,
        order: int,
        kernel_size_sigma: float = 5.0,
    ) -> np.ndarray:
        """Convolve ``spec`` on uniform ``wave`` with this fiber/order LSF."""
        from ..primitives.lsf.lsf_convolve import convolve_numba_1d_chromatic

        _, kernels = self.make_kernels(
            wave=wave,
            fiber=fiber,
            order=order,
            kernel_size_sigma=kernel_size_sigma,
        )
        return convolve_numba_1d_chromatic(spec, kernels)


@njit(nogil=True)
def _make_kernel_grid(
    dw: float,
    fwhms: np.ndarray,
    kernel_size_sigma: float = 5.0,
) -> np.ndarray:
    sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_max = np.nanmax(np.abs(fwhms))
    if (not np.isfinite(fwhm_max)) or fwhm_max <= 0.0 or dw <= 0.0:
        nx_lsf = 1
    else:
        nx_lsf = int(2 * np.ceil(fwhm_max / sigma_to_fwhm * kernel_size_sigma / dw))
        if nx_lsf < 1:
            nx_lsf = 1
        if nx_lsf % 2 == 0:
            nx_lsf += 1
    return dw * (np.arange(nx_lsf) - nx_lsf // 2)


@njit(nogil=True)
def _make_gaussians_chromatic(
    kernel_grid: np.ndarray,
    fwhms: np.ndarray,
) -> np.ndarray:
    nx_lsf = len(kernel_grid)
    n_lsfs = len(fwhms)
    sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    kernels = np.zeros((n_lsfs, nx_lsf), dtype=np.float32)
    for i in range(n_lsfs):
        sig = abs(fwhms[i]) / sigma_to_fwhm
        if sig <= 0.0:
            kernels[i, nx_lsf // 2] = 1.0
            continue
        kernels[i, :] = np.exp(-0.5 * (kernel_grid / sig) ** 2)
        kernels[i, :] /= np.sum(kernels[i, :])
    return kernels
