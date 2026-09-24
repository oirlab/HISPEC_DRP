"""Generic LSF utilities: PSF stamps → 1-D kernels and spectrum convolution.

This module has no dependency on ``hispecdrp.simulations``. HDF / pyechelle
instrument-model loading lives in ``hispecdrp.simulations.lsf``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.interpolate import interp1d


def _psf_oversample(block: Mapping[str, Any], pixelsize: float) -> float:
    spacing = float(block.get("psf_data_spacing", pixelsize))
    if spacing <= 0.0:
        return 1.0
    return float(pixelsize) / spacing


def lsf_1d_from_psf_stamp(
    stamp: np.ndarray,
    *,
    oversample: float,
    half_width_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse an oversampled PSF stamp to a 1-D spectral LSF.

    Stamp layout matches deposit: axis 0 = cross-dispersion (``y``), axis 1 =
    spectral (``x``). Returns ``(offsets_pix, kernel)`` with ``offsets`` in
    detector spectral pixels relative to the stamp center, and ``kernel``
    normalized to unit sum.
    """
    psf = np.asarray(stamp, dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError(f"PSF stamp must be 2-D; got {psf.shape}")
    _h, w = psf.shape
    # Spectral profile: sum over cross-dispersion.
    prof = np.sum(np.maximum(psf, 0.0), axis=0)
    total = float(np.sum(prof))
    if not np.isfinite(total) or total <= 0.0:
        offs = np.array([0.0], dtype=np.float64)
        return offs, np.ones(1, dtype=np.float64)

    i = np.arange(w, dtype=np.float64)
    offs = (i - 0.5 * (w - 1)) / float(oversample)
    ker = prof / total
    if half_width_pix is not None and float(half_width_pix) > 0.0:
        m = np.abs(offs) <= float(half_width_pix)
        if not np.any(m):
            m[int(np.argmin(np.abs(offs)))] = True
        offs = offs[m]
        ker = ker[m]
        s = float(np.sum(ker))
        if s > 0.0:
            ker = ker / s
    return offs.astype(np.float64), ker.astype(np.float64)


def lsf_library_from_psf_fiber_block(
    fiber_block: Mapping[Any, Any],
    *,
    pixelsize: float = 1.0,
) -> dict[int, dict[str, Any]]:
    """
    Build a per-order 1-D LSF library from one fiber's PSF stamp block.

    ``fiber_block`` is ``{order: {psf_stamps, psf_wavelengths, ...}}`` (the
    layout written by the pyechelle HDF reader). Returns
    ``{order: {"wavelengths", "offsets_list", "kernels"}}``.
    """
    out: dict[int, dict[str, Any]] = {}
    for od_s, block in fiber_block.items():
        if not isinstance(block, Mapping):
            continue
        if "psf_stamps" not in block or "psf_wavelengths" not in block:
            continue
        od = int(od_s)
        oversample = _psf_oversample(block, float(pixelsize))
        waves = np.asarray(block["psf_wavelengths"], dtype=np.float64)
        stamps = block["psf_stamps"]
        offsets_list = []
        kernels = []
        for i in range(len(waves)):
            offs, ker = lsf_1d_from_psf_stamp(stamps[i], oversample=oversample)
            offsets_list.append(offs)
            kernels.append(ker)
        out[od] = {
            "wavelengths": waves,
            "offsets_list": offsets_list,
            "kernels": kernels,
        }
    return out


def sigma_pix_from_lsf_library(
    lib: Mapping[int, Mapping[str, Any]],
    order: int,
    wave: np.ndarray,
) -> np.ndarray:
    """
    Effective Gaussian σ (detector pixels) vs spectral sample from an LSF library.

    Uses the second moment of each collapsed 1-D kernel, interpolated in
    wavelength. Missing orders return σ = 1 pixel.
    """
    if int(order) not in lib:
        return np.full(np.asarray(wave).shape, 1.0, dtype=np.float64)
    block = lib[int(order)]
    waves = np.asarray(block["wavelengths"], dtype=np.float64)
    sigs = np.empty(waves.size, dtype=np.float64)
    for i, (offs, ker) in enumerate(zip(block["offsets_list"], block["kernels"])):
        offs = np.asarray(offs, dtype=np.float64)
        ker = np.asarray(ker, dtype=np.float64)
        mu = float(np.sum(offs * ker))
        var = float(np.sum(ker * (offs - mu) ** 2))
        sigs[i] = np.sqrt(max(var, 1.0e-8))
    wave = np.asarray(wave, dtype=np.float64)
    order_idx = np.argsort(waves)
    waves_s = waves[order_idx]
    sigs_s = sigs[order_idx]
    if waves_s.size == 1:
        return np.full(wave.shape, float(sigs_s[0]))
    f = interp1d(waves_s, sigs_s, kind="linear", bounds_error=False, fill_value="extrapolate")
    return np.asarray(f(wave), dtype=np.float64)


def convolve_order_spectrum_gaussian_lsf(
    wave: np.ndarray,
    flux: np.ndarray,
    sigma_pix: np.ndarray,
    *,
    num_sigma: float = 5.0,
) -> np.ndarray:
    """
    Convolve a 1-D order spectrum with a wavelength-dependent Gaussian LSF.

    ``sigma_pix`` is in detector spectral pixels; converted to wavelength using
    local ``dλ/dpix`` from ``wave``.
    """
    from ..telluric_correction.telluric_models import _convolve_and_resample

    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    sigma_pix = np.asarray(sigma_pix, dtype=np.float64)
    dw = np.gradient(wave)
    abs_dw = np.abs(dw)
    pos = np.isfinite(abs_dw) & (abs_dw > 0.0)
    if np.any(pos):
        abs_dw = np.where(pos, abs_dw, float(np.median(abs_dw[pos])))
    else:
        abs_dw = np.full_like(abs_dw, 1.0e-6)
    sigma_wave = abs_dw * np.maximum(sigma_pix, 1.0e-6)
    # Work on unique sorted wavelength for the cubic interpolator.
    order = np.argsort(wave)
    wave_s = wave[order]
    flux_s = flux[order]
    sigma_s = sigma_wave[order]
    uniq, first = np.unique(wave_s, return_index=True)
    conv_u = _convolve_and_resample(
        uniq,
        flux_s[first],
        uniq,
        sigma_s[first],
        num_sigma=int(max(3, int(np.ceil(num_sigma)))),
    )
    conv_s = np.interp(wave_s, uniq, conv_u)
    out = np.empty_like(flux)
    out[order] = conv_s
    return out


def convolve_l2_with_gaussian_lsf(
    wave_2d: np.ndarray,
    flux_2d: np.ndarray,
    sigma_pix_2d: np.ndarray,
    *,
    num_sigma: float = 5.0,
) -> np.ndarray:
    """
    Apply a per-pixel Gaussian LSF to a 2-D ``(order, pix)`` spectrum.

    ``sigma_pix_2d`` has the same shape as ``wave_2d`` / ``flux_2d`` and is in
    detector spectral pixels.
    """
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    flux_2d = np.asarray(flux_2d, dtype=np.float64)
    sigma_pix_2d = np.asarray(sigma_pix_2d, dtype=np.float64)
    if wave_2d.shape != flux_2d.shape or wave_2d.shape != sigma_pix_2d.shape:
        raise ValueError(
            "wave_2d, flux_2d, and sigma_pix_2d must have the same shape; "
            f"got {wave_2d.shape}, {flux_2d.shape}, {sigma_pix_2d.shape}"
        )
    out = np.empty_like(flux_2d)
    for i in range(wave_2d.shape[0]):
        out[i] = convolve_order_spectrum_gaussian_lsf(
            wave_2d[i], flux_2d[i], sigma_pix_2d[i], num_sigma=num_sigma
        )
    return out


try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*_args, **_kwargs):
        def _wrap(fn):
            return fn
        return _wrap


@njit(nogil=True)
def _convolve_numba_1d_chromatic(
    spec_padded: np.ndarray,
    kernels: np.ndarray,
    spec_out: np.ndarray,
) -> np.ndarray:
    kernel_size = kernels.shape[1]
    for i in range(len(spec_out)):
        s = 0.0
        for j in range(kernel_size):
            s += spec_padded[i + j] * kernels[i, j]
        spec_out[i] = s
    return spec_out


def convolve_numba_1d_chromatic(
    spec: np.ndarray,
    kernels: np.ndarray,
    spec_out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convolve a 1-D spectrum with a wavelength-dependent LSF.

    ``kernels`` is ``(n_wave, n_kernel)`` with an odd ``n_kernel``. Each row is
    the normalized kernel at that sample. The wavelength grid must be uniform
    (same assumption as :meth:`~hispecdrp.datamodels.gausslsf.GaussianLSFModel.make_kernels`).
    """
    spec = np.asarray(spec)
    kernels = np.asarray(kernels)
    kernel_size = int(kernels.shape[1])
    if kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd; got {kernel_size}")
    if kernels.shape[0] != spec.shape[0]:
        raise ValueError(
            f"Kernels shape {kernels.shape} does not match spectrum length {spec.shape[0]}"
        )
    n_pad = kernel_size // 2
    spec_padded = np.pad(spec, n_pad, mode="edge")
    if spec_out is None:
        spec_out = np.full_like(spec, np.nan)
    elif spec_out.shape != spec.shape:
        raise ValueError(
            f"Output array shape {spec_out.shape} does not match spectrum shape {spec.shape}"
        )
    return _convolve_numba_1d_chromatic(spec_padded, kernels, spec_out)
