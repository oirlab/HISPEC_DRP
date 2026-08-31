"""LSF utilities: HDF PSF → 1-D kernels and spectrum convolution."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ...datamodels import hispec_config


def _psf_oversample(block: dict, pixelsize: float) -> float:
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


def hdf_psf_lsf_library(
    spectrograph: str,
    fiber: int = 2,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Per-order library of 1-D LSFs from HDF PSF stamps.

    Returns ``{order: {"wavelengths", "offsets_list", "kernels"}}`` where
    ``offsets_list`` / ``kernels`` are length-``n_stamp`` lists of 1-D arrays.
    """
    from ...simulations.e2e_pipeline import _load_hdf_model_data

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    _, model_data = _load_hdf_model_data(spectrograph)
    pixelsize = float(model_data.get("pixelsize", 1.0))
    fiber_block = model_data.get(f"fiber_{fiber}", {})
    out: dict[int, dict[str, np.ndarray]] = {}
    for od_s, block in fiber_block.items():
        if "psf_stamps" not in block or "psf_wavelengths" not in block:
            continue
        od = int(od_s)
        oversample = _psf_oversample(block, pixelsize)
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


def sigma_pix_from_hdf_lsf(
    spectrograph: str,
    fiber: int,
    order: int,
    wave: np.ndarray,
) -> np.ndarray:
    """
    Effective Gaussian σ (detector pixels) vs spectral sample from HDF PSF LSF.

    Uses second-moment of the collapsed 1-D kernel, interpolated in wavelength.
    """
    lib = hdf_psf_lsf_library(spectrograph, fiber)
    if int(order) not in lib:
        return np.full(np.asarray(wave).shape, 1.0, dtype=np.float64)
    block = lib[int(order)]
    waves = block["wavelengths"]
    sigs = np.empty(waves.size, dtype=np.float64)
    for i, (offs, ker) in enumerate(zip(block["offsets_list"], block["kernels"])):
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


def convolve_l2_with_hdf_lsf(
    wave_2d: np.ndarray,
    flux_2d: np.ndarray,
    *,
    spectrograph: str,
    fiber: int,
    order_start: int | None = None,
) -> np.ndarray:
    """
    Apply per-order HDF-derived Gaussian LSF to a 2-D (order, pix) spectrum.
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    flux_2d = np.asarray(flux_2d, dtype=np.float64)
    n_ord = wave_2d.shape[0]
    if order_start is None:
        order_start = int(
            hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"]
        )
    out = np.empty_like(flux_2d)
    for i in range(n_ord):
        od = int(order_start) + i
        sig = sigma_pix_from_hdf_lsf(spectrograph, fiber, od, wave_2d[i])
        out[i] = convolve_order_spectrum_gaussian_lsf(wave_2d[i], flux_2d[i], sig)
    return out
