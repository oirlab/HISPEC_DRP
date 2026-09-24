"""HDF / pyechelle LSF helpers used by simulated L2 paint.

Generic convolution and PSF-stamp collapse live in
:mod:`hispecdrp.primitives.lsf.lsf_convolve`. This module only loads the
instrument HDF model and maps it onto those primitives.
"""

from __future__ import annotations

import numpy as np

from ..datamodels import hispec_config
from ..primitives.lsf.lsf_convolve import (
    convolve_l2_with_gaussian_lsf,
    lsf_library_from_psf_fiber_block,
    sigma_pix_from_lsf_library,
)


def hdf_psf_lsf_library(
    spectrograph: str,
    fiber: int = 2,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Per-order library of 1-D LSFs from the simulated HDF PSF stamps.

    Returns ``{order: {"wavelengths", "offsets_list", "kernels"}}``.
    """
    from .e2e_pipeline import _load_hdf_model_data

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    _, model_data = _load_hdf_model_data(spectrograph)
    pixelsize = float(model_data.get("pixelsize", 1.0))
    fiber_block = model_data.get(f"fiber_{fiber}", {})
    return lsf_library_from_psf_fiber_block(fiber_block, pixelsize=pixelsize)


def sigma_pix_from_hdf_lsf(
    spectrograph: str,
    fiber: int,
    order: int,
    wave: np.ndarray,
) -> np.ndarray:
    """
    Effective Gaussian σ (detector pixels) vs spectral sample from HDF PSF LSF.
    """
    lib = hdf_psf_lsf_library(spectrograph, fiber)
    return sigma_pix_from_lsf_library(lib, order, wave)


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
    lib = hdf_psf_lsf_library(spectrograph, fiber)
    sigma_2d = np.empty_like(wave_2d)
    for i in range(n_ord):
        od = int(order_start) + i
        sigma_2d[i] = sigma_pix_from_lsf_library(lib, od, wave_2d[i])
    return convolve_l2_with_gaussian_lsf(wave_2d, flux_2d, sigma_2d)
