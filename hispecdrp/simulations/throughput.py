"""
Telescope + instrument throughput and echelle blaze efficiency for simulated spectra.

The end-to-end transmission curves (without blaze) live in ``hispecdrp/data/throughput/``.
The blaze efficiency uses a sinc^2 approximation (:func:`blaze_efficiency_sinc`) and is
applied separately from the throughput.
"""

from __future__ import annotations

import importlib.resources
import logging

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

__all__ = [
    "apply_throughput_total",
    "blaze_efficiency_sinc",
    "blaze_phase",
    "diffracted_angle",
    "get_total_throughput",
]

# BSPEC curve below this wavelength, RSPEC curve at and above it.
_BSPEC_RSPEC_SPLIT = 1.4 * u.micron


def _throughput_files(mode: str):
    throughput_directory = importlib.resources.files('hispecdrp.data.throughput')
    bspec_file = throughput_directory / f'transmission_total_BSPEC_endTOend_{mode}_woBlaze.txt'
    rspec_file = throughput_directory / f'transmission_total_RSPEC_endTOend_{mode}_woBlaze.txt'
    return bspec_file, rspec_file


def _read_throughput_curve(path) -> interp1d:
    # Two-column CSV: wavelength (nm), throughput (0-1).
    data = np.genfromtxt(path, delimiter=",", skip_header=1).T
    wave_um = (data[0] * u.nm).to_value(u.micron)
    return interp1d(wave_um, data[1], fill_value=0.0, bounds_error=False)


def get_total_throughput(
    wavelength: Quantity,
    mode: str = "NGS",
    quantum_efficiency: float = 0.9,
) -> np.ndarray:
    """
    Telescope + instrument throughput (without blaze) at each wavelength.

    Parameters
    ----------
    wavelength : Quantity
        Wavelengths to evaluate the throughput at.
    mode : str
        "NGS" (Natural Guide Star) or "LGS" (Laser Guide Star).
    quantum_efficiency : float
        Scalar multiplier applied to the total throughput.

    Returns
    -------
    np.ndarray
        Dimensionless throughput, same shape as ``wavelength``. Zero outside the tabulated curves.
    """
    mode = mode.upper()
    if mode not in ("NGS", "LGS"):
        raise ValueError("Mode must be NGS or LGS!")

    bspec_file, rspec_file = _throughput_files(mode)
    bspec_interp = _read_throughput_curve(bspec_file)
    rspec_interp = _read_throughput_curve(rspec_file)

    wave_um = wavelength.to_value(u.micron)
    is_bspec = wavelength < _BSPEC_RSPEC_SPLIT
    throughput = np.ones_like(wave_um, dtype=np.float64)
    throughput[is_bspec] *= bspec_interp(wave_um[is_bspec])
    throughput[~is_bspec] *= rspec_interp(wave_um[~is_bspec])
    throughput *= quantum_efficiency
    return throughput


def apply_throughput_total(
    wavelength: Quantity,
    flux: Quantity,
    mode: str = "NGS",
    quantum_efficiency: float = 0.9,
) -> Quantity:
    """
    Apply telescope + instrument throughput to a spectrum.

    This does NOT include the blaze efficiency, which is applied separately by the rate map simulators.
    Apply throughput AFTER any radial velocity shifts.

    Parameters
    ----------
    wavelength : Quantity
        Wavelengths of ``flux``.
    flux : Quantity
        Spectrum to apply the throughput to.
    mode : str
        "NGS" (Natural Guide Star) or "LGS" (Laser Guide Star).
    quantum_efficiency : float
        Scalar multiplier applied to the total throughput.

    Returns
    -------
    Quantity
        ``flux`` times the throughput, in the units of ``flux``.
    """
    logger.warning("Throughput application is in development and may not reflect true HISPEC throughput.")
    return flux * get_total_throughput(wavelength, mode=mode, quantum_efficiency=quantum_efficiency)


# -----------------------------------------------------------------------------
# Blaze efficiency (sinc^2 approximation)
# -----------------------------------------------------------------------------

def diffracted_angle(blaze_angle, order, groove_spacing, wavelength):
    """Calculate diffracted angle using grating equation. Used for blaze efficiency calculation. Assume incidence angle = blaze angle."""
    return np.arcsin(-np.sin(blaze_angle) + order * wavelength / groove_spacing)


def blaze_phase(blaze_angle, order, groove_spacing, wavelength):
    """Calculate blaze phase for blaze efficiency calculation. Assume incidence angle = blaze angle."""
    # assume blaze and incidence angles are equal
    beta = diffracted_angle(blaze_angle, order, groove_spacing, wavelength)
    one = order * np.cos(blaze_angle)
    two = np.cos(blaze_angle) - np.sin(blaze_angle) / np.tan((beta + blaze_angle)/2)
    return one * two


def blaze_efficiency_sinc(blaze_angle, order, groove_spacing, wavelength):
    """Calculate blaze efficiency using sinc^2 approximation. Assume incidence angle = blaze angle."""
    # assume blaze and incidence angles are equal
    bp = blaze_phase(blaze_angle, order, groove_spacing, wavelength)
    sinc = np.sinc(bp*u.rad)
    sinc[np.abs(bp) > 1.0] = 0.0  # put all values outside +/- pi in sinc to zero
    return sinc**2
