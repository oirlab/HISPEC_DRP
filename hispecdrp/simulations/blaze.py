"""Echelle blaze efficiency (sinc^2 approximation) for the simulator."""

import astropy.units as u
import numpy as np

__all__ = ["diffracted_angle", "blaze_phase", "blaze_efficiency_sinc"]


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
