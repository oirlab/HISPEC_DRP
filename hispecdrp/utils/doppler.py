import numpy as np
import scipy.constants

__all__ = ['delta_wave_to_delta_v', 'delta_v_to_delta_wave', 'delta_pix_to_delta_wave']


def delta_wave_to_delta_v(dw, w):
    return dw / w * scipy.constants.c

def delta_v_to_delta_wave(dv, w):
    return dv / scipy.constants.c * w

def delta_pix_to_delta_wave(dx, x, wavesol=None, dispersion=None):
    if dispersion is None:
        dispersion = np.gradient(wavesol)
    disp = np.interp(x, dispersion, disp)
    dw = disp * dx
    return dw