import astropy.units as u
import astropy.constants as const
import numpy as np

def unresolved_etalon(photon_flux=100*u.ph/u.s,
                      wavelength_range = np.array([950,1550])*u.nm,
                      free_spectral_range_nu = None,
                      n = 1.,
                      d = 5 * u.mm,
                      reference_wavelength=None,
                      preset=None):

    wl_min = wavelength_range[0]
    wl_max = wavelength_range[1]
    nu_max = (const.c / wl_min).to(u.GHz)
    nu_min = (const.c / wl_max).to(u.GHz)

    if preset=='BSPEC':
        free_spectral_range_nu = 30 * u.GHz
        reference_wavelength = 1000*u.nm
        upper_part = np.arange(const.c/reference_wavelength.to(u.GHz), nu_max, free_spectral_range_nu)
        lower_part = np.arange(const.c/reference_wavelength.to(u.GHz), nu_min, free_spectral_range_nu)
        wavelength_peaks = const.c / np.concatenate((lower_part, upper_part))
        return wavelength_peaks.to(u.micron), np.ones_like(wavelength_peaks.value) * photon_flux

    if free_spectral_range_nu != None:
        if reference_wavelength != None:
            wavelength_peaks = const.c / np.arange(nu_min, nu_max, free_spectral_range_nu)
            return wavelength_peaks.to(u.micron), np.ones_like(wavelength_peaks.value) * photon_flux
        else:
            upper_part = np.arange(const.c / reference_wavelength.to(u.GHz), nu_max, free_spectral_range_nu)
            lower_part = np.arange(const.c / reference_wavelength.to(u.GHz), nu_min, free_spectral_range_nu)
            wavelength_peaks = const.c / np.concatenate((lower_part, upper_part))
            return wavelength_peaks.to(u.micron), np.ones_like(wavelength_peaks.value) * photon_flux
    else:
        free_spectral_range_nu = (const.c / 2 / n / d).to(u.GHz)
        if reference_wavelength != None:
            wavelength_peaks = const.c / np.arange(nu_min, nu_max, free_spectral_range_nu)
            return wavelength_peaks.to(u.micron), np.ones_like(wavelength_peaks.value) * photon_flux
        else:
            upper_part = np.arange(const.c / reference_wavelength.to(u.GHz), nu_max, free_spectral_range_nu)
            lower_part = np.arange(const.c / reference_wavelength.to(u.GHz), nu_min, free_spectral_range_nu)
            wavelength_peaks = const.c / np.concatenate((lower_part, upper_part))
            return wavelength_peaks.to(u.micron), np.ones_like(wavelength_peaks.value) * photon_flux

    return