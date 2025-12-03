#import the necessary modules from pyechelle
from pyechelle.simulator import Simulator
from pyechelle.telescope import Telescope
from pyechelle.sources import ConstantPhotonFlux,CSVSource, IdealEtalon, Phoenix
from pyechelle.spectrograph import ZEMAX

import os
import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import fourier_shift
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
import h5py
from numba import njit
from scipy.signal import fftconvolve


class SimulateRateMap():
    def __init__(self):
        self.bspec_hdf_path = '/home/aidan/code/HISPEC_DRP/hispecdrp/simulations/fixed_20231006A-1_HISPEC_SPECTRO_YJ_pyechelle.hdf'
        self.rsepc_hdf_path = '/home/aidan/code/HISPEC_DRP/hispecdrp/simulations/fixed_20231006B-1_HISPEC_SPECTRO_HK_pyechelle.hdf'
        self.output_directory = ''
        self.pyechelle_outname = '_pyechelle_out.fits'
        self.rate_map_outname = '_rate_map_out.fits'
        self.telescope_primary_diameter = 9.96 * u.m
        self.quantum_efficiency = 0.9
        self.fibers = {}
        return

    def set_fiber_spectrum(self, fiber_number=1,
                           spectrum_wavelengths=np.linspace(0.9,2.4,100)*u.micron,
                           spectrum_flux=np.ones(100)*1e-12*u.erg/u.s/u.cm**2/u.micron,
                           continuous=True):

        if fiber_number not in [1,2,3]:
            print('Fiber must be 1,2 or 3! No spectrum set.')
            return

        if continuous and not _check_wavelength_sampling(spectrum_wavelengths, min_resolution=3e5):
            print('Input sectrum does not have sufficient sampling for HISPEC simulation.\n',
                  'The input spectrum will be interpolated to a higher sampling.\n',
                  'If the input is intended to be a list of unresolved lines, set continuous=False.')
            spectrum_wavelengths, spectrum_flux = _upsample_spectrum(spectrum_wavelengths,
                                                                     spectrum_flux,
                                                                     desired_sampling_resolution=3e5)

        self.fibers[fiber_number] = {
            'wavelength': spectrum_wavelengths,
            'flux': spectrum_flux
        }
        return

    def apply_throughput_total(self, fiber_number=1, spectrograph='BSPEC'):
        spectrograph = spectrograph.upper()
        if spectrograph not in ['BSPEC', 'RSPEC']:
            print('!Spectrograph must be BSPEC or RSPEC!')
            return None

        throughput_files_bspec = ['feicom_throughput.csv',
                                 'ao_throughput.csv',
                                 'bspec_throughput.csv',
                                 'feiblue_throughput.csv',
                                 'fibblue_throughput.csv',
                                 'tel_throughput.csv']

        throughput_files_rspec = ['feicom_throughput.csv',
                                  'ao_throughput.csv',
                                  'feired_throughput.csv',
                                  'fibred_throughput.csv',
                                  'rspec_throughput.csv',
                                  'tel_throughput.csv']

        throughput_files = throughput_files_rspec if spectrograph.upper() == 'RSPEC' else throughput_files_bspec

        total_throughput = np.ones_like(self.fibers[fiber_number]['wavelength'].value)
        for file in throughput_files:
            th_data = np.genfromtxt('throughput/'+file, delimiter=',',skip_header=1).T
            th_wv = th_data[0] * u.micron
            th = th_data[1]

            intp_func = interp1d(th_wv, th)
            total_throughput *= intp_func(self.fibers[fiber_number]['wavelength'])

        total_throughput *= self.quantum_efficiency
        self.fibers[fiber_number]['flux'] *= total_throughput

        return self.fibers[fiber_number]['wavelength'], self.fibers[fiber_number]['flux']

    def apply_tellurics(self, fiber_number=1):
        print('Applying tellurics currently does nothing.')
        return

    def apply_radial_velocity(self, fiber_number=1, rv=0*u.km/u.s):
        self.fibers[fiber_number]['wavelength'] *= (1 + (rv / const.c).to(u.dimensionless_unscaled))
        return self.fibers[fiber_number]['wavelength'], self.fibers[fiber_number]['flux']

    def create_pyechelle_rate_map(self, spectrograph='BSPEC', exposure_time=300*u.s, save=True):
        spectrograph = spectrograph.upper()
        if spectrograph not in ['BSPEC', 'RSPEC']:
            print('!Spectrograph must be BSPEC or RSPEC!')
            return None

        model_path = self.bspec_hdf_path if spectrograph == 'BSPEC' else self.rsepc_hdf_path

        sim = Simulator(ZEMAX(path=model_path))
        sim.set_ccd(1)  # always leave as 1
        sim.set_bias(0)
        sim.set_read_noise(0)
        sim.set_telescope(Telescope(self.telescope_primary_diameter.value, 0))  # Telescope(primary diameter, secondary diameter)
        sim.set_exposure_time(exposure_time.to(u.s).value)

        sim.set_fibers([f for f in self.fibers])
        fiber_sources = []
        for fiber in self.fibers:
            fiber_csv_path = self.write_fiber_spectrum_to_pyechelle_csv(fiber)
            source = CSVSource(file_path=fiber_csv_path,
                               list_like=False,
                               wavelength_units='micron',
                               flux_units='ph/s/micron')
            fiber_sources.append(source)

        sim.set_sources(fiber_sources)
        sim.set_output(self.output_directory+spectrograph+self.pyechelle_outname, overwrite=True)
        sim.run()

        self.rate_map = fits.getdata(self.output_directory+spectrograph+self.pyechelle_outname)  / exposure_time.to(u.s).value

        if save:
            fits.writeto(self.output_directory+spectrograph+'_pyechelle'+self.rate_map_outname, data=self.rate_map)

        return self.rate_map

    def create_native_rate_map(self, spectrograph='BSPEC',
                               save=True,
                               method='bilinear_weights'):
        spectrograph = spectrograph.upper()
        if spectrograph not in ['BSPEC', 'RSPEC']:
            print('!Spectrograph must be BSPEC or RSPEC!')
            return None

        if method not in ['bilinear_weights', 'prf_convolution']:
            print('!Method must be bilinear_weights (slower) or prf_convolution (psf convolved with prf, faster)!')

        model_path = self.bspec_hdf_path if spectrograph == 'BSPEC' else self.rsepc_hdf_path
        model_data_dict = read_pyechelle_hdf(model_path)

        for fiber in self.fibers:
            wavelength = self.fibers[fiber]['wavelength'].to(u.micron)
            tmp_flux = self.fibers[fiber]['flux'] * np.pi * (self.telescope_primary_diameter ** 2) / 4
            flux = tmp_flux.to(u.ph / u.s / u.micron, equivalencies=u.spectral_density(wavelength))

            if method=='bilinear_weights':
                self.rate_map = bilinear_weights_rate_map(wavelength.value,
                                                          flux.value,
                                                          model_data_dict['fiber_'+str(fiber)],
                                                          detector_shape=(model_data_dict['nx'], model_data_dict['ny']),
                                                          pixelsize=model_data_dict['pixelsize']
                                                          )
            elif method=='prf_convolution':
                self.rate_map = psf_prf_convolution_rate_map(wavelength.value,
                                                          flux.value,
                                                          model_data_dict['fiber_' + str(fiber)],
                                                          detector_shape=(model_data_dict['nx'], model_data_dict['ny']),
                                                          pixelsize=model_data_dict['pixelsize']
                                                          )


        if save:
            fits.writeto(self.output_directory+spectrograph+self.rate_map_outname, data=self.rate_map, overwrite=True)

        return self.rate_map

    def write_fiber_spectrum_to_pyechelle_csv(self, fiber):
        wavelength = self.fibers[fiber]['wavelength'].to(u.micron)
        tmp_flux = self.fibers[fiber]['flux'] * np.pi*(self.telescope_primary_diameter**2)/4
        flux = tmp_flux.to(u.ph/u.s/u.micron, equivalencies=u.spectral_density(wavelength))

        arr = np.array([wavelength.value, flux.value]).T
        np.savetxt(self.output_directory+'fiber_'+str(fiber)+'.csv', arr, delimiter=',')
        return os.path.abspath(self.output_directory+'fiber_'+str(fiber)+'.csv')


def bilinear_weights_rate_map(
    wave, flux,
    orders,
    detector_shape=(4096, 4096),
    pixelsize=10 # microns
):
    """
    Simulate detector rate map (photons/s/pixel)
    using nearest PSF, oversampled PSF stamps, and optimized Numba deposition.
    """
    det = np.zeros(detector_shape, dtype=np.float64)

    for od in orders:
        print('Simulating order ', od)
        ow = orders[od]['xy_wavelengths']
        ox = orders[od]["x"]
        oy = orders[od]["y"]
        psfw = orders[od]['psf_wavelengths']
        psf_list = orders[od]['psf_stamps']
        oversample = pixelsize/orders[od]['psf_data_spacing']

        # Interpolate trace centers
        fx = PchipInterpolator(ow, ox, extrapolate=True)
        fy = PchipInterpolator(ow, oy, extrapolate=True)
        #fx = interp1d(ow, ox, bounds_error=False, fill_value="extrapolate", kind='linear')
        #fy = interp1d(ow, oy, bounds_error=False, fill_value="extrapolate", kind='linear')

        xc_arr = fx(wave)
        yc_arr = fy(wave)
        dws = np.gradient(wave)

        for i in range(len(wave) - 1):
            lam0 = wave[i]
            dw = dws[i]
            photons_per_s = flux[i] * dw

            if photons_per_s <= 0:
                continue

            if lam0 < ow.min() or lam0 > ow.max():
                continue

            xc = xc_arr[i]
            yc = yc_arr[i]

            # nearest PSF
            idx = np.argmin(np.abs(psfw - lam0))
            psf = psf_list[idx] / np.sum(psf_list[idx]) * photons_per_s   # scale PSF

            _deposit_oversampled_psf_numba(det, psf, xc, yc, oversample)

    return det


def psf_prf_convolution_rate_map(wave,
                                 flux,
                                 orders,
                                 detector_shape=(4096, 4096),
                                 pixelsize=10 # microns
                                 ):

    det = np.zeros(detector_shape, dtype=np.float64)

    for od in orders:
        print('Simulating order ', od)
        ow = orders[od]['xy_wavelengths']
        ox = orders[od]["x"]
        oy = orders[od]["y"]
        psfw = orders[od]['psf_wavelengths']
        psf_list = orders[od]['psf_stamps']
        oversample = pixelsize / orders[od]['psf_data_spacing']
        oversample = int(oversample)

        # Interpolate trace centers
        fx = PchipInterpolator(ow, ox, extrapolate=True)
        fy = PchipInterpolator(ow, oy, extrapolate=True)
        # fx = interp1d(ow, ox, bounds_error=False, fill_value="extrapolate", kind='linear')
        # fy = interp1d(ow, oy, bounds_error=False, fill_value="extrapolate", kind='linear')

        prf_kernel = np.array([np.ones((oversample, oversample))]*len(psf_list))
        psf_eff_list = fftconvolve(psf_list, prf_kernel, mode='same', axes=[1,2])
        pad = (psf_eff_list[0].shape[0] // 2, psf_eff_list[1].shape[1] // 2)

        xc_arr = fx(wave)
        yc_arr = fy(wave)
        xc_floor_arr = np.floor(xc_arr)
        yc_floor_arr = np.floor(yc_arr)
        subpixel_shifts_x_arr = xc_arr - xc_floor_arr
        subpixel_shifts_y_arr = yc_arr - yc_floor_arr

        dws = np.gradient(wave)

        for i in range(len(wave) - 1):
            lam0 = wave[i]
            dw = dws[i]
            photons_per_s = flux[i] * dw

            if photons_per_s <= 0:
                continue

            if lam0 < ow.min() or lam0 > ow.max():
                continue

            xc_floor = int(xc_floor_arr[i])
            dx = subpixel_shifts_x_arr[i]
            yc_floor = int(yc_floor_arr[i])
            dy = subpixel_shifts_y_arr[i]

            # pad, shift, then crop
            # nearest PSF
            idx = np.argmin(np.abs(psfw - lam0))
            psf_eff = psf_eff_list[idx]
            psf_eff_pad = np.pad(psf_eff, ((pad[0],pad[0]), (pad[1],pad[1])),
                                 mode='constant', constant_values=0.0)
            shifted_psf_eff_pad = np.fft.ifftn(fourier_shift(np.fft.fftn(psf_eff_pad),
                                                         shift=[dy, dx])).real
            cy, cx = shifted_psf_eff_pad.shape[0] // 2, shifted_psf_eff_pad.shape[1] // 2
            halfy, halfx = psf_eff.shape[0] // 2, psf_eff.shape[1] // 2
            shifted_psf_eff = shifted_psf_eff_pad[cy - halfy:cy + halfy + 1, cx - halfx:cx + halfx + 1]

            # 4) downsample by summing non-overlapping s x s blocks to get detector pixels
            ny_f, nx_f = shifted_psf_eff.shape
            ny = ny_f // oversample
            nx = nx_f // oversample
            shifted_psf_eff = shifted_psf_eff[:ny * oversample, :nx * oversample]  # trim to exact multiple
            # reshape and sum blocks
            down = shifted_psf_eff.reshape(ny, oversample, nx, oversample).sum(axis=(1, 3))

            down /= np.sum(down)
            flux_to_add = down * photons_per_s

            #print(flux_to_add.shape)
            if yc_floor > 100 and xc_floor > 100 and yc_floor<4000 and xc_floor<4000:
                det[yc_floor-ny_f//2:yc_floor+ny_f//2+1, xc_floor-nx_f//2:xc_floor+nx_f//2+1] += flux_to_add

    return det






@njit(fastmath=True)
def _deposit_oversampled_psf_numba(det, psf, xc, yc, oversample):
    """
    Numba-accelerated deposition of an oversampled PSF into detector pixels
    using bilinear subpixel weighting. Flux is conserved.
    """
    H, W = psf.shape
    ny, nx = det.shape

    # Precompute center offset once
    y0 = yc * oversample - (H * 0.5 + 1)
    x0 = xc * oversample - (W * 0.5 + 1)

    for j in range(H):
        # detector-space y position of this oversampled pixel
        y_s = (j + y0) / oversample
        i0 = int(np.floor(y_s))
        dy = y_s - i0

        for i in range(W):
            x_s = (i + x0) / oversample
            j0 = int(np.floor(x_s))
            dx = x_s - j0

            val = psf[j, i]

            # Skip if entire stencil is off-chip
            if (i0 < -1 or i0 > ny or j0 < -1 or j0 > nx):
                continue

            # Contribute to 4 nearest neighbors
            # Pixel (i0, j0)
            if 0 <= i0 < ny and 0 <= j0 < nx:
                det[i0, j0] += val * (1 - dy) * (1 - dx)

            # Pixel (i0, j0+1)
            if 0 <= i0 < ny and 0 <= j0+1 < nx:
                det[i0, j0+1] += val * (1 - dy) * dx

            # Pixel (i0+1, j0)
            if 0 <= i0+1 < ny and 0 <= j0 < nx:
                det[i0+1, j0] += val * dy * (1 - dx)

            # Pixel (i0+1, j0+1)
            if 0 <= i0+1 < ny and 0 <= j0+1 < nx:
                det[i0+1, j0+1] += val * dy * dx


def _check_wavelength_sampling(wavelengths, min_resolution=1e6):
    wl = np.array(wavelengths, dtype=float)

    # Ensure sorted input
    if not np.all(np.diff(wl) > 0):
        raise ValueError("Wavelengths must be strictly increasing.")

    delta_wl = np.diff(wl)
    resolution = wl[:-1] / delta_wl

    # Returns false if resolution below threshold anywhere
    return np.all(resolution >= min_resolution)


def _upsample_spectrum(wavelengths, flux, desired_sampling_resolution=3e5):
    # Step size in ln(lambda)
    dloglam = 1.0 / desired_sampling_resolution

    # Number of steps required
    lmax = np.max(wavelengths)
    lmin = np.min(wavelengths)
    N = int(np.floor(np.log(lmax / lmin) / dloglam)) + 1

    # Logarithmically spaced wavelength array
    new_wavelengths = lmin * np.exp(np.arange(N) * dloglam)
    interp_func = interp1d(wavelengths, flux, kind='linear', fill_value=0.)
    new_flux = interp_func(new_wavelengths) * flux.unit

    return new_wavelengths, new_flux


def read_pyechelle_hdf(hdf_model_path) -> dict:
    """Read key data from a pyechelle HDF model file into a nested dictionary."""
    with h5py.File(hdf_model_path, "r") as hdf_data:
        ccd1_data = hdf_data["CCD_1"]

        base_dict = {
            'nx': ccd1_data.attrs['Nx'],
            'ny': ccd1_data.attrs['Ny'],
            'pixelsize': ccd1_data.attrs['pixelsize']
        }

        for fiber in ccd1_data:
            if 'fiber' in fiber:
                base_dict[fiber] = _parse_fiber(ccd1_data[fiber])

        return base_dict


def _parse_fiber(fiber_group) -> dict:
    """Parse one fiber (fiber_x) group and return its orders/PSF data."""
    fiber_dict = {}

    # First create order entries (non-PSF)
    for key in fiber_group:
        if "psf" not in key:
            order = key.replace("order", "")
            fiber_dict[order] = {}

    # Fill in data
    for key in fiber_group:
        if "psf" in key:
            _parse_psf_order(key, fiber_group[key], fiber_dict)
        else:
            _parse_order(key, fiber_group[key], fiber_dict)

    return fiber_dict


def _parse_psf_order(key: str, group, fiber_dict: dict) -> None:
    """Parse a PSF order group like 'psf_order_12'."""
    order = key.replace("psf_order_", "")

    wavelengths = np.array([
        float(k.replace("wavelength_", ""))
        for k in group
    ])
    stamps = np.array([group[k][()] for k in group])

    fiber_dict[order]["psf_wavelengths"] = wavelengths
    fiber_dict[order]["psf_stamps"] = stamps
    fiber_dict[order]['psf_data_spacing'] = [group[k].attrs['dataSpacing'] for k in group][0]


def _parse_order(key: str, dataset, fiber_dict: dict) -> None:
    """Parse a regular order dataset like 'order12'."""
    order = key.replace("order", "")
    arr = np.vstack([list(k) for k in dataset])

    fiber_dict[order]["xy_wavelengths"] = arr[:, -1]
    fiber_dict[order]["x"] = arr[:, 4]
    fiber_dict[order]["y"] = arr[:, 5]
