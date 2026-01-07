from hispecdrp.utils import download_gdrive_file

try:
    from pyechelle.simulator import Simulator
    from pyechelle.telescope import Telescope
    from pyechelle.sources import CSVSource
    from pyechelle.spectrograph import ZEMAX
except ImportError:
    print("pyechelle package not found. pyechelle simulations will not run.")

import os
import h5py

import numpy as np
from numba import njit

from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity
from astropy.io import fits
import astropy.utils.data

__all__ = ['SimulateRateMap', 'read_pyechelle_hdf', 'download_zemax_files']


simulations_directory = os.path.dirname(os.path.abspath(__file__))
default_bspec_hdf = 'HISPEC_yJ_20231006A-1_all_traces.hdf'
default_rspec_hdf = 'HISPEC_HK_20231006B-1_all_traces.hdf'

class SimulateRateMap:
    """Class for simulating rate maps using pyechelle fiber spectra."""

    def __init__(self) -> None:
        self.hdf_file_directory: str = str(astropy.utils.data._get_download_cache_loc()) + '/'
        self.bspec_hdf_model: str = 'HISPEC_yJ_20231006A-1_all_traces.hdf'
        self.rspec_hdf_model: str = 'HISPEC_HK_20231006B-1_all_traces.hdf'
        self.output_directory: str = os.getcwd()
        self.pyechelle_outname: str = '_pyechelle_out.fits'
        self.rate_map_outname: str = 'rate_map_out.fits'

        if os.path.isdir(self.output_directory) is False:
            os.mkdir(self.output_directory)

        self.telescope_primary_diameter: Quantity = 9.96 * u.m
        self.quantum_efficiency: float = 0.9

        self.fibers: dict = {}

        self.bspec_orders: list[int] = list(range(110, 150))
        self.rspec_orders: list[int] = list(range(59, 99))

        self.rate_map: np.ndarray[np.float64] = None


    def set_fiber_spectrum(
        self,
        fiber_number: int = 1,
        spectrum_wavelengths: Quantity = np.linspace(0.9, 2.4, 100) * u.micron,
        spectrum_flux: Quantity = (np.ones(100) * 1e-12 * u.erg / u.s / u.cm**2 / u.micron),
        continuous: bool = True,
    ) -> None:
        """Assign spectral input to a fiber."""

        if fiber_number not in (1, 2, 3):
            raise ValueError('Fiber must be 1, 2, or 3!')

        if continuous and not _check_wavelength_sampling(
            spectrum_wavelengths, min_resolution=3e5
        ):
            print(
                "Input spectrum does not have sufficient sampling for HISPEC "
                "simulation.\nThe input spectrum will be interpolated to a higher "
                "sampling.\nIf the input is intended to be a list of unresolved lines, "
                " or you do not want to interpolate to higher sampling, set continuous=False."
            )
            spectrum_wavelengths, spectrum_flux = _upsample_spectrum(
                spectrum_wavelengths,
                spectrum_flux,
                desired_sampling_resolution=3e5,
            )

        self.fibers[fiber_number] = {
            "wavelength": spectrum_wavelengths,
            "flux": spectrum_flux,
            "continuous": continuous
        }


    def apply_throughput_total(
        self,
        fiber_number: int = 1,
        spectrograph: str = "BSPEC"
    ) -> tuple[Quantity, Quantity]:
        """Apply telescope + instrument throughput to fiber flux."""
        
        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")

        throughput_files_bspec: list[str] = [
            "feicom_throughput.csv",
            "ao_throughput.csv",
            "bspec_throughput.csv",
            "feiblue_throughput.csv",
            "fibblue_throughput.csv",
            "tel_throughput.csv",
        ]

        throughput_files_rspec: list[str] = [
            "feicom_throughput.csv",
            "ao_throughput.csv",
            "feired_throughput.csv",
            "fibred_throughput.csv",
            "rspec_throughput.csv",
            "tel_throughput.csv",
        ]

        throughput_files: list[str] = (
            throughput_files_rspec
            if spectrograph == "RSPEC"
            else throughput_files_bspec
        )

        wavelengths = self.fibers[fiber_number]["wavelength"]
        total_throughput = np.ones_like(wavelengths.value)

        for file in throughput_files:
            file_path = os.path.join(simulations_directory, "throughput", file)
            th_data = np.genfromtxt(file_path, delimiter=",", skip_header=1).T
            th_wv = th_data[0] * u.micron
            th_val = th_data[1]

            interp_func = interp1d(th_wv, th_val, fill_value=0.0,
                                    bounds_error=False)
            total_throughput *= interp_func(wavelengths)

        total_throughput *= self.quantum_efficiency
        self.fibers[fiber_number]["flux"] *= total_throughput

        return wavelengths, self.fibers[fiber_number]["flux"]


    def apply_tellurics(self, fiber_number: int = 1) -> None:
        """Placeholder for atmospheric absorption effects."""
        raise NotImplementedError("Telluric application not yet implemented.")


    def apply_radial_velocity(
        self,
        fiber_number: int = 1,
        rv: Quantity = 0 * u.km / u.s
    ) -> tuple[Quantity, Quantity]:
        """Apply Doppler shift to wavelengths."""

        factor = (1 + (rv / const.c).to(u.dimensionless_unscaled))
        self.fibers[fiber_number]["wavelength"] *= factor

        return (
            self.fibers[fiber_number]["wavelength"],
            self.fibers[fiber_number]["flux"],
        )


    def create_pyechelle_rate_map(
        self,
        spectrograph: str = "BSPEC",
        exposure_time: Quantity = 300 * u.s,
        save: bool = True,
    ) -> np.ndarray[np.float64]:
        """Run pyechelle simulator to generate pixel-space rate map."""

        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")

        model = self.bspec_hdf_model if spectrograph == "BSPEC" else self.rspec_hdf_model
        model_path = self.hdf_file_directory + model

        if os.path.isfile(model_path) is False and model in [default_bspec_hdf, default_rspec_hdf]:
            print('Model file not found locally. Downloading to ', model_path)
            download_zemax_files()

        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f"ZEMAX model file not found at {model_path}!")

        sim = Simulator(ZEMAX(path=model_path))
        sim.set_ccd(1)
        sim.set_bias(0)
        sim.set_read_noise(0)
        sim.set_telescope(
            Telescope(self.telescope_primary_diameter.to(u.m).value, 0)
        )
        sim.set_exposure_time(exposure_time.to(u.s).value)
        sim.set_fibers(list(self.fibers))
        sim.set_orders(
            self.bspec_orders if spectrograph == "BSPEC" else self.rspec_orders
        )
        
        fiber_sources = []
        for fiber in self.fibers:
            csv_path = self.write_fiber_spectrum_to_pyechelle_csv(fiber)
            fiber_sources.append(
                CSVSource(
                    file_path=csv_path,
                    list_like=not self.fibers[fiber]["continuous"],
                    wavelength_units="micron",
                    flux_units="ph/s/micron",
                )
            )

        sim.set_sources(fiber_sources)
        #sim.set_efficiency(ConstantEfficiency(1.0))

        output_path = os.path.join(
            self.output_directory, f"{spectrograph}{self.pyechelle_outname}"
        )
        sim.set_output(output_path, overwrite=True)
        sim.run()

        self.rate_map = (
            fits.getdata(output_path) / exposure_time.to(u.s).value
        )

        if save:
            save_path = os.path.join(
                self.output_directory,
                f"{spectrograph}_pyechelle_{self.rate_map_outname}",
            )
            print('Saving rate map to ', save_path)
            fits.writeto(save_path, data=self.rate_map, overwrite=True)
        else:
            os.remove(csv_path)
            os.remove(output_path)

        return self.rate_map
    

    def create_native_rate_map(
        self,
        spectrograph: str = "BSPEC",
        save: bool = True,
        method: str = "prf_convolution",
    ) -> np.ndarray[np.float64]:
        """Create rate map using native model interpolation/convolution.
          Faster than pyechelle for bright sources."""

        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")

        if method not in ("bilinear_weights", "prf_convolution"):
            raise ValueError(
                "Method must be bilinear_weights or "
                "prf_convolution (psf convolved with prf)!"
            )

        model = self.bspec_hdf_model if spectrograph == "BSPEC" else self.rspec_hdf_model
        model_path = self.hdf_file_directory + model

        if os.path.isfile(model_path) is False and model in [default_bspec_hdf, default_rspec_hdf]:
            print('Model file not found locally. Downloading to ', model_path)
            download_zemax_files()

        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f"ZEMAX model file not found at {model_path}!")

        model_data = read_pyechelle_hdf(model_path)

        order_list = (
            self.bspec_orders if spectrograph == "BSPEC" else self.rspec_orders
        )
        order_list_str = [str(o) for o in order_list]


        detector_shape = (model_data["nx"], model_data["ny"])
        pixelsize = model_data["pixelsize"]


        rate_map_array = np.empty((len(self.fibers), detector_shape[1], detector_shape[0]), dtype=np.float64)
        for i, fiber in enumerate(self.fibers):
            fiber_key = f"fiber_{fiber}"

            # remove orders in hdf model that will not be used
            for key in list(model_data[fiber_key]):
                if key not in order_list_str:
                    del model_data[fiber_key][key]

            wavelength = self.fibers[fiber]["wavelength"].to(u.micron)
            tmp_flux = (
                self.fibers[fiber]["flux"] *
                np.pi * (self.telescope_primary_diameter ** 2) / 4
            )
            flux = tmp_flux.to(
                u.ph / u.s / u.micron,
                equivalencies=u.spectral_density(wavelength),
            )



            if method == "bilinear_weights":
                self.rate_map = bilinear_weights_rate_map(
                    wavelength,
                    flux,
                    model_data[fiber_key],
                    detector_shape=detector_shape,
                    pixelsize=pixelsize,
                )
            else:
                self.rate_map = psf_prf_convolution_rate_map(
                    wavelength,
                    flux,
                    model_data[fiber_key],
                    detector_shape=detector_shape,
                    pixelsize=pixelsize,
                )

            rate_map_array[i, :, :] = self.rate_map

        self.rate_map = np.sum(rate_map_array, axis=0)

        if save and self.rate_map is not None:
            save_path = os.path.join(
                self.output_directory, f"{spectrograph}_{self.rate_map_outname}"
            )
            print('Saving rate map to ', save_path)
            fits.writeto(save_path, data=self.rate_map, overwrite=True)

        return self.rate_map


    def write_fiber_spectrum_to_pyechelle_csv(
        self,
        fiber: int
    ) -> str:
        """Write fiber's wavelength + flux into a CSV formatted for pyechelle."""

        wavelength = self.fibers[fiber]["wavelength"].to(u.micron)
        tmp_flux = (
            self.fibers[fiber]["flux"] *
            np.pi * (self.telescope_primary_diameter ** 2) / 4
        )
        flux = tmp_flux.to(
            u.ph / u.s / u.micron,
            equivalencies=u.spectral_density(wavelength),
        )

        arr = np.column_stack(
            [wavelength.value, flux.value]
        )
        fname = f"fiber_{fiber}.csv"
        out_path = os.path.join(self.output_directory, fname)

        np.savetxt(out_path, arr, delimiter=",")
        return os.path.abspath(out_path)
    

def download_zemax_files() -> None:
    """Download HISPEC ZEMAX model files from Google Drive."""

    zemax_files = [
        'HISPEC_HK_20231006B-1_all_traces.hdf',
        'HISPEC_yJ_20231006A-1_all_traces.hdf'
    ]

    for zfile in zemax_files:
        download_gdrive_file(zfile, use_cached=True)


def bilinear_weights_rate_map(
    wave: Quantity,
    flux: Quantity,
    orders: dict[str, dict],
    detector_shape: tuple[int, int] = (4096, 4096),
    pixelsize: float = 10.0
) -> np.ndarray[np.float64]:
    """
    Simulate detector rate map (photons/s/pixel) by depositing
    oversampled PSFs using bilinear interpolation.

    Parameters
    ----------
    wave : Quantity
        Input wavelengths corresponding to flux samples.
    flux : Quantity
        Flux array with photon rate density (photons/s/unit wavelength).
    orders : dict
        Nested structure containing wavelength trace and PSF stamps for orders. 
        Top level keys are order numbers (str).
    detector_shape : tuple of int, optional
        Detector (ny, nx) pixel size, default = (4096, 4096).
    pixelsize : float, optional
        Pixel size in microns (affects oversampling), default = 10.

    Returns
    -------
    det : ndarray[float64]
        Detector photon rate map (photons/s/pixel).
        Modified in-place by Numba-accelerated PSF deposit function.
    """
    det = np.zeros(detector_shape, dtype=np.float64)

    for od in orders:
        print('Simulating order', od)

        ow = orders[od]['xy_wavelengths']
        ox = orders[od]['x']
        oy = orders[od]['y']
        psfw = orders[od]['psf_wavelengths']
        psf_list = orders[od]['psf_stamps']
        oversample = pixelsize / orders[od]['psf_data_spacing']

        # Interpolate detector trace positions
        fx = PchipInterpolator(ow, ox, extrapolate=True)
        fy = PchipInterpolator(ow, oy, extrapolate=True)

        xc_arr = fx(wave)
        yc_arr = fy(wave)
        dws = np.gradient(wave)

        for i in range(len(wave) - 1):
            lam0 = wave[i]
            dw = dws[i]
            photons_per_s = flux[i] * dw

            if photons_per_s.value <= 0:
                continue

            if lam0.value < ow.min() or lam0.value > ow.max():
                continue

            xc = float(xc_arr[i])
            yc = float(yc_arr[i])

            # Nearest PSF by wavelength
            idx = int(np.argmin(np.abs(psfw - lam0.value)))

            # Normalize and scale PSF by total photons
            psf = psf_list[idx]
            psf = (psf / np.sum(psf)) * photons_per_s.value  

            _deposit_oversampled_psf_numba(det, psf, xc, yc, oversample)

    return det


@njit(fastmath=True)
def _deposit_oversampled_psf_numba(
    det: np.ndarray[np.float64],
    psf: np.ndarray[np.float64],
    xc: float,
    yc: float,
    oversample: float
) -> None:
    """
    Numba-accelerated deposition of an oversampled PSF into detector pixels
    using bilinear subpixel weighting. Flux is conserved.

    Parameters
    ----------
    det : ndarray[float64]
        The detector image array to update in-place (ny x nx).
    psf : ndarray[float64]
        The oversampled PSF stamp to deposit (H x W).
    xc, yc : float
        Center of PSF in detector pixel coordinates (x, y).
    oversample : float
        Oversampling factor (sampled PSF pixels per detector pixel).

    Returns
    -------
    None
        The detector array `det` is modified in place.
    """
    H, W = psf.shape
    ny, nx = det.shape

    # Precompute center offset 
    y0 = yc * oversample - (H * 0.5 + 1.0)
    x0 = xc * oversample - (W * 0.5 + 1.0)

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

            # Skip if off-chip
            if (i0 < -1 or i0 > ny or j0 < -1 or j0 > nx):
                continue

            # Contribute to 4 nearest neighbors
            if 0 <= i0 < ny and 0 <= j0 < nx:
                det[i0, j0] += val * (1.0 - dy) * (1.0 - dx)

            if 0 <= i0 < ny and 0 <= j0 + 1 < nx:
                det[i0, j0 + 1] += val * (1.0 - dy) * dx

            if 0 <= i0 + 1 < ny and 0 <= j0 < nx:
                det[i0 + 1, j0] += val * dy * (1.0 - dx)

            if 0 <= i0 + 1 < ny and 0 <= j0 + 1 < nx:
                det[i0 + 1, j0 + 1] += val * dy * dx


def psf_prf_convolution_rate_map(
    wave: Quantity,
    flux: Quantity,
    orders: dict[str, dict],
    detector_shape: tuple[int, int] = (4096, 4096),
    pixelsize: float = 10.0
) -> np.ndarray[np.float64]:
    """
    Generate a detector rate map by convolving oversampled PSFs with
    the pixel response function (PRF), and interpolating for many
    trace positions across each order.

    Parameters
    ----------
    wave : Quantity
        Input wavelength samples.
    flux : Quantity
        Flux samples expressed as photon-rate density.
    orders : dict
        Nested dictionary containing trace coordinates and PSF stamps.
    detector_shape : tuple of int, optional
        Detector pixels (ny, nx). Default = (4096, 4096).
    pixelsize : float, optional
        Pixel size in microns, used to determine oversampling.

    Returns
    -------
    det : ndarray[float64]
        Detector photon rate map in photons/s/pixel.
    """
    det = np.zeros(detector_shape, dtype=np.float64)

    for od in orders:
        print('Simulating order', od)

        ow = orders[od]['xy_wavelengths']
        ox = orders[od]["x"]
        oy = orders[od]["y"]
        psfw = orders[od]['psf_wavelengths']
        psf_list = orders[od]['psf_stamps']

        oversample = pixelsize / orders[od]['psf_data_spacing']
        if not float(oversample).is_integer():
            raise ValueError("Oversample factor must be an integer")
        oversample = int(oversample)

        # Interpolate detector trace positions
        fx = PchipInterpolator(ow, ox, extrapolate=True)
        fy = PchipInterpolator(ow, oy, extrapolate=True)

        # Create list of convolved effective PSFs across the order
        prf_kernel = np.ones((oversample, oversample))
        psf_eff_list = fftconvolve(
            psf_list,
            prf_kernel[None, ...],
            mode='same',
            axes=(1, 2)
        )

        order_wave_min = ow.min() * u.micron
        order_wave_max = ow.max() * u.micron
        order_wave = wave[(wave >= order_wave_min) & (wave <= order_wave_max)]   
        order_flux = flux[(wave >= order_wave_min) & (wave <= order_wave_max)] 

        xc_arr = fx(order_wave)
        yc_arr = fy(order_wave)
        xc_floor_arr = np.floor(xc_arr).astype(int)
        yc_floor_arr = np.floor(yc_arr).astype(int)
        subpixel_dx = xc_arr - xc_floor_arr
        subpixel_dy = yc_arr - yc_floor_arr

        dws = np.gradient(order_wave)
        photons_per_s = order_flux * dws

        psf_height, psf_width = psf_eff_list[0].shape
        x_psf_eff = np.arange(psf_width) - psf_width // 2
        y_psf_eff = np.arange(psf_height) - psf_height // 2

        max_pixels_x = psf_width // oversample
        max_pixels_y = psf_height // oversample

        # Create interpolation function for effective psf
        psf_eff_interp = RegularGridInterpolator((psfw, y_psf_eff, x_psf_eff), psf_eff_list, bounds_error=False, fill_value=None)

        pixel_xs = (np.arange(max_pixels_x) - (max_pixels_x) // 2) * oversample
        pixel_ys = (np.arange(max_pixels_y) - (max_pixels_y) // 2) * oversample


        for i in range(len(order_wave) - 1):
            # Get x,y grid to interpolate the effective psf on
            pixels_xs_shifted = pixel_xs - subpixel_dx[i] * oversample
            pixels_ys_shifted = pixel_ys - subpixel_dy[i] * oversample
            xx, yy = np.meshgrid(pixels_xs_shifted, pixels_ys_shifted)

            # Interpolate effective psf at this wavelength and normalize
            stamp = psf_eff_interp((order_wave[i], yy, xx))
            stamp_norm = stamp / np.sum(stamp)

            xlo_idx = xc_floor_arr[i] - max_pixels_x // 2
            xhi_idx = xc_floor_arr[i] + max_pixels_x // 2 + 1
            ylo_idx = yc_floor_arr[i] - max_pixels_y // 2
            yhi_idx = yc_floor_arr[i] + max_pixels_y // 2 + 1

            # Calculate the slice boundaries for the source (stamp)
            src_x_start = 0
            src_x_end = max_pixels_x
            src_y_start = 0
            src_y_end = max_pixels_y

            # Adjust x indices if they go out of bounds
            if xlo_idx < 0:
                src_x_start = -xlo_idx  # Skip the first part of the stamp
                xlo_idx = 0
            if xhi_idx > detector_shape[0]:
                src_x_end = max_pixels_x - (xhi_idx - detector_shape[0])
                xhi_idx = detector_shape[0]

            # Adjust y indices if they go out of bounds
            if ylo_idx < 0:
                src_y_start = -ylo_idx  # Skip the first part of the stamp
                ylo_idx = 0
            if yhi_idx > detector_shape[1]:
                src_y_end = max_pixels_y - (yhi_idx - detector_shape[1])
                yhi_idx = detector_shape[1]

            # Only add stamp if there's overlap
            if xlo_idx < xhi_idx and ylo_idx < yhi_idx and src_x_start < src_x_end and src_y_start < src_y_end:
                # Extract the valid portion of the stamp
                valid_stamp = stamp_norm[src_y_start:src_y_end, src_x_start:src_x_end]
                
                # Add to det array
                det[ylo_idx:yhi_idx, xlo_idx:xhi_idx] += valid_stamp * photons_per_s[i].value

    return det


def _check_wavelength_sampling(
    wavelengths: Quantity,
    min_resolution: float = 3e5
) -> bool:
    """Return True if all wavelength sampling meets resolution threshold."""

    wl = np.array(wavelengths, dtype=float)

    # Ensure strictly increasing wavelengths
    if not np.all(np.diff(wl) > 0):
        raise ValueError("Wavelengths must be strictly increasing.")

    delta_wl = np.diff(wl)
    resolution = wl[:-1] / delta_wl

    return bool(np.all(resolution >= min_resolution))


def _upsample_spectrum(
    wavelengths: Quantity,
    flux: Quantity,
    desired_sampling_resolution: float = 3e5
) -> tuple[Quantity, Quantity]:
    """
    Upsample spectrum in log-wavelength space to reach desired resolution.
    """

    dloglam = 1.0 / desired_sampling_resolution

    lmax = np.max(wavelengths)
    lmin = np.min(wavelengths)
    n_steps = int(np.floor(np.log(lmax / lmin) / dloglam)) + 1

    new_wavelengths = (
        lmin * np.exp(np.arange(n_steps) * dloglam)
    )

    interp_func = interp1d(wavelengths, flux, kind="linear", fill_value=0.)
    new_flux = interp_func(new_wavelengths) * flux.unit

    return new_wavelengths, new_flux


def read_pyechelle_hdf(hdf_model_path: str) -> dict:
    """
    Read pyechelle HDF model into nested dictionary structure.
    """

    with h5py.File(hdf_model_path, "r") as hdf_data:
        ccd1_data = hdf_data["CCD_1"]

        base_dict = {
            "nx": ccd1_data.attrs["Nx"],
            "ny": ccd1_data.attrs["Ny"],
            "pixelsize": ccd1_data.attrs["pixelsize"],
        }

        for fiber_key in ccd1_data:
            if "fiber" in fiber_key:
                base_dict[fiber_key] = _parse_fiber(ccd1_data[fiber_key])

        return base_dict


def _parse_fiber(fiber_group: h5py.Group) -> dict:
    """
    Parse one fiber group and return its orders & PSF data.
    """

    fiber_dict = {}

    # Create order placeholders (non-PSF groups)
    for key in fiber_group:
        if "psf" not in key:
            order = key.replace("order", "")
            fiber_dict[order] = {}

    # Fill in data for each entry
    for key in fiber_group:
        if "psf_order_" in key:
            _parse_psf_order(key, fiber_group[key], fiber_dict)
        else:
            _parse_order(key, fiber_group[key], fiber_dict)

    return fiber_dict


def _parse_psf_order(
    key: str,
    group: h5py.Group,
    fiber_dict: dict
) -> None:
    """
    Parse a PSF order group like 'psf_order_59' into arrays.
    """

    order = key.replace("psf_order_", "")

    wavelengths = np.array([
        float(k.replace("wavelength_", "")) for k in group
    ])

    stamps = np.array([group[k][()] for k in group])
    fiber_dict[order]["psf_wavelengths"] = wavelengths
    fiber_dict[order]["psf_stamps"] = stamps
    fiber_dict[order]["psf_data_spacing"] = [
        group[k].attrs["dataSpacing"] for k in group
    ][0]


def _parse_order(
    key: str,
    dataset: h5py.Dataset,
    fiber_dict: dict
) -> None:
    """
    Parse a regular trace dataset like 'order59' into X, Y and wavelength arrays.
    """

    order = key.replace("order", "")
    arr = np.vstack([list(k) for k in dataset])
    fiber_dict[order]["xy_wavelengths"] = arr[:, -1]
    fiber_dict[order]["x"] = arr[:, 4]
    fiber_dict[order]["y"] = arr[:, 5]