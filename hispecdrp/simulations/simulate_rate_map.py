from hispecdrp.utils import download_gdrive_file
from hispecdrp.datamodels import hispec_config

try:
    from pyechelle.simulator import Simulator
    from pyechelle.telescope import Telescope
    from pyechelle.sources import CSVSource
    from pyechelle.spectrograph import ZEMAX
except ImportError:
    Simulator = None  # type: ignore[misc, assignment]
    Telescope = None
    CSVSource = None
    ZEMAX = None

import logging
import os
import importlib.resources
import h5py

import numpy as np

logger = logging.getLogger(__name__)

# Default output filenames for saved products
DEFAULT_RATE_MAP_FILENAME = "rate_map.fits"
DEFAULT_PYECHELLE_FILENAME = "pyechelle_out.fits"

logger = logging.getLogger(__name__)

# Default output filenames for saved products
DEFAULT_RATE_MAP_FILENAME = "rate_map.fits"
DEFAULT_PYECHELLE_FILENAME = "pyechelle_out.fits"
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

__all__ = [
    "DEFAULT_RATE_MAP_FILENAME",
    "DEFAULT_PYECHELLE_FILENAME",
    "SimulateRateMap",
    "read_pyechelle_hdf",
]
__all__ = [
    "DEFAULT_RATE_MAP_FILENAME",
    "DEFAULT_PYECHELLE_FILENAME",
    "SimulateRateMap",
    "read_pyechelle_hdf",
]


default_bspec_hdf = 'HISPEC_yJ_20231006A-1_all_traces.hdf'
default_rspec_hdf = 'HISPEC_HK_20231006B-1_all_traces.hdf'

class SimulateRateMap:
    """Class for simulating rate maps using pyechelle fiber spectra.

    Spectra are set onto fibers with `set_fiber_spectrum`, and the rate map
    can then be simulated either with pyechelle or with a native interpolation/convolution function.
    All HDF model files are in a format usable by pyechelle. pyechelle contains photon noise, 
    and takes increased time to run for bright sources. The runtime for the native method is only dependent on the
    number of wavelength bins simulated. Note that the native method does not include photon noise.
    Both pyechelle and the bilinear_weights native method do not interpolate the psfs, which can result in abrupt jumps
    in the resulting extracted spectra which may or may not be acceptable depending on the use case.
    The prf_convolution native method interpolates the psf and should be viewed as the default method.


    Attributes
    ----------
    hdf_file_directory : str
        Directory where HDF files that contain trace and psf data are stored. Defaults to astropy cache.
    bspec_hdf_model : str
        Name of the Blue (yJ) spectrograph HDF model file.
    rspec_hdf_model : str
        Name of the Red (HK) spectrograph HDF model file.
    output_directory : str
        Directory where output files are saved. Defaults to current working directory.
    pyechelle_outname : str
        Name of the pyechelle output file if saved. This is used to generate the rate map, but is not the rate map.
        The spectrograph name is prepended when saving.
    rate_map_outname : str
        Name of the final rate map output file if saved. The spectrograph name is prepended when saving.
    telescope_primary_diameter : Quantity
        Effective diameter of the telescope primary mirror. Default is 9.96 m (Keck).
    fibers : dict
        Dictionary storing fiber spectra information set by `set_fiber_spectrum`.
    bspec_orders : list of int
        List of echelle orders to simulate for the Blue (yJ) spectrograph.
    rspec_orders : list of int
        List of echelle orders to simulate for the Red (HK) spectrograph.
    rate_map : ndarray[float64]
        The resulting rate map (photons/s/pixel) after simulation.

    """

    def __init__(self) -> None:
        self.hdf_file_directory = os.getenv('HISPECDRP_TEST_DATA_DIR')
        if self.hdf_file_directory is None:
            self.hdf_file_directory = os.path.join(
                str(astropy.utils.data._get_download_cache_loc()),
                'HISPECDRP_TEST_DATA'
            )
        self.hdf_file_directory += os.sep

        self.bspec_hdf_model: str = 'HISPEC_yJ_20231006A-1_all_traces.hdf'
        self.rspec_hdf_model: str = 'HISPEC_HK_20231006B-1_all_traces.hdf'
        self.output_directory: str = os.getcwd()
        self.pyechelle_outname: str = DEFAULT_PYECHELLE_FILENAME
        self.pyechelle_outname: str = DEFAULT_PYECHELLE_FILENAME
        self.rate_map_outname: str = DEFAULT_RATE_MAP_FILENAME

        self.telescope_primary_diameter: Quantity = 9.96 * u.m
        self.fibers: dict = {}

        self.bspec_orders: list[int] = hispec_config.HISPECConfig.get_order_list("BSPEC")
        self.rspec_orders: list[int] = hispec_config.HISPECConfig.get_order_list("RSPEC")

        self.rate_map: np.ndarray[np.float64] = None


    def set_fiber_spectrum(
        self,
        fiber_number: int = 1,
        spectrum_wavelengths: Quantity = np.linspace(0.9, 2.4, 100) * u.micron,
        spectrum_flux: Quantity = (np.ones(100) * 1e-12 * u.erg / u.s / u.cm**2 / u.micron),
        continuous: bool = True,
    ) -> None:
        """Assign spectral input to a fiber.

        spectrum_flux is assumed to a flux density per unit of wavelength"""

        if fiber_number not in (1, 2, 3):
            raise ValueError('Fiber must be 1, 2, or 3!')

        if continuous and not _check_wavelength_sampling(
            spectrum_wavelengths, min_resolution=3e5
        ):
            logger.warning(
                "Input spectrum does not have sufficient sampling for HISPEC simulation. "
                "The input spectrum will be interpolated to a higher sampling. "
                "If the input is intended to be a list of unresolved lines, or you do not want "
                "to interpolate to higher sampling, set continuous=False."
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
        mode: str = "NGS",
        quantum_efficiency: float = 0.9,
        peak_blaze_efficiency: float = 1.0
    ) -> tuple[Quantity, float]:
        """Apply telescope + instrument throughput to fiber flux. This does NOT include the blaze efficiency which
        is applied separately by pyechelle/native simulator.
        
        Parameters 
        ----------
        fiber_number: int
            should be either 1, 2, or 3. Must have been set with `set_fiber_spectrum` first.
        mode: str
            should be either "NGS" (Natural Guide Star) or "LGS" (Laser Guide Star).
        quantum_efficiency: float
            simple scalar multiplier applied to the total throughput.
        peak_blaze_efficiency: float
            peak blaze efficiency to be used in the blaze calculation.

        Throughput should be applied AFTER any radial velocity shifts"""

        warnings.warn("Throughput application is in development and may not reflect true HISPEC throughput.", UserWarning)

        mode = mode.upper()
        if mode not in ("NGS", "LGS"):
            raise ValueError("Mode must be NGS or LGS!")

        throughput_directory = importlib.resources.files('hispecdrp.data.throughput')
        if mode == "NGS":
            bspec_throughput_file = throughput_directory / 'transmission_total_BSPEC_endTOend_NGS_woBlaze.txt'
            rspec_throughput_file = throughput_directory / 'transmission_total_RSPEC_endTOend_NGS_woBlaze.txt'
        elif mode == "LGS":
            bspec_throughput_file = throughput_directory / 'transmission_total_BSPEC_endTOend_LGS_woBlaze.txt'
            rspec_throughput_file = throughput_directory / 'transmission_total_RSPEC_endTOend_LGS_woBlaze.txt'

        wavelengths = self.fibers[fiber_number]["wavelength"]
        total_throughput_wo_blaze = np.ones_like(wavelengths.value)

        # expecting csv files with two columns of wavelength (nm) and throughput (0-1)
        bspec_th_data = np.genfromtxt(bspec_throughput_file, delimiter=",", skip_header=1).T
        bspec_th_wv = (bspec_th_data[0] * u.nm).to(u.micron)
        bspec_th_val = bspec_th_data[1]
        rspec_th_data = np.genfromtxt(rspec_throughput_file, delimiter=",", skip_header=1).T
        rspec_th_wv = (rspec_th_data[0] * u.nm).to(u.micron)
        rspec_th_val = rspec_th_data[1]

        bspec_interp_func = interp1d(bspec_th_wv, bspec_th_val, fill_value=0.0,
                                bounds_error=False)
        rspec_interp_func = interp1d(rspec_th_wv, rspec_th_val, fill_value=0.0,
                                bounds_error=False)
        
        total_throughput_wo_blaze[wavelengths < 1.4*u.micron] *= bspec_interp_func(wavelengths[wavelengths < 1.4*u.micron])
        total_throughput_wo_blaze[wavelengths >= 1.4*u.micron] *= rspec_interp_func(wavelengths[wavelengths >= 1.4*u.micron])
        total_throughput_wo_blaze *= quantum_efficiency

        self.fibers[fiber_number]["flux"] *= total_throughput_wo_blaze

        return wavelengths, self.fibers[fiber_number]["flux"]


    def apply_tellurics(self, fiber_number: int = 1) -> None:
        """Placeholder for atmospheric absorption effects."""
        raise NotImplementedError("Telluric application not yet implemented.")


    def apply_radial_velocity(
        self,
        fiber_number: int = 1,
        rv: Quantity = 0 * u.km / u.s
    ) -> tuple[Quantity, Quantity]:
        """Apply Doppler shift to wavelengths.
        
        Must be applied BEFORE throughput/telluric application."""

        factor = (1 + (rv / const.c).to(u.dimensionless_unscaled))
        self.fibers[fiber_number]["wavelength"] *= factor

        return (
            self.fibers[fiber_number]["wavelength"],
            self.fibers[fiber_number]["flux"],
        )
    

    def get_wavelength_solution(self,
                                spectrograph: str = "BSPEC",
                                ) -> dict[int, dict[str, np.ndarray]]:
        """Get wavelength solution from HDF model file as a nested dictionary of the form:
        {order_number: {'wavelengths': np.ndarray, 'x': np.ndarray, 'y': np.ndarray}}
        where the wavelength, x, and y arrays are all 1D and correspond to each other.
        """
        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")
        
        model = self.bspec_hdf_model if spectrograph == "BSPEC" else self.rspec_hdf_model
        model_path = self.hdf_file_directory + model

        if os.path.isfile(model_path) is False and model in [default_bspec_hdf, default_rspec_hdf]:
            logger.info("Model file not found locally. Attempting to download to %s", model_path)
            download_gdrive_file(model, use_cached=True)

        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f".hdf model file not found at {model_path}!")

        model_data = read_pyechelle_hdf(model_path)

        order_list = (
            self.bspec_orders if spectrograph == "BSPEC" else self.rspec_orders
        )
        order_list_str = [str(o) for o in order_list]

        self.wavelength_solution = {}
        for fiber in [*range(1,4)]:
            self.wavelength_solution[fiber] = {}
            fiber_key = f"fiber_{fiber}"
            for od in order_list_str:
                self.wavelength_solution[fiber][int(od)] = {}
                ow = model_data[fiber_key][od]['xy_wavelengths']
                ox = model_data[fiber_key][od]["x"]

                # Interpolate wavelength as function of detector x pixel.
                # HDF traces may not be strictly sorted in x and may contain duplicate x values,
                # which breaks PchipInterpolator's requirement that x be strictly increasing.
                ox = np.asarray(ox, dtype=np.float64)
                ow = np.asarray(ow, dtype=np.float64)
                good = np.isfinite(ox) & np.isfinite(ow)
                ox = ox[good]
                ow = ow[good]
                if ox.size < 2:
                    raise ValueError(f"Not enough valid samples for fiber {fiber} order {od} wavelength solution.")

                # Sort by x
                sidx = np.argsort(ox)
                oxs = ox[sidx]
                ows = ow[sidx]

                # Merge duplicates in x by averaging wavelength at the same pixel
                ux, inv = np.unique(oxs, return_inverse=True)
                if ux.size < 2:
                    raise ValueError(f"Not enough unique x samples for fiber {fiber} order {od} wavelength solution.")
                wsum = np.bincount(inv, weights=ows)
                wcnt = np.bincount(inv)
                uow = wsum / np.maximum(wcnt, 1)

                fx = PchipInterpolator(ux, uow, extrapolate=True)

                self.wavelength_solution[fiber][int(od)] = {
                    "pixel": np.arange(4096),
                    "wavelength": fx(np.arange(4096))[::-1] # reversed to account for mirrored hdf file
                }

        return self.wavelength_solution


    def create_pyechelle_rate_map(
        self,
        spectrograph: str = "BSPEC",
        exposure_time: Quantity = 300 * u.s,
        save: bool = True,
    ) -> np.ndarray[np.float64]:
        """Run pyechelle simulator to generate pixel-space rate map.

        Parameters
        ----------
        spectrograph : str
            'BSPEC' for Blue (yJ) spectrograph, 'RSPEC' for Red (HK) spectrograph.
        exposure_time : Quantity
            Exposure time of the observation. Default is 300 seconds. Necessary for pyechelle simulation. Increase to improve SNR.
        save : bool
            If True, saves the resulting rate map to output_directory. Default is True.
        Returns
        -------
        rate_map : ndarray[float64]
            Simulated rate map in units of photons/s/pixel.

        Temporary CSV files are created during this process to feed pyechelle custom spectra.
        A sinc^2 blaze efficiency is applied internally by pyechelle, it cannot be turned off."""

        if Simulator is None:
            raise ImportError(
                "pyechelle is not installed. Install the HISPEC fork "
                "(see docs/source/installation.rst) or use "
                "SimulateRateMap.create_native_rate_map instead."
            )

        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")

        model = self.bspec_hdf_model if spectrograph == "BSPEC" else self.rspec_hdf_model
        model_path = self.hdf_file_directory + model

        if os.path.isfile(model_path) is False and model in [default_bspec_hdf, default_rspec_hdf]:
            logger.info("Model file not found locally. Attempting to download to %s", model_path)
            download_gdrive_file(model, use_cached=True)

        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f".hdf model file not found at {model_path}!")
        
        if os.path.isdir(self.output_directory) is False:
            os.mkdir(self.output_directory)

        logger.info("Running pyechelle simulation (spectrograph=%s, exposure_time=%s).", spectrograph, exposure_time)
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

        output_path = os.path.join(
            self.output_directory, self.pyechelle_outname
        )

        sim.set_output(output_path, overwrite=True)
        sim.run()

        self.rate_map = (
            fits.getdata(output_path) / exposure_time.to(u.s).value
        )

        # correct orientation from hdf file
        self.flip_rate_map_horizontal()
        self.rotate_rate_map_vertical()
        self.add_channel_rate_map(where='right')

        if save:
            save_path = os.path.join(
                self.output_directory,
                self.rate_map_outname,
            )
            logger.info("Saving pyechelle rate map to %s", save_path)
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
        apply_blaze: bool = True
    ) -> np.ndarray[np.float64]:
        """Create rate map using native model interpolation/convolution.

        Parameters
        ----------
        spectrograph : str
            'BSPEC' for Blue (yJ) spectrograph, 'RSPEC' for Red (HK) spectrograph.
        save : bool
            If True, saves the resulting rate map to output_directory. Default is True.
        method : str
            Method to use for rate map generation. Options are 'bilinear_weights' or 'prf_convolution'. 'prf_convolution' is recommended.
        apply_blaze : bool
            If True, applies a sinc^2 blaze efficiency to the fiber flux to each order before simulation. Default is True.
        Returns
        -------
        rate_map : ndarray[float64]
            Simulated rate map in units of photons/s/pixel.

        The 'prf_convolution' method interpolates the PSFs, is faster, and is recommended as the default."""

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
            logger.info("Model file not found locally. Attempting to download to %s", model_path)
            download_gdrive_file(model, use_cached=True)

        if os.path.isfile(model_path) is False:
            raise FileNotFoundError(f".hdf model file not found at {model_path}!")

        model_data = read_pyechelle_hdf(model_path)

        order_list = (
            self.bspec_orders if spectrograph == "BSPEC" else self.rspec_orders
        )
        order_list_str = [str(o) for o in order_list]

        logger.info("Creating native rate map (spectrograph=%s, method=%s).", spectrograph, method)
        detector_shape = (model_data["nx"], model_data["ny"])
        pixelsize = model_data["pixelsize"]
        blaze_angle = model_data["blaze_angle"] * u.deg
        groove_spacing = model_data["groove_spacing"] / u.mm

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
                    apply_blaze=apply_blaze,
                    blaze_angle=blaze_angle,
                    groove_spacing=groove_spacing
                )
            else:
                self.rate_map = psf_prf_convolution_rate_map(
                    wavelength,
                    flux,
                    model_data[fiber_key],
                    detector_shape=detector_shape,
                    pixelsize=pixelsize,
                    apply_blaze=apply_blaze,
                    blaze_angle=blaze_angle,
                    groove_spacing=groove_spacing
                )

            rate_map_array[i, :, :] = self.rate_map

        self.rate_map = np.sum(rate_map_array, axis=0)

        # correct orientation from hdf file
        self.flip_rate_map_horizontal()
        self.rotate_rate_map_vertical()
        self.add_channel_rate_map(where='right')

        if save and self.rate_map is not None:
            if os.path.isdir(self.output_directory) is False:
                os.mkdir(self.output_directory)

            save_path = os.path.join(
                self.output_directory, self.rate_map_outname
            )
            logger.info("Saving rate map to %s", save_path)
            fits.writeto(save_path, data=self.rate_map, overwrite=True)

        return self.rate_map


    def write_fiber_spectrum_to_pyechelle_csv(
        self,
        fiber: int
    ) -> str:
        """Write fiber's wavelength + flux into a CSV formatted for pyechelle.
        
        CSV spectra are in units of photons/s/micron."""

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

    
    def flip_rate_map_horizontal(self):
        """Flip the rate map horizontally to correct for backwards hdf file"""
        if self.rate_map is not None:
            self.rate_map = np.flip(self.rate_map, axis=1)
        return self.rate_map

    def rotate_rate_map_vertical(self):
        """Rotate the rate map so orders are vertical"""
        if self.rate_map is not None:
            self.rate_map = np.rot90(self.rate_map, 3)
        return self.rate_map
    
    def add_channel_rate_map(self, where='right'):
        """Add extra channel to rate map at left or right."""
        if self.rate_map is None:
            raise ValueError("Rate map must be generated before adding channel!")
        
        new_channel = np.zeros((4096, 64))

        if where == 'left':
            self.rate_map = np.hstack((new_channel, self.rate_map))
        elif where == 'right':
            self.rate_map = np.hstack((self.rate_map, new_channel))
        else:
            raise ValueError("where must be 'left' or 'right'!")
        
        return self.rate_map


def bilinear_weights_rate_map(
    wave: Quantity,
    flux: Quantity,
    orders: dict[str, dict],
    detector_shape: tuple[int, int] = (4096, 4096),
    pixelsize: float = 10.0,
    apply_blaze: bool = True,
    blaze_angle: Quantity = None,
    groove_spacing: Quantity = None
) -> np.ndarray[np.float64]:
    """
    Simulate detector rate map (photons/s/pixel) by depositing
    oversampled PSFs using bilinear weighting.

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
        logger.debug("Simulating order %s", od)

        ow = orders[od]['xy_wavelengths']
        ox = orders[od]['x']
        oy = orders[od]['y']
        psfw = orders[od]['psf_wavelengths']
        psf_list = orders[od]['psf_stamps']
        oversample = pixelsize / orders[od]['psf_data_spacing']

        # Interpolate detector trace positions
        fx = PchipInterpolator(ow, ox, extrapolate=True)
        fy = PchipInterpolator(ow, oy, extrapolate=True)

        order_wave_min = ow.min() * u.micron
        order_wave_max = ow.max() * u.micron
        order_wave = wave[(wave >= order_wave_min) & (wave <= order_wave_max)]   
        order_flux = flux[(wave >= order_wave_min) & (wave <= order_wave_max)] 

        xc_arr = fx(order_wave)
        yc_arr = fy(order_wave)
        dws = np.gradient(order_wave)

        # apply blaze efficiency if requested
        if apply_blaze and blaze_angle is not None and groove_spacing is not None:
            if blaze_angle.value < 0:
                blaze_angle *= -1
            blaze_eff = _blaze_efficiency_sinc(
                blaze_angle,
                int(od),
                1.0 / groove_spacing,
                order_wave
            )
            order_flux *= blaze_eff

        for i in range(len(order_wave) - 1):
            lam0 = order_wave[i]
            dw = dws[i]
            photons_per_s = order_flux[i] * dw

            if photons_per_s.value <= 0:
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
    pixelsize: float = 10.0,
    apply_blaze: bool = True,
    blaze_angle: Quantity = None,
    groove_spacing: Quantity = None
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
        Top level keys are order numbers (str).
    detector_shape : tuple of int, optional
        Detector pixels (ny, nx). Default = (4096, 4096).
    pixelsize : float, optional
        Pixel size in microns, used to determine oversampling.
    apply_blaze : bool, optional
        If True, applies a sinc^2 blaze efficiency to the flux for each order.
    blaze_angle : Quantity, optional
        Blaze angle of the echelle grating. Required if apply_blaze is True.
    groove_spacing : Quantity, optional
        Groove spacing of the echelle grating. Required if apply_blaze is True.

    Returns
    -------
    det : ndarray[float64]
        Detector photon rate map in photons/s/pixel.
    """
    det = np.zeros(detector_shape, dtype=np.float64)

    for od in orders:
        logger.debug("Simulating order %s", od)

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

        # apply blaze efficiency if requested
        if apply_blaze and blaze_angle is not None and groove_spacing is not None:
            if blaze_angle.value < 0:
                blaze_angle *= -1
            blaze_eff = _blaze_efficiency_sinc(
                blaze_angle,
                int(od),
                1.0 / groove_spacing,
                order_wave
            )
            order_flux *= blaze_eff

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
    """Check if spectrum is sampled with enough resolution to avoid aliasing for HISPEC.
      Return True if all sampled wavelength spacings meets resolution threshold."""

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
    Read pyechelle HDF model into nested dictionary structure for native simulation methods.

    Expected input HDF structure: 
        CCD_1/Spectrograph with attributes: blaze, gpmm 
        CCD_1/fiber_#/order# and psf_order_# groups
        CCD_1 with attributes: Nx, Ny, pixelsize (microns)
        order# groups: tuples with columns including x, y, wavelength of trace
        psf_order_# groups: datasets named wavelength_# with 2D PSF stamps with dataSpacing attribute for sampling in microns

    Output dictionary structure:
    {
        "nx": int,
        "ny": int,
        "pixelsize": float,
        "blaze_angle": float,
        "groove_spacing": float,
        "fiber_1": {
            "59": {
                "x": ndarray,
                "y": ndarray,
                "xy_wavelengths": ndarray,
                "psf_wavelengths": ndarray,
                "psf_stamps": ndarray,
                "psf_data_spacing": float
            },
            ...
        },
        "fiber_2": { ... },
        "fiber_3": { ... }
    }
    """

    with h5py.File(hdf_model_path, "r") as hdf_data:
        ccd1_data = hdf_data["CCD_1"]

        base_dict = {
            "nx": ccd1_data.attrs["Nx"],
            "ny": ccd1_data.attrs["Ny"],
            "pixelsize": ccd1_data.attrs["pixelsize"],
            "blaze_angle": ccd1_data['Spectrograph'].attrs["blaze"],
            'groove_spacing': ccd1_data['Spectrograph'].attrs["gpmm"]
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


def _diffracted_angle(blaze_angle, order, groove_spacing, wavelength):
    """Calculate diffracted angle using grating equation. Used for blaze efficiency calculation. Assume incidence angle = blaze angle."""
    return np.arcsin(-np.sin(blaze_angle) + order * wavelength / groove_spacing)


def _blaze_phase(blaze_angle, order, groove_spacing, wavelength):
    """Calculate blaze phase for blaze efficiency calculation. Assume incidence angle = blaze angle."""
    # assume blaze and incidence angles are equal
    beta = _diffracted_angle(blaze_angle, order, groove_spacing, wavelength)
    one = order * np.cos(blaze_angle)
    two = np.cos(blaze_angle) - np.sin(blaze_angle) / np.tan((beta + blaze_angle)/2)
    return one * two


def _blaze_efficiency_sinc(blaze_angle, order, groove_spacing, wavelength):
    """Calculate blaze efficiency using sinc^2 approximation. Assume incidence angle = blaze angle."""
    # assume blaze and incidence angles are equal
    bp = _blaze_phase(blaze_angle, order, groove_spacing, wavelength)
    sinc = np.sinc(bp*u.rad)
    sinc[np.abs(bp) > 1.0] = 0.0  # put all values outside +/- pi in sinc to zero
    return sinc**2