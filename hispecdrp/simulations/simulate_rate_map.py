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
from functools import lru_cache
import h5py

import numpy as np

logger = logging.getLogger(__name__)

# Default output filenames for saved products
DEFAULT_RATE_MAP_FILENAME = "rate_map.fits"
DEFAULT_PYECHELLE_FILENAME = "pyechelle_out.fits"

from numba import njit, prange

from hispecdrp.primitives.utils.cubic_spline import (
    cubic_spline_build,
    cubic_spline_eval,
    cubic_spline_eval_unsorted,
)
from hispecdrp.primitives.utils.linear_interp import linear_interp2d_eval_cell

from scipy.interpolate import PchipInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from .throughput import blaze_efficiency_sinc

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

        self.bspec_hdf_model: str = default_bspec_hdf
        self.rspec_hdf_model: str = default_rspec_hdf
        self.output_directory: str = os.getcwd()
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

        If continuous=True, spectrum_flux is assumed to be a flux density per unit of
        wavelength (e.g. erg/s/cm^2/micron), sampled at spectrum_wavelengths.

        If continuous=False, spectrum_wavelengths is treated as a list of discrete line
        centers and spectrum_flux must instead be a RATE per line (NOT a density),
        given as either a photon count rate (e.g. u.ph / u.s) or an energy/radiant-power
        rate (e.g. u.erg / u.s, u.W). Energy-rate line fluxes are converted to a photon
        count rate downstream using each line's wavelength (E_photon = h*c/wavelength).
        No telescope-area conversion is applied to line fluxes downstream."""

        if fiber_number not in (1, 2, 3):
            raise ValueError('Fiber must be 1, 2, or 3!')

        if continuous and spectrum_flux.unit.is_equivalent(u.ph/u.s):
            logger.warning(
                "Spectrum is specified as continuous, but units match those expected from a line list. "
                "Continuing assuming spectrum is a non-continuous line list."
            )
            continuous=False
        elif not continuous and spectrum_flux.unit.is_equivalent(u.erg/u.s/u.cm**2/u.AA):
            logger.warning(
                "Spectrum is specified as NOT continuous, but units are a spectral density. "
                "Continuing assuming spectrum is a continuous spectrum."
            )
            continuous=True

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

    def get_wavelength_solution(self,
                                spectrograph: str = "BSPEC",
                                reverse_spectral_axis : bool = True
                                ) -> dict[int, dict[str, np.ndarray]]:
        """Get wavelength solution from HDF model file as a nested dictionary of the form:
        {fiber: {order_number: {'pixel': np.ndarray, 'wavelength': np.ndarray}}}
        giving the wavelength at each detector x pixel.

        The wavelength at each pixel comes from a not-a-knot cubic spline of wavelength vs. x through
        the HDF trace samples. Pixels outside the trace samples are extrapolated with the end cubics.
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

        self.wavelength_solution = {}
        for fiber in [*range(1,4)]:
            self.wavelength_solution[fiber] = {}
            fiber_key = f"fiber_{fiber}"
            for od in order_list:
                ow = model_data[fiber_key][od]['xy_wavelengths']
                ox = model_data[fiber_key][od]["x"]

                # Interpolate wavelength as function of detector x pixel with a cubic spline through
                # the trace samples. HDF traces may not be sorted in x, and the spline needs x
                # strictly increasing.
                ox = np.asarray(ox, dtype=np.float64)
                ow = np.asarray(ow, dtype=np.float64)
                good = np.isfinite(ox) & np.isfinite(ow)
                ox = ox[good]
                ow = ow[good]
                if ox.size < 3:
                    raise ValueError(f"Not enough valid samples for fiber {fiber} order {od} wavelength solution.")
                sort = np.argsort(ox)
                ox = ox[sort]
                ow = ow[sort]
                if np.any(np.diff(ox) <= 0):
                    logger.warning(f"Duplicate x values found for fiber {fiber} order {od}. This order will be skipped.")
                    continue

                # Not-a-knot cubic spline; pixels outside the trace samples are extrapolated with the end cubics
                inv_w = cubic_spline_eval(cubic_spline_build(ox, ow), np.arange(4096, dtype=np.float64))

                self.wavelength_solution[fiber][od] = {
                    "pixel": np.arange(4096),
                    "wavelength": inv_w[::-1] # reverse to match detector orientation
                }
                if reverse_spectral_axis:
                    self.wavelength_solution[fiber][od]['wavelength'] = self.wavelength_solution[fiber][od]['wavelength'][::-1]

        return self.wavelength_solution

    @staticmethod
    @lru_cache(maxsize=4)
    def _instrument_wavelength_solution(spectrograph: str) -> dict[int, dict[int, dict[str, np.ndarray]]]:
        """Cached `get_wavelength_solution` of a default SimulateRateMap, per spectrograph (expensive HDF read)."""
        return SimulateRateMap().get_wavelength_solution(spectrograph=spectrograph)

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
                    flux_units=(
                        "ph/s/micron" if self.fibers[fiber]["continuous"] else "ph/s"
                    ),
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
        apply_blaze: bool = True,
        sampling_factor: float | None = 4,
    ) -> np.ndarray[np.float64]:
        """Create rate map using native model interpolation/convolution.

        Parameters
        ----------
        spectrograph : str
            'BSPEC' for Blue (yJ) spectrograph, 'RSPEC' for Red (HK) spectrograph.
        save : bool
            If True, saves the resulting rate map to output_directory. Default is True.
        method : str
            Method to use for rate map generation. Options are 'bilinear_weights', 'prf_convolution' or 'prf_convolution_old'.
            'prf_convolution_old' is the previous, much slower 'prf_convolution' implementation (PCHIP trace, one interpolator call per sample).
        apply_blaze : bool
            If True, applies a sinc^2 blaze efficiency to the fiber flux to each order before simulation. Default is True.
        sampling_factor : float or None
            'prf_convolution' only: number of input-spectrum samples per detector pixel along each order's trace.
            A continuous input spectrum is resampled onto this grid with a cubic spline, whether it is over- or
            undersampled. If None, the input samples are used as is. Ignored for line lists. Default is 4.
        Returns
        -------
        rate_map : ndarray[float64]
            Simulated rate map in units of photons/s/pixel.

        The 'prf_convolution' method interpolates the PSFs, is faster, and is recommended as the default."""

        spectrograph = spectrograph.upper()

        if spectrograph not in ("BSPEC", "RSPEC"):
            raise ValueError("Spectrograph must be BSPEC or RSPEC!")

        if method not in ("bilinear_weights", "prf_convolution", "prf_convolution_old"):
            raise ValueError(
                "Method must be bilinear_weights, "
                "prf_convolution (psf convolved with prf) or prf_convolution_old!"
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

        logger.info("Creating native rate map (spectrograph=%s, method=%s).", spectrograph, method)
        detector_shape = (model_data["nx"], model_data["ny"])
        pixelsize = model_data["pixelsize"]
        blaze_angle = model_data["blaze_angle"] * u.deg
        groove_spacing = model_data["groove_spacing"] / u.mm

        rate_map_array = np.empty((len(self.fibers), detector_shape[1], detector_shape[0]), dtype=np.float64)
        for i, fiber in enumerate(self.fibers):
            fiber_key = f"fiber_{fiber}"

            # remove orders in hdf model that will not be used
            for od in list(model_data[fiber_key]):
                if od not in order_list:
                    del model_data[fiber_key][od]

            wavelength = self.fibers[fiber]["wavelength"].to(u.micron)
            continuous = self.fibers[fiber]["continuous"]

            if continuous:
                tmp_flux = (
                    self.fibers[fiber]["flux"] *
                    np.pi * (self.telescope_primary_diameter ** 2) / 4
                )
                flux = tmp_flux.to(
                    u.ph / u.s / u.micron,
                    equivalencies=u.spectral_density(wavelength),
                )
            else:
                # Line lists are already a photon count RATE (or energy RATE),
                # not a density, so no telescope-area collecting-area scaling
                # or spectral-density (per-wavelength) conversion is applied.
                # Energy-rate fluxes are converted to a photon rate per line
                # using each line's wavelength.
                flux = _line_flux_to_photon_rate(
                    self.fibers[fiber]["flux"], wavelength
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
                    groove_spacing=groove_spacing,
                    continuous=continuous
                )
            elif method == "prf_convolution_old":
                self.rate_map = psf_prf_convolution_rate_map_old(
                    wavelength,
                    flux,
                    model_data[fiber_key],
                    detector_shape=detector_shape,
                    pixelsize=pixelsize,
                    apply_blaze=apply_blaze,
                    blaze_angle=blaze_angle,
                    groove_spacing=groove_spacing,
                    continuous=continuous
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
                    groove_spacing=groove_spacing,
                    continuous=continuous,
                    sampling_factor=sampling_factor,
                    bad_psf_indices=_BAD_ZEMAX_PSFS.get(fiber),
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

        CSV spectra are in units of photons/s/micron for continuous spectra,
        or photons/s per line for line lists (continuous=False). Line-list
        flux may be supplied as a photon count rate or an energy rate; it is
        converted to a photon count rate before being written out."""

        wavelength = self.fibers[fiber]["wavelength"].to(u.micron)

        if self.fibers[fiber]["continuous"]:
            tmp_flux = (
                self.fibers[fiber]["flux"] *
                np.pi * (self.telescope_primary_diameter ** 2) / 4
            )
            flux = tmp_flux.to(
                u.ph / u.s / u.micron,
                equivalencies=u.spectral_density(wavelength),
            )
        else:
            # Line lists are already a photon count RATE (or energy RATE),
            # not a flux density, so no telescope-area or spectral-density
            # conversion is applied. Energy-rate fluxes are converted to a
            # photon rate per line using each line's wavelength.
            flux = _line_flux_to_photon_rate(
                self.fibers[fiber]["flux"], wavelength
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
    orders: dict[int, dict],
    detector_shape: tuple[int, int] = (4096, 4160),
    pixelsize: float = 10.0,
    apply_blaze: bool = True,
    blaze_angle: Quantity = None,
    groove_spacing: Quantity = None,
    continuous: bool = True
) -> np.ndarray[np.float64]:
    """
    Simulate detector rate map (photons/s/pixel) by depositing
    oversampled PSFs using bilinear weighting.

    Parameters
    ----------
    wave : Quantity
        Input wavelengths corresponding to flux samples.
    flux : Quantity
        Flux array. If continuous=True, this is a photon rate density
        (photons/s/unit wavelength) and is integrated over each wavelength bin.
        If continuous=False, this is a photon count RATE (photons/s) per discrete
        line and is used directly, with no bin-width integration.
    orders : dict
        Nested structure containing wavelength trace and PSF stamps for orders. 
        Top level keys are order numbers (int).
    detector_shape : tuple of int, optional
        Detector (ny, nx) pixel size, default = (4096, 4160).
    pixelsize : float, optional
        Pixel size in microns (affects oversampling), default = 10.
    continuous : bool, optional
        If True (default), flux is treated as a wavelength density and multiplied
        by the local wavelength bin width to get photons/s. If False, flux is
        treated as an already-integrated photon rate (photons/s) per discrete line,
        e.g. for injecting unresolved emission lines. Default is True.

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

        # apply blaze efficiency if requested
        if apply_blaze and blaze_angle is not None and groove_spacing is not None:
            if blaze_angle.value < 0:
                blaze_angle *= -1
            blaze_eff = blaze_efficiency_sinc(
                blaze_angle,
                od,
                1.0 / groove_spacing,
                order_wave
            )
            order_flux *= blaze_eff

        if continuous:
            # order_flux is a photon rate DENSITY (photons/s/micron); integrate
            # over each local wavelength bin to get a photon rate (photons/s).
            dws = np.gradient(order_wave)
            photons_per_s_arr = order_flux * dws
            # last sample has no trailing bin to integrate over in this scheme
            n_samples = len(order_wave) - 1
        else:
            # order_flux is already a photon RATE (photons/s) per discrete line;
            # use it directly, and inject every line (don't drop the last one).
            photons_per_s_arr = order_flux
            n_samples = len(order_wave)

        for i in range(n_samples):
            lam0 = order_wave[i]
            photons_per_s = photons_per_s_arr[i]

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


def psf_prf_convolution_rate_map_old(
    wave: Quantity,
    flux: Quantity,
    orders: dict[int, dict],
    detector_shape: tuple[int, int] = (4096, 4160),
    pixelsize: float = 10.0,
    apply_blaze: bool = True,
    blaze_angle: Quantity = None,
    groove_spacing: Quantity = None,
    continuous: bool = True
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
        Flux samples. If continuous=True, expressed as a photon rate density
        (photons/s/unit wavelength) and integrated over each wavelength bin.
        If continuous=False, expressed as a photon count RATE (photons/s) per
        discrete line and used directly, with no bin-width integration.
    orders : dict
        Nested dictionary containing trace coordinates and PSF stamps.
        Top level keys are order numbers (int).
    detector_shape : tuple of int, optional
        Detector pixels (ny, nx). Default = (4096, 4160).
    pixelsize : float, optional
        Pixel size in microns, used to determine oversampling.
    apply_blaze : bool, optional
        If True, applies a sinc^2 blaze efficiency to the flux for each order.
    blaze_angle : Quantity, optional
        Blaze angle of the echelle grating. Required if apply_blaze is True.
    groove_spacing : Quantity, optional
        Groove spacing of the echelle grating. Required if apply_blaze is True.
    continuous : bool, optional
        If True (default), flux is treated as a wavelength density and multiplied
        by the local wavelength bin width to get photons/s. If False, flux is
        treated as an already-integrated photon rate (photons/s) per discrete line,
        e.g. for injecting unresolved emission lines. Default is True.

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
            blaze_eff = blaze_efficiency_sinc(
                blaze_angle,
                od,
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

        if continuous:
            # order_flux is a photon rate DENSITY (photons/s/micron); integrate
            # over each local wavelength bin to get a photon rate (photons/s).
            dws = np.gradient(order_wave)
            photons_per_s = order_flux * dws
        else:
            # order_flux is already a photon RATE (photons/s) per discrete line;
            # use it directly, with no bin-width integration.
            photons_per_s = order_flux

        psf_height, psf_width = psf_eff_list[0].shape
        x_psf_eff = np.arange(psf_width) - psf_width // 2
        y_psf_eff = np.arange(psf_height) - psf_height // 2

        max_pixels_x = psf_width // oversample
        max_pixels_y = psf_height // oversample

        # Create interpolation function for effective psf
        psf_eff_interp = RegularGridInterpolator((psfw, y_psf_eff, x_psf_eff), psf_eff_list, bounds_error=False, fill_value=None)

        pixel_xs = (np.arange(max_pixels_x) - (max_pixels_x) // 2) * oversample
        pixel_ys = (np.arange(max_pixels_y) - (max_pixels_y) // 2) * oversample


        # continuous: last sample has no trailing bin to integrate over in this
        # scheme, so it is skipped. line list: every discrete line is injected.
        n_samples = len(order_wave) - 1 if continuous else len(order_wave)
        for i in range(n_samples):
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


# PSF stamps in the RSPEC pyechelle HDF model (HISPEC_HK_20231006B-1_all_traces.hdf) that differ from the
# rest of their order and fiber, as {fiber: {order: [stamp indices]}}. Indices count the order's stamps
# sorted by increasing wavelength. A stamp is either entirely good or entirely bad, so whole stamps are
# compared: with the stamps normalized to unit sum, the distance between two stamps is 0.5 * sum|P_i - P_j|
# over all pixels. The kept stamps are the longest chain in wavelength order whose consecutive members are
# within 1.5x the largest distance over the same stamp gaps in clean orders (59-60, 76-98); the rest are
# listed. The bad stamps lie along a single straight band across the detector.
# psf_prf_convolution_rate_map drops them before interpolating in wavelength.
_BAD_ZEMAX_PSFS = {
    1: {
        64: [14],
        65: [13],
        66: [12],
        67: [11],
        68: [9, 10],
        69: [8],
        70: [6],
        71: [3, 4],
        72: [2],
    },
    2: {
        63: [14],
        64: [13],
        65: [12],
        66: [11],
        67: [9, 10],
        68: [8],
        69: [6],
        70: [3, 4],
        71: [2],
    },
    3: {
        62: [14],
        63: [13],
        64: [12],
        65: [11],
        66: [9, 10],
        67: [8],
        68: [6],
        69: [4],
        70: [2],
    },
}


def psf_prf_convolution_rate_map(
    wave: Quantity,
    flux: Quantity,
    orders: dict[int, dict],
    detector_shape: tuple[int, int] = (4096, 4160),
    pixelsize: float = 10.0,
    apply_blaze: bool = True,
    blaze_angle: Quantity = None,
    groove_spacing: Quantity = None,
    continuous: bool = True,
    sampling_factor: float | None = 4,
    bad_psf_indices: dict[int, list[int]] | None = None,
) -> np.ndarray[np.float64]:
    """
    Generate a detector rate map by convolving oversampled PSFs with the pixel response function (PRF),
    and interpolating them for many trace positions across each order.

    Same inputs and flux bookkeeping as the previous implementation, `psf_prf_convolution_rate_map_old`.
    Differences from it:

    - Trace positions x(lambda), y(lambda) use a not-a-knot cubic spline instead of PCHIP.
    - Stamps are built (in parallel) and deposited in Numba rather than one interpolator call per
      wavelength sample. As before, each stamp is the effective PSF (PSF
      convolved with the PRF) shifted to the sample's pixel phase by bilinear interpolation.
    - The PSF stamps in bad_psf_indices are left out. As before, between PSF wavelengths the effective
      PSF is a linear blend of the two bracketing (kept) PSFs, and beyond the end PSFs it extrapolates
      linearly from the end pair.
    - For an even oversampling factor, the effective PSFs are bilinearly interpolated by half a PSF
      sample and cropped to an odd size, so they are centred on the trace position. The previous
      implementation leaves them half a PSF sample (0.05 px at 10x) off.
    - By default a continuous spectrum is resampled with a cubic spline onto sampling_factor samples
      per detector pixel along each order's trace (see sampling_factor).

    Parameters
    ----------
    wave : Quantity
        Input wavelength samples.
    flux : Quantity
        Flux samples. If continuous=True, expressed as a photon rate density
        (photons/s/unit wavelength) and integrated over each wavelength bin.
        If continuous=False, expressed as a photon count RATE (photons/s) per
        discrete line and used directly, with no bin-width integration.
    orders : dict
        Nested dictionary containing trace coordinates and PSF stamps.
        Top level keys are order numbers (int).
    detector_shape : tuple of int, optional
        Detector pixels (ny, nx). Default = (4096, 4160).
    pixelsize : float, optional
        Pixel size in microns, used to determine oversampling.
    apply_blaze : bool, optional
        If True, applies a sinc^2 blaze efficiency to the flux for each order.
    blaze_angle : Quantity, optional
        Blaze angle of the echelle grating. Required if apply_blaze is True.
    groove_spacing : Quantity, optional
        Groove spacing of the echelle grating. Required if apply_blaze is True.
    continuous : bool, optional
        If True (default), flux is treated as a wavelength density and multiplied
        by the local wavelength bin width to get photons/s. If False, flux is
        treated as an already-integrated photon rate (photons/s) per discrete line,
        e.g. for injecting unresolved emission lines. Default is True.
    sampling_factor : float or None, optional
        Number of samples per detector pixel along each order's trace. If given and continuous=True,
        the input spectrum is resampled with a not-a-knot cubic spline onto a grid uniform in
        detector x along the trace (wavelengths from the trace's lambda(x) cubic spline), whether
        the input is over- or undersampled. If None, the input samples are used as is. Default is 4.
    bad_psf_indices : dict or None, optional
        PSF stamps to drop, as {order: [stamp indices]} with the indices counting each order's stamps
        sorted by increasing wavelength (see ``_BAD_ZEMAX_PSFS``). If None, all stamps are used.

    Returns
    -------
    det : ndarray[float64]
        Detector photon rate map in photons/s/pixel.
    """
    det = np.zeros(detector_shape, dtype=np.float64)
    if bad_psf_indices is None:
        bad_psf_indices = {}

    for od in orders:
        logger.debug("Simulating order %s", od)

        ow = np.asarray(orders[od]['xy_wavelengths'], dtype=np.float64)
        ox = np.asarray(orders[od]["x"], dtype=np.float64)
        oy = np.asarray(orders[od]["y"], dtype=np.float64)

        oversample = pixelsize / orders[od]['psf_data_spacing']
        if not float(oversample).is_integer():
            raise ValueError("Oversample factor must be an integer")
        oversample = int(oversample)

        if continuous and sampling_factor is not None:
            order_wave, order_flux = _resample_order_spectrum(wave, flux, ow, ox, sampling_factor)
        else:
            order_wave_min = ow.min() * u.micron
            order_wave_max = ow.max() * u.micron
            in_order = (wave >= order_wave_min) & (wave <= order_wave_max)
            order_wave = wave[in_order]
            order_flux = flux[in_order]

        # continuous: last sample has no trailing bin to integrate over in this
        # scheme, so it is skipped. line list: every discrete line is injected.
        n_samples = len(order_wave) - 1 if continuous else len(order_wave)
        if n_samples <= 0:
            continue

        # apply blaze efficiency if requested
        if apply_blaze and blaze_angle is not None and groove_spacing is not None:
            if blaze_angle.value < 0:
                blaze_angle *= -1
            blaze_eff = blaze_efficiency_sinc(
                blaze_angle,
                od,
                1.0 / groove_spacing,
                order_wave
            )
            order_flux *= blaze_eff

        if continuous:
            # order_flux is a photon rate DENSITY (photons/s/micron); integrate
            # over each local wavelength bin to get a photon rate (photons/s).
            photons_per_s = (order_flux * np.gradient(order_wave)).to_value(u.ph / u.s)
        else:
            # order_flux is already a photon RATE (photons/s) per discrete line;
            # use it directly, with no bin-width integration.
            photons_per_s = order_flux.to_value(u.ph / u.s)

        # Trace positions
        lam = order_wave.to_value(u.micron)[:n_samples]
        xc_arr = cubic_spline_eval_unsorted(cubic_spline_build(ow, ox), lam)
        yc_arr = cubic_spline_eval_unsorted(cubic_spline_build(ow, oy), lam)
        xc_floor_arr = np.floor(xc_arr).astype(np.int64)
        yc_floor_arr = np.floor(yc_arr).astype(np.int64)

        # Effective PSFs on increasing wavelength, without the outlier stamps in bad_psf_indices so the
        # interpolation goes around them
        psfw_all = np.asarray(orders[od]['psf_wavelengths'], dtype=np.float64)
        psf_sort = np.delete(np.argsort(psfw_all), bad_psf_indices.get(od, []))
        psfw = psfw_all[psf_sort]
        prf_kernel = np.ones((oversample, oversample))
        psf_eff_list = fftconvolve(
            np.asarray(orders[od]['psf_stamps'], dtype=np.float64)[psf_sort],
            prf_kernel[None, ...],
            mode='same',
            axes=(1, 2)
        )
        if oversample % 2 == 0:
            # An even PRF box puts effective-PSF sample m at m - N//2 - 1/2 (N = stamp size). Bilinear
            # interpolation at the half-sample phase moves the samples onto m - N//2; then crop to an odd
            # stamp centred on 0.
            Nrows, Ncols = psf_eff_list.shape[1:]
            ry = min(Nrows // 2, Nrows - 2 - Nrows // 2)
            rx = min(Ncols // 2, Ncols - 2 - Ncols // 2)
            psf_eff_list = _half_sample_shift_numba(psf_eff_list)[
                :, Nrows // 2 - ry:Nrows // 2 + ry + 1, Ncols // 2 - rx:Ncols // 2 + rx + 1
            ]
        psf_height, psf_width = psf_eff_list[0].shape
        x_psf_eff = np.arange(psf_width, dtype=np.float64) - psf_width // 2
        y_psf_eff = np.arange(psf_height, dtype=np.float64) - psf_height // 2

        # Linear interpolation in wavelength through the PSFs, at every effective-PSF grid point
        # (trilinear with the bilinear pixel-phase step)
        psf_a, psf_b, psf_c, psf_d = _build_psf_wavelength_linear_numba(psfw, psf_eff_list)

        # Detector-pixel sample offsets within a stamp, in PSF samples
        max_pixels_x = psf_width // oversample
        max_pixels_y = psf_height // oversample
        pixel_xs = (np.arange(max_pixels_x) - max_pixels_x // 2) * float(oversample)
        pixel_ys = (np.arange(max_pixels_y) - max_pixels_y // 2) * float(oversample)

        stamps = np.empty((n_samples, max_pixels_y, max_pixels_x))
        _interp_prf_stamps_numba(
            x_psf_eff, y_psf_eff, psf_a, psf_b, psf_c, psf_d, psfw, lam,
            (xc_arr - xc_floor_arr) * oversample, (yc_arr - yc_floor_arr) * oversample,
            pixel_xs, pixel_ys, stamps,
        )
        _deposit_stamps_numba(det, stamps, xc_floor_arr, yc_floor_arr, photons_per_s[:n_samples])

    return det


def _resample_order_spectrum(
    wave: Quantity, flux: Quantity, ow: np.ndarray, ox: np.ndarray, sampling_factor: float
) -> tuple[Quantity, Quantity]:
    """
    Resample a continuous spectrum onto sampling_factor samples per detector pixel along one order.

    The grid is uniform in detector x over the trace samples (ow, ox), converted to wavelength with
    the trace's not-a-knot cubic spline lambda(x), and kept within the input wavelength range. The
    flux is a not-a-knot cubic spline through the input samples covering the order (plus one on
    each side). wave must be strictly increasing. Returns empty arrays if the input does not
    overlap the order.
    """
    w = np.asarray(wave.to_value(u.micron), dtype=np.float64)
    f = np.asarray(flux.value, dtype=np.float64)
    s = np.argsort(ox)
    x = np.arange(ox.min(), ox.max(), 1.0 / sampling_factor)
    lam = np.sort(cubic_spline_eval(cubic_spline_build(ox[s], ow[s]), x))
    lam = lam[(lam >= max(ow.min(), w[0])) & (lam <= min(ow.max(), w[-1]))]
    i0 = max(int(np.searchsorted(w, lam[0])) - 1, 0) if lam.size else 0
    i1 = min(int(np.searchsorted(w, lam[-1])) + 2, w.size) if lam.size else 0
    if i1 - i0 < 3:
        return lam[:0] * u.micron, f[:0] * flux.unit
    fl = cubic_spline_eval(cubic_spline_build(w[i0:i1], f[i0:i1]), lam)
    return lam * u.micron, fl * flux.unit


@njit(parallel=True, cache=True)
def _half_sample_shift_numba(z: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of each z[k] (Npsf, Nrows, Ncols) at the half-sample midpoints; returns (Npsf, Nrows - 1, Ncols - 1)."""
    Npsf, Nrows, Ncols = z.shape
    out = np.empty((Npsf, Nrows - 1, Ncols - 1))
    for k in prange(Npsf):
        for j in range(Nrows - 1):
            for i in range(Ncols - 1):
                out[k, j, i] = linear_interp2d_eval_cell(z[k], i, j, 0.5, 0.5)
    return out


@njit(parallel=True, cache=True)
def _build_psf_wavelength_linear_numba(
    psfw: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear interpolation in wavelength between consecutive PSFs z (Npsf, Nrows, Ncols), as per-segment
    Horner coefficients (a, b, c, d), each (Npsf - 1, Nrows, Ncols), with c = d = 0: on segment k,
    z(lam) = a + t*(b + t*(c + t*d)) with t = lam - psfw[k].
    `_interp_prf_stamps_numba` uses the first/last segment beyond the end PSFs, so it extrapolates linearly there.
    """
    Npsf, Nrows, Ncols = z.shape
    a = np.empty((Npsf - 1, Nrows, Ncols))
    b = np.empty((Npsf - 1, Nrows, Ncols))
    c = np.zeros((Npsf - 1, Nrows, Ncols))
    d = np.zeros((Npsf - 1, Nrows, Ncols))
    for k in prange(Npsf - 1):
        h = psfw[k + 1] - psfw[k]
        for j in range(Nrows):
            for i in range(Ncols):
                a[k, j, i] = z[k, j, i]
                b[k, j, i] = (z[k + 1, j, i] - z[k, j, i]) / h
    return a, b, c, d


@njit(parallel=True, cache=True)
def _interp_prf_stamps_numba(
    xk: np.ndarray,
    yk: np.ndarray,
    za: np.ndarray,
    zb: np.ndarray,
    zc: np.ndarray,
    zd: np.ndarray,
    psfw: np.ndarray,
    lam: np.ndarray,
    shift_x: np.ndarray,
    shift_y: np.ndarray,
    pixel_xs: np.ndarray,
    pixel_ys: np.ndarray,
    stamps: np.ndarray,
) -> None:
    """
    Fill stamps (Nsamples, Nys, Nxs) in place with unit-sum effective PSF stamps.

    Stamp n is built in two steps:

    1. The chromatic effective PSF at wavelength lam[n], from the per-segment polynomials in wavelength
       through the PSFs (za, zb, zc, zd from `_build_psf_wavelength_linear_numba`; the end segments
       extrapolate), clamped at zero. It is evaluated only on the grid points that bracket each
       stamp pixel, a (2 Nys, 2 Nxs) block, not the whole PSF grid.
    2. The pixel-phase shift, by bilinear interpolation of that block (`linear_interp2d_eval_cell`)
       at x = pixel_xs - shift_x[n], y = pixel_ys - shift_y[n] in PSF samples.

    The PSF grid xk, yk must be unit spaced, so the cell of each point is found directly (clamped to
    the edge cells, which then extrapolate linearly) instead of by a search.
    """
    Nseg, Nrows, Ncols = za.shape
    Nys = pixel_ys.shape[0]
    Nxs = pixel_xs.shape[0]
    for n in prange(lam.shape[0]):
        # Bracketing grid cell and fractional offset of every stamp row and column
        jj = np.empty(Nys, dtype=np.int64)
        tys = np.empty(Nys)
        for a in range(Nys):
            y = pixel_ys[a] - shift_y[n] - yk[0]
            jj[a] = min(max(int(np.floor(y)), 0), Nrows - 2)
            tys[a] = y - jj[a]
        ii = np.empty(Nxs, dtype=np.int64)
        txs = np.empty(Nxs)
        for b in range(Nxs):
            x = pixel_xs[b] - shift_x[n] - xk[0]
            ii[b] = min(max(int(np.floor(x)), 0), Ncols - 2)
            txs[b] = x - ii[b]

        # 1. Chromatic PSF at lam[n] on those grid points: psf[2a + r, 2b + c] is grid point (jj[a] + r, ii[b] + c)
        k = min(max(np.searchsorted(psfw, lam[n]) - 1, 0), Nseg - 1)
        t = lam[n] - psfw[k]
        a0, b0, c0, d0 = za[k], zb[k], zc[k], zd[k]
        psf = np.empty((2 * Nys, 2 * Nxs))
        for a in range(Nys):
            for r in range(2):
                j = jj[a] + r
                for b in range(Nxs):
                    for c in range(2):
                        i = ii[b] + c
                        # Clamp at zero where the linear extrapolation beyond the end PSFs goes negative
                        psf[2 * a + r, 2 * b + c] = max(a0[j, i] + t * (b0[j, i] + t * (c0[j, i] + t * d0[j, i])), 0.0)

        # 2. Pixel-phase shift by bilinear interpolation
        total = 0.0
        for a in range(Nys):
            for b in range(Nxs):
                v = linear_interp2d_eval_cell(psf, 2 * b, 2 * a, txs[b], tys[a])
                stamps[n, a, b] = v
                total += v
        for a in range(Nys):
            for b in range(Nxs):
                stamps[n, a, b] /= total


@njit(cache=True)
def _deposit_stamps_numba(
    det: np.ndarray,
    stamps: np.ndarray,
    x_floor: np.ndarray,
    y_floor: np.ndarray,
    photons_per_s: np.ndarray,
) -> None:
    """Add stamps[n] * photons_per_s[n] to det, centred on pixel (y_floor[n], x_floor[n]); off-chip pixels are dropped."""
    Nrows, Ncols = det.shape
    Nys, Nxs = stamps.shape[1], stamps.shape[2]
    for n in range(stamps.shape[0]):
        y0 = y_floor[n] - Nys // 2
        x0 = x_floor[n] - Nxs // 2
        w = photons_per_s[n]
        for a in range(Nys):
            yy = y0 + a
            if yy < 0 or yy >= Nrows:
                continue
            for b in range(Nxs):
                xx = x0 + b
                if 0 <= xx < Ncols:
                    det[yy, xx] += stamps[n, a, b] * w


def _line_flux_to_photon_rate(flux: Quantity, wavelength: Quantity) -> Quantity:
    """Convert a per-line flux (already a RATE, not a density) to a photon
    count rate (photons/s).

    Used for continuous=False (line list) spectra, where each flux value is
    assumed to already be integrated over the (unresolved) line rather than
    a density that needs to be multiplied by a wavelength bin width.

    Accepts flux given either as:
      - a photon count rate, e.g. u.ph / u.s, or
      - an energy/radiant-power rate, e.g. u.erg / u.s, u.W.
    Energy-rate fluxes are converted to a photon rate using each line's
    photon energy, E_photon = h * c / wavelength, since photon energy is
    wavelength-dependent.

    Parameters
    ----------
    flux : Quantity
        Per-line flux, as a photon count rate or an energy rate.
    wavelength : Quantity
        Wavelength of each line (broadcastable against flux), used to
        convert energy-rate fluxes to a photon count rate.

    Returns
    -------
    Quantity
        Flux converted to a photon count rate (u.ph / u.s).
    """
    if flux.unit.is_equivalent(u.ph / u.s):
        return flux.to(u.ph / u.s)
    elif flux.unit.is_equivalent(u.erg / u.s):
        photon_energy = (const.h * const.c / wavelength).to(u.erg)
        return (flux.to(u.erg / u.s) / photon_energy).value * (u.ph / u.s)
    else:
        raise ValueError(
            f"Unsupported flux unit '{flux.unit}' for continuous=False. "
            "Line-list flux must be given as a photon count rate "
            "(e.g. ph/s) or an energy rate (e.g. erg/s, W)."
        )


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
    # Allow a small float margin: densify inserts at exact log-lambda steps that can
    # land ~0.5 below ``min_resolution`` from rounding (~2e-6 relative at R=3e5).
    return bool(np.all(resolution >= float(min_resolution) * (1.0 - 1e-5)))


def _upsample_spectrum(
    wavelengths: Quantity,
    flux: Quantity,
    desired_sampling_resolution: float = 3e5
) -> tuple[Quantity, Quantity]:
    """
    Densify undersampled regions to a minimum resolving power.

    Preserves existing wavelength samples and only inserts new points where the
    local resolving power falls below ``desired_sampling_resolution``.  A global
    linear resample to fixed ``R`` is avoided: for BT-Settl in the BSPEC band the
    native grid is already denser than ``R~3e5``, and a global rebuild would
    *downsample* those lines (a cm/s-m/s systematic for small Doppler tests).

    Inserted fluxes use PCHIP on the native samples (shape-preserving).
    """
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    fl = np.asarray(flux, dtype=np.float64).ravel()
    if wl.size < 2:
        return wavelengths, flux
    if not np.all(np.diff(wl) > 0.0):
        raise ValueError("Wavelengths must be strictly increasing.")

    min_R = float(desired_sampling_resolution)
    out_wl: list[float] = [float(wl[0])]
    for i in range(wl.size - 1):
        w0 = float(wl[i])
        w1 = float(wl[i + 1])
        local_R = w0 / (w1 - w0) if w1 > w0 else np.inf
        if local_R < min_R and np.isfinite(local_R) and w0 > 0.0:
            dloglam = 1.0 / min_R
            n_fill = int(np.floor(np.log(w1 / w0) / dloglam))
            for k in range(1, n_fill + 1):
                w_ins = w0 * np.exp(k * dloglam)
                if w_ins < w1 * (1.0 - 1e-15):
                    out_wl.append(float(w_ins))
        out_wl.append(w1)

    new_wl = np.asarray(out_wl, dtype=np.float64)
    if new_wl.size == wl.size:
        return wavelengths, flux

    interp = PchipInterpolator(wl, fl, extrapolate=False)
    new_fl = np.asarray(interp(new_wl), dtype=np.float64)
    # Exact natives (avoid tiny PCHIP roundoff on retained samples).
    native_idx = np.searchsorted(new_wl, wl)
    native_idx = np.clip(native_idx, 0, new_wl.size - 1)
    match = np.abs(new_wl[native_idx] - wl) <= (
        np.finfo(np.float64).eps * np.maximum(1.0, np.abs(wl))
    )
    new_fl[native_idx[match]] = fl[match]

    wl_unit = getattr(wavelengths, "unit", None) or u.micron
    fl_unit = getattr(flux, "unit", None)
    new_wavelengths = new_wl * wl_unit
    new_flux = new_fl * fl_unit if fl_unit is not None else new_fl
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
            59: {
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
                base_dict[fiber_key] = _read_fiber(ccd1_data[fiber_key])

        return base_dict


def _read_fiber(fiber_group: h5py.Group) -> dict:
    """
    Read one fiber group into {order: trace and PSF data}, with integer orders.
    """

    fiber_dict = {}

    # Create order placeholders (non-PSF groups)
    for key in fiber_group:
        if "psf" not in key:
            order = int(key.replace("order", ""))
            fiber_dict[order] = {}

    # Fill in data for each entry
    for key in fiber_group:
        if "psf_order_" in key:
            _read_order_psfs(key, fiber_group[key], fiber_dict)
        else:
            _read_order_trace(key, fiber_group[key], fiber_dict)

    return fiber_dict


def _read_order_psfs(
    key: str,
    group: h5py.Group,
    fiber_dict: dict
) -> None:
    """
    Read the PSF stamps of a group like 'psf_order_59' into fiber_dict[order].
    """

    order = int(key.replace("psf_order_", ""))

    wavelengths = np.array([
        float(k.replace("wavelength_", "")) for k in group
    ])

    stamps = np.array([group[k][()] for k in group])
    fiber_dict[order]["psf_wavelengths"] = wavelengths
    fiber_dict[order]["psf_stamps"] = stamps
    fiber_dict[order]["psf_data_spacing"] = [
        group[k].attrs["dataSpacing"] for k in group
    ][0]


def _read_order_trace(
    key: str,
    dataset: h5py.Dataset,
    fiber_dict: dict
) -> None:
    """
    Read a trace dataset like 'order59' into fiber_dict[order] as x, y and wavelength arrays.
    """
    order = int(key.replace("order", ""))
    arr = np.vstack([list(k) for k in dataset])
    fiber_dict[order]["xy_wavelengths"] = arr[:, -1]
    fiber_dict[order]["x"] = arr[:, 4]
    fiber_dict[order]["y"] = arr[:, 5]

