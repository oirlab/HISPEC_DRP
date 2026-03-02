"""
Simulate up-the-ramp (UTR) readouts and run Stage 0 (and optionally Stage 1) pipeline.

This module provides functions to build simulated calibration models, generate
ramp data from a rate map, and run the Stage 0 pipeline to produce rate/slope
products. Calibrations can be (1) dummy/simulated via
:func:`create_simulated_calibration_models`, or (2) master calibrations
downloaded from the calibration store via :func:`run_caldb.get_master_calibrations`
when ``use_master_calibrations=True`` in :func:`simulate_up_the_ramp`. Optional
helpers for the fiberflat and Stage 1 pipelines are included for full workflows.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

# SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps3/)
SVO_FILTER_URL = "https://svo2.cab.inta-csic.es/theory/fps3/getdata.php?format=ascii&id=2MASS/2MASS.{band}"

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

from .. import datamodels
from ..datamodels.dqflags import DQ_FLAGS
from ..pipelines import Stage0Pipeline, ProcessFiberflatsPipeline, Stage1Pipeline
from .create_ramp import create_ramp
from .run_caldb import get_master_calibrations
from .simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap

# Directory containing bundled spectrum data (e.g. BT-Settl .h5 files)
SIMULATIONS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Default output filenames for simulated products
DEFAULT_SIMULATED_RAMP_FILENAME = "simulated_ramp.fits"

def _fiberflat_rate_filename(spectrograph: str, fiber: int) -> str:
    """Default filename for a single-fiber fiberflat rate FITS (matches KOA-style naming)."""
    prefix = "HB" if spectrograph.upper() == "BSPEC" else "HR"
    return f"{prefix}.20231006.00000.00_rate-FIBERFLAT-FIB{fiber}.fits"


def get_expected_ramp_path(output_dir: str) -> str:
    """Return the path where the simulated ramp is saved (same convention as run_simulated_pipeline)."""
    return os.path.join(os.path.abspath(output_dir), DEFAULT_SIMULATED_RAMP_FILENAME)


def get_expected_rate_map_path(output_dir: str) -> str:
    """Return the path where the rate map is saved by SimulateRateMap.create_native_rate_map."""
    return os.path.join(os.path.abspath(output_dir), DEFAULT_RATE_MAP_FILENAME)


def get_expected_fiberflat_rate_paths(
    output_dir: str,
    spectrograph: str,
    fibers: list[int],
    fiberflat_filename: callable | None = None,
) -> list[str]:
    """Return paths where dummy fiber flat rate FITS would be saved (one per fiber in 1–3)."""
    if fiberflat_filename is None:
        fiberflat_filename = _fiberflat_rate_filename
    out_abs = os.path.abspath(output_dir)
    return [
        os.path.join(out_abs, fiberflat_filename(spectrograph, f))
        for f in sorted(fibers)
        if f in (1, 2, 3)
    ]


def ramp_file_exists(output_dir: str) -> bool:
    """Return True if the simulated ramp file already exists in output_dir."""
    return os.path.isfile(get_expected_ramp_path(output_dir))


def rate_map_file_exists(output_dir: str) -> bool:
    """Return True if the rate map file already exists in output_dir."""
    return os.path.isfile(get_expected_rate_map_path(output_dir))


def fiberflat_rate_files_exist(
    output_dir: str,
    spectrograph: str,
    fibers: list[int],
    fiberflat_filename: callable | None = None,
) -> bool:
    """Return True if all expected dummy fiber flat rate files exist for the given fibers."""
    paths = get_expected_fiberflat_rate_paths(
        output_dir, spectrograph, fibers, fiberflat_filename=fiberflat_filename
    )
    return len(paths) > 0 and all(os.path.isfile(p) for p in paths)


__all__ = [
    "create_simulated_calibration_models",
    "simulate_ramp_from_rate_map",
    "fetch_svo_2mass_filter",
    "get_spectrum_path",
    "load_btsettl_spectrum",
    "create_ramp_and_calibrations",
    "create_dummy_fiberflat_rate_files",
    "get_expected_ramp_path",
    "get_expected_rate_map_path",
    "get_expected_fiberflat_rate_paths",
    "ramp_file_exists",
    "rate_map_file_exists",
    "fiberflat_rate_files_exist",
    "get_default_stage0_prim_args",
    "run_stage0_simulation",
    "simulate_up_the_ramp",
    "run_fiberflat_pipeline",
    "run_stage1_simulation",
    "create_wavelength_solution_model_from_sim",
]


def create_wavelength_solution_model_from_sim(spectrograph: str):
    """
    Build an LFCWavesolModel from the SimulateRateMap HDF wavelength solution for the given spectrograph.

    Used so Stage 1 set_wavesol can run with a known wavelength solution when using simulated data.

    Parameters
    ----------
    spectrograph : str
        'BSPEC' or 'RSPEC'.

    Returns
    -------
    LFCWavesolModel
        Model with wave_fib# and spec_dq_fib# set from the HDF trace solution (micron; DQ zeros).
    """
    from ..datamodels import hispec_config
    from ..datamodels.lfcwavesol import LFCWavesolModel

    srm = SimulateRateMap()
    ws_dict = srm.get_wavelength_solution(spectrograph=spectrograph)
    spectrograph = spectrograph.upper()
    order_list = hispec_config.HISPECConfig.get_order_list(spectrograph)

    # Use the real HISPEC instrument config (provides metadata schema, num_fibers, etc.)
    wavesol_model = LFCWavesolModel(
        config=hispec_config.HISPECConfig,
        meta={"spectrograph": spectrograph},
    )
    for fiber in (1, 2, 3):
        if fiber not in ws_dict or not ws_dict[fiber]:
            continue
        wave_rows = []
        for o in order_list:
            if o not in ws_dict[fiber]:
                break
            wave_rows.append(ws_dict[fiber][o]["wavelength"])
        if wave_rows:
            wave_arr = np.stack(wave_rows, axis=0).astype(np.float64)
            setattr(wavesol_model, f"wave_fib{fiber}", wave_arr)
            setattr(
                wavesol_model,
                f"spec_dq_fib{fiber}",
                np.zeros(wave_arr.shape, dtype=np.uint32),
            )
    return wavesol_model


def create_simulated_calibration_models(
    shape: tuple[int, int] = (4096, 4096),
    *,
    gain: float = 1.0,
    bias_dn: float | None = None,
    dark_current: float = 10.0,
    read_noise: float = 0.0,
    saturation_dn: float | None = None,
    nonlin_coeffs: np.ndarray | None = None,
    dq_bad_rows: list[int] | None = None,
    detflat_scale: float = 0.1,
    seed: int | None = None,
) -> dict[str, datamodels.CalibrationModel]:
    """
    Create a set of calibration data models for use in up-the-ramp simulation.

    All arrays are 2D with shape (ny, nx). Defaults match a typical HISPEC
    detector simulation (e.g. gain 1 e-/ADU, dark 10 e-/s, zero read noise).

    Parameters
    ----------
    shape : tuple of int, optional
        (ny, nx) detector dimensions. Default is (4096, 4096).
    gain : float, optional
        Detector gain in e-/ADU. Default is 1.0.
    bias_dn : float or None, optional
        Bias level in DN. If None, set to 1000 / gain so bias is 1000 e-.
        Default is None.
    dark_current : float, optional
        Dark current in e-/s/pixel. Default is 10.0.
    read_noise : float, optional
        Read noise in e-. Default is 0.0.
    saturation_dn : float or None, optional
        Saturation threshold in DN. If None, use max uint16. Default is None.
    nonlin_coeffs : np.ndarray or None, optional
        Polynomial coefficients for nonlinearity (e.g. [c0, c1, c2] for
        np.polyval). Default is None (no nonlinearity).
    dq_bad_rows : list of int or None, optional
        Row indices to mark as DO_NOT_USE in the DQ model. Default is None.
    detflat_scale : float, optional
        Scale (std) of random detector flat field (mean 1.0). Default is 0.1.
    seed : int or None, optional
        Random seed for detflat and dark maps. Default is None.

    Returns
    -------
    dict
        Dictionary with keys: ``dq``, ``saturation``, ``gain``, ``detflat``,
        ``bias``, ``dark``, ``rn``. Each value is the corresponding
        calibration data model instance.
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)

    if bias_dn is None:
        bias_dn = 1000.0 / gain
    if saturation_dn is None:
        saturation_dn = float(np.iinfo(np.uint16).max)
    if nonlin_coeffs is None:
        nonlin_coeffs = np.array([-1e-6, 1.0, 0.0])

    # DQ
    dq_map = np.zeros((ny, nx), dtype=np.uint32)
    if dq_bad_rows is not None:
        for r in dq_bad_rows:
            dq_map[r, :] |= DQ_FLAGS["DO_NOT_USE"]
    dq_model = datamodels.DQModel(DQ=dq_map)

    # Saturation
    sat_map = np.full((ny, nx), saturation_dn, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(
        SAT_THRESH=sat_map,
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Gain
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(
        GAIN=gain_map,
        RN=np.zeros_like(gain_map),
        COV=np.zeros((2, 2, ny, nx), dtype=np.float32),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Detector flat
    detflat_map = rng.normal(loc=1.0, scale=detflat_scale, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(
        FLAT=detflat_map,
        FLAT_ERR=np.zeros_like(detflat_map),
        FLAT_VAR_RNOISE=np.zeros_like(detflat_map),
        FLAT_VAR_POISSON=np.zeros_like(detflat_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Bias (in DN)
    bias_map = np.full((ny, nx), bias_dn, dtype=np.float32)
    bias_model = datamodels.BiasModel(
        BIAS=bias_map,
        kTC=np.zeros_like(bias_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Dark (rate in e-/s, store as DN/s via gain)
    dark_map = rng.normal(loc=dark_current, scale=dark_current / 10.0, size=(ny, nx)).astype(np.float32)
    dark_model = datamodels.DarkModel(
        RATE=dark_map / gain_map,
        RATE_ERR=np.zeros_like(dark_map),
        RATE_VAR_RNOISE=np.zeros_like(dark_map),
        RATE_VAR_POISSON=np.zeros_like(dark_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    # Read noise
    rn_map = np.full((ny, nx), read_noise, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(
        RN=rn_map / gain_map,
        RN_ERR=np.zeros_like(rn_map),
        DQ=np.zeros((ny, nx), dtype=np.uint32),
    )

    return {
        "dq": dq_model,
        "saturation": saturation_model,
        "gain": gain_model,
        "detflat": detflat_model,
        "bias": bias_model,
        "dark": dark_model,
        "rn": rn_model,
    }


def simulate_ramp_from_rate_map(
    rate_map: np.ndarray,
    calibration_models: dict[str, datamodels.CalibrationModel],
    *,
    readtime: float = 1.7,
    n_reads: int = 10,
    nonlin_coeffs: np.ndarray | None = None,
    max_cores: int = 1,
    spectrograph: str = "RSPEC",
    convert_to_uint16: bool = True,
    clip_ramps: bool = True,
    poisson_noise: bool = True,
    kTC_noise: float = 50.0,
) -> datamodels.RampModel:
    """
    Simulate a ramp from a rate map (e-/s/pixel) using create_ramp and calibration models.

    Parameters
    ----------
    rate_map : np.ndarray
        2D rate map in e-/s/pixel (e.g. from SimulateRateMap).
    calibration_models : dict
        Output of :func:`create_simulated_calibration_models`. Must contain
        ``gain``, ``detflat``, ``bias``, ``dark``, ``rn``.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads in the ramp. Default is 10.
    nonlin_coeffs : np.ndarray or None, optional
        Nonlinearity polynomial coefficients. If None, use [-1e-6, 1, 0].
        Default is None.
    max_cores : int, optional
        Number of cores for create_ramp. Default is 1.
    spectrograph : str, optional
        Spectrograph name for ramp meta. Default is "RSPEC".
    convert_to_uint16 : bool, optional
        Whether to convert ramp data to uint16. Default is True.
    clip_ramps : bool, optional
        Whether to clip ramp to uint16 range. Default is True.
    poisson_noise : bool, optional
        Whether to add Poisson noise in create_ramp. Default is True.
    kTC_noise : float, optional
        kTC noise in e- for first read. Default is 50.0.

    Returns
    -------
    RampModel
        Ramp data model with ``data`` and ``dq_raw`` and meta set.
    """
    gain_model = calibration_models["gain"]
    detflat_model = calibration_models["detflat"]
    bias_model = calibration_models["bias"]
    dark_model = calibration_models["dark"]
    rn_model = calibration_models["rn"]

    gain_map = gain_model.gain if hasattr(gain_model, "gain") else gain_model["GAIN"]
    detflat_map = detflat_model.flat if hasattr(detflat_model, "flat") else detflat_model["FLAT"]
    bias_map = bias_model.bias if hasattr(bias_model, "bias") else bias_model["BIAS"]
    dark_map = dark_model.rate if hasattr(dark_model, "rate") else dark_model["RATE"]
    rn_map = rn_model.rn if hasattr(rn_model, "rn") else rn_model["RN"]

    if nonlin_coeffs is None:
        nonlin_coeffs = np.array([-1e-6, 1.0, 0.0])

    # create_ramp expects bias in e-, dark in e-/s, read_noise in e-; calibration models store DN and DN/s
    gain_arr = np.asarray(gain_map, dtype=np.float64)
    bias_e = np.asarray(bias_map, dtype=np.float64) * gain_arr
    dark_e_per_s = np.asarray(dark_map, dtype=np.float64) * gain_arr
    rn_dn = np.asarray(rn_map, dtype=np.float64)
    read_noise_e = rn_dn * gain_arr

    ramp_dict = create_ramp(
        rate_map,
        readtime=readtime,
        n_reads=n_reads,
        nonlin_coeffs=nonlin_coeffs,
        gain=gain_map,
        flat=detflat_map,
        bias=bias_e,
        dark=dark_e_per_s,
        read_noise=read_noise_e,
        kTC_noise=kTC_noise,
        poisson_noise=poisson_noise,
        convert_to_uint16=convert_to_uint16,
        clip_ramps=clip_ramps,
        max_cores=max_cores,
    )

    ramp_dict["meta"]["spectrograph"] = spectrograph
    ramp_model = datamodels.RampModel(
        data=ramp_dict["data"],
        dq_raw=ramp_dict["dq_raw"],
        meta=ramp_dict["meta"],
    )
    ramp_model.meta.filename = DEFAULT_SIMULATED_RAMP_FILENAME
    return ramp_model


def get_default_stage0_prim_args(
    *,
    max_cores: int = 1,
    ramp_fit_method: str = "jwst_likely",
    run_bias_sub: bool = False,
    run_nonlin_corr: bool = False,
    run_jump_detec: bool = False,
) -> dict:
    """
    Return default primitive arguments for the Stage 0 pipeline.

    Parameters
    ----------
    max_cores : int, optional
        Number of cores for parallel primitives. Default is 1.
    ramp_fit_method : str, optional
        Ramp fit method: 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'.
        Default is 'jwst_likely'.
    run_bias_sub : bool, optional
        Whether to run bias subtraction. Default is False.
    run_nonlin_corr : bool, optional
        Whether to run nonlinearity correction. Default is False.
    run_jump_detec : bool, optional
        Whether to run jump detection. Default is False.

    Returns
    -------
    dict
        Nested dict of primitive names to keyword arguments for
        Stage0Pipeline.apply(..., prim_args=...).
    """
    return {
        "dq_init": {"max_cores": max_cores},
        "saturation_check": {"max_cores": max_cores},
        "bias_sub": {"run": run_bias_sub, "max_cores": max_cores},
        "nonlin_corr": {"run": run_nonlin_corr, "max_cores": max_cores},
        "jump_detec": {
            "run": run_jump_detec,
            "method": "tpd",
            "max_cores": max_cores,
            "rejection_threshold": 4.0,
        },
        "ramp_fit": {
            "method": ramp_fit_method,
            "max_cores": max_cores,
            "jwst_jump_detection": True,
            "jwst_jump_rejection_threshold": 4.0,
        },
        "dark_sub": {"max_cores": max_cores},
        "gain_corr": {"max_cores": max_cores},
        "detflat_corr": {"max_cores": max_cores},
    }


def create_dummy_fiberflat_rate_files(
    fibers: list[int],
    spectrograph: str,
    output_dir: str,
    calibration_models: dict[str, datamodels.CalibrationModel],
    *,
    flat_scale: float = 1e-14,
    flat_temperature_k: float = 1000.0,
    readtime: float = 1.7,
    n_reads: int = 10,
    max_cores: int = 1,
    stage0_prim_args: dict | None = None,
    fiberflat_filename: callable | None = None,
) -> list[str]:
    """
    Create dummy fiber flat rate FITS files using the same create_ramp / Stage 0 path as the science ramp.

    For each requested fiber (1, 2, 3), a blackbody spectrum at the given temperature (0.8–3.0 µm,
    optionally scaled) is placed on that fiber only, a rate map is built, a ramp is simulated,
    Stage 0 is run, and the rate product is saved to output_dir. The resulting file paths can be
    passed to :func:`run_fiberflat_pipeline`.

    Parameters
    ----------
    fibers : list of int
        Fiber numbers (1, 2, 3) to create fiber flat rate files for.
    spectrograph : str
        Spectrograph name (e.g. "BSPEC", "RSPEC").
    output_dir : str
        Directory to save each rate FITS file.
    calibration_models : dict
        Calibration models (same as used for the science ramp). Must contain
        dq, saturation, bias, gain, rn, dark, detflat.
    flat_scale : float, optional
        Scale factor for the fiber flat spectrum flux. Default is 1e-14.
    flat_temperature_k : float, optional
        Blackbody temperature in Kelvin for the fiber flat spectrum. Default is 1000.0.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads in each ramp. Default is 10.
    max_cores : int, optional
        Cores for create_ramp and Stage 0. Default is 1.
    stage0_prim_args : dict or None, optional
        Primitive arguments for Stage 0. If None, use :func:`get_default_stage0_prim_args`(max_cores).
    fiberflat_filename : callable or None, optional
        (spectrograph, fiber) -> filename. If None, use default KOA-style naming.

    Returns
    -------
    list of str
        Paths to the saved rate FITS files (one per fiber), in fiber order.
    """
    import astropy.units as u
    from astropy.modeling.models import BlackBody

    if stage0_prim_args is None:
        stage0_prim_args = get_default_stage0_prim_args(max_cores=max_cores)
    if fiberflat_filename is None:
        fiberflat_filename = _fiberflat_rate_filename

    # Blackbody spectrum 0.8–3.0 µm at high resolution to avoid interpolation errors in the simulation
    n_wl_flat = 100_000
    wl_flat = np.linspace(0.8, 3.0, n_wl_flat) * u.micron
    wl_aa = wl_flat.to(u.AA)
    bb_scale = 1.0 * u.erg / (u.cm**2 * u.s * u.AA * u.sr)
    bb = BlackBody(temperature=flat_temperature_k * u.K, scale=bb_scale)
    bb_lam_aa = bb(wl_aa)
    # BlackBody returns surface brightness (per sr); use .value and attach flux-density unit (sr folded into flat_scale)
    # Convert per-Angstrom to per-micron: 1 micron = 1e4 AA => same flux density => multiply by 1e4
    flux_flat = (flat_scale * np.asarray(bb_lam_aa.value) * 1e4) * (u.erg / u.s / u.cm**2 / u.micron)

    out_dir_abs = os.path.abspath(output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    paths: list[str] = []

    for f in sorted(fibers):
        if f not in (1, 2, 3):
            continue
        logger.info("Creating dummy fiber flat rate file for fiber %s.", f)
        srm = SimulateRateMap()
        srm.output_directory = out_dir_abs
        srm.set_fiber_spectrum(
            fiber_number=f,
            spectrum_wavelengths=wl_flat,
            spectrum_flux=flux_flat,
            continuous=True,
        )
        rate_map = srm.create_native_rate_map(spectrograph=spectrograph, method="prf_convolution")
        ramp_model = simulate_ramp_from_rate_map(
            rate_map,
            calibration_models,
            readtime=readtime,
            n_reads=n_reads,
            max_cores=max_cores,
            spectrograph=spectrograph,
            poisson_noise=True,
            convert_to_uint16=True,
        )
        rate_model = run_stage0_simulation(
            ramp_model,
            calibration_models,
            output_dir=out_dir_abs,
            prim_args=stage0_prim_args,
        )
        # Set object_fiber meta so group_by_illuminated_fiber (fiber flat pipeline) finds this fiber
        obj_flat = "FIBERFLAT-B" if spectrograph.upper() == "BSPEC" else "FIBERFLAT-R"
        for fib in (1, 2, 3):
            setattr(rate_model.meta, f"object_fiber{fib}", obj_flat if fib == f else "BLOCKED")
        filename = fiberflat_filename(spectrograph, f)
        filepath = os.path.join(out_dir_abs, filename)
        rate_model.save(filepath=filepath)
        paths.append(filepath)
        logger.debug("Saved dummy fiber flat rate file: %s", filepath)

    return paths


def run_stage0_simulation(
    ramp_model: datamodels.RampModel,
    calibration_models: dict[str, datamodels.CalibrationModel],
    output_dir: str = ".",
    prim_args: dict | None = None,
) -> datamodels.DataModel:
    """
    Run the Stage 0 pipeline on a simulated ramp with the given calibrations.

    Parameters
    ----------
    ramp_model : RampModel
        Simulated ramp from :func:`simulate_ramp_from_rate_map`.
    calibration_models : dict
        Output of :func:`create_simulated_calibration_models`. Must contain
        ``dq``, ``saturation``, ``bias``, ``gain``, ``rn``, ``dark``, ``detflat``.
    output_dir : str, optional
        Directory for pipeline outputs. Default is ".".
    prim_args : dict or None, optional
        Primitive arguments for the pipeline. If None, use
        :func:`get_default_stage0_prim_args`(). Default is None.

    Returns
    -------
    DataModel
        Result of Stage0Pipeline.apply (e.g. rate or slope model).
    """
    if prim_args is None:
        prim_args = get_default_stage0_prim_args()

    logger.info("Running Stage 0 pipeline (output_dir=%s).", output_dir)
    pipe = Stage0Pipeline()
    pipe.dq_init.dq = calibration_models["dq"]
    pipe.saturation_check.saturation = calibration_models["saturation"]
    pipe.bias_sub.bias = calibration_models["bias"]
    pipe.ramp_fit.gain = calibration_models["gain"]
    pipe.ramp_fit.rn = calibration_models["rn"]
    pipe.dark_sub.dark = calibration_models["dark"]
    pipe.gain_corr.gain = calibration_models["gain"]
    pipe.detflat_corr.detflat = calibration_models["detflat"]

    return pipe.apply(ramp_model, prim_args=prim_args, output_dir=output_dir)


def fetch_svo_2mass_filter(band: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch 2MASS filter profile (wavelength in Angstrom, transmission) from the
    SVO Filter Profile Service.

    Parameters
    ----------
    band : str
        One of 'J', 'H', 'K'.

    Returns
    -------
    wavelength_angstrom : np.ndarray
        Wavelength in Angstrom.
    transmission : np.ndarray
        Dimensionless transmission (0--1).
    """
    band = band.upper()
    if band not in ("J", "H", "K"):
        raise ValueError("band must be one of 'J', 'H', 'K'")
    url = SVO_FILTER_URL.format(band=band)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = np.loadtxt(resp, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected SVO filter data shape for 2MASS.{band}")
    wavelength_angstrom = data[:, 0]
    transmission = data[:, 1]
    return wavelength_angstrom, transmission


def _ab_mag_through_filter(
    wavelength_um: np.ndarray,
    flux_erg_scm2_angstrom: np.ndarray,
    filter_wl_angstrom: np.ndarray,
    filter_transmission: np.ndarray,
) -> float:
    """
    Compute AB magnitude of a spectrum through a filter.

    Spectrum: wavelength in micron, flux in erg/s/cm^2/Angstrom (same units as
    BT-Settl file). Filter: wavelength in Angstrom, dimensionless transmission.
    Uses the bandpass-weighted mean f_nu and m_AB = -2.5*log10(f_nu) - 48.60
    (f_nu in erg/s/cm^2/Hz).

    Parameters
    ----------
    wavelength_um : np.ndarray
        Spectrum wavelength in micron.
    flux_erg_scm2_angstrom : np.ndarray
        Spectrum flux in erg/s/cm^2/Angstrom (no units object).
    filter_wl_angstrom : np.ndarray
        Filter wavelength in Angstrom.
    filter_transmission : np.ndarray
        Filter transmission (dimensionless).

    Returns
    -------
    float
        AB magnitude in the band.
    """
    # Common grid: use spectrum wavelength in Angstrom
    wl_A = np.asarray(wavelength_um, dtype=np.float64) * 1.0e4
    flux = np.asarray(flux_erg_scm2_angstrom, dtype=np.float64)
    # Interpolate filter transmission onto spectrum wavelength
    trans = np.interp(wl_A, filter_wl_angstrom, filter_transmission)
    # Restrict to where filter has support
    valid = trans > 1.0e-6
    if not np.any(valid):
        raise ValueError("Spectrum and filter wavelength ranges do not overlap")
    wl = wl_A[valid]
    fl = flux[valid]
    tr = trans[valid]
    # f_nu(lambda) = F_lambda * lambda^2 / c  (c in A/s -> f_nu in erg/s/cm^2/Hz)
    c_angstrom_per_s = 2.99792458e18
    f_nu = fl * (wl**2) / c_angstrom_per_s
    # Bandpass-weighted mean f_nu
    num = np.trapz(f_nu * tr, wl)
    den = np.trapz(tr, wl)
    if den <= 0:
        raise ValueError("Filter transmission integral is non-positive")
    f_nu_band = num / den
    # AB mag: m_AB = -2.5*log10(f_nu) - 48.60  (Oke & Gunn 1983, f_nu in erg/s/cm^2/Hz)
    return float(-2.5 * np.log10(f_nu_band) - 48.60)


def get_spectrum_path(path: str) -> str:
    """
    Resolve a spectrum file path relative to the simulations data directory.

    If ``path`` is absolute, it is returned unchanged. Otherwise the path is
    resolved relative to ``hispecdrp/simulations/data/`` (e.g. for bundled
    BT-Settl .h5 files).

    Parameters
    ----------
    path : str
        Filename (e.g. lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5) or absolute path.

    Returns
    -------
    str
        Absolute path to the spectrum file.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(SIMULATIONS_DATA_DIR, path)


def load_btsettl_spectrum(
    filepath: str,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    mag_ab: float | None = None,
    filter_band: str | None = None,
):
    """
    Load wavelength and flux from a BT-Settl spectrum HDF5 file.

    Adapted from create_rate_map.py. The file is expected to have 'wavelength'
    (in Angstrom) and 'flux' (in 1e-23 erg/cm^2/s/Angstrom) datasets. Values
    are converted to micron and astropy units, and clipped to the given
    wavelength range.

    If ``mag_ab`` and ``filter_band`` are set, the spectrum is normalized to
    the given AB magnitude in the corresponding 2MASS filter (J, H, or K).
    The 2MASS profile is fetched from the SVO Filter Profile Service.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file (e.g. lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5).
    wl_min_um : float, optional
        Minimum wavelength in microns. Default is 0.8.
    wl_max_um : float, optional
        Maximum wavelength in microns. Default is 3.0.
    mag_ab : float or None, optional
        If set with ``filter_band``, normalize the spectrum to this AB magnitude
        in that band. Default is None.
    filter_band : str or None, optional
        One of 'J', 'H', 'K'. Used with ``mag_ab`` to normalize. Default is None.

    Returns
    -------
    wavelength : astropy.units.Quantity
        Wavelength array in micron.
    flux : astropy.units.Quantity
        Flux density in erg/s/cm^2/Angstrom (same as file).
    """
    if h5py is None:
        raise ImportError("h5py is required to load BT-Settl spectrum; install with pip install h5py")
    import astropy.units as u

    logger.debug("Loading BT-Settl spectrum from %s (wl=%.2f--%.2f um)", filepath, wl_min_um, wl_max_um)
    with h5py.File(filepath, "r") as f:
        wavelength = f["wavelength"][:] * 1.0e-4 * u.micron
        flux = f["flux"][:] * 1.0e-23 * (u.erg / u.cm**2 / u.s / u.Angstrom)

    mask = (wavelength >= wl_min_um * u.micron) & (wavelength <= wl_max_um * u.micron)
    wavelength = wavelength[mask]
    flux = flux[mask]
    logger.debug("Loaded %d wavelength points from %s", len(wavelength), filepath)

    if mag_ab is not None and filter_band is not None:
        filter_wl_A, filter_trans = fetch_svo_2mass_filter(filter_band)
        wl_um = wavelength.to(u.micron).value
        flux_val = flux.to(u.erg / u.cm**2 / u.s / u.Angstrom).value
        m_current = _ab_mag_through_filter(wl_um, flux_val, filter_wl_A, filter_trans)
        scale = 10.0 ** (-0.4 * (mag_ab - m_current))
        flux = flux * scale
        logger.info(
            "Normalized spectrum to %s=%.2f (AB mag); scale=%.4e (was %.2f mag)",
            filter_band,
            mag_ab,
            scale,
            m_current,
        )

    return wavelength, flux


def create_ramp_and_calibrations(
    rate_map: np.ndarray | None = None,
    shape: tuple[int, int] = (4096, 4096),
    *,
    fibers: list[int] | None = None,
    spectrum_wavelengths=None,
    spectrum_flux=None,
    output_dir: str = ".",
    calibration_models: dict[str, datamodels.CalibrationModel] | None = None,
    use_master_calibrations: bool = False,
    readtime: float = 1.7,
    n_reads: int = 10,
    spectrograph: str = "RSPEC",
    max_cores: int = 1,
    poisson_noise: bool = True,
    convert_to_uint16: bool = True,
    instrument_name: str | None = None,
    instrument_era: str = "0.0.1",
    mjd_start: float | None = None,
    datetime_obs: str | None = None,
    connect_remote: bool = True,
    sync_caldb_first: bool = True,
    **cal_kw,
) -> tuple[datamodels.RampModel, dict[str, datamodels.CalibrationModel]]:
    """
    Create calibration models and a simulated ramp (no Stage 0 pipeline).

    Use this to separate simulation setup from running Stage 0. If ``rate_map``
    is None, a default rate map is generated. Calibrations are either fetched
    from the store (when ``use_master_calibrations`` is True), passed in via
    ``calibration_models``, or created via
    :func:`create_simulated_calibration_models`(shape, **cal_kw).

    Parameters
    ----------
    rate_map : np.ndarray or None, optional
        2D rate map in e-/s/pixel. If None, a default is created via SimulateRateMap.
    shape : tuple of int, optional
        (ny, nx) for dummy calibrations when ``calibration_models`` is None and
        ``use_master_calibrations`` is False. Default is (4096, 4096).
    fibers : list of int or None, optional
        Fiber numbers (1, 2, 3, and/or 4) to place spectra on when building the
        default rate map. Only 1, 2, 3 are supported by SimulateRateMap; others
        are ignored for rate map creation. If None, default is [1].
    spectrum_wavelengths : array-like with units or None, optional
        Wavelength array for the input spectrum when building the default rate
        map. If None (and spectrum_flux is None), a flat spectrum is used.
    spectrum_flux : array-like with units or None, optional
        Flux array for the input spectrum. Must be provided with
        spectrum_wavelengths to use an input spectrum.
    output_dir : str, optional
        Directory for saving the rate map when building the default rate map
        (when ``rate_map`` is None). Default is ".".
    calibration_models : dict or None, optional
        Calibration models to use. If None, master or dummy models are used.
    use_master_calibrations : bool, optional
        If True and ``calibration_models`` is None, fetch from calibration store.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads. Default is 10.
    spectrograph : str, optional
        Spectrograph name. Default is "RSPEC".
    max_cores : int, optional
        Cores for ramp creation. Default is 1.
    poisson_noise : bool, optional
        Whether to add Poisson noise in create_ramp. Default is True.
    convert_to_uint16 : bool, optional
        Whether to convert ramp data to uint16. Default is True.
    instrument_name, instrument_era, mjd_start, datetime_obs : optional
        Used when ``use_master_calibrations`` is True.
    connect_remote : bool, optional
        Whether to connect to remote when fetching master calibrations.
    sync_caldb_first : bool, optional
        When using master calibrations, sync store first.
    **cal_kw
        Passed to :func:`create_simulated_calibration_models` when creating dummy calibrations.

    Returns
    -------
    tuple of (RampModel, dict)
        (ramp_model, calibration_models). Pass these to :func:`run_stage0_simulation`
        to run the Stage 0 pipeline.
    """
    if calibration_models is None:
        if use_master_calibrations:
            logger.info("Fetching master calibrations from calibration store (spectrograph=%s).", spectrograph)
            calibration_models = get_master_calibrations(
                meta=None,
                instrument_name=instrument_name,
                instrument_era=instrument_era,
                spectrograph=spectrograph,
                mjd_start=mjd_start,
                datetime_obs=datetime_obs,
                connect_remote=connect_remote,
                sync_first=sync_caldb_first,
            )
        else:
            logger.debug("Creating simulated calibration models (shape=%s).", shape)
            calibration_models = create_simulated_calibration_models(shape, **cal_kw)

    if rate_map is None:
        if fibers is None:
            fibers = [1]
        logger.info("Building default rate map (fibers=%s, spectrograph=%s, output_dir=%s).", fibers, spectrograph, output_dir)
        srm = SimulateRateMap()
        srm.output_directory = os.path.abspath(output_dir)
        srm.rate_map_outname = DEFAULT_RATE_MAP_FILENAME
        import astropy.units as u

        if spectrum_wavelengths is not None and spectrum_flux is not None:
            wl = spectrum_wavelengths
            flux = spectrum_flux
        else:
            wl = np.arange(0.9, 2.5, 2.5e-6) * u.micron
            flux = np.ones_like(wl.value) * 1e-12 * u.erg / u.s / u.cm**2 / u.micron
        # SimulateRateMap only supports fibers 1, 2, 3
        for f in fibers:
            if f in (1, 2, 3):
                srm.set_fiber_spectrum(fiber_number=f, spectrum_wavelengths=wl, spectrum_flux=flux, continuous=True)
        rate_map = srm.create_native_rate_map(spectrograph=spectrograph, method="prf_convolution")
        logger.debug("Rate map shape: %s", getattr(rate_map, "shape", None))

    logger.info("Simulating ramp (n_reads=%s, readtime=%s s).", n_reads, readtime)
    ramp_model = simulate_ramp_from_rate_map(
        rate_map,
        calibration_models,
        readtime=readtime,
        n_reads=n_reads,
        max_cores=max_cores,
        spectrograph=spectrograph,
        poisson_noise=poisson_noise,
        convert_to_uint16=convert_to_uint16,
    )
    return ramp_model, calibration_models


def simulate_up_the_ramp(
    rate_map: np.ndarray | None = None,
    shape: tuple[int, int] = (4096, 4096),
    *,
    fibers: list[int] | None = None,
    spectrum_wavelengths=None,
    spectrum_flux=None,
    output_dir: str = ".",
    calibration_models: dict[str, datamodels.CalibrationModel] | None = None,
    use_master_calibrations: bool = False,
    readtime: float = 1.7,
    n_reads: int = 10,
    spectrograph: str = "RSPEC",
    max_cores: int = 1,
    poisson_noise: bool = True,
    convert_to_uint16: bool = True,
    stage0_prim_args: dict | None = None,
    instrument_name: str | None = None,
    instrument_era: str = "0.0.1",
    mjd_start: float | None = None,
    datetime_obs: str | None = None,
    connect_remote: bool = True,
    sync_caldb_first: bool = True,
    **cal_kw,
) -> tuple[datamodels.RampModel, datamodels.DataModel, dict]:
    """
    High-level up-the-ramp simulation: create calibrations, ramp, and run Stage 0.

    If ``rate_map`` is None, a simple rate map is generated using
    :class:`SimulateRateMap` with a flat spectrum on fiber 1 (for testing).
    Otherwise the provided rate map (e-/s/pixel) is used.

    Calibrations are either (1) taken from the calibration store (master
    calibrations) when ``use_master_calibrations`` is True, (2) passed in via
    ``calibration_models``, or (3) created as dummy models via
    :func:`create_simulated_calibration_models`(shape, **cal_kw).

    Parameters
    ----------
    rate_map : np.ndarray or None, optional
        2D rate map in e-/s/pixel. If None, a default rate map is created
        via SimulateRateMap. Default is None.
    shape : tuple of int, optional
        (ny, nx) for creating dummy calibration models when
        ``calibration_models`` is None and ``use_master_calibrations`` is False.
        Default is (4096, 4096).
    fibers : list of int or None, optional
        Fiber numbers (1, 2, 3, and/or 4) for default rate map. Passed to
        :func:`create_ramp_and_calibrations`. If None, default is [1].
    spectrum_wavelengths : array-like with units or None, optional
        Input spectrum wavelengths for default rate map. Passed to
        :func:`create_ramp_and_calibrations`.
    spectrum_flux : array-like with units or None, optional
        Input spectrum flux. Passed to :func:`create_ramp_and_calibrations`.
    output_dir : str, optional
        Output directory for Stage 0. Default is ".".
    calibration_models : dict or None, optional
        Calibration models to use. If None, either master calibrations are
        fetched (when ``use_master_calibrations`` is True) or dummy models
        are created. Default is None.
    use_master_calibrations : bool, optional
        If True and ``calibration_models`` is None, fetch DQ, saturation,
        gain, detflat, bias, dark, and read noise from the calibration store
        via :func:`run_caldb.get_master_calibrations`. Requires KOA_CALIBRATION_CACHE
        and (for remote) synced/downloaded calibrations. Default is False.
    readtime : float, optional
        Time between reads in seconds. Default is 1.7.
    n_reads : int, optional
        Number of reads. Default is 10.
    spectrograph : str, optional
        Spectrograph name. Default is "RSPEC".
    max_cores : int, optional
        Cores for ramp creation and Stage 0. Default is 1.
    poisson_noise : bool, optional
        Whether to add Poisson noise when creating the ramp. Default is True.
    convert_to_uint16 : bool, optional
        Whether to convert ramp data to uint16. Default is True.
    stage0_prim_args : dict or None, optional
        Primitive args for Stage 0. If None, use default. Default is None.
    instrument_name : str or None, optional
        Instrument name for the calibration store. Used when
        ``use_master_calibrations`` is True. Default is None.
    instrument_era : str, optional
        Instrument era for calibration selection. Used when
        ``use_master_calibrations`` is True. Default is "0.0.1".
    mjd_start : float or None, optional
        MJD start for calibration selection. Used when
        ``use_master_calibrations`` is True. Default is None (current time).
    datetime_obs : str or None, optional
        ISO datetime for calibration selection. Used when
        ``use_master_calibrations`` is True if mjd_start is None.
        Default is None.
    connect_remote : bool, optional
        Whether to connect to remote when fetching master calibrations.
        Default is True.
    sync_caldb_first : bool, optional
        When ``use_master_calibrations`` is True, sync from remote and add
        local cache files to the DB before selecting. Default is True.
    **cal_kw
        Passed to :func:`create_simulated_calibration_models` when
        ``calibration_models`` is None and ``use_master_calibrations`` is False.

    Returns
    -------
    tuple of (RampModel, DataModel, dict)
        (ramp_model, stage0_result_model, calibration_models).
    """
    logger.info(
        "simulate_up_the_ramp: rate_map=%s, use_master_calibrations=%s, output_dir=%s.",
        "provided" if rate_map is not None else "default",
        use_master_calibrations,
        output_dir,
    )
    ramp_model, calibration_models = create_ramp_and_calibrations(
        rate_map=rate_map,
        shape=shape,
        fibers=fibers,
        spectrum_wavelengths=spectrum_wavelengths,
        spectrum_flux=spectrum_flux,
        output_dir=output_dir,
        calibration_models=calibration_models,
        use_master_calibrations=use_master_calibrations,
        readtime=readtime,
        n_reads=n_reads,
        spectrograph=spectrograph,
        max_cores=max_cores,
        poisson_noise=poisson_noise,
        convert_to_uint16=convert_to_uint16,
        instrument_name=instrument_name,
        instrument_era=instrument_era,
        mjd_start=mjd_start,
        datetime_obs=datetime_obs,
        connect_remote=connect_remote,
        sync_caldb_first=sync_caldb_first,
        **cal_kw,
    )

    if stage0_prim_args is None:
        stage0_prim_args = get_default_stage0_prim_args(max_cores=max_cores)

    logger.info("Running Stage 0 simulation.")
    result = run_stage0_simulation(
        ramp_model,
        calibration_models,
        output_dir=output_dir,
        prim_args=stage0_prim_args,
    )
    return ramp_model, result, calibration_models


def _patch_fiberflat_model_object_fiber_meta(model: datamodels.DataModel, filepath: str) -> None:
    """
    Set object_fiber1/2/3 on model.meta from filename so group_by_illuminated_fiber finds fibers.

    Existing FITS (e.g. from disk or GDrive) may lack these; the fiber flat pipeline and
    trace primitives require them. Infers fiber from ...FIB1.../FIB2.../FIB3... and
    FIBERFLAT-B vs -R from HB vs HR in the path.
    """
    name = os.path.basename(filepath).upper()
    # Infer illuminated fiber from filename (e.g. rate-FIBERFLAT-FIB1.fits or ...FIB1...)
    match = re.search(r"FIB(?:ERFLAT-)?FIB([123])", name) or re.search(r"FIB([123])\b", name)
    fiber = int(match.group(1)) if match else 1
    obj_flat = "FIBERFLAT-B" if "HB" in name or "BSPEC" in name else "FIBERFLAT-R"
    for f in (1, 2, 3):
        setattr(model.meta, f"object_fiber{f}", obj_flat if f == fiber else "BLOCKED")


def run_fiberflat_pipeline(
    fiberflat_filepaths: list[str],
    prim_args: dict | None = None,
    *,
    yrange: tuple[int, int] = (500, 4000),
    xrange: tuple[int, int] = (100, 4050),
    deg_pos: int = 4,
    deg_width: int = 2,
    extraction_aperture: int = 10,
) -> dict[str, datamodels.DataModel]:
    """
    Run the ProcessFiberflatsPipeline on a list of fiberflat rate files.

    Parameters
    ----------
    fiberflat_filepaths : list of str
        Paths to fiberflat rate FITS files (one per fiber).
    prim_args : dict or None, optional
        Primitive arguments. If None, built from the other keyword arguments.
        Default is None.
    yrange : tuple of int, optional
        Fitting range in spectral direction (pixels). Default is (500, 4000).
    xrange : tuple of int, optional
        XD direction range. Default is (100, 4050).
    deg_pos : int, optional
        Polynomial degree for trace position. Default is 4.
    deg_width : int, optional
        Polynomial degree for trace width. Default is 2.
    extraction_aperture : int, optional
        Extraction aperture in pixels. Default is 10.

    Returns
    -------
    dict
        Pipeline results with keys such as ``trace_pos``, ``trace_prof``.
    """
    logger.info("Running fiberflat pipeline on %d file(s).", len(fiberflat_filepaths))
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]
    for model, fp in zip(ff_models, fiberflat_filepaths):
        _patch_fiberflat_model_object_fiber_meta(model, fp)
    if prim_args is None:
        do_orders = ff_models[0].make_order_vec() if hasattr(ff_models[0], "make_order_vec") else None
        # Restrict extraction to fibers present in the fiberflat inputs (trace pos/prof only exist for these)
        ff_by_fiber = datamodels.group_by_illuminated_fiber(ff_models)
        do_fibers = sorted(ff_by_fiber.keys(), key=int)
        prim_args = {
            "trace_pos": {"deg_pos": deg_pos, "deg_width": deg_width, "yrange": yrange, "xrange": xrange},
            "spec_extract": {
                "extraction_aperture": extraction_aperture,
                "do_orders": do_orders,
                "do_fibers": do_fibers,
            },
            "set_wavesol": {"run": False},
            "register_fiberflat_cal": {"run": False},
        }
    pipe = ProcessFiberflatsPipeline(prim_args=prim_args)
    return pipe.apply(ff_models)


def run_stage1_simulation(
    rate_model: datamodels.DataModel,
    trace_pos_model: datamodels.DataModel,
    trace_prof_model: datamodels.DataModel,
    prim_args: dict | None = None,
    *,
    extraction_aperture: int = 10,
    do_orders: list | None = None,
    do_fibers: list[str] | None = None,
) -> datamodels.DataModel:
    """
    Run the Stage 1 pipeline on a rate (or slope) model with trace calibrations.

    Parameters
    ----------
    rate_model : DataModel
        Stage 0 output (rate/slope product).
    trace_pos_model : DataModel
        Trace position model from fiberflat pipeline.
    trace_prof_model : DataModel
        Trace profile model from fiberflat pipeline.
    prim_args : dict or None, optional
        Primitive arguments. If None, built from the other keyword arguments.
        Default is None.
    extraction_aperture : int, optional
        Extraction aperture in pixels. Default is 10.
    do_orders : list or None, optional
        Orders to extract. If None, use order vector from rate model. Default is None.
    do_fibers : list of str or None, optional
        Fibers to extract (e.g. ['1','2','3']). Default is None.

    Returns
    -------
    DataModel
        Stage 1 spectral extraction result.
    """
    if do_orders is None and hasattr(rate_model, "make_order_vec"):
        do_orders = rate_model.make_order_vec()
    if do_orders is None:
        do_orders = []
    if do_fibers is None:
        do_fibers = ["1", "2", "3"]
    
    if prim_args is not None:
        method = prim_args.get("spec_extract", {}).get("method", "box")
    else:
        method = "box"

    logger.info("Running Stage 1 simulation (do_orders=%s, do_fibers=%s, method=%s).", do_orders, do_fibers, method)
    if prim_args is None:
        prim_args = {
            "spec_extract": {
                "method": method,
                "extraction_aperture": extraction_aperture,
                "do_orders": do_orders,
                "do_fibers": do_fibers,
                "tracepos": trace_pos_model,
                "traceprof": trace_prof_model,
            },
            "set_wavesol": {"run": False},
        }

    pipe = Stage1Pipeline(prim_args=prim_args)
    return pipe.apply(rate_model)
