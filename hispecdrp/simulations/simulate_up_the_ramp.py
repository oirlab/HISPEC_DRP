"""
Simulate up-the-ramp (UTR) readouts and run Stage 0–2 pipelines for end-to-end tests.

This module provides functions to build simulated calibration models, generate
ramp data from a rate map, and run the Stage 0 pipeline to produce rate/slope
products. Calibrations can be (1) dummy/simulated via
:func:`create_simulated_calibration_models`, or (2) master calibrations
downloaded from the calibration store via :func:`run_caldb.get_master_calibrations`
when ``use_master_calibrations=True`` in :func:`simulate_up_the_ramp`. Optional
helpers for the fiberflat and Stage 1 pipelines are included for full workflows.
"""

from __future__ import annotations

import copy
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

from .. import datamodels
from ..utils.spectral_templates_utils import (
    fetch_svo_2mass_filter,
    get_spectrum_path,
    load_btsettl_spectrum,
)
from ..datamodels.dqflags import DQ_FLAGS
from ..pipelines import Stage0Pipeline, ProcessFiberflatsPipeline
from . import e2e_support
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
    out_abs = e2e_support.calibration_dir(output_dir)
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


def load_rate_map_array(filepath: str) -> np.ndarray:
    """
    Load a 2D rate map from disk.

    Files written by :class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`
    are plain FITS images (no ``DATAMODL`` header), so this uses ``astropy.io.fits``
    rather than :func:`~hispecdrp.datamodels.open`.
    """
    from astropy.io import fits

    data = fits.getdata(filepath, memmap=False)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"Rate map at {filepath!r} must be 2D; got shape {arr.shape}"
        )
    return arr


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
    "load_rate_map_array",
    "get_expected_fiberflat_rate_paths",
    "ramp_file_exists",
    "rate_map_file_exists",
    "fiberflat_rate_files_exist",
    "get_default_stage0_prim_args",
    "run_stage0_simulation",
    "simulate_up_the_ramp",
    "run_fiberflat_pipeline",
    "run_stage1_simulation",
    "run_stage2_simulation",
    "get_default_stage1_prim_args",
    "get_default_stage2_prim_args",
    "create_wavelength_solution_model_from_sim",
    "prepare_fiberflat_models",
    "get_fiberflat_prim_args",
]

# Re-export Stage 1/2 E2E helpers from e2e_support.
run_stage1_simulation = e2e_support.run_stage1_simulation
run_stage2_simulation = e2e_support.run_stage2_simulation
get_default_stage1_prim_args = e2e_support.get_default_stage1_prim_args
get_default_stage2_prim_args = e2e_support.get_default_stage2_prim_args


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
    max_cores: int | None = 1,
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
    mc = max_cores if max_cores is not None else 1
    return {
        "dq_init": {"max_cores": mc},
        "saturation_check": {"max_cores": mc},
        "bias_sub": {"run": run_bias_sub, "max_cores": mc},
        "nonlin_corr": {"run": run_nonlin_corr, "max_cores": mc},
        "jump_detec": {
            "run": run_jump_detec,
            "method": "tpd",
            "max_cores": mc,
            "rejection_threshold": 4.0,
        },
        "ramp_fit": {
            "method": ramp_fit_method,
            "max_cores": mc,
            "jwst_jump_detection": True,
            "jwst_jump_rejection_threshold": 4.0,
        },
        "dark_sub": {"max_cores": mc},
        "gain_corr": {"max_cores": mc},
        "detflat_corr": {"max_cores": mc},
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

    pipeline_root = e2e_support.ensure_output_directory(output_dir)
    out_dir_abs = e2e_support.calibration_dir(pipeline_root)
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
            max_cores=max_cores,
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
    *,
    max_cores: int | None = None,
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
    if max_cores is not None:
        prim_args = e2e_support.apply_max_cores_to_stage0_prim_args(prim_args, max_cores)

    output_dir = e2e_support.ensure_output_directory(output_dir)
    logger.info("Running Stage 0 pipeline (output_dir=%s).", output_dir)
    ny, nx = int(ramp_model.data.shape[1]), int(ramp_model.data.shape[2])
    nonlin_model = calibration_models.get("nonlin") or e2e_support.identity_nonlin_calibration_model(
        ny, nx
    )

    pipe = Stage0Pipeline()
    pipe.dq_init.dq = calibration_models["dq"]
    pipe.saturation_check.saturation = calibration_models["saturation"]
    pipe.bias_sub.bias = calibration_models["bias"]
    pipe.nonlin_corr.nonlin = nonlin_model
    pipe.jump_detec.gain = calibration_models["gain"]
    pipe.jump_detec.rn = calibration_models["rn"]
    pipe.ramp_fit.gain = calibration_models["gain"]
    pipe.ramp_fit.rn = calibration_models["rn"]
    pipe.dark_sub.dark = calibration_models["dark"]
    pipe.gain_corr.gain = calibration_models["gain"]
    pipe.detflat_corr.detflat = calibration_models["detflat"]

    return pipe.apply(ramp_model, prim_args=prim_args, output_dir=output_dir)


def create_ramp_and_calibrations(
    rate_map: np.ndarray | None = None,
    shape: tuple[int, int] = (4096, 4096),
    *,
    fibers: list[int] | None = None,
    spectrum_wavelengths=None,
    spectrum_flux=None,
    radial_velocity_km_s: float = 0.0,
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
    template_mag_ab: float | None = None,
    template_2mass_band: str | None = None,
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
    radial_velocity_km_s : float, optional
        Radial velocity in km/s to apply to the input spectrum before generating
        the rate map (Doppler shift). Must be in the range -500 to 500. Applied
        via SimulateRateMap.apply_radial_velocity before create_native_rate_map.
        Default is 0.0.
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
    template_mag_ab, template_2mass_band : optional
        When building the default rate map from the bundled BT-Settl template, normalize
        to this AB magnitude in the given 2MASS band (``'J'``, ``'H'``, or ``'K'``).
    **cal_kw
        Passed to :func:`create_simulated_calibration_models` when creating dummy calibrations.

    Returns
    -------
    tuple of (RampModel, dict, ndarray)
        ``(ramp_model, calibration_models, rate_map)``. Pass the first two to
        :func:`run_stage0_simulation` to run the Stage 0 pipeline. ``rate_map`` is the
        2D e-/s/pixel array used for ramp simulation (same data as ``rate_map.fits``).
    """
    if (template_mag_ab is not None) ^ (template_2mass_band is not None):
        raise ValueError(
            "template_mag_ab and template_2mass_band must both be set or both be None."
        )
    _band: str | None = None
    if template_2mass_band is not None:
        _band = str(template_2mass_band).strip().upper()
        if _band not in ("J", "H", "K"):
            raise ValueError("template_2mass_band must be one of 'J', 'H', 'K'")

    output_dir = e2e_support.ensure_output_directory(output_dir)
    e2e_support.ensure_koa_calibration_cache_env(output_dir)
    ramp_path = get_expected_ramp_path(output_dir)

    if rate_map is None:
        if fibers is None:
            fibers = list(e2e_support.DEFAULT_RATE_MAP_FIBERS)
        logger.info(
            "Building default rate map (fibers=%s, spectrograph=%s, output_dir=%s).",
            fibers,
            spectrograph,
            output_dir,
        )
        srm = SimulateRateMap()
        srm.output_directory = output_dir
        srm.rate_map_outname = DEFAULT_RATE_MAP_FILENAME
        import astropy.units as u

        if spectrum_wavelengths is not None and spectrum_flux is not None:
            wl = spectrum_wavelengths
            flux = spectrum_flux
        else:
            template_path = get_spectrum_path(
                e2e_support.DEFAULT_TEMPLATE_FILENAME,
                base_dir=e2e_support.templates_data_dir(),
            )
            wl, flux = load_btsettl_spectrum(
                template_path,
                mag_ab=template_mag_ab,
                filter_band=_band,
            )
            wl_val = np.asarray(getattr(wl, "value", wl), dtype=np.float64)
            sort_idx = np.argsort(wl_val)
            wl = wl[sort_idx]
            flux = flux[sort_idx]
        for f in fibers:
            if f in (1, 2, 3):
                srm.set_fiber_spectrum(
                    fiber_number=f,
                    spectrum_wavelengths=wl,
                    spectrum_flux=flux,
                    continuous=True,
                )
        if radial_velocity_km_s != 0.0:
            if not (-500.0 <= radial_velocity_km_s <= 500.0):
                raise ValueError(
                    f"radial_velocity_km_s must be in [-500, 500], got {radial_velocity_km_s}"
                )
            rv_quantity = radial_velocity_km_s * u.km / u.s
            for f in fibers:
                if f in (1, 2, 3):
                    srm.apply_radial_velocity(fiber_number=f, rv=rv_quantity)
        rate_map = srm.create_native_rate_map(
            spectrograph=spectrograph, method="prf_convolution"
        )
        logger.debug("Rate map shape: %s", getattr(rate_map, "shape", None))

    if calibration_models is None:
        if use_master_calibrations:
            logger.info(
                "Fetching master calibrations from calibration store (spectrograph=%s).",
                spectrograph,
            )
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
            rm = np.asarray(rate_map, dtype=np.float64)
            if rm.ndim != 2:
                raise ValueError(
                    f"rate_map must be 2D when creating dummy calibrations; got shape {rm.shape}"
                )
            eff_shape = (int(rm.shape[0]), int(rm.shape[1]))
            keys = ("dq", "saturation", "gain", "detflat", "bias", "dark", "rn")
            cal_models: dict[str, datamodels.CalibrationModel] = {}
            for k in keys:
                fp = e2e_support.dummy_cal_path(output_dir, k)
                if os.path.isfile(fp):
                    cal_models[k] = datamodels.open(fp)
            missing = [k for k in keys if k not in cal_models]
            if missing:
                logger.debug(
                    "Creating simulated calibration models (shape=%s); missing=%s",
                    eff_shape,
                    missing,
                )
                new_models = create_simulated_calibration_models(eff_shape, **cal_kw)
                cal_models |= new_models
                for k in missing:
                    try:
                        cal_models[k].save(filepath=e2e_support.dummy_cal_path(output_dir, k))
                    except Exception as e:
                        logger.warning("Could not save dummy calibration %s: %s", k, e)
            calibration_models = cal_models

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
    if not os.path.isfile(ramp_path):
        ramp_model.save(filepath=ramp_path)
    return ramp_model, calibration_models, np.asarray(rate_map, dtype=np.float64)


def simulate_up_the_ramp(
    rate_map: np.ndarray | None = None,
    shape: tuple[int, int] = (4096, 4096),
    *,
    fibers: list[int] | None = None,
    spectrum_wavelengths=None,
    spectrum_flux=None,
    radial_velocity_km_s: float = 0.0,
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
    radial_velocity_km_s : float, optional
        Radial velocity in km/s (-500 to 500) to apply to the spectrum when
        building the rate map. Passed to :func:`create_ramp_and_calibrations`.
        Default is 0.0.
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
    ramp_model, calibration_models, _rate_map = create_ramp_and_calibrations(
        rate_map=rate_map,
        shape=shape,
        fibers=fibers,
        spectrum_wavelengths=spectrum_wavelengths,
        spectrum_flux=spectrum_flux,
        radial_velocity_km_s=radial_velocity_km_s,
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


def prepare_fiberflat_models(fiberflat_filepaths: list[str]) -> list[datamodels.DataModel]:
    """
    Open fiberflat rate FITS files and patch object_fiber meta so pipelines can group by fiber.

    Parameters
    ----------
    fiberflat_filepaths : list of str
        Paths to fiberflat rate FITS files (one per fiber).

    Returns
    -------
    list of DataModel
        Opened models with object_fiber# meta set.
    """
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]
    for model, fp in zip(ff_models, fiberflat_filepaths):
        _patch_fiberflat_model_object_fiber_meta(model, fp)
    return ff_models


def get_fiberflat_prim_args(
    ff_models: list[datamodels.DataModel],
    *,
    yrange: tuple[int, int] = (500, 4000),
    xrange: tuple[int, int] = (100, 4050),
    deg_pos: int = 4,
    deg_width: int = 2,
    extraction_aperture: int = 10,
) -> dict:
    """Build default primitive arguments for ProcessFiberflatsPipeline from opened fiberflat models."""
    do_orders = ff_models[0].make_order_vec() if hasattr(ff_models[0], "make_order_vec") else None
    ff_by_fiber = datamodels.group_by_illuminated_fiber(ff_models)
    do_fibers = sorted(ff_by_fiber.keys(), key=int)
    return {
        "trace_pos": {"deg_pos": deg_pos, "deg_width": deg_width, "yrange": yrange, "xrange": xrange},
        "spec_extract": {
            "extraction_aperture": extraction_aperture,
            "do_orders": do_orders,
            "do_fibers": do_fibers,
        },
        "set_wavesol": {"run": False},
        "register_fiberflat_cal": {"run": False},
    }


def run_fiberflat_pipeline(
    fiberflat_filepaths: list[str],
    prim_args: dict | None = None,
    *,
    output_dir: str | None = None,
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
    output_dir : str or None, optional
        If set, pipeline products are written under this directory. Default is None.
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
    if output_dir is not None:
        e2e_support.ensure_koa_calibration_cache_env(output_dir)
    logger.info("Running fiberflat pipeline on %d file(s).", len(fiberflat_filepaths))
    ff_models = prepare_fiberflat_models(fiberflat_filepaths)
    if prim_args is None:
        prim_args = get_fiberflat_prim_args(
            ff_models,
            yrange=yrange,
            xrange=xrange,
            deg_pos=deg_pos,
            deg_width=deg_width,
            extraction_aperture=extraction_aperture,
        )
    pipe = ProcessFiberflatsPipeline(prim_args=prim_args)
    apply_kw: dict = {}
    if output_dir is not None:
        apply_kw["output_dir"] = e2e_support.ensure_output_directory(output_dir)
    return pipe.apply(ff_models, **apply_kw)
