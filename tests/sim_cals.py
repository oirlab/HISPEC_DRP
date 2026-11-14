"""Build the simulated HISPEC calibrations shared across the pytest suite.

Port of the scripts ``create_hispec_detector_cals.py``, ``create_hispec_wavesol_cals.py``,
``create_hispec_fiberflat_cals.py`` and ``create_hispec_lfc_cals.py``. Every calibration is
registered in the calibration store at ``KOA_CALIBRATION_CACHE``, which writes it to disk, so
only file paths are kept. ``pytest_sessionstart`` in ``tests/conftest.py`` points the store at this
cal DB, building it if needed, so tests find the cals through the store.

Calibrations built by `build_hispec_cals`, in dependency order:

1. Simulated wavelength solution from the HDF model (BSPEC and RSPEC), needed by the fiberflat pipeline.
2. Trace positions, trace profiles and fiberflat (BSPEC and RSPEC) from simulated fiberflat rate maps.
3. LFC wavesol and LSF (BSPEC and RSPEC) from noisy simulated LFC rate maps, extracted with the cals above.

Not built yet: the detector cals (`make_stage0_cals`, BSPEC).

`make_gascell_L0` simulates gas cell inputs for tests; nothing from it goes in the cal DB.
"""
from __future__ import annotations

import json
import os
import shutil

import astropy.time
import astropy.units as u
import numpy as np

from hispecdrp import (
    ProcessFiberflatsPipeline,
    ProcessLFCPipeline,
    Stage1Pipeline,
    datamodels,
)
from hispecdrp.calibrations import HISPECCalibrationStore
from hispecdrp.primitives.blaze_correction.blaze_correction_primitive import (
    lamp_spectrum_shape,
)
from hispecdrp.primitives.line_fitting.fit_lfc_lines_primitive import gen_lfc_modes
from hispecdrp.primitives.wavesol.gascell_wavesol_primitive import _load_fts_reference
from hispecdrp.simulations.simulate_rate_map import SimulateRateMap
from tests.utils import add_noise_L0

#DATETIME_OBS = "2026-08-26T00:00:00.000"
from datetime import datetime, timezone
DATETIME_OBS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
SHAPE = datamodels.HISPECConfig.DETECTOR_SIZE
FIBERS = [i for i in range(1, datamodels.HISPECConfig.num_fibers+1)]
LFC_NFRAMES = 3
LFC_MAX_SNR = 200


def _register(model, register: bool = True):
    """Register model in the calibration store and return its path, or return the model if register is False."""
    if not register:
        return model
    with HISPECCalibrationStore(connect_remote=False) as store:
        path, _ = store.register_calibration(model)
    return path

#########################
#### Detector cals ######
#########################

def make_dq_array_refchan(
    shape : tuple[int, int] | None = None,
    refpix : bool = True
) -> np.ndarray:
    """DQ map with the 4 reference pixels on every edge and the 64-column reference channel flagged."""
    if shape is None:
        shape = SHAPE
    dq = np.zeros(shape, dtype=np.uint32)
    if refpix:
        Nrefpix = 4
        channel_size = 64
        refpix_dq_val = datamodels.DQ_FLAGS['REFERENCE_PIXEL']
        dq[:, :Nrefpix] |= refpix_dq_val
        dq[:, -Nrefpix:] |= refpix_dq_val
        dq[:Nrefpix, :] |= refpix_dq_val
        dq[-Nrefpix:, :] |= refpix_dq_val
        dq[:, :channel_size] |= refpix_dq_val
    return dq


def make_stage0_cals(
    shape : tuple[int, int],
    nonlin_coeffs : np.ndarray | None = None,
    meta : dict | None = None,
    read_noise : float | None = 5.0,
    gain : float | None = 1.1,
    bias : float | None = 1000.0,
    kTC_noise : float | None = 50.0,
    dark_current : float | None = 0.0,
    flat_level : float | None = 1.0,
    dq : np.ndarray | None = None,
    sat_thresh : int | None = None,
    register : bool = True,
) -> dict:
    """Stage 0 detector cals of shape (Nrows, Ncols), each a constant map.

    read_noise, bias, kTC_noise and dark_current are in electrons and are divided by the gain to
    store them in DN. Set any of read_noise, gain, bias, dark_current or flat_level to None to skip
    that cal; the nonlinearity cal is only made when nonlin_coeffs (highest order first) is given.

    If dq is given, it is the DQ map of the dq cal and is attached to every other cal; otherwise the
    dq cal is all zeros. If sat_thresh is given, the saturation threshold (DN) is that constant;
    otherwise the SaturationModel default is used.

    Returns ``{name: path}`` of the registered cals, or ``{name: model}`` if register is False.
    """
    cals = {}

    # --- Create DQ init map ---
    if dq is not None:
        cals['dq'] = datamodels.DQModel(dq=dq, meta=meta)
    else:
        cals['dq'] = datamodels.DQModel(shape=shape, meta=meta)
    dq_kwargs = {'dq': cals['dq'].dq} if dq is not None else {}

    if nonlin_coeffs is not None:
        nonlin_coeffs = np.broadcast_to(nonlin_coeffs[::-1, None, None], (np.size(nonlin_coeffs), *shape))
        nonlin_max = np.full(shape, fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
        cals['nonlin'] = datamodels.NonlinearCorrectionModel(
            coeffs=nonlin_coeffs,
            nonlin_thresh=nonlin_max,
            nonlin_max=nonlin_max,
            dq=cals['dq'].dq,
            meta=meta,
        )

    # --- Create saturation threshold map ---
    if sat_thresh is not None:
        cals['saturation'] = datamodels.SaturationModel(
            sat_thresh=np.full(shape, sat_thresh, dtype=np.uint16),
            meta=meta,
            **dq_kwargs,
        )
    else:
        cals['saturation'] = datamodels.SaturationModel(shape=shape, meta=meta, **dq_kwargs)

    # --- Create gain map ---
    if gain is not None:
        cals['gain'] = datamodels.GainModel(
            gain=np.full(shape, gain, dtype=np.float32),
            cov=np.zeros((2, 2, *shape), dtype=np.float32),
            meta=meta,
            **dq_kwargs,
        )

    gain_map = cals['gain'].gain if 'gain' in cals else np.ones(shape, dtype=np.float32)

    # --- Create bias map ---
    if bias is not None:
        cals['bias'] = datamodels.BiasModel(
            bias=np.full(shape, bias / gain_map, dtype=np.float32),  # Bias DataModel in DN
            ktc=np.full(shape, kTC_noise / gain_map, dtype=np.float32),  # KTC noise in DN RMS
            meta=meta,
            **dq_kwargs,
        )

    # --- Create read noise map ---
    if read_noise is not None:
        cals['rn'] = datamodels.ReadNoiseModel(
            rn=np.full(shape, read_noise, dtype=np.float32) / gain_map,  # read noise DataModel in DN RMS
            meta=meta,
            **dq_kwargs,
        )

    # --- Create dark current map ---
    if dark_current is not None:
        cals['dark'] = datamodels.DarkModel(
            rate=np.full(shape, dark_current / gain_map, dtype=np.float32),  # dark current DataModel in DN
            meta=meta,
            **dq_kwargs,
        )

    # --- Create detector flat ---
    if flat_level is not None:
        cals['detflat'] = datamodels.DetectorFlatModel(
            flat=np.full(shape, flat_level, dtype=np.float32),
            meta=meta,
            **dq_kwargs,
        )

    if register:
        for name, model in cals.items():
            cals[name] = _register(model)

    return cals


#######################################
#### Simulated wavelength solution ####
#######################################

def make_sim_wavesol(spectrograph: str, register : bool = True):
    """The wavelength solution of the HDF model as a SimulatedWavesolModel.

    Returns the path of the registered cal, or the model if register is False.
    """
    wave = SimulateRateMap().get_wavelength_solution(spectrograph)
    model = datamodels.SimulatedWavesolModel(
        meta={'spectrograph': spectrograph, 'datetime_obs': DATETIME_OBS, 'origin': 'KECK'},
    )
    for fiber in wave:
        orders = list(wave[fiber].keys())
        arr2d = np.full((len(orders), 4096), np.nan, dtype=np.float64)
        for i, order in enumerate(orders):
            arr2d[i, :] = wave[fiber][order]['wavelength']
        model[fiber].wave = arr2d
    return _register(model, register)


###################
#### Fiberflats ####
###################

def normalize_whitelight(wave: np.ndarray, spec: np.ndarray, flux_int: float = 100E-9) -> np.ndarray:
    """Scale spec so its integral over wave is flux_int."""
    return spec * (flux_int / np.trapezoid(spec, wave))


def make_fiberflat_L0(spectrograph: str, do_fibers: int | list[int], output_dir: str) -> datamodels.L0Model:
    """Simulate a noiseless 1000 K white-light rate map in the fibers do_fibers, save it as L0 and return the model.

    The file goes in output_dir with the suffix ``FIBERFLAT-FIB<fibers>`` (e.g. ``FIBERFLAT-FIB2``).
    """
    do_fibers = [int(f) for f in np.atleast_1d(do_fibers)]
    dq = make_dq_array_refchan()
    meta = {'datetime_obs': DATETIME_OBS, 'spectrograph': spectrograph, 'exposure_type': 'SPEC', 'exposure_time': 60.0}
    mjd_start = astropy.time.Time(meta['datetime_obs'], format='isot', scale='utc').mjd

    input_wave = np.arange(0.9, 2.5, 2.5e-6) * u.micron
    input_flux = normalize_whitelight(input_wave.value, lamp_spectrum_shape(input_wave.value, temperature_k=1000.0))
    srm = SimulateRateMap()
    for fiber in do_fibers:
        srm.set_fiber_spectrum(
            fiber_number=fiber,
            spectrum_wavelengths=input_wave,
            spectrum_flux=input_flux * u.erg / u.s / u.cm**2 / u.micron,
            continuous=True,
        )
    rate = srm.create_native_rate_map(
        spectrograph=spectrograph,
        method='prf_convolution',
        save=False
    ).astype(np.float32)

    model = datamodels.L0Model(
        rate=rate,
        rate_err=np.zeros_like(rate),
        rate_var_rnoise=np.zeros_like(rate),
        rate_var_poisson=np.zeros_like(rate),
        dq=dq,
        mid_time=np.full(rate.shape, mjd_start + (meta['exposure_time'] / 2) / 86400, dtype=np.float64),
        meta=meta,
    )
    model.meta.target_name = 'FIBERFLAT'
    for fiber in range(1, model.num_fibers + 1):
        setattr(
            model.meta,
            f'object_fiber{fiber}',
            'WHITELIGHT' if fiber in do_fibers else 'BLOCKED'
        )
    #model.config.set_meta_defaults(model)
    model.save(output_dir=output_dir, suffix='FIBERFLAT-FIB' + ''.join(str(f) for f in do_fibers))
    return model


def make_fiberflat_cals(fiberflats_L0: list, register: bool = True) -> dict:
    """Run ProcessFiberflatsPipeline with default settings on the fiberflat L0s (models or paths) to make the
    trace position, trace profile and fiberflat cals. The wavesol comes from the store.

    Returns ``{name: path}`` of the registered cals, or ``{name: model}`` if register is False.
    """
    ff_models_L0 = [datamodels.open(ff) for ff in fiberflats_L0]
    primitives = {
        'register_tracepos_cal': {'run': False},
        'register_traceprof_cal': {'run': False},
        'register_fiberflat_cal': {'run': False},
    }
    ff_results = ProcessFiberflatsPipeline.call(ff_models_L0, primitives=primitives, save_result=False)

    paths = {}
    for key, name in (('trace_pos', 'tracepos'), ('trace_prof', 'traceprof'), ('fiberflat', 'fiberflat')):
        model = ff_results[key]
        model.meta.origin = 'KECK'
        paths[name] = _register(model, register)
    return paths


############
#### LFC ####
############

def make_lfc_L0(
    spectrograph: str,
    do_fibers: int | list[int] | None = None,
) -> datamodels.L0Model:
    """
    Simulate a noiseless LFC rate map in all three fibers.
    """
    SPEC = spectrograph[0].upper()
    lfc_params = datamodels.HISPECConfig.lfc_params[f'LFC-{SPEC}']
    lfc_wavelengths = gen_lfc_modes(
        nu0=lfc_params['nu0'],
        delta_nu=lfc_params['delta_nu'],
        wave_min=0.96,
        wave_max=2.5,
    )
    srm = SimulateRateMap()
    if do_fibers is None:
        do_fibers = range(1, datamodels.HISPECConfig.num_fibers + 1)
    for fiber in do_fibers:
        srm.set_fiber_spectrum(
            fiber_number=fiber,
            spectrum_wavelengths=lfc_wavelengths * u.um,
            spectrum_flux=np.ones_like(lfc_wavelengths) * u.erg / u.s,
            continuous=False,
        )
    rate = srm.create_native_rate_map(
        spectrograph=spectrograph,
        save=False
    ).astype(np.float32)
    model = datamodels.L0Model(
        rate=rate,
        meta={
            'spectrograph': spectrograph,
            'datetime_obs': DATETIME_OBS,
            'target_name': f'LFC-{SPEC}',
            'object_fiber1': f'LFC-{SPEC}',
            'object_fiber2': f'LFC-{SPEC}',
            'object_fiber3': f'LFC-{SPEC}',
            'origin': 'KECK',
        },
    )
    return model


def make_lfc_L0s(
    spectrograph: str,
    output_dir: str | None = None,
    Nframes: int = LFC_NFRAMES,
    max_snr: float = LFC_MAX_SNR,
) -> list[datamodels.L0Model]:
    """Nframes noisy copies of the LFC rate map in all three fibers, each with peak SNR max_snr.

    If output_dir is given, each is saved there with the suffix ``LFC-<i>``.
    """
    model_L0 = make_lfc_L0(spectrograph)
    models_L0 = [add_noise_L0(model_L0, max_snr=max_snr) for _ in range(Nframes)]
    if output_dir is not None:
        for i, model in enumerate(models_L0, start=1):
            model.save(output_dir=output_dir, suffix=f'LFC-{i}')
    return models_L0


def make_lfc_cals(lfc_L0s: list, register: bool = True) -> dict:
    """Run Stage1Pipeline and then ProcessLFCPipeline with default settings on the LFC L0s (models or paths)
    to make the LFC wavesol and LSF cals.

    All three fibers carry the LFC, so there is one fiber group with one wavesol and one LSF.
    Returns ``{name: path}`` of the registered cals, or ``{name: model}`` if register is False.
    """
    models_L1 = Stage1Pipeline.call([datamodels.open(m) for m in lfc_L0s], save_result=False)
    primitives = {
        'register_wave_cal': {'run': False},
        'register_lsf_cal': {'run': False},
    }
    result = ProcessLFCPipeline.call(models_L1, primitives=primitives, save_result=False)
    (wavesol,) = result['wavesol']
    (lsf,) = result['lsf']
    paths = {}
    for name, model in (('lfc_wavesol', wavesol), ('lsf', lsf)):
        model.meta.origin = 'KECK'
        paths[name] = _register(model, register)
    return paths


#################
#### Gas cell ####
#################

def make_gascell_L0(do_fibers: int | list[int] | None = None) -> datamodels.L0Model:
    """Simulate a noiseless RSPEC rate map of a 1000 K white light through the gas cells in the fibers do_fibers
    (default all three).

    The transmission is the product of the FTS scans that FitGasCellWavesolPrimitive uses by default.
    The gas cell is only used with RSPEC.
    """
    spectrograph = 'RSPEC'
    wave, transmission, _ = _load_fts_reference()
    flux = normalize_whitelight(wave, lamp_spectrum_shape(wave, temperature_k=1000.0)) * transmission
    srm = SimulateRateMap()
    if do_fibers is None:
        do_fibers = FIBERS
    do_fibers = [int(f) for f in np.atleast_1d(do_fibers)]
    for fiber in do_fibers:
        srm.set_fiber_spectrum(
            fiber_number=fiber,
            spectrum_wavelengths=wave * u.micron,
            spectrum_flux=flux * u.erg / u.s / u.cm**2 / u.micron,
            continuous=True,
        )
    rate = srm.create_native_rate_map(
        spectrograph=spectrograph,
        method='prf_convolution',
        save=False,
    ).astype(np.float32)
    model = datamodels.L0Model(
        rate=rate,
        meta={
            'spectrograph': spectrograph,
            'datetime_obs': DATETIME_OBS,
            'exposure_type': 'SPEC',
            'exposure_time': 60.0,
            'target_name': 'GASCELL',
            'origin': 'KECK',
        },
    )
    for fiber in range(1, model.num_fibers + 1):
        setattr(model.meta, f'object_fiber{fiber}', 'GASCELL' if fiber in do_fibers else 'BLOCKED')
    return model


def build_hispec_cals(output_dir: str) -> dict[str, dict[str, str]]:
    """Build and register all shared calibrations; L0/L1 inputs are saved under output_dir.

    For now this is the simulated wavesol, the fiberflat cals and the LFC wavesol and LSF for BSPEC and
    RSPEC. The detector cal functions above are not built yet. Returns ``{spectrograph: {name: path}}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for spectrograph in ('BSPEC', 'RSPEC'):
        out = paths[spectrograph] = {}
        out['sim_wavesol'] = make_sim_wavesol(spectrograph)
        ff_models_L0 = [make_fiberflat_L0(spectrograph, fiber, output_dir) for fiber in FIBERS]
        out['fiberflat_L0'] = [model._filepath for model in ff_models_L0]
        out.update(make_fiberflat_cals(ff_models_L0))
        del ff_models_L0
        lfc_models_L0 = make_lfc_L0s(spectrograph, output_dir)
        out['lfc_L0'] = [model._filepath for model in lfc_models_L0]
        out.update(make_lfc_cals(lfc_models_L0))
        del lfc_models_L0
    return paths


########################
#### Persistent cal DB ####
########################

CAL_DB_CACHE = 'KOA_CALIBRATION_CACHE'
CAL_DB_INPUTS = 'inputs'
CAL_DB_MANIFEST = 'manifest.json'


def _map_paths(paths, func):
    if isinstance(paths, dict):
        return {k: _map_paths(v, func) for k, v in paths.items()}
    if isinstance(paths, list):
        return [_map_paths(v, func) for v in paths]
    return func(paths)


def _all_paths(paths) -> list[str]:
    out = []
    _map_paths(paths, out.append)
    return out


def load_cal_db(root: str) -> dict | None:
    """Return the paths of a complete cal DB at root, or None if it is missing or incomplete."""
    manifest_path = os.path.join(root, CAL_DB_MANIFEST)
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path) as f:
        paths = _map_paths(json.load(f), lambda p: os.path.join(root, p))
    if not all(os.path.isfile(p) for p in _all_paths(paths)):
        return None
    return paths


def get_cal_db_paths() -> dict | None:
    """Paths of the cal DB that KOA_CALIBRATION_CACHE points at (``{spectrograph: {name: path}}``), or None."""
    cache = os.getenv('KOA_CALIBRATION_CACHE')
    if not cache:
        return None
    return load_cal_db(os.path.dirname(os.path.normpath(cache)))


def make_cal_db(root: str) -> dict:
    """Build the cal DB at root, replacing any previous one there, and write its manifest last.

    Only the cache, inputs and manifest under root are removed, never anything else in root.
    Expects KOA_CALIBRATION_CACHE to point at ``<root>/KOA_CALIBRATION_CACHE``.
    """
    for name in (CAL_DB_MANIFEST, CAL_DB_CACHE, CAL_DB_INPUTS):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    paths = build_hispec_cals(os.path.join(root, CAL_DB_INPUTS))
    with open(os.path.join(root, CAL_DB_MANIFEST), 'w') as f:
        json.dump(_map_paths(paths, lambda p: os.path.relpath(p, root)), f, indent=2)
    return paths
