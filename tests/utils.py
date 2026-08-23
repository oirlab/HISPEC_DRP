from pathlib import Path

import numpy as np
import yaml
from hispecdrp import datamodels
from hispecdrp.simulations.utr import create_ramp


def repo_root() -> Path:
    """Repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parents[1]


# Fixed epoch for cal-db / sim metadata in tests: inside IERS leap-second tables
# so Astropy/ERFA does not emit "dubious year" warnings (unlike Time.now() in 2026+
# or packaged wavesol placeholder MJD 0 / 1858-11-17).
SAFE_TEST_MJD = 60000.0  # 2023-02-25 UTC


def save_config_yaml(config_dict : dict, config_path : str):
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)


def make_ramp_model(
    source : np.array,
    meta : dict | None = None,
    **kwargs
) -> datamodels.RampModel:
    ramp_data = create_ramp(
        source=source,
        **kwargs,
    )
    ramp_model = datamodels.RampModel(
        data=ramp_data['data'],
        dq=np.zeros(ramp_data['data'].shape[1:], dtype=np.uint32),
        dq_raw=np.zeros(ramp_data['data'].shape, dtype=np.uint8),
        meta={**(meta or {}), **ramp_data['meta']},
    )
    return ramp_model


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
    saturation_threshold : int | None = 65535
):

    cals = {}

    # --- Create DQ init map ---
    dqinit_map = np.zeros(shape, dtype=np.uint32)
    
    # set row to bad
    dqinit_model = datamodels.DQModel(
        dq=dqinit_map,
        meta=meta
    )
    cals['dq'] = dqinit_model
    
    if nonlin_coeffs is not None:
        nonlin_coeffs = np.broadcast_to(nonlin_coeffs[::-1, None, None], (np.size(nonlin_coeffs), *shape))
        max = np.full(shape, fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
        thresh = max
        nonlin_model = datamodels.NonlinearCorrectionModel(
            coeffs=nonlin_coeffs,
            nonlin_thresh=thresh,
            nonlin_max=max,
            meta=meta,
            dq=dqinit_map,
        )
        cals['nonlin'] = nonlin_model

    # --- Create saturation threshold map ---
    if saturation_threshold is not None:
        saturation_threshold = np.iinfo(np.uint16).max  # DN
        saturation_map = np.full(shape, saturation_threshold, dtype=np.uint16)
        saturation_model = datamodels.SaturationModel(
            sat_thresh=saturation_map,
            meta=meta,
            dq=dqinit_map,
        )
        cals['saturation'] = saturation_model

    # --- Create random gain map ---
    if gain is not None:
        gain_map = np.full(shape, gain, dtype=np.float32)
        gain_model = datamodels.GainModel(
            gain=gain_map,
            cov=np.zeros((2, 2, *shape), dtype=np.float32),
            meta=meta,
            dq=dqinit_map,
        )
        cals['gain'] = gain_model

    gain_map = cals['gain'].gain if 'gain' in cals else np.ones(shape, dtype=np.float32)

    # --- Create bias map ---
    if bias is not None:
        bias_map = np.full(shape, bias, dtype=np.float32)
        bias_model = datamodels.BiasModel(
            bias=bias_map / gain_map,  # Bias DataModel in DN
            ktc=np.full_like(bias_map, kTC_noise / gain_map),  # KTC noise in DN RMS
            meta=meta,
            dq=dqinit_map,
        )
        cals['bias'] = bias_model

    # --- Create random read noise map ---
    if read_noise is not None:
        rn_map = np.full(shape, read_noise, dtype=np.float32)
        rn_model = datamodels.ReadNoiseModel(
            rn=rn_map / gain_map,  # read noise DataModel in DN RMS
            rn_err=np.zeros_like(rn_map),
            meta=meta,
            dq=dqinit_map,
        )
        cals['rn'] = rn_model

    # --- Create random dark current map ---
    if dark_current is not None:
        dark_map = np.full(shape, dark_current, dtype=np.float32)
        dark_model = datamodels.DarkModel(
            rate=dark_map / gain_map,  # dark current DataModel in DN RMS
            rate_err=np.zeros_like(dark_map),
            rate_var_rnoise=np.zeros_like(dark_map),
            rate_var_poisson=np.zeros_like(dark_map),
            meta=meta,
            dq=dqinit_map,
        )
        cals['dark'] = dark_model

    if flat_level is not None:
        flat_map = np.full(shape, flat_level, dtype=np.float32)
        flat_model = datamodels.DetectorFlatModel(
            flat=flat_map,
            flat_err=np.zeros_like(flat_map),
            flat_var_rnoise=np.zeros_like(flat_map),
            flat_var_poisson=np.zeros_like(flat_map),
            meta=meta,
            dq=dqinit_map,
        )
        cals['detflat'] = flat_model

    return cals

def fast_ccf_orders(spectrograph: str) -> list[int]:
    """Three mid-band echelle orders for fast RV-recovery unit tests."""
    from hispecdrp.datamodels import hispec_config

    orders = list(hispec_config.HISPECConfig.get_order_list(spectrograph))
    if len(orders) <= 3:
        return orders
    mid = len(orders) // 2
    return [orders[mid - 1], orders[mid], orders[mid + 1]]
