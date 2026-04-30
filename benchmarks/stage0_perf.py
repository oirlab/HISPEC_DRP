import os
import yaml
import numpy as np

from datetime import datetime

from hispecdrp import datamodels
from hispecdrp.pipelines import Stage0Pipeline
from hispecdrp.simulations import create_ramp

from .utils import _run_pipeline_perf


def _create_stage0_cals(
    nonlin_coeffs : np.ndarray,
    RN : float  = 1.0,
    bias : float = 0.0,
    dark_current : float = 0.0,
    gain : float = 1.0,
    output_dir : str | None = None
) -> dict:
    
    shape = (4096, 4096)
    ny, nx = shape

    dq_map = np.zeros(shape, dtype=np.uint32)

    dq_model = datamodels.DQModel(dq=dq_map)

    # --- Create non-linearity correction model ---
    coeffs = np.broadcast_to(nonlin_coeffs[::-1, None, None], (np.size(nonlin_coeffs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.full(shape=(ny, nx), fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
    nonlin_model = datamodels.NonlinearCorrectionModel(
        coeffs=coeffs,
        nonlin_thresh=thresh,
        nonlin_max=max,
        dq=dq_map,
    )

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(
        sat_thresh=saturation_map,
        dq=np.zeros_like(saturation_map, dtype=np.uint32)
    )

    # --- Create random gain map ---
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(
        gain=gain_map,
        rn=np.full_like(gain_map, 0.0),
        cov=np.full((2, 2, ny, nx), 0, dtype=np.float32),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    # --- Create random dark map ---
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    dark_model = datamodels.DarkModel(
        rate=dark_map / gain_map,  # dark in DN/s
        rate_err=np.zeros_like(dark_map),
        rate_var_rnoise=np.zeros_like(dark_map),
        rate_var_poisson=np.zeros_like(dark_map),
        dq=np.zeros_like(dark_map, dtype=np.uint32)
    )

    # --- Create bias map ---
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    bias_model = datamodels.BiasModel(
        bias=bias_map / gain_map,  # bias in DN
        ktc=np.zeros_like(bias_map),  # kTC in DN
        dq=np.zeros_like(bias_map, dtype=np.uint32)
    )

    # --- Create random flat map ---
    detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(
        flat=detflat_map,
        flat_err=np.zeros_like(detflat_map),
        flat_var_rnoise=np.zeros_like(detflat_map),
        flat_var_poisson=np.zeros_like(detflat_map),
        dq=np.zeros_like(detflat_map, dtype=np.uint32)
    )

    # --- Create random read noise map ---
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(
        rn=rn_map / gain_map,
        rn_err=np.zeros_like(rn_map),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    cals = {
        'dq': dq_model,
        'nonlin': nonlin_model,
        'saturation': saturation_model,
        'gain': gain_model,
        'dark': dark_model,
        'bias': bias_model,
        'detflat': detflat_model,
        'rn': rn_model,
    }

    if output_dir is not None:
        for cal in cals:
            cals[cal].save(filepath=os.path.join(output_dir, f"simulated_{cal}.fits"))

    return cals


def _get_stage0_prim_args(cals : dict, max_cores : int, quicklook : bool = False) -> dict:
    prim_args = {prim: {} for prim in Stage0Pipeline.PRIMITIVES.keys()}
    prim_args['dq_init']['dq'] = cals['dq']
    prim_args['saturation_check']['saturation'] = cals['saturation']
    prim_args['bias_sub']['bias'] = cals['bias']
    prim_args['nonlin_corr']['nonlin'] = cals['nonlin']
    prim_args['jump_detec']['method'] = 'tpd'
    prim_args['jump_detec']['gain'] = cals['gain']
    prim_args['jump_detec']['rn'] = cals['rn']
    prim_args['ramp_fit']['jwst_jump_detection'] = False
    prim_args['ramp_fit']['jwst_jump_rejection_threshold'] = 4.0
    prim_args['ramp_fit']['gain'] = cals['gain']
    prim_args['ramp_fit']['rn'] = cals['rn']
    prim_args['dark_sub']['dark'] = cals['dark']
    prim_args['gain_corr']['gain'] = cals['gain']
    prim_args['detflat_corr']['detflat'] = cals['detflat']
    
    for prim in prim_args:
        prim_args[prim]['max_cores'] = max_cores
    
    if quicklook:
        prim_args['jump_detec']['run'] = False
        prim_args['ramp_fit']['method'] = 'ols'

    return prim_args


def _make_stage0_config_file(
    config_filepath : str,
    cals : dict,
    max_cores : int,
    output_dir : str | None = None,
    save_result : bool = True,
    quicklook : bool = False,
):
    prim_args = _get_stage0_prim_args(cals=cals, max_cores=max_cores, quicklook=quicklook)
    config = {
        'class' : 'Stage0Pipeline',
        'arguments' : {
            'save_result' : save_result,
            'output_dir' : output_dir,
        },
        'primitives' : {},
    }
    for prim in Stage0Pipeline.PRIMITIVES.keys():
        config['primitives'][prim] = {'arguments' : {}}
        config['primitives'][prim]['arguments'] = prim_args[prim]
        for arg in config['primitives'][prim]['arguments']:
            if arg in cals:
                cal = config['primitives'][prim]['arguments'][arg]
                config['primitives'][prim]['arguments'][arg] = cal._filepath
    
    with open(config_filepath, 'w') as f:
        yaml.dump(config, f)


def _create_stage0_input(cals : dict, nonlin_coeffs : np.ndarray, n_reads : int, output_dir : str | None = None) -> datamodels.RampModel:

    # --- Create a fake UTR cube ---
    max_slope = 1000
    read_noise = 1.0
    readtime = 1.7
    ny, nx = (4096, 4096)
    electron_rate_map = np.linspace(
        0, max_slope, nx, dtype=np.float32
    )[None, :] * np.ones((ny, 1), dtype=np.float32)
    
    max_cores_ramp = os.cpu_count()
    ramp_data = create_ramp(
        electron_rate_map,  # in e-/s
        readtime=readtime, n_reads=n_reads,
        nonlin_coeffs=nonlin_coeffs,
        gain=cals['gain'].gain,  # in e-/s
        flat=cals['detflat'].flat,  # unitless
        dark=cals['dark'].rate,  # in e-/s
        bias=cals['bias'].bias,  # in e-/s
        kTC_noise=50,  # in e-/s
        poisson_noise=False, read_noise=read_noise,  # in e-/s
        convert_to_uint16=True, clip_ramps=True,
        max_cores=max_cores_ramp,
    )

    ramp_model = datamodels.RampModel(
        data=ramp_data['data'],
        dq_raw=ramp_data['dq_raw'],
        meta=ramp_data['meta']
    )

    if output_dir is not None:
        ramp_model.save(filepath=os.path.join(output_dir, "simulated_ramp.fits"))

    return ramp_model


def run_perf_stage0(
    output_dir : str,
    n_reads : int = 1,
    max_cores : int = 1,
    save_result : bool = True,
    quicklook : bool = False
):

    # Create dummy cals
    nonlin_coeffs = np.array([-1e-6, 1, 0]) # NOTE: Needed for create_ramp.
    cals = _create_stage0_cals(
        nonlin_coeffs,
        output_dir=output_dir,
    )

    # Create the ramp model
    ramp_model = _create_stage0_input(cals, nonlin_coeffs, n_reads, output_dir=output_dir)
    ramp_filepath = ramp_model._filepath
    del ramp_model

    config_filepath = os.path.join(output_dir, "stage0_config.yaml")
    _make_stage0_config_file(
        config_filepath=config_filepath,
        cals=cals,
        max_cores=max_cores,
        output_dir=output_dir,
        save_result=save_result,
        quicklook=quicklook,
    )

    del cals

    results = _run_pipeline_perf(
        pipeline_class=Stage0Pipeline,
        input=ramp_filepath,
        config_file=config_filepath,
    )
    
    # Save timing results
    output_path = os.path.join(
        output_dir,
        f'perf_Stage0Pipeline{'FULL' if not quicklook else 'QL'}_nreads={n_reads}_max_cores={max_cores}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )
    results.write(output_path, format='csv', overwrite=True)
