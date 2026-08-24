import os
import yaml
import numpy as np
from datetime import datetime

from hispecdrp import datamodels
from hispecdrp.datamodels import hispec_config
from hispecdrp.pipelines import Stage2Pipeline
from hispecdrp.primitives.utils.spectral_utils import load_btsettl_spectrum, get_spectrum_path

from .utils import _run_pipeline_perf

import hispecdrp
TEMPLATES_DATA_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), 'data', 'templates')
DEFAULT_TEMPLATE_FILENAME = 'lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5'
LAMBDA_ORDER_COEF_UM = 147.0

def _order_wavelength_grid(order: int, n_pix: int) -> np.ndarray:
    lam_center = LAMBDA_ORDER_COEF_UM / order
    delta_lam = lam_center / order
    return np.linspace(
        lam_center - delta_lam / 2.0,
        lam_center + delta_lam / 2.0,
        n_pix,
        dtype=np.float64,
    )


def _create_stage2_input(
    output_dir : str,
    spectrograph : str = 'RSPEC',
) -> str:
    """Create an L1 model with a real BT-Settl stellar spectrum on fiber 1 and save to disk."""
    order_list = hispec_config.HISPECConfig.get_order_list(spectrograph)
    lam_centers = [LAMBDA_ORDER_COEF_UM / m for m in order_list]
    wl_min_um = min(lam_centers) * 0.95
    wl_max_um = max(lam_centers) * 1.05

    # Load BT-Settl template
    template_path = get_spectrum_path(DEFAULT_TEMPLATE_FILENAME, base_dir=TEMPLATES_DATA_DIR)
    wl_q, flux_q = load_btsettl_spectrum(template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    wl_template = np.asarray(wl_q.to_value('micron'), dtype=np.float64)
    flux_template = np.asarray(flux_q.value, dtype=np.float64)
    ok = np.isfinite(wl_template) & np.isfinite(flux_template) & (wl_template > 0)
    wl_template, flux_template = wl_template[ok], flux_template[ok]
    idx = np.argsort(wl_template)
    wl_template, flux_template = wl_template[idx], flux_template[idx]

    n_orders = len(order_list)
    n_pix = 4096
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(order_list):
        w = _order_wavelength_grid(int(m), n_pix)
        wave[oi] = w
        spec[oi] = np.interp(w, wl_template, flux_template, left=0.0, right=0.0).astype(np.float32)

    order_max = np.nanmax(spec, axis=1, keepdims=True)
    order_max = np.where(order_max == 0, 1.0, order_max)
    spec = (spec / order_max).astype(np.float32)
    err = (spec / 100.0).astype(np.float32)

    order_start = int(min(order_list))
    order_end = int(max(order_list))
    meta = {'order_start': order_start, 'order_end': order_end,
            'spectrograph': spectrograph, 'instrument_name': 'HISPEC'}
    model = datamodels.L1Model(meta=meta, config=hispec_config.HISPECConfig)
    model.meta.order_start = order_start
    model.meta.order_end = order_end
    model.spec_uncal_fib1 = spec
    model.spec_uncal_err_fib1 = err
    model.spec_uncal_var_poisson_fib1 = np.zeros_like(spec)
    model.spec_uncal_var_rnoise_fib1 = np.zeros_like(spec)
    model.wave_fib1 = wave
    model.spec_dq_fib1 = np.zeros((n_orders, n_pix), dtype=np.uint32)

    filepath = os.path.join(output_dir, 'simulated_l1.fits')
    model.save(filepath=filepath)
    return filepath


def _create_stage2_cals(
    input_filepath : str,
    output_dir : str | None = None
) -> dict:
    """Create a dummy FiberflatModel filled with ones (flat blaze = no correction)."""
    input_model = datamodels.open(input_filepath)
    n_orders, n_px = input_model.traces[1]['spec_uncal'].shape
    n_fibers = len(input_model.traces)

    ff_kwargs = {}
    for fiber in range(1, n_fibers + 1):
        ff_kwargs[f'SPEC_FIB{fiber}']            = np.ones((n_orders, n_px), dtype=np.float32)
        ff_kwargs[f'SPEC_ERR_FIB{fiber}']        = np.ones((n_orders, n_px), dtype=np.float32)
        ff_kwargs[f'SPEC_VAR_POISSON_FIB{fiber}']= np.zeros((n_orders, n_px), dtype=np.float32)
        ff_kwargs[f'SPEC_VAR_RNOISE_FIB{fiber}'] = np.zeros((n_orders, n_px), dtype=np.float32)
        ff_kwargs[f'SPEC_DQ_FIB{fiber}']         = np.zeros((n_orders, n_px), dtype=np.uint32)

    fiberflat_model = datamodels.FiberflatModel(**ff_kwargs)

    cals = {'fiberflat': fiberflat_model}

    if output_dir is not None:
        fiberflat_model.save(filepath=os.path.join(output_dir, 'simulated_fiberflat.fits'))

    return cals


def _get_stage2_prim_args(cals : dict, max_cores : int, do_orders : list[int], do_fibers : list[str], quicklook : bool = False, save_rv_preview : bool = False) -> dict:
    prim_args = {prim: {} for prim in Stage2Pipeline.PRIMITIVES.keys()}
    prim_args['blaze_corr']['fiberflat'] = cals['fiberflat']

    for prim in prim_args:
        prim_args[prim]['max_cores'] = max_cores
        prim_args[prim]['do_orders'] = do_orders
        prim_args[prim]['do_fibers'] = do_fibers

    prim_args['stellar_rv']['save_preview'] = save_rv_preview
    prim_args['stellar_rv']['v_min_coarse'] = -20_000.0
    prim_args['stellar_rv']['v_max_coarse'] = 20_000.0
    prim_args['stellar_rv']['v_step_coarse'] = 50.0

    if quicklook:
        pass

    return prim_args


def _make_stage2_config_file(
    cals : dict,
    max_cores : int,
    do_orders : list[int],
    do_fibers : list[str],
    output_dir : str | None = None,
    save_result : bool = True,
    quicklook : bool = False,
    save_rv_preview : bool = False,
):
    prim_args = _get_stage2_prim_args(
        cals=cals,
        max_cores=max_cores,
        do_orders=do_orders,
        do_fibers=do_fibers,
        quicklook=quicklook,
        save_rv_preview=save_rv_preview,
    )
    config = {
        'class' : 'Stage2Pipeline',
        'arguments' : {
            'save_result' : save_result,
            'output_dir' : output_dir,
        },
        'primitives' : {},
    }
    for prim in Stage2Pipeline.PRIMITIVES.keys():
        config['primitives'][prim] = {'arguments' : {}}
        config['primitives'][prim]['arguments'] = prim_args[prim]
        for arg in list(config['primitives'][prim]['arguments']):
            if arg in cals:
                cal = config['primitives'][prim]['arguments'][arg]
                config['primitives'][prim]['arguments'][arg] = cal._filepath

    return config


def run_perf_stage2(
    output_dir : str,
    max_cores : int = 1,
    do_orders : list[int] | None = None,
    do_fibers : list[str] | None = None,
    save_result : bool = True,
    quicklook : bool = False,
    save_rv_preview : bool = False,
):
    input_filepath = _create_stage2_input(output_dir=output_dir)

    cals = _create_stage2_cals(
        input_filepath=input_filepath,
        output_dir=output_dir,
    )

    config = _make_stage2_config_file(
        cals=cals,
        max_cores=max_cores,
        do_fibers=do_fibers,
        do_orders=do_orders,
        output_dir=output_dir,
        save_result=save_result,
        quicklook=quicklook,
        save_rv_preview=save_rv_preview,
    )

    config_filepath = os.path.join(output_dir, "stage2_config.yaml")
    with open(config_filepath, 'w') as f:
        yaml.dump(config, f)

    # NOTE: Delete for better RAM estimate
    # since filepaths are passed, not models.
    del cals

    results = _run_pipeline_perf(
        pipeline_class=Stage2Pipeline,
        input=input_filepath,
        config_file=config_filepath,
    )

    # Save timing results
    output_path = os.path.join(
        output_dir,
        f'perf_Stage2Pipeline{'FULL' if not quicklook else 'QL'}_max_cores={max_cores}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )
    results.write(output_path, format='csv', overwrite=True)