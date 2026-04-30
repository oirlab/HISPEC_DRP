import os
import yaml

from datetime import datetime

from hispecdrp import datamodels
from hispecdrp.utils import download_gdrive_file
from hispecdrp.pipelines import Stage1Pipeline

from .utils import _run_pipeline_perf


def _create_stage1_cals(
    output_dir : str | None = None
) -> dict:

    tracepos_path = download_gdrive_file('HISPEC/Cals/HR.20231006.00000.00.mastercal-tracepos.fits', output_dir=output_dir)
    traceprof_path = download_gdrive_file('HISPEC/Cals/HR.20231006.00000.00.mastercal-traceprof.fits', output_dir=output_dir)

    cals = {
        'tracepos' : datamodels.open(tracepos_path),
        'traceprof' : datamodels.open(traceprof_path),
    }

    return cals


def _get_stage1_prim_args(cals : dict, max_cores : int, quicklook : bool = False) -> dict:
    prim_args = {prim: {} for prim in Stage1Pipeline.PRIMITIVES.keys()}
    prim_args['spec_extract']['method'] = 'optimal'
    prim_args['spec_extract']['extraction_aperture'] = 7
    prim_args['spec_extract']['tracepos'] = cals['tracepos']
    prim_args['spec_extract']['traceprof'] = cals['traceprof']
    #prim_args['set_wavesol']['wavesol'] = cals['wavesol']
    prim_args['set_wavesol']['run'] = False

    for prim in prim_args:
        prim_args[prim]['max_cores'] = max_cores
    
    if quicklook:
        prim_args['spec_extract']['method'] = 'box'
        prim_args['spec_extract']['extraction_aperture'] = 3

    return prim_args


def _make_stage1_config_file(
    cals : dict,
    max_cores : int,
    output_dir : str | None = None,
    save_result : bool = True,
    quicklook : bool = False,
):
    prim_args = _get_stage1_prim_args(cals=cals, max_cores=max_cores, quicklook=quicklook)
    config = {
        'class' : 'Stage1Pipeline',
        'arguments' : {
            'save_result' : save_result,
            'output_dir' : output_dir,
        },
        'primitives' : {},
    }
    for prim in Stage1Pipeline.PRIMITIVES.keys():
        config['primitives'][prim] = {'arguments' : {}}
        config['primitives'][prim]['arguments'] = prim_args[prim]
        for arg in config['primitives'][prim]['arguments']:
            if arg in cals:
                cal = config['primitives'][prim]['arguments'][arg]
                config['primitives'][prim]['arguments'][arg] = cal._filepath
    
    return config

def run_perf_stage1(
    output_dir : str,
    max_cores : int = 1,
    save_result : bool = True,
    quicklook : bool = False
):
    
    # Download cals
    cals = _create_stage1_cals(output_dir=output_dir)

    # Download a fiberflat as a dummy input model
    input_filepath = download_gdrive_file('HISPEC/L0/HR.20231006.00000.00_rate-FIBERFLAT-FIB1.fits', output_dir=output_dir)

    config = _make_stage1_config_file(
        cals=cals,
        max_cores=max_cores,
        output_dir=output_dir,
        save_result=save_result,
        quicklook=quicklook,
    )

    config_filepath = os.path.join(output_dir, "stage1_config.yaml")
    with open(config_filepath, 'w') as f:
        yaml.dump(config, f)

    del cals

    results = _run_pipeline_perf(
        pipeline_class=Stage1Pipeline,
        input=input_filepath,
        config_file=config_filepath,
    )
    
    # Save timing results
    output_path = os.path.join(
        output_dir,
        f'perf_Stage1Pipeline{'FULL' if not quicklook else 'QL'}_max_cores={max_cores}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )
    results.write(output_path, format='csv', overwrite=True)