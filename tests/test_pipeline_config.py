from hispecdrp.pipelines import Stage0Pipeline, BasePipeline
from hispecdrp import datamodels

from hispecdrp.driver import ConfigFile

from .utils import save_config_yaml, make_ramp_model, make_stage0_cals

import numpy as np
import os

def test_pipeline_auto_config(tmp_path):
    input_model = make_ramp_model(source=np.ones((10, 10)))

    # Auto select config file based on dataset and context
    pipe = Stage0Pipeline(dataset=input_model)

    # Test ramp fit primitive properties
    assert pipe.ramp_fit.method =='cython_likely'

    # Test that all primitives in the pipeline have config files
    for prim_alias in pipe.PRIMITIVES:
        prim = getattr(pipe, prim_alias)
        assert isinstance(prim.config, ConfigFile)
        assert prim.config.context['quicklook'] in (False, None)

    # Change quicklook context to True
    pipe = Stage0Pipeline(dataset=input_model, context=dict(quicklook=True))
    assert pipe.ramp_fit.method =='ols'

    # Test that all primitives in the pipeline have config files with quicklook context set to True
    for prim_alias in pipe.PRIMITIVES:
        prim = getattr(pipe, prim_alias)
        assert isinstance(prim.config, ConfigFile)
        assert prim.config.context['quicklook'] in (True, None)


def test_pipeline_custom_config_file(tmp_path):

    input_model = make_ramp_model(source=np.ones((10, 10)))

    # Make a config file to override these params
    method = 'mcds'
    mcds_num_coadd = 5
    config = {
        'class' : 'Stage0Pipeline',
        'context' : {},
        'primitives' : {
            'ramp_fit': {
                'parameters' : {
                    'method': method,
                    'mcds_num_coadd': mcds_num_coadd,
                }
            }
        },
    }
    config_path = os.path.join(tmp_path, 'Stage0PipelineConfig-Custom.yaml')
    save_config_yaml(config, config_path)

    # Test calling from_config_file (class name determined from config file)
    pipe = BasePipeline.from_config_file(
        dataset=input_model,
        config=config_path,
    )
    assert pipe.ramp_fit.method == method
    assert pipe.ramp_fit.mcds_num_coadd == mcds_num_coadd

    # Test passing directly to constructor
    pipe = Stage0Pipeline(
        dataset=input_model,
        config=config_path,
    )
    assert pipe.ramp_fit.method == method
    assert pipe.ramp_fit.mcds_num_coadd == mcds_num_coadd

    # Test with config file only in memory
    pipe = Stage0Pipeline(
        dataset=input_model,
        config=config,
    )
    assert pipe.ramp_fit.method == method
    assert pipe.ramp_fit.mcds_num_coadd == mcds_num_coadd


def test_pipeline_call_interface(tmp_path):

    shape = (10, 10)
    nonlin_coeffs = np.array([0.0, 1.0])
    gain = 1
    bias = 1000
    kTC_noise = 0
    read_noise = 1

    cals = make_stage0_cals(
        shape=shape,
        nonlin_coeffs=nonlin_coeffs,
        read_noise=read_noise,
        gain=gain,
        bias=bias,
        kTC_noise=kTC_noise,
    )

    input_model = make_ramp_model(
        source=np.ones((10, 10)),
        gain=gain,
        bias=bias,
        nonlin_coeffs=nonlin_coeffs,
        read_noise=read_noise,
        kTC_noise=kTC_noise
    )

    primitives = {
        'dq_init': {
            'dq': cals['dq'],
        },
        'saturation_check': {
            'saturation': cals['saturation'],
        },
        'bias_sub': {
            'bias': cals['bias'],
        },
        'nonlin_corr': {
            'nonlin': cals['nonlin'],
        },
        'jump_detec': {
            'rn': cals['rn'],
            'gain': cals['gain'],
        },
        'ramp_fit': {
            'rn': cals['rn'],
            'gain': cals['gain'],
        },
        'dark_sub' : {
            'dark': cals['dark'],
        },
        'gain_corr' : {
            'gain': cals['gain'],
        },
        'detflat_corr' : {
            'detflat': cals['detflat'],
        }
    }

    result, pipe = Stage0Pipeline.call(
        input_model,
        primitives=primitives,
        return_instance=True
    )
    assert isinstance(result, datamodels.L0Model)
    assert isinstance(pipe, Stage0Pipeline)


def test_effective_pipeline_config(tmp_path):
    input_model = make_ramp_model(source=np.ones((10, 10)))

    method = 'mcds'
    mcds_num_coadd = 5
    primitives = {
        'ramp_fit': {
            'method': method,
            'mcds_num_coadd': mcds_num_coadd
        }
    }

    # Auto configure the pipeline and override with primitives above
    pipe = Stage0Pipeline(dataset=input_model, primitives=primitives)

    # Effective config for pipeline + primitives
    effective_config = pipe.get_effective_config()

    assert effective_config.primitives['ramp_fit']['parameters']['method'] == method
    assert effective_config.primitives['ramp_fit']['parameters']['mcds_num_coadd'] == mcds_num_coadd

    # Reuse the effective config
    effective_config_filepath = str(tmp_path / f"{effective_config.name}.yaml")
    effective_config.save(effective_config_filepath)
    pipe = Stage0Pipeline.from_config_file(
        dataset=input_model,
        config=effective_config_filepath,
    )

    assert pipe.ramp_fit.method == method
    assert pipe.ramp_fit.mcds_num_coadd == mcds_num_coadd