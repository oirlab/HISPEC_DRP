from hispecdrp.primitives import FitRampPrimitive
from hispecdrp.primitives import BasePrimitive
from hispecdrp import datamodels

from hispecdrp.driver import ConfigFile

from .utils import save_config_yaml, make_ramp_model

import numpy as np
import os


def test_primitive_auto_config(tmp_path):
    input_model = make_ramp_model(source=np.ones((10, 10)))

    # Auto select config file based on dataset and context
    prim = FitRampPrimitive(dataset=input_model)
    assert prim.method =='cython_likely'
    assert isinstance(prim.config, ConfigFile)
    assert prim.config.context.get('quicklook') in (None, False)

    # Change quicklook context to True
    prim = FitRampPrimitive(dataset=input_model, context=dict(quicklook=True))
    assert prim.method =='ols'
    assert isinstance(prim.config, ConfigFile)
    assert prim.config.context['quicklook'] is True


def test_primitive_custom_config_file(tmp_path):

    # Override these params
    method = 'mcds'
    mcds_num_coadd = 5

    # Make config file
    config = {
        'class' : 'FitRampPrimitive',
        'context' : {},
        'parameters' : {
            'method': method,
            'mcds_num_coadd': mcds_num_coadd,
        }
    }
    config_path = os.path.join(tmp_path, 'RampFitConfig-Custom.yaml')
    save_config_yaml(config, config_path)

    # Test passing directly to constructor
    prim = FitRampPrimitive(
        config=config_path,
        parent=None,
    )
    assert prim.method == method
    assert prim.mcds_num_coadd == mcds_num_coadd

    # Test calling from_config_file
    # (class name determined from config file)
    prim = BasePrimitive.from_config_file(config=config_path)
    assert prim.method == method
    assert prim.mcds_num_coadd == mcds_num_coadd

    # Test again with class constructor
    prim = FitRampPrimitive(config=config)
    assert prim.method == method
    assert prim.mcds_num_coadd == mcds_num_coadd


def test_primitive_kwargs(tmp_path):

    # Override these params
    method = 'mcds'
    mcds_num_coadd = 5

    input_model = make_ramp_model(source=np.ones((10, 10)))

    # Test passing directly to constructor with kwargs
    prim = FitRampPrimitive(
        dataset=input_model,
        method=method,
        mcds_num_coadd=mcds_num_coadd,
    )
    assert prim.method == method
    assert prim.mcds_num_coadd == mcds_num_coadd


def test_primitive_cals(tmp_path):

    input_model = make_ramp_model(source=np.ones((10, 10)))

    rn = datamodels.ReadNoiseModel(rn=np.ones(input_model.data.shape[1:]))

    # Test passing directly to constructor with kwargs
    prim = FitRampPrimitive(
        dataset=input_model,
        rn=rn,
        gain=None,
    )
    assert isinstance(prim.rn, datamodels.ReadNoiseModel)
    assert prim.gain is None


def test_primitive_call_interface(tmp_path):
    input_model = make_ramp_model(source=np.ones((10, 10)))
    method = 'mcds'
    mcds_num_coadd = 5

    model_L0, prim = FitRampPrimitive.call(
        input_model,
        rn=datamodels.ReadNoiseModel(rn=np.ones(input_model.data.shape[1:])),
        gain=datamodels.GainModel(gain=np.ones(input_model.data.shape[1:])),
        method=method,
        mcds_num_coadd=mcds_num_coadd,
        return_instance=True
    )
    assert isinstance(model_L0, datamodels.L0Model)
    assert isinstance(prim, FitRampPrimitive)
    assert prim.method == method
    assert prim.mcds_num_coadd == mcds_num_coadd


def test_effective_primitive_config(tmp_path):
    input_model = make_ramp_model(source=np.ones((10, 10)))

    # Auto select config file based on dataset and context
    prim = FitRampPrimitive(dataset=input_model)
    effective_config = prim.get_effective_config()
    assert effective_config.parameters['method'] == prim.method
    assert effective_config.parameters['mcds_num_coadd'] == prim.mcds_num_coadd