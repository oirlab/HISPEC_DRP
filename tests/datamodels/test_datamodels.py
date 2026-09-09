"""Tests for HISPEC / PARVI datamodel I/O.
"""
from hispecdrp import datamodels
import numpy as np
import os
from hispecdrp.utils import download_gdrive_file


def test_datamodel_default_extensions():
    """
    Single-extension-family calibration/data models (BiasModel, DarkModel,
    DetectorFlatModel, GainModel, ReadNoiseModel, SaturationModel, RampModel,
    L0Model, DQModel, NonlinearCorrectionModel) should NOT override
    `primary_extension_name`: the base `DataModel` default (the first
    non-special key declared in `EXTENSIONS`) already resolves to the right
    extension for these. Multi-fiber models (e.g. L1Model, L2Model,
    GaussianLSFModel) must keep their dynamic override, since the base
    default can't know which fiber was actually populated.
    """
    primary_ext_names = {
        datamodels.BiasModel: 'bias',
        datamodels.DarkModel: 'rate',
        datamodels.DetectorFlatModel: 'flat',
        datamodels.GainModel: 'gain',
        datamodels.ReadNoiseModel: 'rn',
        datamodels.SaturationModel: 'sat_thresh',
        datamodels.RampModel: 'data',
        datamodels.L0Model: 'rate',
        datamodels.DQModel: 'dq',
        datamodels.NonlinearCorrectionModel: 'coeffs',
    }
    for cls, expected in primary_ext_names.items():
        assert 'primary_extension_name' not in cls.__dict__, (
            f"{cls.__name__} should rely on DataModel's default primary_extension_name"
        )
        model = cls()
        assert model.primary_extension_name.lower() == expected
        # DETECTOR_SIZE_Y/X fallback: with no data or explicit `shape=` given
        # anywhere on the model, sibling extensions declared with
        # shape=(DETECTOR_SIZE_Y, DETECTOR_SIZE_X) (e.g. DQ, present on every
        # one of these models) still auto-initialize to the real detector size.
        # (`model.shape` itself is a separate, simpler property that only
        # reflects the primary extension's actually-assigned data -- it has no
        # detector-size fallback of its own.)
        assert model.dq.shape == model.config.DETECTOR_SIZE

    # Multi-fiber models must keep their dynamic override (returns None until a
    # fiber's extension is actually set -- the static base default would guess wrong).
    multi_fiber_models = [
        datamodels.GaussianLSFModel,
        datamodels.L1Model,
        datamodels.L2Model,
        datamodels.ReferenceEtalonModel,
        datamodels.FiberflatModel,
    ]
    for cls in multi_fiber_models:
        assert 'primary_extension_name' in cls.__dict__, (
            f"{cls.__name__} must keep its dynamic primary_extension_name override"
        )
        model = cls()
        assert model.primary_extension_name is None

def test_hispec_level0(tmp_path, datadir):
    
    # Science file (just tellurics)
    filepath = download_gdrive_file('HB.20231006.00000.00_rate-TELLURICS-FIB1.fits', output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument_name == 'HISPEC'
        assert model.meta.exposure_type == "SPEC"
        assert model.meta.spectrograph == 'BSPEC'
        assert model.meta.model_type == 'L0Model'
        assert model.meta.instrument_mode == 'YJHK'
        shape = (4096, 4096 + 64)
        assert model.rate.shape == shape
        assert model.rate_err.shape == shape
        assert model.rate_var_poisson.shape == shape
        assert model.rate_var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.rate.dtype, np.float32)
        assert np.issubdtype(model.rate_err.dtype, np.float32)
        assert np.issubdtype(model.rate_var_poisson.dtype, np.float32)
        assert np.issubdtype(model.rate_var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as HISPECL0
    model = datamodels.L0Model(filepath)

    # Set arbitrary metadata
    model.meta.exposure_time = 10.0 # valid

    # Set invalid metadata
    try:
        model.meta.exposure_time = 'abc'
    except ValueError:
        pass
    else:
        raise RuntimeError("Invalid metadata was set.")
    
    # Print model metadata, shown with pytest -s
    model.meta.pprint(show_none=True)

    # Save model
    model.save(output_dir=tmp_path)

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(filepath)
    _test_model_attrs(model)


def test_parvi_level0(tmp_path, datadir):

    datamodels.set_instrument('PARVI')

    # Science file (just tellurics)
    filepath = download_gdrive_file('PR.20260131.80205.77_rate-SOLAR.fits', output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument_name == 'PARVI'
        assert model.meta.exposure_type == "SPEC"
        assert model.meta.model_type == 'L0Model'
        shape = (2048, 2048)
        assert model.rate.shape == shape
        assert model.rate_err.shape == shape
        assert model.rate_var_poisson.shape == shape
        assert model.rate_var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.rate.dtype, np.float32)
        assert np.issubdtype(model.rate_err.dtype, np.float32)
        assert np.issubdtype(model.rate_var_poisson.dtype, np.float32)
        assert np.issubdtype(model.rate_var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as PARVIL0
    model = datamodels.L0Model(filepath)

    # Save model
    model.save(output_dir=tmp_path)

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(filepath)
    _test_model_attrs(model)

    del os.environ['HISPECDRP_INSTRUMENT_NAME']