from hispecdrp import FitRampPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels
import numpy as np

from .utils import add_metadata


def create_ramp_model(slope):
    # Create ramp
    times = np.arange(10).astype(np.float64)
    ramps = np.zeros(shape=(100, 100, 10), dtype=np.float32)
    for i in range(10):
        ramps[:, :, i] = slope * times[i]
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramps, times=times))
    ramp_model.meta.target.target_name = 'My-Target'
    ramp_model.meta.target.object_fiber1 = 'Sci'
    ramp_model.meta.target.object_fiber1 = 'Sky'
    ramp_model.meta.target.object_fiber1 = 'LFC'
    ramp_model.meta.exposure.exposure_type = "SCI"
    ramp_model.meta.exposure.jd_start = 2460577.5
    ramp_model.meta.instrument.spectrograph = 'BSPEC'
    ramp_model.meta.instrument.mode = 'YJHK'
    ramp_model.meta.model_type = 'HISPECRampModel'
    add_metadata(ramp_model)
    return ramp_model


def test_ramp_fit(tmp_path):
    
    slope = 10
    ramp_model = create_ramp_model(slope=slope)

    # Config file
    rampfit_config_file = str(tmp_path / 'ramp_config.yaml')
    rampfit_config_raw = """
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      save_results: False
      suffix: 'ramp-fit'
    """
    with open(rampfit_config_file, 'w') as f:
        f.write(textwrap.dedent(rampfit_config_raw))

    # Setup and call the flat primitive
    rampfit_prim = FitRampPrimitive(config_file=rampfit_config_file)
    model_result = rampfit_prim.apply(ramp_model)

    # Test manual results
    np.testing.assert_array_equal(model_result.data, slope)

    # Test CDS
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='cds')
    np.testing.assert_array_equal(model_result.data, slope)

    # Test MCDS
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='mcds', num_coadd=3)
    np.testing.assert_array_equal(model_result.data, slope)