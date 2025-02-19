from hispecdrp import FitRampPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels
import numpy as np

from .utils import add_metadata

def create_rampfit_config():
    conf = """
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      save_results: True
      suffix: 'ramp-fit'
    """
    return conf


def create_ramp_model(slope):
    # Create ramp
    times = np.arange(10).astype(np.float64)
    ramps = np.zeros(shape=(100, 100, 10), dtype=np.float32)
    for i in range(10):
        ramps[:, :, i] = slope * times[i]
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramps, times=times))
    ramp_model.meta.exposure.exposure_type = "SCI"
    ramp_model.meta.exposure.jd_start = 2460577.5
    ramp_model.meta.instrument.spectrograph = 'BSPEC'
    ramp_model.meta.instrument.mode = 'YJHK'
    add_metadata(ramp_model)
    return ramp_model


def test_ramp_fit(tmp_path):
    
    slope = 10
    ramp_model = create_ramp_model(slope=slope)

    # Config file
    rampfit_config_file = str(tmp_path / 'ramp_config.yaml')
    rampfit_config_raw = create_rampfit_config()
    with open(rampfit_config_file, 'w') as f:
        f.write(textwrap.dedent(rampfit_config_raw))

    # Setup and call the flat primitive
    rampfit_prim = FitRampPrimitive(config_file=rampfit_config_file)
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path)

    # Test manual results
    np.testing.assert_array_equal(model_result.data, slope)
    np.testing.assert_array_equal(model_result.err_all, 0)
    np.testing.assert_array_equal(model_result.err_floor, 0)