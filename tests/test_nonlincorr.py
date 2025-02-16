from hispecdrp import FitRampPrimitive, NonlinearCorrectionPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels

import numpy as np

from .utils import add_metadata

import matplotlib.pyplot as plt


def create_nonlincorr_config():
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
      suffix: 'nonlincorr'
    """
    return conf

def create_ramp_config():
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
      suffix: 'rampfit'
    """
    return conf


def create_ramp_model(slope, nonlin_coeffs_inv):
    # Create ramp
    times = np.arange(1, 11).astype(np.float64)
    ramps = np.zeros(shape=(100, 100, 10), dtype=np.float32)
    counts_true = slope * times
    counts_observed = np.polyval(nonlin_coeffs_inv[::-1], counts_true)
    nonlin_coeffs = np.polyfit(counts_observed.astype(np.float64), counts_true.astype(np.float64), len(nonlin_coeffs_inv)-1)
    nonlin_coeffs = nonlin_coeffs[::-1]
    ramps[:, :, :] = counts_observed
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramps, times=times))
    ramp_model.meta.exposure.exposure_type = "SCI"
    ramp_model.meta.exposure.jd_start = 2460577.5
    ramp_model.meta.instrument.spectrograph = 'BSPEC'
    ramp_model.meta.instrument.mode = 'YJHK'
    add_metadata(ramp_model)
    return ramp_model, nonlin_coeffs


def test_ramp_fit(tmp_path):
    
    # Science file (just tellurics)
    #git_file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    #cache_dir = astropy.utils.data._get_download_cache_loc()
    #file_path = download_github_lfs_file(git_file_path, output_dir=cache_dir, preserve_github_path=True, use_cached=True)
    #file_path = '/Users/cale/Desktop/HS_BSPEC_20240924T000000_RAW-DETFLAT.fits'

    slope = 10
    nonlin_coeffs_inv = np.array([0.0, 1.0, 1E-8, -1E-10], dtype=np.float64) # NOTE: zeroth order coeff always zero, but we should keep for consistency
    #nonlin_coeffs_inv = np.array([0.0, 1.0, -1E-10], dtype=np.float64) # NOTE: zeroth order coeff always zero, but we should keep for consistency
    ramp_model, nonlin_coeffs = create_ramp_model(slope, nonlin_coeffs_inv)

    # Nonlin config file
    nonlin_config_file = str(tmp_path / 'nonlin_config.yaml')
    nonlin_config_raw = create_nonlincorr_config()
    with open(nonlin_config_file, 'w') as f:
        f.write(textwrap.dedent(nonlin_config_raw))

    # Rmap config file
    rampfit_config_file = str(tmp_path / 'ramp_config.yaml')
    rampfit_config_raw = create_ramp_config()
    with open(rampfit_config_file, 'w') as f:
        f.write(textwrap.dedent(rampfit_config_raw))

    # Setup and call the nonlin primitive
    coeffs = np.broadcast_to(nonlin_coeffs, (4096, 4096, len(nonlin_coeffs)))
    thresh = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    max = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    nonlin_model = datamodels.HISPECNonlinearCoeffModel(data=dict(coeffs=coeffs, thresh=thresh, max=max))
    nonlin_prim = NonlinearCorrectionPrimitive(config_file=nonlin_config_file)
    model_result = nonlin_prim.apply(ramp_model, output_dir=tmp_path, coeffs=nonlin_model)

    # Setup and call the flat primitive
    rampfit_prim = FitRampPrimitive(config_file=rampfit_config_file)
    model_result = rampfit_prim.apply(model_result, output_dir=tmp_path)

    # Test manual results
    np.testing.assert_array_equal(model_result.data, slope)
    np.testing.assert_array_equal(model_result.err_all, 0)
    np.testing.assert_array_equal(model_result.err_floor, 0)