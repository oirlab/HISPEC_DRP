from hispecdrp import Stage0Pipeline
from hispecdrp import datamodels
import numpy as np


def test_stage0(tmp_path):
    
    coeffs = np.broadcast_to(np.array([[[0]], [[1]]], dtype=np.float32), (2, 4096, 4096))
    thresh = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    max = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    nonlin_model = datamodels.HISPECNonlinearCorrectionModel(data=dict(coeffs=coeffs, thresh=thresh, max=max))
    nonlin_model.meta.instrument.spectrograph = 'BSPEC'
    datamodels.set_hispec_meta(nonlin_model)
    
    slope = 10
    source = np.full((4096, 4096), slope, dtype=np.float32)
    ramp_data = datamodels.create_ramp(source, readtime=1.0, n_reads=10, bias=0, read_noise=0, poisson_noise=False, nonlin_coeffs=None)
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramp_data['data'], dq=ramp_data['dq'], times=ramp_data['times']))
    ramp_model.meta.instrument.spectrograph = 'BSPEC'
    datamodels.set_hispec_meta(ramp_model)

    # Setup and call the stage 0 pipeline
    stage0_pipe = Stage0Pipeline()

    # Set any primitive args
    stage0_pipe.nonlin_corr.nonlin = nonlin_model

    # Run pipeline
    model_result = stage0_pipe.apply(ramp_model, output_dir=tmp_path)
    
    # Test against manual correction? (True for now)
    np.testing.assert_array_equal(model_result.data, slope)
