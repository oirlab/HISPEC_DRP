from hispecdrp import datamodels
from hispecdrp import jump_detection_difference
from hispecdrp import jump_detection_jwst
import numpy as np
import os

def test_jumps(tmp_path):
    
    slope = 10
    source = np.full((4096, 4096), slope, dtype=np.float32)
    ramp_data = datamodels.create_ramp(source, readtime=1.0, n_reads=16, bias=0, read_noise=0, poisson_noise=False, nonlin_coeffs=None)
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramp_data['data'], dq=ramp_data['dq'], times=ramp_data['times']))
    ramp_model.meta.spectrograph = 'BSPEC'
    datamodels.set_default_meta(ramp_model)

    # Modify rows to insert a jump of 100 on specified reads in pixel at specified column
    # Also, create a comparison array with JUMP_DET flag set
    jump_val = 100
    reads = [2, 3, 4, 5, 6]
    rows = [101, 502, 1023, 2031, 4054]
    col = 1002
    dq_comp = np.zeros(ramp_model.dq.shape, dtype=np.uint32)
    for read, row in zip(reads, rows):
        ramp_model.data[read:, row, col] += jump_val
        dq_comp[read, row, col] = np.uint32(datamodels.DQ_FLAGS['JUMP_DET'])

    dq_out = jump_detection_jwst(ramp_model.data, ramp_model.dq)
    np.testing.assert_equal(dq_out, dq_comp)

    dq_out = jump_detection_difference(ramp_model.data, ramp_model.dq)
    np.testing.assert_equal(dq_out, dq_comp)
