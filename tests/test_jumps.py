from hispecdrp import datamodels
from hispecdrp import simulations
from hispecdrp import jump_detection_difference
from hispecdrp import jump_detection_jwst
import numpy as np
import os

def test_jumps(tmp_path):
    
    slope = 10
    source = np.full((4096, 4096), slope, dtype=np.float32)
    ramp_data = simulations.create_ramp(source, readtime=1.0, n_reads=16, bias=0, read_noise=None, kTC_noise=None, poisson_noise=False, nonlin_coeffs=None)
    ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']), meta=ramp_data['meta'])

    # Modify rows to insert a jump of 100 on specified reads in pixel at specified column
    # Also, create a comparison array with JUMP_DET flag set
    jump_val = 100
    reads = [2, 3, 4, 5, 6]
    rows = [101, 502, 1023, 2031, 4054]
    col = 1002
    dq_comp = np.zeros(ramp_model.dq_raw.shape, dtype=np.uint8)
    for read, row in zip(reads, rows):
        ramp_model.data[read:, row, col] += jump_val
        dq_comp[read, row, col] = np.uint8(datamodels.DQ_FLAGS['JUMP_DET'])

    dq_pixel = np.zeros((4096, 4096), dtype=np.uint32)

    jump_detection_jwst(ramp_model.data, ramp_model.dq_raw, dq_pixel)
    np.testing.assert_equal(dq_comp, ramp_model.dq_raw)

    # Reinitialize the dq_raw array
    ramp_model.dq_raw = np.zeros(ramp_model.data.shape, dtype=np.uint8)

    jump_detection_difference(ramp_model.data, ramp_model.dq_raw, dq_pixel)
    np.testing.assert_equal(dq_comp, ramp_model.dq_raw)
