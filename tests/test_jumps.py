from hispecdrp import datamodels
from hispecdrp import simulations
from hispecdrp import jump_detection_difference
from hispecdrp import jump_detection_jwst
import numpy as np

def test_jumps(tmp_path, datadir):
    np.random.seed(12345)

    slope = 10
    ny,nx = 10,10
    source = np.full((ny,nx), slope, dtype=np.float32)

    gain_val = 1.0
    read_noise_val = 10.0

    gain_data = np.ones((ny, nx), dtype=np.float32) * gain_val
    read_noise_data = np.ones((ny, nx), dtype=np.float32) * read_noise_val

    ramp_data = simulations.create_ramp(source, readtime=1.0, n_reads=16,
                                       gain=gain_val, flat = 1, dark = 0, bias = 1000,
                                       kTC_noise = 50, read_noise=read_noise_val, poisson_noise=True, convert_to_uint16 = True,
                                       nonlin_coeffs=None, clip_ramps = False)
    ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']),
                                      meta=ramp_data['meta'])

    # Modify rows to insert a jump of 100 on specified reads in pixel at specified column
    # Also, create a comparison array with JUMP_DET flag set
    jump_val = 100
    reads = [2, 8, 2, 5]
    rows = [5, 5, 2, 4]
    cols = [5, 5, 6, 7]
    dq_comp = np.zeros(ramp_model.dq_raw.shape, dtype=np.uint8)
    for read, row, col in zip(reads, rows, cols):
        ramp_model.data[read:, row, col] += jump_val
        dq_comp[read, row, col] = np.uint8(datamodels.DQ_FLAGS['JUMP_DET'])

    dq_pixel = np.zeros((ny,nx), dtype=np.uint32)

    # At this threshold, both methods should work
    thresh = 6.0

    jump_detection_jwst(ramp_model.data, 
                            ramp_model.dq_raw, 
                            dq_pixel, 
                            gain=gain_data,
                            rdnoise=read_noise_data,
                            rejection_threshold=thresh)

    np.testing.assert_array_equal(dq_comp, ramp_model.dq_raw)

    # Reinitialize the dq_raw array
    ramp_model.dq_raw = np.zeros(ramp_model.data.shape, dtype=np.uint8)

    thresh = 5.0

    jump_detection_difference(ramp_model.data, 
                              ramp_model.dq_raw, 
                              dq_pixel, 
                              gain=gain_data,
                              rdnoise=read_noise_data,
                              rejection_threshold=thresh)

    np.testing.assert_array_equal(dq_comp, ramp_model.dq_raw)

    # At this threshold, no jumps should be identified
    thresh = 100.0
    dq_comp = np.zeros(ramp_model.dq_raw.shape, dtype=np.uint8)
    ramp_model.dq_raw = np.zeros(ramp_model.data.shape, dtype=np.uint8)

    jump_detection_jwst(ramp_model.data, 
                            ramp_model.dq_raw, 
                            dq_pixel, 
                            gain=gain_data,
                            rdnoise=read_noise_data,
                            rejection_threshold=thresh)

    np.testing.assert_array_equal(dq_comp, ramp_model.dq_raw)

    # Reinitialize the dq_raw array
    ramp_model.dq_raw = np.zeros(ramp_model.data.shape, dtype=np.uint8)

    jump_detection_difference(ramp_model.data, 
                              ramp_model.dq_raw, 
                              dq_pixel, 
                              gain=gain_data,
                              rdnoise=read_noise_data,
                              rejection_threshold=thresh)

    np.testing.assert_array_equal(dq_comp, ramp_model.dq_raw)
