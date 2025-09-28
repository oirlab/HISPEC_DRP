from hispecdrp import Stage0Pipeline
from hispecdrp import datamodels
import numpy as np
import os
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.ramp_fitting.utils import get_mid_delta_times_ramp,get_ramp_chunks

from hispecdrp.primitives.dq_init.dqinit_primitive import apply_dq_mask
from hispecdrp.primitives.saturation.saturation_primitive import _check_saturation_numba
from hispecdrp.primitives.bias_subtraction.bias_primitive import _bias_subtraction_numba
from hispecdrp.primitives.dark_subtraction.dark_primitive import _dark_subtraction_rate_numba
from hispecdrp.primitives.detector_gain.detgain_primitive import _detector_gain_correction_numba
from hispecdrp.primitives.detector_flat.detflat_primitive import _detector_flat_correction_numba

def test_stage0_units(tmp_path):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    if 1:
        dq_raw = np.zeros((3, 2, 2), dtype=np.uint8)
        dq_model = np.zeros((2, 2), dtype=np.uint32)
        dq_model[1, 1] |= DQ_FLAGS['DO_NOT_USE']
        apply_dq_mask(dq_raw, dq_model)
        np.testing.assert_array_equal(np.sum(dq_raw & DQ_FLAGS["DO_NOT_USE"]), 3 * DQ_FLAGS["DO_NOT_USE"],
                                      err_msg='Error in DQ init check')
        # exit()

    if 1:
        saturation_threshold = np.iinfo(np.uint16).max  # DN
        dq_raw = np.zeros((3, 2, 2), dtype=np.uint8)
        dq = np.zeros((2, 2), dtype=np.uint32)
        data_raw = np.zeros((3, 2, 2), dtype=np.uint16)
        data_raw[2::, :, :] = saturation_threshold
        data_raw[1::, 1, 1] = saturation_threshold
        saturation_map = np.full((2, 2), saturation_threshold, dtype=np.float32)
        sat_dq = np.zeros((2, 2), dtype=np.uint32)
        _check_saturation_numba(
            data_raw,
            dq_raw,
            dq,
            saturation_map,
            sat_dq,
        )
        # print(dq_raw)
        np.testing.assert_array_equal(np.sum(dq_raw & DQ_FLAGS["SATURATED"]), 5 * DQ_FLAGS["SATURATED"],
                                      err_msg='Error in mid_time calculation with no DQ flags')
        # exit()

        if 1:
            # Fake ramp data: shape (n_read=3, ny=2, nx=2)
            data_raw = np.array(
                [
                    [[10, 20],
                     [30, 40]],
                    [[15, 25],
                     [35, 45]],
                    [[20, 30],
                     [40, 50]]
                ],
                dtype=np.float32
            )

            dq = np.zeros((2, 2), dtype=np.uint32)

            # Bias image (ny=2, nx=2)
            bias_map = np.array([[1, 2],
                                 [3, 4]], dtype=np.float32)

            # Bias DQ map
            bias_dq = np.array([[0, 1],
                                [2, 4]], dtype=np.uint32)

            # Run the numba kernel
            _bias_subtraction_numba(
                data_raw,
                dq,
                bias_map,
                bias_dq
            )

            # print("Data after bias subtraction:\n", data_out)
            # print("DQ after bias propagation:\n", dq_out)

            # --- Assertions ---
            # Check subtraction: data_out[0,0,0] should be 10 - 1 = 9
            np.testing.assert_array_equal(
                data_raw[0, :, :],
                np.array([[9, 18],
                          [27, 36]], dtype=np.float32),
                err_msg="Bias subtraction failed on first read"
            )

            # Check DQ propagation: all reads at (1,1) should have bias_dq[1,1] = 4
            np.testing.assert_array_equal(
                dq[1, 1],
                np.full(3, 4, dtype=np.uint32),
                err_msg="Bias DQ propagation failed"
            )

            # exit()
    if 1:
        # Fake 2D rate image
        rate = np.array([[10.0, 20.0],
                         [30.0, 40.0]], dtype=np.float32)

        # Error map
        err = np.array([[1.0, 2.0],
                        [3.0, 4.0]], dtype=np.float32)

        # DQ map
        dq = np.zeros((2, 2), dtype=np.uint32)

        # Dark calibration
        dark_rate = np.array([[1.0, 2.0],
                              [3.0, 4.0]], dtype=np.float32)
        dark_err = np.array([[0.5, 0.5],
                             [1.0, 1.0]], dtype=np.float32)
        dark_dq = np.array([[0, 1],
                            [2, 4]], dtype=np.uint32)

        # Run kernel
        rate_out, err_out, dq_out = _dark_subtraction_rate_numba(
            rate.copy(), err.copy(), dq.copy(),
            dark_rate, dark_err, dark_dq
        )

        # print("Rate after dark subtraction:\n", rate_out)
        # print("Error after propagation:\n", err_out)
        # print("DQ after propagation:\n", dq_out)

        # Assertions
        np.testing.assert_array_equal(
            rate_out,
            np.array([[9.0, 18.0],
                      [27.0, 36.0]], dtype=np.float32),
            err_msg="Dark subtraction failed"
        )

        np.testing.assert_array_equal(
            dq_out,
            np.array([[0, 1],
                      [2, 4]], dtype=np.uint32),
            err_msg="Dark DQ propagation failed"
        )

        np.testing.assert_almost_equal(
            err_out[1, 0],
            np.sqrt(3.0 ** 2 + 1.0 ** 2), decimal=6,
            err_msg="Error propagation incorrect"
        )

        # exit()

    if 1:
        # Fake rate image (2D)
        rate = np.array([[10.0, 20.0],
                         [30.0, 40.0]], dtype=np.float32)

        # Error map
        err = np.array([[1.0, 2.0],
                        [3.0, 4.0]], dtype=np.float32)

        # DQ map
        dq = np.zeros((2, 2), dtype=np.uint32)

        # Gain calibration
        gain = np.array([[2.0, 4.0],
                         [0.5, 1.0]], dtype=np.float32)
        # Gain error
        gain_err = np.array([[0.2, 0.4],
                             [0.05, 0.1]], dtype=np.float32)
        gain_dq = np.array([[0, 1],
                            [2, 4]], dtype=np.uint32)

        # Run kernel
        rate_out, err_out, dq_out = _detector_gain_correction_numba(
            rate.copy(), err.copy(), dq.copy(),
            gain, gain_err, gain_dq
        )

        print("Rate after gain correction:\n", rate_out)
        print("Error after scaling:\n", err_out)
        print("DQ after propagation:\n", dq_out)

        # Assertions
        np.testing.assert_array_equal(
            rate_out,
            np.array([[20.0, 80.0],
                      [15.0, 40.0]], dtype=np.float32),
            err_msg="Gain correction failed"
        )

        np.testing.assert_array_equal(
            dq_out,
            np.array([[0, 1],
                      [2, 4]], dtype=np.uint32),
            err_msg="Gain DQ propagation failed"
        )

        # Error scaling check
        for i in range(2):
            for j in range(2):
                np.testing.assert_almost_equal(
                    err_out[i, j],
                    rate[i, j] * gain[i, j] * np.sqrt(
                        (err[i, j] / rate[i, j]) ** 2 + (gain_err[i, j] / gain[i, j]) ** 2), decimal=6,
                    err_msg="Error propagation incorrect in gain correction"
                )
        # exit()

    if 1:

        # 2D inputs (ny=2, nx=2)
        rate = np.array([[10.0, 20.0],
                         [30.0, 40.0]], dtype=np.float32)

        # Error map
        err = np.array([[1.0, 2.0],
                        [3.0, 4.0]], dtype=np.float32)

        # DQ map
        dq = np.zeros((2, 2), dtype=np.uint32)

        # Flat calibration
        flat = np.array([[2.0, 4.0],
                         [0.5, 1.0]], dtype=np.float32)

        # Flat error
        flat_err = np.array([[0.2, 0.4],
                             [0.05, 0.1]], dtype=np.float32)

        # Flat DQ
        flat_dq = np.array([[0, 1],
                            [2, 4]], dtype=np.uint32)

        # Run kernel
        rate_out, err_out, dq_out = _detector_flat_correction_numba(
            rate.copy(), err.copy(), dq.copy(),
            flat, flat_err, flat_dq
        )

        print("Rate after flat correction:\n", rate_out)
        print("Error after propagation:\n", err_out)
        print("DQ after propagation:\n", dq_out)

        # --- Assertions ---

        # Rate: elementwise division rate/flat
        np.testing.assert_array_equal(
            rate_out,
            np.array([[10.0 / 2.0, 20.0 / 4.0],
                      [30.0 / 0.5, 40.0 / 1.0]], dtype=np.float32),
            err_msg="Flat correction (rate division) failed"
        )

        # DQ propagation: OR with flat_dq
        np.testing.assert_array_equal(
            dq_out,
            np.array([[0, 1],
                      [2, 4]], dtype=np.uint32),
            err_msg="Flat DQ propagation failed"
        )

        for i in range(2):
            for j in range(2):
                np.testing.assert_almost_equal(
                    err_out[i, j],
                    rate[i, j] / flat[i, j] * np.sqrt(
                        (err[i, j] / rate[i, j]) ** 2 + (flat_err[i, j] / flat[i, j]) ** 2), decimal=6,
                    err_msg="Error propagation incorrect in flat correction"
                )


    # --- Test get_mid_delta_times_ramp function with various DQ flags ---
    if 1:
        n_reads = 4
        times = np.arange(n_reads)
        dq_raw = np.zeros((n_reads), dtype=np.uint32)
        mid_time = get_mid_delta_times_ramp(times, dq_raw)
        np.testing.assert_array_equal(mid_time, 1.5, err_msg='Error in mid_time calculation with no DQ flags')

        n_reads = 6
        times = np.arange(n_reads)
        dq_raw = np.zeros((n_reads), dtype=np.uint32)
        dq_raw[[0, 3]] |= DQ_FLAGS['DO_NOT_USE']
        dq_raw[[5]] |= DQ_FLAGS['SATURATED']
        print(dq_raw[:])
        mid_time = get_mid_delta_times_ramp(times, dq_raw)
        np.testing.assert_array_equal(mid_time, 1.5, err_msg='Error in mid_time calculation with DO_NOT_USE flags')

        n_reads = 5
        times = np.arange(n_reads)
        dq_raw = np.zeros((n_reads), dtype=np.uint32)
        dq_raw[[2]] |= DQ_FLAGS['JUMP_DET']
        print(dq_raw[:])
        mid_time = get_mid_delta_times_ramp(times, dq_raw)
        np.testing.assert_array_equal(mid_time, (0.5 + 2.5 + 3.5) / 3.0,
                                      err_msg='Error in mid_time calculation with JUMP_DET flag')


    # --- Test get_ramp_chunks function with various DQ flags ---

    if 1:
        # test all saturated return empty
        n_reads = 4
        dq_raw = np.ones((n_reads), dtype=np.uint8)
        chunks = get_ramp_chunks(dq_raw)
        np.testing.assert_array_equal(len(chunks[0]), 0, err_msg='Error: chunks should be all empty when all bad')
        np.testing.assert_array_equal(len(chunks[1]), 0, err_msg='Error: chunks should be all empty when all bad')

        # test all good
        n_reads = 4
        dq_raw = np.zeros((n_reads), dtype=np.uint8)
        chunks = get_ramp_chunks(dq_raw)
        np.testing.assert_array_equal(chunks[0], [0], err_msg='Error when all reads are good')
        np.testing.assert_array_equal(chunks[1], [4], err_msg='Error when all reads are good')

        # test all good
        n_reads = 4
        dq_raw = np.zeros((n_reads), dtype=np.uint8)
        dq_raw[[3]] |= DQ_FLAGS['SATURATED']
        chunks = get_ramp_chunks(dq_raw)
        np.testing.assert_array_equal(chunks[0], [0], err_msg='Error when saturation')
        np.testing.assert_array_equal(chunks[1], [3], err_msg='Error when saturation')

        # test maximum number of jumps
        n_reads = 6
        dq_raw = np.zeros((n_reads), dtype=np.uint8)
        dq_raw[[2, 4]] |= DQ_FLAGS['JUMP_DET']
        chunks = get_ramp_chunks(dq_raw)
        np.testing.assert_array_equal(chunks[0], [0, 2, 4], err_msg='Error when maximum number of jumps')
        np.testing.assert_array_equal(chunks[1], [2, 4, 6], err_msg='Error when maximum number of jumps')
        dq_raw[[2, 4]] |= DQ_FLAGS['JUMP_DET']
        dq_raw[[3]] |= DQ_FLAGS['DO_NOT_USE']
        chunks = get_ramp_chunks(dq_raw)
        np.testing.assert_array_equal(chunks[0], [0, 4], err_msg='Error when maximum number of jumps')
        np.testing.assert_array_equal(chunks[1], [2, 6], err_msg='Error when maximum number of jumps')

