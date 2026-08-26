"""Tests for ``Stage1Pipeline`` (HISPEC; ignored in default pytest).
"""
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import Stage1Pipeline
from hispecdrp.utils import download_gdrive_file

def test_stage1_pipeline_hispec(tmp_path, datadir):

    trace_pos_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-tracepos.001.fits', output_dir=datadir)
    trace_prof_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-traceprof.001.fits', output_dir=datadir)
    wave_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-sim_wavesol.001.fits', output_dir=datadir)
    telluric_L0_file = download_gdrive_file('HR.20231006.00000.00_rate-TELLURICS-FIB1.fits', output_dir=datadir)

    for method in ('box', 'optimal'):
        primitives = {
            'spec_extract' : {
                'method' : method,
                'tracepos' : trace_pos_cal,
                'traceprof' : trace_prof_cal,
            },
            'set_wavesol' : {
                'wavesol' : wave_cal,
            },
        }
        spec_model = Stage1Pipeline.call(
            telluric_L0_file,
            primitives=primitives,
            save_result=False
        )
        fiber = '1'
        do_orders = spec_model.make_order_vec()
        for i, order in enumerate(do_orders):
            
            spec = getattr(spec_model, f'spec_uncal_fib{fiber}')[i, :]
            err = getattr(spec_model, f'spec_uncal_err_fib{fiber}')[i, :]
            snr = np.nanmean(spec / err)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr:.2f}")
            thresh = 0
            assert snr > thresh, f"Fiber {fiber} Order {order}: {method} SNR <= {thresh}"