import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import Stage1Pipeline
from hispecdrp.utils import download_gdrive_file

def test_stage1_hispec(tmp_path, datadir):

    trace_pos_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-tracepos.001.fits', output_dir=datadir)
    trace_prof_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-traceprof.001.fits', output_dir=datadir)
    wave_cal = download_gdrive_file('HR.20260606.00000.00.mastercal-sim_wavesol.001.fits', output_dir=datadir)
    telluric_L0_file = download_gdrive_file('HR.20231006.00000.00_rate-TELLURICS-FIB1.fits', output_dir=datadir)

    for method in ('box', 'optimal'):
        pipe = Stage1Pipeline(save_result=False)
        pipe.spec_extract.method = method
        pipe.spec_extract.tracepos = trace_pos_cal
        pipe.spec_extract.traceprof = trace_prof_cal
        pipe.set_wavesol.wavesol = wave_cal
        spec_model = pipe.apply(telluric_L0_file)
        fiber = '1'
        do_orders = spec_model.make_order_vec()
        for i, order in enumerate(do_orders):
            
            spec = getattr(spec_model, f'spec_uncal_fib{fiber}')[i, :]
            err = getattr(spec_model, f'spec_uncal_err_fib{fiber}')[i, :]
            snr = np.nanmean(spec / err)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr:.2f}")
            thresh = 0
            assert snr > thresh, f"Fiber {fiber} Order {order}: {method} SNR <= {thresh}"