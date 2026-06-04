import os
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, ProcessFiberflatsPipeline, Stage1Pipeline
from hispecdrp.utils import download_gdrive_file

def test_stage1_hispec(tmp_path, datadir):
    # Download cals...
    telluric_L0_file = download_gdrive_file('HR.20231006.00000.00_rate-TELLURICS-FIB1.fits', 
    output_dir=datadir)
    for method in ('box', 'optimal'):
        prim_args['spec_extract']['method'] = method
        pipe = Stage1Pipeline(prim_args=prim_args)
        spec_model = pipe.apply(telluric_L0_file)
        fiber = '1'
        for i, order in enumerate(do_orders):
            
            spec = getattr(spec_model, f'spec_uncal_fib{fiber}')[i, :]
            err = getattr(spec_model, f'spec_uncal_err_fib{fiber}')[i, :]
            snr = np.nanmean(spec / err)
            
            print(f"Fiber {fiber} Order {order}: OPT SNR = {snr:.2f}")
            thresh = 20
            assert snr > thresh, f"Fiber {fiber} Order {order}: {method} SNR <= {thresh}"