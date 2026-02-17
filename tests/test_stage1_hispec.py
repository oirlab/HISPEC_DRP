import os
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, ProcessFiberflatsPipeline, Stage1Pipeline
from hispecdrp.utils import download_gdrive_file

def test_stage1_hispec(tmp_path, datadir):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    fiberflat_filepaths = [
        download_gdrive_file('HR.20231006.00000.00_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('HR.20231006.00000.00_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        download_gdrive_file('HR.20231006.00000.00_rate-FIBERFLAT-FIB3.fits', output_dir=datadir)
    ]
    fibers = ['1', '2', '3']
    ff_models = [datamodels.open(fp) for fp in fiberflat_filepaths]

    yrange = (500, 4000) # Fitting range in spectral direction (pixels)
    xrange = (100, 4050) # Which orders to consider (XD direction)
    deg_pos = 4
    deg_width = 2
    order_vec = ff_models[0].make_order_vec()

    # # Extract Spectra per fiber/order
    extraction_aperture = 10
    do_orders = order_vec

    prim_args = {
        'trace_pos' : {
            'deg_pos' : deg_pos,
            'deg_width' : deg_width,
            'yrange' : yrange,
            'xrange' : xrange,
        },
        'spec_extract' : {
            'extraction_aperture' : extraction_aperture,
            'do_orders' : do_orders,
        },
        'set_wavesol' : {
            'run' : False,
        },
        'register_fiberflat_cal' : {
            'run' : False,
        }
    }

    # Test Fiberflat Pipeline
    pipe = ProcessFiberflatsPipeline(prim_args=prim_args)

    ff_results = pipe.apply(ff_models)

    trace_pos_model = ff_results['trace_pos']
    trace_prof_model = ff_results['trace_prof']
    # fiberflat1d = ff_results['fiberflat']

    # Test Stage1 Pipeline
    prim_args = {
        'spec_extract' : {
            'extraction_aperture' : extraction_aperture,
            'do_orders' : do_orders,
            'do_fibers' : fibers,
            'tracepos' : trace_pos_model,
            'traceprof' : trace_prof_model,
        },
        'set_wavesol' : {
            'run' : False,
        },
    }

    telluric_L0_file = download_gdrive_file('HR.20231006.00000.00_rate-TELLURICS-FIB1.fits', output_dir=datadir)
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