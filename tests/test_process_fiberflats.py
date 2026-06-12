import os
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, ProcessFiberflatsPipeline
from hispecdrp.utils import download_gdrive_file


def test_process_fiberflats(tmp_path, datadir):

    # RSPEC
    fiberflats = [
        download_gdrive_file('HR.20260606.00000.00_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('HR.20260606.00000.00_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        download_gdrive_file('HR.20260606.00000.00_rate-FIBERFLAT-FIB3.fits', output_dir=datadir)
    ]

    ff_models = [datamodels.open(fp) for fp in fiberflats]
    spectrograph = ff_models[0].meta.spectrograph
    order_vec = ff_models[0].make_order_vec()
    do_orders = order_vec

    deg_pos = 4
    deg_width = 2
    extraction_aperture = 10
    disp_range = {
        'BSPEC' : (300, 4000), # Fitting range in spectral direction (pixels)
        'RSPEC' : (400, 4000) # Fitting range in spectral direction (pixels)
    }
    xdisp_range = {
        'BSPEC' : (135, 4050), # Which orders to consider (XD direction)
        'RSPEC' : (100, 4050) # Which orders to consider (XD direction)
    }

    prim_args = {
        'trace_pos' : {
            'deg_pos' : deg_pos,
            'deg_width' : deg_width,
            'disp_range' : disp_range[spectrograph],
            'xdisp_range' : xdisp_range[spectrograph],
        },
        'spec_extract' : {
            'extraction_aperture' : extraction_aperture,
            'do_orders' : do_orders,
        },
        'register_tracepos_cal' : {
            'run' : False,
        },
        'register_traceprof_cal' : {
            'run' : False,
        },
        'set_wavesol' : {
            'run' : False,
        },
        'register_fiberflat_cal' : {
            'run' : False,
        }
    }

    pipe = ProcessFiberflatsPipeline(prim_args=prim_args, output_dir=str(tmp_path))
    ff_results = pipe.apply(ff_models)

    assert isinstance(ff_results, dict)
    assert isinstance(ff_results['trace_pos'], datamodels.TracePosModel)
    assert isinstance(ff_results['trace_prof'], datamodels.TraceProfModel)
    assert isinstance(ff_results['fiberflat'], datamodels.FiberflatModel)