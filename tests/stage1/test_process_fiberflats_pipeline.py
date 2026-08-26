"""Tests for ``ProcessFiberflatsPipeline``.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, ProcessFiberflatsPipeline
from hispecdrp.utils import download_gdrive_file

from tests.utils import make_sim_wavecal


def test_process_fiberflats_pipeline(tmp_path, datadir):

    # RSPEC
    fiberflats = [
        download_gdrive_file('HB.20260826.00000.00_rate-FIBERFLAT-FIB1.fits', output_dir=datadir),
        download_gdrive_file('HB.20260826.00000.00_rate-FIBERFLAT-FIB2.fits', output_dir=datadir),
        download_gdrive_file('HB.20260826.00000.00_rate-FIBERFLAT-FIB3.fits', output_dir=datadir)
    ]

    ff_models = [datamodels.open(fp) for fp in fiberflats]
    spectrograph = ff_models[0].meta.spectrograph
    order_vec = ff_models[0].make_order_vec()
    do_orders = order_vec

    wavesol = make_sim_wavecal(spectrograph)

    deg_pos = 4
    deg_width = 2
    extraction_aperture = 10
    disp_range = {
        'BSPEC' : (10, 4096-10), # Fitting range in spectral direction (pixels)
        'RSPEC' : (10, 4096-10) # Fitting range in spectral direction (pixels)
    }
    xdisp_range = {
        'BSPEC' : (70, 3935),
        'RSPEC' : (210, 4030)
    }

    primitives = {
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
            'wavesol': wavesol,
        },
        'register_fiberflat_cal' : {
            'run' : False,
        }
    }

    ff_results = ProcessFiberflatsPipeline.call(
        ff_models,
        primitives=primitives,
        output_dir=str(tmp_path)
    )

    assert isinstance(ff_results, dict)
    assert isinstance(ff_results['trace_pos'], datamodels.TracePosModel)
    assert isinstance(ff_results['trace_prof'], datamodels.TraceProfModel)
    assert isinstance(ff_results['fiberflat'], datamodels.FiberflatModel)