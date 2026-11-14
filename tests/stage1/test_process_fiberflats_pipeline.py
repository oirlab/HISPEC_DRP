"""Tests for ``ProcessFiberflatsPipeline``.
"""
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from hispecdrp import datamodels, ProcessFiberflatsPipeline

from tests.sim_cals import make_fiberflat_L0


@pytest.mark.cal_db
def test_process_fiberflats_pipeline(tmp_path):

    # Simulated BSPEC fiberflat in each fiber
    spectrograph = 'BSPEC'
    do_fibers = [1, 2, 3]
    ff_models = [make_fiberflat_L0(
        spectrograph,
        fiber,
        str(tmp_path)
    ) for fiber in do_fibers]

    primitives = {
        'register_tracepos_cal' : {
            'run' : False,
        },
        'register_traceprof_cal' : {
            'run' : False,
        },
        'register_fiberflat_cal' : {
            'run' : False,
        }
    }

    ff_results = ProcessFiberflatsPipeline.call(
        ff_models,
        primitives=primitives,
        output_dir=str(tmp_path),
        use_derived_trace_cals=True
    )

    assert isinstance(ff_results, dict)
    assert isinstance(ff_results['trace_pos'], datamodels.TracePosModel)
    assert isinstance(ff_results['trace_prof'], datamodels.TraceProfModel)
    assert isinstance(ff_results['fiberflat'], datamodels.FiberflatModel)

    # Test that the generated trace positions fall within the detector
    trace_pos = ff_results['trace_pos']
    for fiber in do_fibers:
        assert np.all(trace_pos.pos[fiber] >= -0.5)
        assert np.all(trace_pos.pos[fiber] <= ff_models[0].rate.shape[1] - 0.5)