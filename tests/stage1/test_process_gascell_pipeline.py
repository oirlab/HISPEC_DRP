import matplotlib
import numpy as np
import pytest
import scipy.constants

from hispecdrp import ProcessGasCellPipeline, Stage1Pipeline, datamodels
from hispecdrp.simulations.simulate_rate_map import SimulateRateMap
from hispecdrp.simulations.simulate_up_the_ramp import stack_wavelength_solution_rows
from tests.sim_cals import make_gascell_L0
from tests.utils import add_noise_L0

import matplotlib.pyplot as plt


@pytest.mark.cal_db
def test_process_gascell_pipeline():

    # The gas cell is only used with RSPEC
    spectrograph = 'RSPEC'
    Nframes = 3
    do_fibers = [1]
    gascell_model_L0 = make_gascell_L0(do_fibers=do_fibers)
    gascell_models_L0 = [add_noise_L0(gascell_model_L0, max_snr=200) for i in range(Nframes)]

    gascell_models_L1 = Stage1Pipeline.call(
        gascell_models_L0,
        save_result=False,
    )

    # The LSF comes from the LFC cals in the cal DB
    result, pipe = ProcessGasCellPipeline.call(
        gascell_models_L1,
        primitives={
            'register_wave_cal': {'run': False},
            # Simulated from the FTS reference itself, so no FTS broadening to remove
            'gascell_wavesol': {'fts_resolution': np.inf, 'continuum_deg_intra_order': 3},
        },
        save_result=False,
        return_instance=True,
    )

    # The gas cell fibers form one group with one wavesol
    assert len(result) == 1
    wavesol_model = result[0]
    assert isinstance(wavesol_model, datamodels.GasCellWavesolModel)
    assert wavesol_model.meta.cal_type == 'gascell_wavesol'

    # True wavelength solution: the HDF solution the rate map was simulated with
    sim_wavesol = SimulateRateMap().get_wavelength_solution(spectrograph=spectrograph)
    order_vec = wavesol_model.make_order_vec()
    Ncols = wavesol_model.wave_fib1.shape[1]

    fit_results = pipe.gascell_wavesol._output['fit_results']
    assert len(fit_results) == len(do_fibers)
    # The HDF solution clamps the first pixel(s) of some orders, so leave out the edge pixels
    edge = slice(4, -4)
    RED_CHI2_TOL = 12.0  # |chi2_red - 1|
    RMS_LIMIT = 75.0  # m/s
    for fiber, fit_result in zip(do_fibers, fit_results):
        assert fit_result['info']['converged'], f"Fiber {fiber} fit did not converge"
        assert abs(fit_result['chi2_red'] - 1) < RED_CHI2_TOL, f"Fiber {fiber} chi2_red {fit_result['chi2_red']:.3f}"

        # RMS of the fitted minus true velocity on the fitted orders
        oi = np.array([np.flatnonzero(order_vec == order)[0] for order in fit_result['orders']])
        wave_true = stack_wavelength_solution_rows(sim_wavesol[fiber], order_vec, n_pix=Ncols)[oi, edge]
        wave = getattr(wavesol_model, f'wave_fib{fiber}')[oi, edge]
        dv = (wave - wave_true) / wave_true * scipy.constants.c
        rms = np.sqrt(np.mean(dv**2))
        assert rms < RMS_LIMIT, f"Fiber {fiber} wavelength RMS vs. truth {rms:.2f} m/s >= {RMS_LIMIT} m/s"
