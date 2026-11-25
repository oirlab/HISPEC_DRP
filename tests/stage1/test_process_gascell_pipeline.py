import numpy as np
import pytest
import scipy.constants

from hispecdrp import ProcessGasCellPipeline, Stage1Pipeline, datamodels
from hispecdrp.primitives import wavesol
from hispecdrp.simulations.simulate_rate_map import SimulateRateMap
from hispecdrp.simulations.simulate_up_the_ramp import stack_wavelength_solution_rows
from tests.sim_cals import make_gascell_L0
from tests.utils import add_noise_L0


@pytest.mark.cal_db
def test_process_gascell_pipeline():

    # The gas cell is only used with RSPEC
    spectrograph = 'RSPEC'
    Nframes = 3
    do_fibers = [1]
    gascell_model_L0 = make_gascell_L0(do_fibers=do_fibers)
    do_orders = gascell_model_L0.orders#[:10]
    gascell_models_L0 = [add_noise_L0(gascell_model_L0, copy=True, max_snr=200) for i in range(Nframes)]

    gascell_models_L1 = Stage1Pipeline.call(
        gascell_models_L0,
        save_result=False,
        primitives={
            'spec_extract': {'do_fibers': do_fibers, 'do_orders': do_orders},
            'set_wavesol': {'do_fibers': do_fibers, 'do_orders': do_orders}
        },
    )

    # The LSF comes from the LFC cals in the cal DB
    result, pipe = ProcessGasCellPipeline.call(
        gascell_models_L1,
        primitives={
            'register_wave_cal': {'run': False},
            'gascell_wavesol': {
                'continuum_deg_intra_order': 1,
                'wave_deg_intra_order': 6,
                'wave_deg_inter_order' : 6,
                'fts_resolution': False,
            },
        },
        save_result=False,
        return_instance=True,
    )

    # The gas cell fibers form one group with one wavesol
    assert len(result) == 1
    wavesol_model = result[0]
    assert isinstance(wavesol_model, datamodels.GasCellWavesolModel)
    assert wavesol_model.meta.cal_type == 'gascell_wavesol'

    # HDF5 wavelength solution for comparison
    sim_wavesol = SimulateRateMap().get_wavelength_solution(spectrograph=spectrograph)

    fit_results = pipe.gascell_wavesol._output['fit_results']
    assert len(fit_results) == len(do_fibers)
    use_pix = slice(4, -4)
    RED_CHI2_THRESH = 6.5
    RMS_THRESH = 80.0  # m/s
    for fiber, fit_result in zip(do_fibers, fit_results):

        # Fit converged
        assert fit_result['info']['converged'], f"Fiber {fiber} fit did not converge"

        # Check each order
        for order in fit_result['orders']:
            oi = np.where(fit_result['orders'] == order)[0][0]
            wave_gascell = wavesol_model.wave[fiber][oi][use_pix]
            wave_zemax = sim_wavesol[fiber][order]['wavelength'][use_pix]

            # Reduced chi2 for this order
            red_chi2 = fit_result['chi2_red_orders'][oi]
            assert red_chi2 < RED_CHI2_THRESH, f"Fiber {fiber} order {order} reduced chi2 {red_chi2} >= {RED_CHI2_THRESH}"

            # Check the accuracy relative to Zemax
            res_vel = (wave_gascell - wave_zemax) / wave_zemax * scipy.constants.c
            rms_vel = np.sqrt(np.mean(res_vel**2))
            assert rms_vel < RMS_THRESH, f"Fiber {fiber} order {order} wavelength RMS vs. truth {rms_vel} m/s >= {RMS_THRESH} m/s"