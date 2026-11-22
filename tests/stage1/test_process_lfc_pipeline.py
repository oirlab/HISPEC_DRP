import numpy as np
import pytest

from hispecdrp import ProcessLFCPipeline, Stage1Pipeline, datamodels
from tests.sim_cals import make_lfc_L0
from tests.utils import add_noise_L0

import matplotlib.pyplot as plt

@pytest.mark.cal_db
def test_process_lfc_pipeline(tmp_path):

    spectrograph = 'RSPEC'
    Nframes = 3
    do_fibers = [1, 2, 3]
    lfc_model_L0 = make_lfc_L0(spectrograph, do_fibers=do_fibers)
    lfc_models_L0 = [add_noise_L0(lfc_model_L0, copy=True, max_snr=200) for i in range(Nframes)]

    lfc_models_L1 = Stage1Pipeline.call(
        lfc_models_L0,
        save_result=False,
    )
    
    result, pipe = ProcessLFCPipeline.call(
        lfc_models_L1,
        primitives={
            'register_wave_cal': {'run': False},
            'register_lsf_cal': {'run': False},
        },
        save_result=False,
        return_instance=True,
    )

    # All three fibers carry the LFC, so they form one group with one wavesol and one LSF
    assert len(result['wavesol']) == 1
    assert len(result['lsf']) == 1
    wavesol_model = result['wavesol'][0]
    lsf_model = result['lsf'][0]
    assert isinstance(wavesol_model, datamodels.LFCWavesolModel)
    assert wavesol_model.meta.cal_type == 'lfc_wavesol'
    assert isinstance(lsf_model, datamodels.GaussianLSFModel)
    assert lsf_model.meta.cal_type == 'lsf'

    # RMS of the line residuals about the wavelength solution, per fiber
    fit_results = pipe.lfc_wavesol.fit_results
    assert len(fit_results) == 3
    # Limit on the RMS of the line residuals about the wavelength solution, per fiber (m/s)
    RMS_LIMIT = 10.0  # m/s
    for fiber, fit_result in zip(
        range(lfc_models_L1[0].num_fibers),
        fit_results
    ):
        residuals_vel = np.asarray(fit_result['residuals_vel'])[fit_result['good']]
        assert len(residuals_vel) > 4000
        rms = np.sqrt(np.mean(residuals_vel**2))
        assert rms < RMS_LIMIT, f"Fiber {fiber} line residual RMS {rms} m/s >= {RMS_LIMIT} m/s"
