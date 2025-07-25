from hispecdrp import datamodels
from hispecdrp import FitLFCLinesPrimitive, WLSChebyshev2DPrimitive, GaussianLSFPrimitive, ProcessLFCPipeline
#from hispecdrp.utils import download_gdrive_file
from hispecdrp.calibrations import PARVICalibrationORM
from koa_middleware import CalibrationStore

import numpy as np
import os

def test_process_lfc_parvi(tmp_path):

    #lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-LFC.fits', use_cached=True)
    lfc_path = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230725.83766.15_spec-LFC.fits'

    # Fit 1D LFC lines with Gaussian
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = [3]
    prim = FitLFCLinesPrimitive()
    line_fit_result = prim.apply(
        lfc_path,
        full_output=True,
        do_orders=do_orders,
        do_fibers=do_fibers,
        lfc_nu0=datamodels.parvi.LFC_NU_0,
        lfc_delta_nu=datamodels.parvi.LFC_DELTA_NU,
    )
    line_fits_model = line_fit_result.name
    assert isinstance(line_fits_model, datamodels.PARVILFCFitParamsModel)
    assert line_fits_model.meta.cal_type == 'lfc_fitparams'
    assert len(line_fits_model.lfc_fitparams_fib3) > 1000 # Number of modes found > 1000. NOTE: REVISIT THIS TEST!

    # Fit 2D Chebyshev polynomial to the LFC modes
    prim = WLSChebyshev2DPrimitive()
    deg_inter_order = 6
    deg_intra_order = 9
    wavesol_result = prim.apply(
        line_fits_model,
        full_output=True,
        fit_orders=do_orders,
        opt_weights=True,
        do_fibers=[3],
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
    )
    wavesol_model = wavesol_result.name
    fit_result = wavesol_result.fit_results[0]
    assert isinstance(wavesol_model, datamodels.PARVILFCWavesolModel)
    assert wavesol_model.meta.cal_type == 'lfc_wavesol'
    assert fit_result['coeffs'].shape == (deg_intra_order + 1, deg_inter_order + 1)
    # Overall contribution from wavesol over bandpass is < 5 m/s (actual ~2 for parvi) # NOTE: REVISIT THIS TEST!
    assert np.std(fit_result['residuals_vel'][fit_result['good']]) / np.sqrt(len(fit_result['good'])) < 5
    assert getattr(wavesol_model, f'wave_fib{do_fibers[0]}').shape == getattr(wavesol_model, f'spec_fib{do_fibers[0]}').shape

    # Fit 2D Chebyshev polynomial to best fit Gaussian sigmas to determine LSF width at each pixel
    prim = GaussianLSFPrimitive()
    deg_inter_order = 3
    deg_intra_order = 3
    result = prim.apply(
        wavesol_model,
        full_output=True,
        do_fibers=do_fibers,
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
        fit_orders=do_orders,
    )
    lsf_model = result.name
    assert isinstance(lsf_model, datamodels.PARVIGaussianLSFModel)
    assert lsf_model.meta.cal_type == 'lsf'
    assert getattr(lsf_model, f'lsf_params_fib{do_fibers[0]}').shape == getattr(lsf_model, f'spec_fib{do_fibers[0]}').shape

    # Test Pipeline
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'
    pipe = ProcessLFCPipeline()
    prim_args = {
        'fit_lfc_lines' : {
            'do_orders': do_orders,
            'do_fibers': do_fibers,
            'lfc_nu0': datamodels.parvi.LFC_NU_0,
            'lfc_delta_nu': datamodels.parvi.LFC_DELTA_NU,
        },
        'lfc_wavesol' : {
            'fit_orders': do_orders,
            'opt_weights': True,
            'do_fibers': do_fibers,
            'deg_inter_order': 6,
            'deg_intra_order': 9,
        },
        'lsf' : {
            'do_fibers': do_fibers,
            'deg_inter_order': 3,
            'deg_intra_order': 3,
            'fit_orders': do_orders,
        },
    }
    model_result = pipe.apply(lfc_path, prim_args=prim_args)
    assert np.all(model_result.wave_fib3 == lsf_model.wave_fib3)
    assert np.all(model_result.lfc_fitparams_fib3 == lsf_model.lfc_fitparams_fib3)
    assert np.all(model_result.lsf_params_fib3 == lsf_model.lsf_params_fib3)

    # Check that the calibration was saved to the cache
    with CalibrationStore(orm_class=PARVICalibrationORM) as store:
        results = store.local_db.query()
        assert len(results) == 2 # lfc_wavesol and lsf
        assert results[0].cal_type == 'lfc_wavesol'
        assert results[1].cal_type == 'lsf'