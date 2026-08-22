from hispecdrp import datamodels
from hispecdrp import FitLFCLinesPrimitive, CalculateChebyshev2DWLSPrimitive, CalculateGaussianLSFPrimitive, ProcessLFCPipeline
from hispecdrp.utils import download_gdrive_file
from koa_middleware import CalibrationStore

import numpy as np
import os

def test_process_lfc_parvi(tmp_path, datadir):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    lfc_path = download_gdrive_file('PR.20230725.83774.17_spec-LFC.fits', output_dir=datadir)

    # Fit 1D LFC lines with Gaussian
    do_orders = np.array([90, 91, 92, 93, 94, 96, 97, 98, 99])
    do_fibers = ['4']
    line_fits_model = FitLFCLinesPrimitive.call(lfc_path, do_orders=do_orders, do_fibers=  do_fibers)
    assert isinstance(line_fits_model, datamodels.LFCFitParamsModel)
    assert line_fits_model.meta.cal_type == 'lfc_fitparams'
    assert len(line_fits_model.lfc_fitparams_fib4) > 1000 # Number of modes found > 1000. NOTE: REVISIT THIS TEST!

    # Fit 2D Chebyshev polynomial to the LFC modes
    deg_inter_order = 6
    deg_intra_order = 9
    wavesol_model, prim = CalculateChebyshev2DWLSPrimitive.call(
        line_fits_model,
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
        fit_orders=do_orders,
        opt_weights=True,
        do_fibers=['4'],
        return_instance=True,
    )
    fit_results = prim.fit_results
    assert isinstance(wavesol_model, datamodels.LFCWavesolModel)
    assert wavesol_model.meta.cal_type == 'lfc_wavesol'
    assert wavesol_model.cheb_coeffs_fib4.shape == (deg_intra_order + 1, deg_inter_order + 1)
    # Overall contribution from wavesol over bandpass is < 5 m/s (actual ~2 for parvi) # NOTE: REVISIT THIS TEST!
    assert np.std(fit_results[0]['residuals_vel'][fit_results[0]['good']]) / np.sqrt(len(fit_results[0]['good'])) < 5
    assert getattr(wavesol_model, f'wave_fib{do_fibers[0]}').shape == getattr(wavesol_model, f'spec_fib{do_fibers[0]}').shape

    # Fit 2D Chebyshev polynomial to best fit Gaussian sigmas to determine LSF width at each pixel
    
    lsf_model, prim = CalculateGaussianLSFPrimitive.call(
        wavesol_model,
        do_fibers=do_fibers,
        deg_inter_order=3,
        deg_intra_order=3,
        fit_orders=do_orders,
        return_instance=True
    )
    assert isinstance(lsf_model, datamodels.GaussianLSFModel)
    assert lsf_model.meta.cal_type == 'lsf'
    assert getattr(lsf_model, f'lsf_params_fib{do_fibers[0]}').shape == getattr(lsf_model, f'spec_fib{do_fibers[0]}').shape

    # TODO: Refactor this test into a pipeline (ProcessLFCPipeline) test once pipeline is implemented, and add more assertions on the final output models.