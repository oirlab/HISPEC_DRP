from hispecdrp import datamodels
from hispecdrp import FitLFCLinesPrimitive, CalculateChebyshev2DWLSPrimitive, CalculateGaussianLSFPrimitive, ProcessLFCPipeline
from hispecdrp.utils import download_gdrive_file
from koa_middleware import CalibrationStore

import numpy as np
import os

def test_process_lfc_parvi(tmp_path, datadir):

    lfc_path = download_gdrive_file('PARVI/L2/PR.20230725.83774.17_spec-LFC.fits', use_cached=True, output_dir=datadir)

    # Fit 1D LFC lines with Gaussian
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    do_fibers = ['2']
    prim = FitLFCLinesPrimitive()
    line_fits_model = prim.apply(
        lfc_path,
        do_orders=do_orders,
        do_fibers=do_fibers,
    )
    assert isinstance(line_fits_model, datamodels.PARVILFCFitParamsModel)
    assert line_fits_model.meta.cal_type == 'lfc_fitparams'
    assert len(line_fits_model.lfc_fitparams_fib2) > 1000 # Number of modes found > 1000. NOTE: REVISIT THIS TEST!

    # Fit 2D Chebyshev polynomial to the LFC modes
    prim = CalculateChebyshev2DWLSPrimitive()
    deg_inter_order = 6
    deg_intra_order = 9
    wavesol_model = prim.apply(
        line_fits_model,
        fit_orders=do_orders,
        opt_weights=True,
        do_fibers=['2'],
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
    )
    fit_results = prim.fit_results
    assert isinstance(wavesol_model, datamodels.PARVILFCWavesolModel)
    assert wavesol_model.meta.cal_type == 'lfc_wavesol'
    assert wavesol_model.cheb_coeffs_fib2.shape == (deg_intra_order + 1, deg_inter_order + 1)
    # Overall contribution from wavesol over bandpass is < 5 m/s (actual ~2 for parvi) # NOTE: REVISIT THIS TEST!
    assert np.std(fit_results[0]['residuals_vel'][fit_results[0]['good']]) / np.sqrt(len(fit_results[0]['good'])) < 5
    assert getattr(wavesol_model, f'wave_fib{do_fibers[0]}').shape == getattr(wavesol_model, f'spec_fib{do_fibers[0]}').shape

    # Fit 2D Chebyshev polynomial to best fit Gaussian sigmas to determine LSF width at each pixel
    prim = CalculateGaussianLSFPrimitive()
    deg_inter_order = 3
    deg_intra_order = 3
    
    lsf_model = prim.apply(
        wavesol_model,
        do_fibers=do_fibers,
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
        fit_orders=do_orders,
    )
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
    result = pipe.apply(lfc_path, prim_args=prim_args)
    lsf_model = result['lsf']

    # Check that the calibration was saved to the cache
    with CalibrationStore(orm_class=lsf_model.get_calibration_orm_class()) as store:
        results = store.local_db.query()
        assert len(results) == 2, f"Found {len(results)} in DB, expected 2." # lfc_wavesol and lsf
        assert results[0].cal_type == 'lfc_wavesol'
        assert results[1].cal_type == 'lsf'