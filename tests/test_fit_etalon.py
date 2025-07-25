from hispecdrp import datamodels
from hispecdrp import FitEtalonLinesPrimitive
from hispecdrp.utils import download_gdrive_file

import numpy as np
import os

def test_fit_etalon_parvi(tmp_path):

    #etalon_path = download_gdrive_file('PARVI/L2/PR.20230725.83766.15_spec-ETALON.fits', use_cached=True)
    etalon_path = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230725.83766.15_spec-ETALON.fits'

    # Fit 1D LFC lines with Gaussian
    do_orders = np.arange(90, 120).astype(int).tolist()
    do_fibers = [1]
    prim = FitEtalonLinesPrimitive()
    line_fit_result = prim.apply(
        etalon_path,
        full_output=True,
        do_orders=do_orders,
        do_fibers=do_fibers,
        fsr_estimate=30,
    )
    line_fits_model = datamodels.open(line_fit_result.name)
    assert isinstance(line_fits_model, datamodels.PARVIEtalonFitParamsModel)
    assert line_fits_model.meta.cal_type == 'etalon_fitparams'
    assert len(line_fits_model.etalon_fitparams_fib1) > 1000 # Number of modes found > 1000. NOTE: REVISIT THIS TEST!