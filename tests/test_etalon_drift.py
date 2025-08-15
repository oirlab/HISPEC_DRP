from hispecdrp import datamodels
from hispecdrp import FitEtalonLinesPrimitive, CalculateEtalonDriftCorrectionPrimitive, ApplyDriftCorrectionPrimitive, SetInitialWLSPrimitive
from hispecdrp.utils import download_gdrive_file

import numpy as np
import scipy.constants
import os

def test_fit_etalon_parvi(tmp_path):

    solar_files = [
        'PARVI/L2/PR.20230803.62022.01_spec-SOLAR.fits',
        'PARVI/L2/PR.20230803.65376.76_spec-SOLAR.fits',
        'PARVI/L2/PR.20230803.68709.72_spec-SOLAR.fits'
    ]
    solar_files = [download_gdrive_file(f, use_cached=True) for f in solar_files]
    lfc_wavesol = download_gdrive_file('PARVI/Cals/PR.20230725.83766.15.lfc_wavesol.fits', use_cached=True)
    ref_etalon = download_gdrive_file('PARVI/Cals/PR.20230803.62022.01.ref_etalon.fits', use_cached=True)
    # solar_files = [
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.62022.01_spec-SOLAR.fits',
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.65376.76_spec-SOLAR.fits',
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.68709.72_spec-SOLAR.fits'
    # ]
    #lfc_wavesol = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/Cals/PR.20230725.83766.15.lfc_wavesol.fits'
    do_orders = np.arange(90, 100).astype(int).tolist()
    models = []

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + os.sep

    # Assign initial WLS
    for filepath in solar_files:
        prim = SetInitialWLSPrimitive()
        model = prim.apply(
            filepath,
            wavesol=lfc_wavesol,
            do_fibers=['1', '3'],
        )
        models.append(model)

    # Fit etalon lines
    for i in range(len(models)):
        prim = FitEtalonLinesPrimitive()
        models[i] = prim.apply(
            models[i],
            do_orders=do_orders,
            do_fibers=['1'],
            fsr_estimate=30,
        )

    # Compare spectrum to itself as well to ensure zero drift and that zero drift doesn't break the statistics
    for i in range(len(models)):
        prim = CalculateEtalonDriftCorrectionPrimitive()
        models[i] = prim.apply(
            models[i],
            do_orders=do_orders,
            ref_etalon=ref_etalon,
            do_fibers=['1'],
        )
        assert np.all(np.abs(model.drift_fib1['DRIFT_VEL']) < 20) # Check drift velocities are reasonable

    # Apply the drift correction
    for i in range(len(models)):
        prim = ApplyDriftCorrectionPrimitive()
        models[i] = prim.apply(
            models[i],
            do_orders=do_orders,
            fiber_reference_map={'3': '1'}, # Drift correction for fiber 3 comes from fiber 1. In other words, apply drift correction for fiber 1 to fiber 3.
        )
        for order in do_orders:
            oi = np.abs(models[i].meta.order_start - order)
            r = models[i].wave_fib3[oi, :] - models[0].wave_fib3[oi, :]
            k = np.where(models[i].drift_fib1['ORDER'] == order)[0][0]
            r *= scipy.constants.c / models[0].wave_fib3[oi, :]
            assert np.allclose(r, -models[i].drift_fib1['DRIFT_VEL'][k])