from hispecdrp import FitEtalonLinesPrimitive, CalculateEtalonDriftCorrectionPrimitive, ApplyDriftCorrectionPrimitive, SetInitialWLSPrimitive
from hispecdrp.utils import download_gdrive_file
from hispecdrp import datamodels
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import os

def test_fit_etalon_parvi(tmp_path, datadir):

    solar_files = [
        download_gdrive_file('PARVI/L2/PR.20230803.64207.77_spec-SOLAR.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L2/PR.20230803.67931.81_spec-SOLAR.fits', output_dir=datadir),
        download_gdrive_file('PARVI/L2/PR.20230803.71633.71_spec-SOLAR.fits', output_dir=datadir)
    ]
    lfc_wavesol = download_gdrive_file('PARVI/Cals/PR.20230725.83774.17.master-lfc_wavesol.fits', output_dir=datadir)
    ref_etalon = download_gdrive_file('PARVI/Cals/PR.20230725.83774.17.master-ref_etalon.fits', output_dir=datadir)
    # solar_files = [
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.64207.77_spec-SOLAR.fits',
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.67931.81_spec-SOLAR.fits',
    #     '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/L2/PR.20230803.71633.71_spec-SOLAR.fits',
    # ]
    #lfc_wavesol = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/Cals/PR.20230725.83774.17.master-lfc_wavesol.fits'
    #ref_etalon = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/PARVI/Cals/PR.20230725.83774.17.master-ref_etalon.fits'


    ref_etalon_model = datamodels.open(ref_etalon)
    lfc_wavesol_model = datamodels.open(lfc_wavesol)
    
    do_orders = np.arange(90, 100).astype(int).tolist()
    models = []

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + os.sep

    # Assign initial WLS
    for filepath in solar_files:
        prim = SetInitialWLSPrimitive()
        model = prim.apply(
            filepath,
            wavesol=lfc_wavesol,
            do_fibers=['1', '2'],
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
        assert np.all(np.abs(models[i].drift_fib1['DRIFT_VEL']) < 1000) # Check drift velocities are reasonable

    # Apply the drift correction
    for i in range(len(models)):
        prim = ApplyDriftCorrectionPrimitive()
        models[i] = prim.apply(
            models[i],
            do_orders=do_orders,
            fiber_reference_map={'2': '1'}, # Drift correction for fiber 2 comes from fiber 1. e.g, apply drift correction measured by fiber 1 to fiber 2.
        )
        for order in do_orders:
            oi = np.abs(models[i].meta.order_start - order)
            r = models[i].wave_fib2[oi, :] - lfc_wavesol_model.wave_fib2[oi, :]
            k = np.where(models[i].drift_fib1['ORDER'] == order)[0][0]
            r *= scipy.constants.c / models[0].wave_fib2[oi, :]
            assert np.allclose(r, -models[i].drift_fib1['DRIFT_VEL'][k], atol=1E-6)