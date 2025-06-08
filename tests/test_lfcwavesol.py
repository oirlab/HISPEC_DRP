from hispecdrp import datamodels
from hispecdrp import FitLFCModesPrimitive, WLSChebyshev2DPrimitive
from hispecdrp.utils import download_gdrive_file

import numpy as np

def test_parvi_lfc_wavesol(tmp_path):

    lfc_path = download_gdrive_file('PARVI/Cals/PR.20230725.83766.15.lfc_wavesol_LFC.fits', use_cached=True)

    # Fit 1D LFC modes with Gaussian
    do_orders = [90, 91, 92, 93, 94, 96, 97, 98, 99]
    prim = FitLFCModesPrimitive()
    lfc_wavesol_model = prim.apply(
        lfc_path,
        full_output=False,
        do_orders=do_orders,
        do_fibers=[3],
        sigma_bounds=(0.1, 2.5),
        mu_bounds=(-1, 1),
        lfc_nu0=datamodels.parvi.LFC_NU_0,
        lfc_delta_nu=datamodels.parvi.LFC_DELTA_NU,
    )

    # Fit 2D Chebyshev polynomial to the LFC modes
    prim = WLSChebyshev2DPrimitive()
    deg_inter_order = 6
    deg_intra_order = 9
    result = prim.apply(
        lfc_wavesol_model,
        full_output=True,
        fit_orders=do_orders,
        opt_weights=True,
        do_fibers=[3],
        deg_inter_order=deg_inter_order,
        deg_intra_order=deg_intra_order,
    )
    lfc_wavesol_model = result.name
    fit_result = result.fit_result
    assert fit_result['coeffs'].shape == (deg_intra_order + 1, deg_inter_order + 1)
    assert np.std(fit_result['residuals_vel'][fit_result['good']]) / np.sqrt(len(fit_result['good'])) < 5 # NOTE: REVISIT THIS TEST!