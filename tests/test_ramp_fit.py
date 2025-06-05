from hispecdrp import FitRampPrimitive
from hispecdrp import datamodels
import numpy as np

def test_ramp_fit(tmp_path):
    
    slope = 10
    source = np.full((100, 100), slope, dtype=np.float32)
    ramp_data = datamodels.create_ramp(source, readtime=1.0, n_reads=10, bias=0, read_noise=0, poisson_noise=False)
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramp_data['data'], dq=ramp_data['dq'], times=ramp_data['times']))

    # Setup and call the flat primitive
    rampfit_prim = FitRampPrimitive()
    model_result = rampfit_prim.apply(ramp_model)

    # Test manual results
    np.testing.assert_array_equal(model_result.data, slope)

    # Test CDS
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='cds')
    np.testing.assert_array_equal(model_result.data, slope)

    # Test MCDS
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='mcds', num_coadd=3)
    np.testing.assert_array_equal(model_result.data, slope)

    # Test JWST Likely
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='jwst_likely', max_cores=1)
    np.testing.assert_array_equal(model_result.data, slope)

    # Test JWST OLS
    model_result = rampfit_prim.apply(ramp_model, output_dir=tmp_path, method='jwst_ols', max_cores=1)
    np.testing.assert_array_equal(model_result.data, slope)
