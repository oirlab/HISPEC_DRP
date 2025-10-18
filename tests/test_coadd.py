from numba.core.datamodel import models
from hispecdrp import CoaddFramesPrimitive
from hispecdrp.utils import download_gdrive_file
from hispecdrp import datamodels
import numpy as np
import os


def test_coadd_frames(tmp_path, datadir):

    # Create input models
    input_models = []
    n = 5
    shape = (512, 512)
    scale = 1E-5
    for f in range(n):
        model = datamodels.L0Model()
        model.rate = scale * np.random.randn(*shape).astype(np.float32) + 1
        model.rate_var_rnoise = np.zeros(shape, dtype=np.float32)
        model.rate_var_poisson = (model.rate / 1000.0)**2
        model.rate_err = np.sqrt(model.rate_var_rnoise + model.rate_var_poisson)
        model.dq = np.zeros(shape, dtype=np.uint32)
        model.mid_time = np.ones(shape, dtype=np.float64)
        input_models.append(model)

    # Setup and call the coadd primitive
    prim = CoaddFramesPrimitive()
    model_coadded = prim.apply(input_models, output_dir=tmp_path)

    # Test values are reasonable
    for m in input_models:
        np.testing.assert_allclose(model_coadded.rate, m.rate, rtol=1E-3, atol=0)
