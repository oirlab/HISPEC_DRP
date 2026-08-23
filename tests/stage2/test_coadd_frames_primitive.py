"""Tests for ``CoaddFramesPrimitive``.
"""
from hispecdrp import CoaddFramesPrimitive
from hispecdrp import datamodels
import numpy as np


def test_coadd_frames_primitive(tmp_path, datadir):

    np.random.seed(1)

    # Create input models
    input_models = []
    n = 5
    shape = (64, 64)
    rate = 1.0
    sigma_rnoise = 0.1
    sigma_poisson = 0.1
    sigma = np.sqrt(sigma_rnoise**2 + sigma_poisson**2)

    data_true = np.full(shape, rate, dtype=np.float32)
    for f in range(n):
        model = datamodels.L0Model()
        model.rate_var_rnoise = np.full(shape, sigma_rnoise**2, dtype=np.float32)
        model.rate_var_poisson = np.full(shape, sigma_poisson**2, dtype=np.float32)
        model.rate_err = np.sqrt(model.rate_var_rnoise + model.rate_var_poisson)
        model.rate = np.random.normal(loc=data_true, scale=sigma).astype(np.float32)
        model.rate[model.rate < 0] = 1.0
        model.dq = np.zeros(shape, dtype=np.uint32)
        model.mid_time = np.ones(shape, dtype=np.float64)
        model.generate_filename()
        input_models.append(model)

    # Setup and call the coadd primitive
    result = CoaddFramesPrimitive.call(
        input_models
    )
    model_coadded = result[0]

    rel_residuals = np.abs(model_coadded.rate - data_true) / model_coadded.rate_err
    assert np.sum(rel_residuals > 5) == 0

    # Check 1 sigma statistics
    confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)
    assert abs(confidence_level - 0.683) < 0.1

    # Test data ancestors is updated
    assert len(model_coadded.data_ancestors) == n

    # Try again with sigma clipping
    result = CoaddFramesPrimitive.call(input_models, do_sigma_clip=True, sigma_thresh_low=4.0, sigma_thresh_high=4.0)
    model_coadded = result[0]

    rel_residuals = np.abs(model_coadded.rate - data_true) / model_coadded.rate_err
    assert np.sum(rel_residuals > 5) == 0

    # Check 1 sigma statistics
    confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)
    assert abs(confidence_level - 0.683) < 0.1