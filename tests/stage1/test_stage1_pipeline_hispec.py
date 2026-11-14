"""Tests for ``Stage1Pipeline`` (HISPEC; ignored in default pytest).
"""
import numpy as np
import pytest

from hispecdrp import Stage1Pipeline
from tests.sim_cals import make_fiberflat_L0
from tests.utils import add_noise_L0


@pytest.mark.cal_db
def test_stage1_pipeline_hispec(tmp_path):

    # The trace positions, trace profiles and wavelength solution come from the cal DB
    spectrograph = 'RSPEC'
    fiber = 1
    # The same 1000 K white-light L0 that the cal DB fiberflats are made from
    L0_model = add_noise_L0(make_fiberflat_L0(spectrograph, fiber, str(tmp_path)), max_snr=200)
    methods = ('box', 'optimal')
    snr_thresh = 90

    for method in methods:
        spec_model = Stage1Pipeline.call(
            L0_model,
            primitives={'spec_extract': {'method': method}},
            save_result=False,
        )
        for i, order in enumerate(spec_model.orders):
            spec = spec_model.spec_uncal[fiber][i]
            err = spec_model.spec_uncal_err[fiber][i]
            snr = np.nanmean(spec / err)
            print(f"Fiber {fiber} Order {order}: {method} SNR = {snr:.2f}")
            assert snr > snr_thresh, f"Fiber {fiber} Order {order}: {method} SNR <= {snr_thresh}"
