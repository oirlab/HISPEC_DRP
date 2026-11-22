"""Smoke test for the shared simulated cal DB (``cal_db`` marker, ``tests/conftest.py``)."""
import os

import pytest

from hispecdrp import Stage1Pipeline, datamodels
from tests.sim_cals import get_cal_db_paths

CALS = [
    'dq', 'saturation', 'gain', 'bias', 'rn', 'dark', 'detflat',
    'sim_wavesol', 'tracepos', 'traceprof', 'fiberflat', 'lfc_wavesol', 'lsf',
]


@pytest.mark.cal_db
def test_hispec_cals():
    """Every shared cal exists for its spectrograph, and Stage1Pipeline finds its cals through the shared store."""
    paths = get_cal_db_paths()
    assert paths is not None
    assert set(paths) == {'BSPEC', 'RSPEC'}
    for spectrograph, cals in paths.items():
        assert len(cals['fiberflat_L0']) == 3
        assert len(cals['lfc_L0']) == 3
        for name in CALS:
            assert os.path.isfile(cals[name])
            model = datamodels.open(cals[name], meta_only=True)
            assert model.meta.spectrograph == spectrograph, name

    # No cals passed: tracepos, traceprof and wavesol come from the shared store
    model_L1 = Stage1Pipeline.call(paths['RSPEC']['fiberflat_L0'][0], save_result=False)
    assert model_L1.meta.spectrograph == 'RSPEC'
