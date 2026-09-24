"""Helpers for the documented simulated-LFC example (mock WAVE + plots)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from hispecdrp.datamodels.hispec_config import HISPECConfig
from hispecdrp.simulations.plotting.lfc_pipeline import (
    plot_extracted_lfc_order,
    plot_lfc_line_fwhm,
    plot_lsf_resolving_power,
)
from hispecdrp.simulations.simulate_up_the_ramp import (
    apply_simulate_rate_map_wavelength_solution,
    stack_wavelength_solution_rows,
)


def test_stack_wavelength_solution_rows_keeps_order_alignment():
    order_list = [110, 111, 112]
    ws = {
        110: {"wavelength": np.linspace(1.0, 1.1, 8)},
        112: {"wavelength": np.linspace(1.2, 1.3, 8)},
    }
    arr = stack_wavelength_solution_rows(ws, order_list, n_pix=8)
    assert arr.shape == (3, 8)
    assert np.allclose(arr[0], np.linspace(1.0, 1.1, 8))
    assert np.all(np.isnan(arr[1]))
    assert np.allclose(arr[2], np.linspace(1.2, 1.3, 8))


class _FakeSRM:
    def get_wavelength_solution(self, spectrograph="BSPEC"):
        n_pix = 16
        return {
            2: {
                110: {"wavelength": np.linspace(1.00, 1.02, n_pix)},
                111: {"wavelength": np.linspace(1.02, 1.04, n_pix)},
            }
        }


class _Meta:
    pass


class _Model:
    def __init__(self):
        self.meta = _Meta()


def test_apply_mock_wavesol_writes_aligned_wave_fib(tmp_path):
    model = _Model()
    arrays = apply_simulate_rate_map_wavelength_solution(
        model, "BSPEC", fibers=[2], srm=_FakeSRM()
    )
    orders = HISPECConfig.get_order_list("BSPEC")
    wave = model.wave_fib2
    assert 2 in arrays
    assert wave.shape == (len(orders), 16)
    assert model.meta.spectrograph == "BSPEC"
    assert model.meta.order_start == orders[0]
    assert model.meta.order_end == orders[-1]
    assert np.allclose(wave[0], np.linspace(1.00, 1.02, 16))
    assert np.all(np.isnan(wave[2]))


class _FitTable:
    def __init__(self):
        self.ORDER = np.array([110, 110, 111])
        self.WAVE = np.array([1.00, 1.01, 1.05])
        self.FWHM_PIX = np.array([2.0, 2.1, 2.2])
        self.FWHM_WAVE = np.array([1.0e-4, 1.1e-4, 1.2e-4])

    def __getitem__(self, key):
        return getattr(self, key)


class _LineFits:
    lfc_fitparams_fib2 = _FitTable()


class _Extracted:
    def __init__(self):
        self.spec_fib2 = np.zeros((2, 32), dtype=float)
        self.spec_fib2[0, 10] = 5.0
        self.wave_fib2 = np.linspace(1.0, 1.1, 32)[None, :].repeat(2, axis=0)
        self.meta = type("m", (), {"order_start": 110, "order_end": 111})()

    def make_order_vec(self):
        return np.array([110, 111])


class _LSF:
    def __init__(self):
        self.wave_fib2 = np.linspace(1.0, 1.1, 20)[None, :].repeat(2, axis=0)
        self.lsf_params_fib2 = np.full((2, 20), 1.0e-4)
        self.meta = type("m", (), {"order_start": 110, "order_end": 111})()

    def make_order_vec(self):
        return np.array([110, 111])


def test_lfc_plot_helpers_write_pngs(tmp_path):
    extracted = plot_extracted_lfc_order(
        _Extracted(),
        fiber=2,
        order=110,
        out_path=tmp_path / "extracted.png",
        display=False,
    )
    fwhm = plot_lfc_line_fwhm(
        _LineFits(),
        fiber=2,
        orders=[110, 111],
        y="fwhm_pix",
        out_path=tmp_path / "fwhm.png",
        display=False,
    )
    lsf = plot_lsf_resolving_power(
        _LSF(),
        fiber=2,
        orders=[110, 111],
        out_path=tmp_path / "lsf.png",
        display=False,
    )
    assert extracted["order"] == 110
    assert fwhm["n_orders"] == 2
    assert lsf["n_orders"] == 2
    assert (tmp_path / "extracted.png").is_file()
    assert (tmp_path / "fwhm.png").is_file()
    assert (tmp_path / "lsf.png").is_file()
