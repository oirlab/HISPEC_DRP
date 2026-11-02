"""Helpers for the documented simulated-LFC example (bootstrap WAVE + plots)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.datamodels.hispec_config import HISPECConfig
from hispecdrp.pipelines import ProcessLFCPipeline
from hispecdrp.simulations.plotting.lfc_pipeline import (
    lfc_vs_hdf_wavesol_shift,
    plot_extracted_lfc_order,
    plot_lfc_line_fwhm,
    plot_lfc_rate_map_order_zooms,
    plot_lfc_vs_hdf_wavesol_shift,
    plot_lfc_vs_hdf_wavesol_vs_wavelength,
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


def test_plot_lfc_rate_map_order_zooms_writes_png(tmp_path):
    rate = np.zeros((160, 140), dtype=float)
    line_list = {
        110: np.array([1.000, 1.010, 1.020, 1.030, 1.040]),
        120: np.array([1.100, 1.112, 1.124, 1.136, 1.148]),
        130: np.array([1.200, 1.215, 1.230, 1.245, 1.260]),
    }
    deposit_xy = {
        110: (np.array([20.0, 40.0, 60.0, 80.0, 100.0]), np.array([25.0, 26.0, 27.0, 28.0, 29.0])),
        120: (np.array([22.0, 42.0, 62.0, 82.0, 102.0]), np.array([70.0, 71.0, 72.0, 73.0, 74.0])),
        130: (np.array([24.0, 44.0, 64.0, 84.0, 104.0]), np.array([110.0, 111.0, 112.0, 113.0, 114.0])),
    }
    for rows, cols in deposit_xy.values():
        for r, c in zip(rows, cols):
            rate[int(r), int(c)] = 1.0e6
    result = plot_lfc_rate_map_order_zooms(
        rate,
        line_list,
        spectrograph="BSPEC",
        fiber=2,
        deposit_xy=deposit_xy,
        out_path=tmp_path / "rate_zooms.png",
        display=False,
    )
    assert result["orders"] == [110, 120, 130]
    assert len(result["zooms"]) == 3
    for zoom, (rows, _cols) in zip(result["zooms"], deposit_xy.values()):
        r0, r1 = zoom["row_slice"]
        n_in = int(np.sum((rows >= r0) & (rows < r1)))
        assert n_in >= 2
    assert result["zooms"][0]["scale_bar_pix"] > 0
    assert result["zooms"][0]["scale_bar_dlam_um"] > 0
    assert (tmp_path / "rate_zooms.png").is_file()


def test_process_lfc_pipeline_apply_returns_wavesol_and_lsf(tmp_path, monkeypatch):
    """The documented example unpacks ``apply()`` as ``products['wavesol']`` / ``lsf``."""
    wavesol = object()
    lsf = object()

    def fake_perform(self):
        return dict(output=dict(wavesol=wavesol, lsf=lsf))

    monkeypatch.setattr(ProcessLFCPipeline, "_perform", fake_perform)
    dataset = datamodels.L0Model(
        rate=np.zeros((8, 8), dtype=np.float32),
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": "BSPEC",
            "datetime_obs": "2027-01-01T00:00:00.000",
            "exposure_type": "SPEC",
        },
    )
    pipe = ProcessLFCPipeline(
        output_dir=str(tmp_path),
        dataset=dataset,
        save_result=False,
    )
    products = pipe.apply()
    assert products["wavesol"] is wavesol
    assert products["lsf"] is lsf
    assert set(ProcessLFCPipeline.PRIMITIVES) == {
        "coadd_spectra",
        "blaze_corr",
        "fit_lfc_lines",
        "lfc_wavesol",
        "register_wave_cal",
        "lsf",
        "register_lsf_cal",
    }


def test_lfc_params_match_ddd_mode_spacings():
    # HISPEC.DRP.DDD.001 Sec. 2.4.5: Menlo/KPF comb (BSPEC) filtered to
    # 20 GHz; electro-optic astrocomb (RSPEC) at 16 GHz.
    assert HISPECConfig.lfc_params["LFC-B"]["delta_nu"] == 20e9
    assert HISPECConfig.lfc_params["LFC-R"]["delta_nu"] == 16e9


def test_lfc_vs_hdf_wavesol_shift_constant_velocity():
    from scipy.constants import c

    v_m_s = 30.0
    hdf = np.vstack(
        [
            np.linspace(1.00, 1.05, 16),
            np.linspace(1.10, 1.15, 16),
        ]
    )
    lfc = hdf * (1.0 + v_m_s / c)
    stats = lfc_vs_hdf_wavesol_shift(lfc, hdf, orders=[110, 111])
    assert stats["dlam_um"].shape == (2, 16)
    assert stats["orders"].tolist() == [110, 111]
    assert abs(stats["median_dvel_m_s"] - v_m_s) < 1.0e-6
    assert stats["mad_dvel_m_s"] < 1.0e-6
    for row in stats["per_order"]:
        assert abs(row["median_dvel_m_s"] - v_m_s) < 1.0e-6
        assert row["n"] == 16


def test_lfc_vs_hdf_wavesol_shift_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="WAVE shapes differ"):
        lfc_vs_hdf_wavesol_shift(np.ones((2, 8)), np.ones((2, 7)))


def test_lfc_vs_hdf_wavesol_shift_clips_large_velocity_spikes():
    from scipy.constants import c

    v_m_s = 5.0
    hdf = np.vstack(
        [
            np.linspace(1.00, 1.05, 16),
            np.linspace(1.10, 1.15, 16),
        ]
    )
    lfc = hdf * (1.0 + v_m_s / c)
    lfc[0, -1] = hdf[0, -1] * (1.0 - 1000.0 / c)
    raw = lfc_vs_hdf_wavesol_shift(lfc, hdf, orders=[110, 111])
    clipped = lfc_vs_hdf_wavesol_shift(
        lfc, hdf, orders=[110, 111], max_abs_dvel_m_s=100.0
    )
    assert raw["n_clipped"] == 0
    assert clipped["n_clipped"] == 1
    assert not clipped["good"][0, -1]
    assert abs(clipped["median_dvel_m_s"] - v_m_s) < 1.0e-6
    assert clipped["per_order"][0]["n"] == 15


def test_plot_lfc_vs_hdf_wavesol_shift_writes_png(tmp_path):
    hdf = np.vstack(
        [
            np.linspace(1.00, 1.05, 16),
            np.linspace(1.10, 1.15, 16),
        ]
    )
    lfc = hdf + 2.0e-7
    out = tmp_path / "lfc_vs_hdf.png"
    stats = plot_lfc_vs_hdf_wavesol_shift(
        lfc,
        hdf,
        orders=[110, 111],
        out_path=out,
        display=False,
    )
    assert out.is_file()
    assert stats["out_path"] == str(out)
    assert np.allclose(stats["dlam_um"], 2.0e-7)

    zoom = tmp_path / "lfc_vs_hdf_center.png"
    zoomed = plot_lfc_vs_hdf_wavesol_shift(
        lfc,
        hdf,
        orders=[110, 111],
        xlim=(4, 12),
        out_path=zoom,
        display=False,
    )
    assert zoom.is_file()
    assert zoomed["xlim"] == (4.0, 12.0)

    vs_lam = tmp_path / "lfc_vs_hdf_vs_lambda.png"
    lam_stats = plot_lfc_vs_hdf_wavesol_vs_wavelength(
        lfc,
        hdf,
        orders=[110, 111],
        lam_xlim=(1.02, 1.04),
        out_path=vs_lam,
        display=False,
    )
    assert vs_lam.is_file()
    assert lam_stats["lam_xlim"] == (1.02, 1.04)
