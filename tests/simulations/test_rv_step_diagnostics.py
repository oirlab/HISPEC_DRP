"""Helpers for the documented RV recovery step-diagnostics example."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import numpy as np

from hispecdrp.simulations.ideal_spectrum_rv import (
    recover_rv_from_ramp,
    recover_rv_from_rate_map,
)
from hispecdrp.simulations.plotting.rv_step_diagnostics import plot_rv_step_comparison
from hispecdrp.simulations.rv_step_diagnostics import (
    NO_LINE_REJECTION_SIGMA,
    STEP_BULK_RV_KM_S,
    STELLAR_RV_NO_REJECT_KW,
    check_lsf_wavesol_fiber_alignment,
    compare_step_summaries,
    doppler_shift_spectrum,
    run_step_grid,
    saved_rate_map_path,
    summarize_step_grid,
)
from hispecdrp.simulations.spectra import build_ccfopt_mask, stellar_rv_kwargs_for_ideal_spectrum


def test_step_grid_and_no_reject_defaults():
    assert list(STEP_BULK_RV_KM_S) == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert STELLAR_RV_NO_REJECT_KW["outlier_sigma"] == NO_LINE_REJECTION_SIGMA
    assert NO_LINE_REJECTION_SIGMA >= 1.0e20


def test_doppler_shift_moves_wavelength_not_flux():
    wl = np.array([1.0, 1.1, 1.2]) * u.um
    flux = np.array([2.0, 3.0, 4.0]) * u.erg / u.s / u.cm**2 / u.um
    wl_s, flux_s = doppler_shift_spectrum(wl, flux, 30.0)
    assert np.allclose(flux_s.value, flux.value)
    assert np.all(wl_s.value > wl.value)
    rest, _ = doppler_shift_spectrum(wl, flux, 0.0)
    assert np.allclose(rest.value, wl.value)


def test_stellar_rv_kwargs_keep_all_lines():
    kw = stellar_rv_kwargs_for_ideal_spectrum(
        0.0,
        template="lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5",
        **STELLAR_RV_NO_REJECT_KW,
    )
    assert kw["outlier_sigma"] == NO_LINE_REJECTION_SIGMA


def test_rate_map_and_ramp_accept_truth_spectrum_and_overrides():
    rate_sig = inspect.signature(recover_rv_from_rate_map)
    ramp_sig = inspect.signature(recover_rv_from_ramp)
    for sig in (rate_sig, ramp_sig):
        assert "spectrum_wavelengths" in sig.parameters
        assert "spectrum_flux" in sig.parameters
        assert sig.parameters["stellar_rv_overrides"].kind is inspect.Parameter.VAR_KEYWORD


def test_compare_largest_increment_is_rate_map(capsys):
    inj = np.array([0.0, 10000.0])
    summaries = {
        "L2": summarize_step_grid(inj, inj + 0.01, np.array([0.05, 0.05]), name="L2"),
        "rate map": summarize_step_grid(inj, inj + 30.0, np.array([2.0, 2.0]), name="rate map"),
        "UTR + Stage 0-2": summarize_step_grid(
            inj, inj + 31.0, np.array([2.2, 2.2]), name="UTR + Stage 0-2"
        ),
    }
    increments = compare_step_summaries(summaries)
    assert abs(increments[1]["delta_mean_bias_m_s"] - 29.99) < 1e-6
    assert increments[1]["delta_mean_bias_m_s"] > increments[2]["delta_mean_bias_m_s"]
    assert "Largest increment: rate map" in capsys.readouterr().out


def test_run_step_grid_without_plot():
    def recover(bulk):
        inj = float(bulk) * 1000.0
        return inj, inj + 0.02, 0.01

    summary = run_step_grid(
        recover,
        name="unit",
        title="unit",
        out_path="unused.png",
        bulk_rv_km_s=np.array([0.0, 10.0]),
        plot=False,
    )
    assert summary["name"] == "unit"
    assert np.allclose(summary["bias_m_s"], 0.02)


def test_saved_rate_map_path_matches_rate_map_run_dir():
    path = saved_rate_map_path("/tmp/out", 10.0)
    assert path.name == "rate_map.fits"
    assert path.parent.name == "rate_map_p10.0kms"


def test_check_lsf_wavesol_fiber_alignment():
    wavesol = SimpleNamespace(wave_fib1=np.ones((39, 4096)))
    lsf = SimpleNamespace(
        meta=SimpleNamespace(FIBER=1),
        lsf_params_fib1=np.ones((39, 4096)),
    )
    ok = check_lsf_wavesol_fiber_alignment(
        fiber=1, wavesol=wavesol, lsf_model=lsf
    )
    assert ok["ok"] is True
    assert ok["wave_shape"] == (39, 4096)
    bad = check_lsf_wavesol_fiber_alignment(
        fiber=2, wavesol=wavesol, lsf_model=lsf
    )
    assert bad["ok"] is False


def test_build_ccfopt_mask_accepts_prebuilt_l2():
    sig = inspect.signature(build_ccfopt_mask)
    assert "l2" in sig.parameters


def test_plot_rv_step_comparison(tmp_path):
    inj = np.array([0.0, 10000.0])
    summaries = {
        "L2": summarize_step_grid(inj, inj + 0.01, np.array([0.05, 0.05]), name="L2"),
        "rate map": summarize_step_grid(inj, inj + 1.0, np.array([0.2, 0.2]), name="rate map"),
    }
    out = tmp_path / "cmp.png"
    result = plot_rv_step_comparison(summaries, out_path=out, display=False)
    assert out.is_file()
    assert result["n_steps"] == 2
