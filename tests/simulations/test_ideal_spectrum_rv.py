"""Helpers for the documented ideal-spectrum RV recovery example."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from hispecdrp.simulations.ideal_spectrum_rv import _remove_run_dir, rate_map_to_l0
from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
    load_packaged_sim_hdf_line_zp_wavesol,
)
from hispecdrp.simulations.plotting.ideal_spectrum_rv_recovery import (
    plot_ideal_spectrum_rv_recovery,
)


def test_plot_ideal_spectrum_rv_recovery_residual_cm_s(tmp_path):
    inj = np.array([0.0, 10000.0])
    rec = np.array([0.01, 10000.02])
    err = np.array([0.001, 0.002])
    out = tmp_path / "residual.png"
    summary = plot_ideal_spectrum_rv_recovery(
        inj,
        rec,
        err,
        title="unit test",
        out_path=out,
        display=False,
    )
    assert abs(summary["mean_bias_cm_s"] - 1.5) < 1e-6
    assert abs(summary["std_bias_cm_s"] - 0.5) < 1e-6
    assert out.is_file()


def test_remove_run_dir_deletes_per_rv_products(tmp_path):
    run_dir = tmp_path / "ramp_p0p0kms"
    (run_dir / "cals").mkdir(parents=True)
    (run_dir / "simulated_ramp.fits").write_bytes(b"x")
    (run_dir / "cals" / "dummy.fits").write_bytes(b"y")
    _remove_run_dir(run_dir)
    assert not run_dir.exists()


def test_packaged_line_zp_wavesol_is_reversed_vs_extract_axis():
    """Rate-map extraction / native paint use decreasing λ along an order."""
    w_zp = np.asarray(load_packaged_sim_hdf_line_zp_wavesol("BSPEC").wave_fib1, dtype=float)
    assert w_zp[0, 0] < w_zp[0, -1]


def test_rate_map_to_l0_wraps_rate_and_error():
    rate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    l0 = rate_map_to_l0(rate, spectrograph="BSPEC", mjd_start=60000.0)
    assert l0.meta["spectrograph"] == "BSPEC"
    assert np.allclose(np.asarray(l0.rate), rate.astype(np.float32))
    assert np.all(np.asarray(l0.rate_err) > 0)
