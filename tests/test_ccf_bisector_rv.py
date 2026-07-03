"""
Tests for CCF bisector RV fitting (:func:`fit_ccf_rv_bisector`).

Run::

  pytest tests/test_ccf_bisector_rv.py -v -s

The bisector test writes a diagnostic plot to ``tmp_path`` and prints measured RV ± σ.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hispecdrp.primitives.utils.ccf.ccf_utils import (
    CCFBisectorConfig,
    effective_ccf_v_step_for_fit,
    fit_ccf_rv,
    fit_ccf_rv_bisector,
)


def _dummy_gaussian_ccf(
    true_rv_m_s: float,
    *,
    sigma_m_s: float = 3500.0,
    amplitude: float = 1.0,
    baseline: float = 0.05,
    v_span_m_s: float = 25000.0,
    v_step_m_s: float = 50.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic peak-up Gaussian CCF + optional noise."""
    v = np.arange(-v_span_m_s, v_span_m_s + 0.5 * v_step_m_s, v_step_m_s, dtype=np.float64)
    ccf = baseline + amplitude * np.exp(-0.5 * ((v - true_rv_m_s) / sigma_m_s) ** 2)
    if noise_sigma > 0.0 and rng is not None:
        ccf = ccf + rng.normal(0.0, noise_sigma, size=ccf.shape)
    ccf_err = np.full_like(ccf, noise_sigma if noise_sigma > 0 else 0.01 * amplitude)
    return v, ccf, ccf_err


def _bisector_mc_calibration(
    *,
    true_rv_m_s: float = 1234.5,
    v_step_m_s: float = 25.0,
    noise_sigma: float = 0.0,
    n_trials: int = 500,
    subgrid_offsets: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Return (bias_std, mean_reported_err, ratio) for bisector RV uncertainty calibration.

    When ``noise_sigma == 0``, scans ``subgrid_offsets`` on the injected RV; otherwise
    draws noisy CCF realizations at fixed ``true_rv_m_s``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if noise_sigma <= 0.0:
        if subgrid_offsets is None:
            subgrid_offsets = np.linspace(-12.0, 12.0, 241)
        biases: list[float] = []
        errs: list[float] = []
        for d in subgrid_offsets:
            v, ccf, _ = _dummy_gaussian_ccf(
                true_rv_m_s + float(d),
                sigma_m_s=3200.0,
                amplitude=2.5,
                baseline=0.08,
                v_step_m_s=v_step_m_s,
            )
            vs = effective_ccf_v_step_for_fit(v)
            rv, rv_err = fit_ccf_rv_bisector(v, ccf, vs, ccf_err=None)
            biases.append(rv - (true_rv_m_s + float(d)))
            errs.append(rv_err)
        bias_std = float(np.std(biases))
        mean_err = float(np.mean(errs))
    else:
        biases = []
        errs = []
        for _ in range(n_trials):
            v, ccf, ccf_err = _dummy_gaussian_ccf(
                true_rv_m_s,
                sigma_m_s=3200.0,
                amplitude=2.5,
                baseline=0.08,
                v_step_m_s=v_step_m_s,
                noise_sigma=noise_sigma,
                rng=rng,
            )
            vs = effective_ccf_v_step_for_fit(v)
            rv, rv_err = fit_ccf_rv_bisector(v, ccf, vs, ccf_err=ccf_err)
            biases.append(rv - true_rv_m_s)
            errs.append(rv_err)
        bias_std = float(np.std(biases))
        mean_err = float(np.mean(errs))
    ratio = bias_std / mean_err if mean_err > 0 else float("nan")
    return bias_std, mean_err, ratio


def _plot_bisector_fit(
    v_m_s: np.ndarray,
    ccf: np.ndarray,
    diag: dict,
    true_rv_m_s: float,
    rv_m_s: float,
    rv_err_m_s: float,
    out_path: str,
) -> None:
    """Plot CCF, bisector cuts, weighted RV, and comparison to truth."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(v_m_s, ccf, "o", color="0.25", ms=2.5, alpha=0.55, label="CCF samples", zorder=2)
    ax.plot(v_m_s, ccf, color="0.35", lw=1.0, alpha=0.85, label="CCF (interpolated)", zorder=1)

    baseline = diag.get("baseline", np.nanmin(ccf))
    ax.axhline(baseline, color="gray", ls=":", lw=0.8, label=f"baseline = {baseline:.3g}")

    for cut in diag.get("cuts", []):
        pct = cut["percentile"]
        target = cut["target_flux"] + baseline
        ax.axhline(target, color="gold", alpha=0.25, lw=0.6)
        ax.plot(
            [cut["v_left_m_s"], cut["v_right_m_s"]],
            [target, target],
            color="darkorange",
            lw=1.0,
            alpha=0.7,
        )
        ax.plot(cut["v_mid_m_s"], target, "o", color="crimson", ms=4, alpha=0.85)
        ax.annotate(
            f"{pct:.0f}%",
            (cut["v_mid_m_s"], target),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            color="crimson",
        )

    ax.axvline(true_rv_m_s, color="green", ls="--", lw=1.5, label=f"true RV = {true_rv_m_s:.1f} m/s")
    ax.axvline(rv_m_s, color="blue", ls="-", lw=2.0, label=f"bisector RV = {rv_m_s:.2f} ± {rv_err_m_s:.2f} m/s")
    ax.axvspan(rv_m_s - rv_err_m_s, rv_m_s + rv_err_m_s, color="blue", alpha=0.15)

    bis = diag.get("bis_m_s", float("nan"))
    bis_err = diag.get("bis_err_m_s", float("nan"))
    title = (
        f"CCF bisector RV fit\n"
        f"measured = {rv_m_s:.2f} ± {rv_err_m_s:.2f} m/s  "
        f"(Δ = {rv_m_s - true_rv_m_s:+.2f} m/s);  "
        f"BIS = {bis:.2f} ± {bis_err:.2f} m/s"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("CCF")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def test_ccf_bisector_rv_on_dummy_gaussian(tmp_path):
    """Bisector RV on a synthetic Gaussian CCF matches truth; write diagnostic plot."""
    true_rv_m_s = 1234.5
    v, ccf, ccf_err = _dummy_gaussian_ccf(
        true_rv_m_s,
        sigma_m_s=3200.0,
        amplitude=2.5,
        baseline=0.08,
        v_step_m_s=25.0,
    )
    v_step = effective_ccf_v_step_for_fit(v)

    rv, rv_err, diag = fit_ccf_rv_bisector(
        v,
        ccf,
        v_step,
        ccf_err=None,
        config=CCFBisectorConfig(rv_percentile_step=5.0),
        return_diagnostics=True,
    )

    assert diag["ok"] is True
    assert diag["n_cuts"] >= 5
    assert np.isfinite(rv) and np.isfinite(rv_err)
    assert abs(rv - true_rv_m_s) < 3.0 * v_step
    assert rv_err > 0.0

    plot_path = tmp_path / "ccf_bisector_rv_dummy.png"
    _plot_bisector_fit(v, ccf, diag, true_rv_m_s, rv, rv_err, str(plot_path))
    assert plot_path.is_file()

    print(
        f"\nCCF bisector: true RV = {true_rv_m_s:.3f} m/s, "
        f"measured = {rv:.3f} ± {rv_err:.3f} m/s "
        f"(offset {rv - true_rv_m_s:+.3f} m/s), "
        f"BIS = {diag['bis_m_s']:.3f} ± {diag['bis_err_m_s']:.3f} m/s"
    )
    print(f"Plot written to {plot_path}")


def test_ccf_bisector_rv_uncertainty_calibration_noiseless():
    """Reported σ should bracket sub-grid RV bias scatter on a noise-free CCF."""
    bias_std, mean_err, ratio = _bisector_mc_calibration(noise_sigma=0.0)
    print(
        f"\nNoiseless calibration: bias_std={bias_std:.4f} m/s, "
        f"mean reported err={mean_err:.4f} m/s, ratio={ratio:.2f}"
    )
    assert mean_err >= bias_std
    assert ratio <= 3.0


def test_ccf_bisector_rv_uncertainty_calibration_noisy():
    """Reported σ should be consistent with RV bias scatter under CCF noise."""
    bias_std, mean_err, ratio = _bisector_mc_calibration(noise_sigma=0.02, n_trials=800)
    print(
        f"\nNoisy calibration: bias_std={bias_std:.3f} m/s, "
        f"mean reported err={mean_err:.3f} m/s, ratio={ratio:.2f}"
    )
    assert 0.5 <= ratio <= 2.0


@pytest.mark.parametrize("true_rv_m_s", [-500.0, 0.0, 2500.0])
def test_ccf_bisector_rv_offset_scan(true_rv_m_s: float):
    """Bisector RV remains near injected center for several offsets."""
    v, ccf, ccf_err = _dummy_gaussian_ccf(true_rv_m_s, v_step_m_s=20.0)
    v_step = effective_ccf_v_step_for_fit(v)
    rv, rv_err = fit_ccf_rv_bisector(v, ccf, v_step, ccf_err=None)
    assert np.isfinite(rv) and np.isfinite(rv_err)
    assert abs(rv - true_rv_m_s) < 5.0 * v_step


def test_ccf_bisector_vs_gaussian_on_clean_ccf():
    """On a noiseless symmetric Gaussian, bisector and Gaussian fits agree."""
    true_rv = 42.0
    v, ccf, _ = _dummy_gaussian_ccf(true_rv, v_step_m_s=10.0, amplitude=1.0, baseline=0.0)
    v_step = effective_ccf_v_step_for_fit(v)
    rv_bis, err_bis = fit_ccf_rv_bisector(v, ccf, v_step)
    rv_g, err_g = fit_ccf_rv(v, ccf, v_step)
    assert abs(rv_bis - true_rv) < 1.0
    assert abs(rv_g - true_rv) < 1.0
    assert abs(rv_bis - rv_g) < 2.0 * v_step


@pytest.mark.skipif(
    os.environ.get("HISPEC_DRP_SHOW_CCF_BISECTOR_PLOT") != "1",
    reason="Set HISPEC_DRP_SHOW_CCF_BISECTOR_PLOT=1 to open interactive plot",
)
def test_ccf_bisector_rv_show_plot(tmp_path):
    """Optional interactive display of the bisector diagnostic plot."""
    true_rv_m_s = -876.0
    v, ccf, ccf_err = _dummy_gaussian_ccf(true_rv_m_s, v_step_m_s=30.0)
    v_step = effective_ccf_v_step_for_fit(v)
    rv, rv_err, diag = fit_ccf_rv_bisector(
        v, ccf, v_step, ccf_err=ccf_err, return_diagnostics=True
    )
    plot_path = tmp_path / "ccf_bisector_show.png"
    _plot_bisector_fit(v, ccf, diag, true_rv_m_s, rv, rv_err, str(plot_path))
    matplotlib.use("TkAgg")
    img = plt.imread(plot_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    plt.show()
