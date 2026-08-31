"""Tests for Stage-1 optimal-extraction residual pull diagnostics."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.spectral_extraction.extract_sheared_sp import (
    forward_sheared_separable,
)
from hispecdrp.primitives.spectral_extraction.extraction_utils import (
    get_extraction_window,
)
from hispecdrp.primitives.spectral_extraction.iso_lambda_shear import (
    psf_stamp_shear_dx_dy,
)
from hispecdrp.simulations.plotting.stage1_extraction_noise import (
    build_profile_image,
    collect_fiber_extraction_pulls,
    order_extraction_residuals,
)


def _synthetic_echelle_frame(
    rng: np.random.Generator,
    *,
    ny: int = 256,
    nx: int = 128,
    spec_amp: float = 50.0,
    read_noise: float = 0.5,
    shear: float = 0.0,
    line_depth: float = 0.0,
    line_row: int = 128,
    line_sigma: float = 1.2,
) -> tuple[np.ndarray, ...]:
    """Separable frame whose forward model matches sheared SP extract."""
    trace_y = np.full(ny, nx / 2.0, dtype=np.float64)
    x = np.arange(ny, dtype=np.float64)
    spec_true = spec_amp * (1.0 + 0.1 * np.sin(x / 17.0))
    if line_depth > 0.0:
        spec_true *= 1.0 - float(line_depth) * np.exp(
            -0.5 * ((x - float(line_row)) / float(line_sigma)) ** 2
        )

    aperture = 8.0
    continuum = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        yc = trace_y[i]
        window_inds, W = get_extraction_window(yc, aperture, nx)
        if window_inds.size == 0:
            continue
        cols = window_inds.astype(float) - yc
        P = np.exp(-0.5 * (cols / 1.2) ** 2)
        continuum[i, window_inds] = P

    profile = build_profile_image(continuum, trace_y, extraction_aperture=aperture)
    shear_arr = np.full(ny, float(shear), dtype=np.float64)
    truth, _ = forward_sheared_separable(
        spec_true,
        profile,
        trace_y,
        extraction_aperture=aperture,
        shear=shear_arr if abs(shear) > 1e-12 else None,
    )

    var_rnoise = np.full((ny, nx), read_noise**2, dtype=np.float64)
    noise = rng.normal(0.0, read_noise, size=truth.shape)
    rate = truth + noise
    rate_err = np.sqrt(var_rnoise)
    rate_dq = np.zeros((ny, nx), dtype=np.uint32)
    var_poisson = np.zeros((ny, nx), dtype=np.float64)
    tracepos = trace_y.reshape(1, -1)
    return (
        rate,
        rate_err,
        rate_dq,
        var_rnoise,
        var_poisson,
        truth,
        continuum,
        tracepos,
        shear_arr,
        profile,
    )


def test_order_extraction_pulls_near_standard_normal():
    rng = np.random.default_rng(0)
    rate, rate_err, rate_dq, var_rnoise, var_poisson, truth, continuum, tracepos, _, _ = (
        _synthetic_echelle_frame(rng)
    )
    result = order_extraction_residuals(
        rate,
        rate_err,
        rate_dq,
        var_rnoise,
        var_poisson,
        continuum,
        tracepos[0],
        extraction_aperture=8.0,
        continuum_smooth_pix=None,
        method="horne",
    )
    pulls = result["pulls_pixel"][result["good_mask_pixel"]]
    pulls = pulls[np.isfinite(pulls)]
    assert pulls.size > 500
    assert abs(float(np.mean(pulls))) < 0.08
    assert abs(float(np.std(pulls)) - 1.0) < 0.15


def test_collect_fiber_extraction_pulls_concatenates_orders():
    rng = np.random.default_rng(1)
    rate, rate_err, rate_dq, var_rnoise, var_poisson, truth, continuum, tracepos, _, _ = (
        _synthetic_echelle_frame(rng)
    )
    trace2 = np.vstack([tracepos[0], tracepos[0]])
    out = collect_fiber_extraction_pulls(
        rate,
        rate_err,
        rate_dq,
        var_rnoise,
        var_poisson,
        continuum,
        trace2,
        orders=[110, 111],
        order_offset=110,
        extraction_aperture=8.0,
        store_maps_orders=[110],
        continuum_smooth_pix=None,
    )
    assert out["n_orders"] == 2
    assert out["n_pix_pixel"] > 1000
    assert out["n_pix_spec"] > 1000
    assert "residual" in out["per_order"][110]
    assert "residual" not in out["per_order"][111]
    assert abs(float(np.mean(out["pulls_pixel"]))) < 0.08


def test_psf_stamp_shear_dx_dy_recovers_input_tilt():
    oversample = 4.0
    h = w = 41
    yy, xx = np.mgrid[0:h, 0:w]
    s_true = 0.25
    x0 = 0.5 * (w - 1)
    y0 = 0.5 * (h - 1)
    dy = (yy - y0) / oversample
    dx = s_true * dy
    xc = x0 + dx * oversample
    stamp = np.exp(-0.5 * (((xx - xc) / 2.0) ** 2 + ((yy - y0) / 2.0) ** 2))
    s_meas = psf_stamp_shear_dx_dy(stamp, oversample=oversample)
    assert abs(s_meas - s_true) < 0.02


def test_sheared_sp_roundtrip_machine_precision():
    """Banded LS invert of sheared scatter recovers noiseless model to ~1e-12."""
    rng = np.random.default_rng(2)
    s_true = 0.5
    rate, rate_err, rate_dq, var_rnoise, var_poisson, truth, continuum, tracepos, shear, profile = (
        _synthetic_echelle_frame(
            rng,
            shear=s_true,
            line_depth=0.8,
            line_row=130,
            line_sigma=1.2,
            read_noise=1e-12,  # effectively noiseless
        )
    )
    # Noiseless rate = exact forward model.
    result = order_extraction_residuals(
        truth,
        np.ones_like(truth),
        rate_dq,
        np.ones_like(truth),
        np.zeros_like(truth),
        continuum,
        tracepos[0],
        extraction_aperture=8.0,
        shear=shear,
        continuum_smooth_pix=None,
        method="sheared_sp",
    )
    m = result["aperture_mask"]
    diff = truth - result["model"]
    assert float(np.max(np.abs(diff[m]))) < 1e-10


def test_sheared_sp_pulls_near_standard_normal():
    rng = np.random.default_rng(3)
    s_true = 0.45
    rate, rate_err, rate_dq, var_rnoise, var_poisson, truth, continuum, tracepos, shear, _ = (
        _synthetic_echelle_frame(
            rng,
            shear=s_true,
            line_depth=0.5,
            read_noise=0.4,
        )
    )
    result = order_extraction_residuals(
        rate,
        rate_err,
        rate_dq,
        var_rnoise,
        var_poisson,
        continuum,
        tracepos[0],
        extraction_aperture=8.0,
        shear=shear,
        continuum_smooth_pix=None,
        method="sheared_sp",
    )
    pulls = result["pulls_pixel"][result["good_mask_pixel"]]
    pulls = pulls[np.isfinite(pulls)]
    assert pulls.size > 500
    assert abs(float(np.mean(pulls))) < 0.1
    assert abs(float(np.std(pulls)) - 1.0) < 0.2
