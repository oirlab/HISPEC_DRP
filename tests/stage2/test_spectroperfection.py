"""Moments + noiseless round-trip for calibration-driven spectroperfection."""

from __future__ import annotations

import numpy as np

from hispecdrp.datamodels.hispec_config import HISPECConfig
from hispecdrp.datamodels.tracepos import TracePosModel
from hispecdrp.datamodels.traceprof import TraceProfModel
from hispecdrp.primitives.spectral_extraction.extract_spectroperfection import (
    extract_spectroperfection,
    forward_spectroperfection,
)
from hispecdrp.primitives.trace_fitting.traceprof_moments import (
    fill_traceprof_spectroperfection_moments,
    spot_window_moments,
)


def _tilted_gaussian_stamp(
    ny: int,
    nx: int,
    *,
    row0: float,
    col0: float,
    sigma_spec: float,
    sigma_spat: float,
    shear: float,
    amp: float = 1.0e3,
) -> np.ndarray:
    """Iso-λ ridge ``row = row0 + shear * (col - col0)`` with Gaussian widths."""
    rr = np.arange(ny, dtype=np.float64)[:, None]
    cc = np.arange(nx, dtype=np.float64)[None, :]
    dc = cc - col0
    dr = rr - (row0 + shear * dc)
    return amp * np.exp(-0.5 * ((dr / sigma_spec) ** 2 + (dc / sigma_spat) ** 2))


def test_spot_window_moments_recover_injected_shear():
    ny, nx = 41, 41
    row0, col0 = 20.0, 20.0
    shear_true = 0.35
    sig_spec, sig_spat = 0.9, 1.4
    rate = _tilted_gaussian_stamp(
        ny,
        nx,
        row0=row0,
        col0=col0,
        sigma_spec=sig_spec,
        sigma_spat=sig_spat,
        shear=shear_true,
    )
    mom = spot_window_moments(rate, row0, col0, half_win=8)
    assert abs(mom["shear"] - shear_true) < 0.03
    assert abs(mom["sigma_spec"] - sig_spec) < 0.08
    assert abs(mom["sigma_spat"] - sig_spat) < 0.08


def test_fill_traceprof_moments_from_fiberflat_and_lfc():
    ny, nx = 96, 48
    ost = int(HISPECConfig.echelle_orders["BSPEC"]["order_start"])
    n_orders = len(list(HISPECConfig.get_order_list("BSPEC")))
    fiber = 2
    order = ost
    shear_true = 0.25
    sig_spec_true = 0.7
    sig_spat_true = 1.6
    yc = 22.0

    lfc = np.zeros((ny, nx), dtype=np.float64)
    spot_rows = np.array([18.0, 40.0, 62.0, 80.0])
    for r0 in spot_rows:
        lfc += _tilted_gaussian_stamp(
            ny,
            nx,
            row0=float(r0),
            col0=yc,
            sigma_spec=sig_spec_true,
            sigma_spat=sig_spat_true,
            shear=shear_true,
            amp=2.0e3,
        )
    ff = np.zeros((ny, nx), dtype=np.float64)
    cc = np.arange(nx, dtype=np.float64)
    spat = np.exp(-0.5 * ((cc - yc) / sig_spat_true) ** 2)
    ff[:] = spat[None, :]

    pos = np.full((n_orders, ny), np.nan, dtype=np.float64)
    pos[0, :] = yc
    tracepos = TracePosModel(config=HISPECConfig, meta={"spectrograph": "BSPEC"})
    setattr(tracepos, f"pos_fib{fiber}", pos)

    traceprof = TraceProfModel(config=HISPECConfig, meta={"spectrograph": "BSPEC"})
    setattr(traceprof, f"prof_fib{fiber}", ff)

    expected = {int(order): [(float(r), yc) for r in spot_rows]}
    fill_traceprof_spectroperfection_moments(
        traceprof,
        fiberflat_rate=ff,
        lfc_rate=lfc,
        tracepos=tracepos,
        spectrograph="BSPEC",
        fiber=fiber,
        orders=[int(order)],
        extraction_aperture=8.0,
        expected_xy_by_order=expected,
    )
    shear = np.asarray(getattr(traceprof, f"shear_fib{fiber}"))
    sig_s = np.asarray(getattr(traceprof, f"sigma_spec_fib{fiber}"))
    sig_p = np.asarray(getattr(traceprof, f"sigma_spat_fib{fiber}"))
    assert abs(float(np.nanmedian(shear[0])) - shear_true) < 0.05
    assert abs(float(np.nanmedian(sig_s[0])) - sig_spec_true) < 0.12
    assert abs(float(np.nanmedian(sig_p[0, 10:-10])) - sig_spat_true) < 0.15


def test_spectroperfection_noiseless_round_trip():
    ny, nx = 72, 36
    yc = 18.0
    aperture = 8.0
    shear = np.full(ny, 0.22, dtype=np.float64)
    sigma_spec = np.full(ny, 0.65, dtype=np.float64)
    cc = np.arange(nx, dtype=np.float64)
    profile = np.repeat(
        np.exp(-0.5 * ((cc - yc) / 1.3) ** 2)[None, :],
        ny,
        axis=0,
    )
    trace_y = np.full(ny, yc, dtype=np.float64)
    x = np.arange(ny, dtype=np.float64)
    f_true = 30.0 * (1.0 + 0.2 * np.sin(x / 6.0))
    f_true *= 1.0 - 0.55 * np.exp(-0.5 * ((x - 36.0) / 1.8) ** 2)

    det = forward_spectroperfection(
        f_true,
        profile,
        trace_y,
        extraction_aperture=aperture,
        shear=shear,
        sigma_spec=sigma_spec,
        spectral_kernel="gaussian",
    )
    rec = extract_spectroperfection(
        det,
        np.ones_like(det),
        profile,
        trace_y,
        extraction_aperture=aperture,
        shear=shear,
        sigma_spec=sigma_spec,
        spectral_kernel="gaussian",
        ridge_frac=0.0,
    )
    f_hat = rec["spec"]
    good = np.isfinite(f_hat) & np.isfinite(f_true)
    good[:8] = False
    good[-8:] = False
    assert int(np.sum(good)) > ny // 2
    rel = np.max(np.abs(f_hat[good] - f_true[good]) / np.maximum(np.abs(f_true[good]), 1.0))
    assert rel < 2.0e-6, f"max relative error {rel}"


def test_spectroperfection_bilinear_round_trip():
    ny, nx = 72, 36
    yc = 18.0
    aperture = 8.0
    shear = np.full(ny, 0.22, dtype=np.float64)
    cc = np.arange(nx, dtype=np.float64)
    profile = np.repeat(
        np.exp(-0.5 * ((cc - yc) / 1.3) ** 2)[None, :],
        ny,
        axis=0,
    )
    trace_y = np.full(ny, yc, dtype=np.float64)
    x = np.arange(ny, dtype=np.float64)
    f_true = 30.0 * (1.0 + 0.2 * np.sin(x / 6.0))
    f_true *= 1.0 - 0.55 * np.exp(-0.5 * ((x - 36.0) / 1.8) ** 2)

    det = forward_spectroperfection(
        f_true,
        profile,
        trace_y,
        extraction_aperture=aperture,
        shear=shear,
        spectral_kernel="bilinear",
    )
    rec = extract_spectroperfection(
        det,
        np.ones_like(det),
        profile,
        trace_y,
        extraction_aperture=aperture,
        shear=shear,
        spectral_kernel="bilinear",
        ridge_frac=0.0,
    )
    f_hat = rec["spec"]
    good = np.isfinite(f_hat) & np.isfinite(f_true)
    good[:8] = False
    good[-8:] = False
    assert int(np.sum(good)) > ny // 2
    rel = np.max(np.abs(f_hat[good] - f_true[good]) / np.maximum(np.abs(f_true[good]), 1.0))
    assert rel < 2.0e-6, f"max relative error {rel}"
