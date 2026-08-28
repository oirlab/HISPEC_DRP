"""Mask-CCF interpolation error and photon-like per-line RV uncertainties."""
from __future__ import annotations

import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_common import science_centroid_spec_err
from hispecdrp.primitives.utils.ccf.ccf_utils import (
    median_pixel_dv_m_s,
    rv_per_chunks_line_mask,
)
from hispecdrp.utils.line_mask_utils import MaskLines
from hispecdrp.utils.physical_constants import C_M_S


def _hispec_like_wave(n: int = 256, w0: float = 1.210, dv_pix_m_s: float = 500.0) -> np.ndarray:
    dlam = float(w0) * float(dv_pix_m_s) / float(C_M_S)
    return float(w0) + dlam * (np.arange(n, dtype=np.float64) - 0.5 * (n - 1))


def _gaussian_line(wave: np.ndarray, lam0: float, depth: float, sig_um: float) -> np.ndarray:
    return 1.0 - float(depth) * np.exp(-0.5 * ((wave - lam0) / sig_um) ** 2)


def _line_mask_rv(*, spec_err: float, v_step_m_s: float, depth: float = 0.40):
    wave = _hispec_like_wave()
    lam0 = float(np.median(wave))
    sig_um = 2.0e-5
    spec = _gaussian_line(wave, lam0, depth, sig_um)
    err = np.full_like(spec, float(spec_err))
    dq = np.zeros(spec.size, dtype=bool)
    mask = MaskLines.from_delta_centers_um(np.array([lam0]), np.array([1.0]))
    rvs, errs, *_rest = rv_per_chunks_line_mask(
        wave,
        spec,
        err,
        dq,
        mask,
        window_size_um=0.02,
        v_min_m_s=-20_000.0,
        v_max_m_s=20_000.0,
        v_step_m_s=float(v_step_m_s),
        window_line_by_line=True,
        line_mask_line_chunk_v_center_m_s=0.0,
        min_good_pixels=9,
        max_cores=1,
        return_ccf=True,
    )
    assert rvs.size >= 1
    assert np.any(np.isfinite(rvs) & np.isfinite(errs) & (errs > 0))
    i = int(np.nanargmin(np.abs(rvs)))
    return float(rvs[i]), float(errs[i]), median_pixel_dv_m_s(wave)


def test_science_centroid_spec_err_drops_blaze_floor():
    spec_uncal = np.full(8, 100.0)
    spec_uncal_err = np.full(8, 1.0)
    blaze = 0.5
    spec = spec_uncal / blaze
    # Post-blaze SPEC_ERR includes a 50% fiberflat term.
    spec_err = np.abs(spec) * np.sqrt((0.01) ** 2 + (0.50) ** 2)
    phot = science_centroid_spec_err(spec, spec_err, spec_uncal, spec_uncal_err)
    np.testing.assert_allclose(phot, np.abs(spec) * 0.01, rtol=1e-10)
    assert np.all(phot < 0.2 * spec_err)


def test_mask_ccf_rv_err_scales_with_flux_noise():
    """Per-line σ_μ must track interpolated science noise, not sit at the LSF width."""
    _rv_lo, err_lo, pix_dv = _line_mask_rv(spec_err=0.05, v_step_m_s=100.0)
    _rv_hi, err_hi, _ = _line_mask_rv(spec_err=0.005, v_step_m_s=100.0)
    assert err_lo < pix_dv
    assert err_hi < err_lo / 5.0
    ratio = err_lo / err_hi
    assert 6.0 < ratio < 15.0


def test_mask_ccf_rv_err_stable_under_velocity_oversampling():
    """Finer CCF grids reuse pixels; √nsp keeps σ_μ from collapsing as 1/√n_grid."""
    _rv_a, err_coarse, _ = _line_mask_rv(spec_err=0.02, v_step_m_s=100.0)
    _rv_b, err_fine, _ = _line_mask_rv(spec_err=0.02, v_step_m_s=20.0)
    assert err_coarse > 0.0 and err_fine > 0.0
    np.testing.assert_allclose(err_fine, err_coarse, rtol=0.35)


def test_mask_ccf_err_is_interpolated_pixel_noise_not_line_scatter():
    """Two lines of different depth must not set σ_CCF to RMS(flux)."""
    wave = _hispec_like_wave(n=512)
    lam_a = float(wave[180])
    lam_b = float(wave[340])
    sig_um = 2.0e-5
    spec = _gaussian_line(wave, lam_a, 0.50, sig_um) * _gaussian_line(wave, lam_b, 0.10, sig_um)
    spec_err = 0.01
    err = np.full_like(spec, spec_err)
    dq = np.zeros(spec.size, dtype=bool)
    mask = MaskLines.from_delta_centers_um(np.array([lam_a, lam_b]), np.array([1.0, 1.0]))
    _rvs, _errs, _c, v_grid, _ccf, _lo, _hi, _ng, ccf_std, *_rest = rv_per_chunks_line_mask(
        wave,
        spec,
        err,
        dq,
        mask,
        window_size_um=0.05,
        v_min_m_s=-8_000.0,
        v_max_m_s=8_000.0,
        v_step_m_s=100.0,
        window_line_by_line=False,
        min_good_pixels=9,
        max_cores=1,
        return_ccf=True,
    )
    assert ccf_std is not None and ccf_std.size > 0
    sig = np.asarray(ccf_std, dtype=np.float64).ravel()
    sig = sig[np.isfinite(sig) & (sig > 0)]
    assert sig.size > 10
    # RMS of the two line depths is O(0.1); pixel-propagated σ stays ~spec_err.
    assert float(np.median(sig)) < 0.05
    assert float(np.median(sig)) < 0.4


def test_mask_ccf_rv_err_converges_at_high_snr():
    """Per-line σ_μ must fall with flux noise and plateau, not explode, at tiny σ."""
    errs = []
    pix_dv = None
    for spec_err in (5.0e-2, 5.0e-3, 5.0e-4, 5.0e-8):
        _rv, err, pix_dv = _line_mask_rv(spec_err=spec_err, v_step_m_s=100.0)
        assert np.isfinite(err) and err > 0.0
        assert err < pix_dv
        errs.append(err)
    assert errs[1] < errs[0] / 5.0
    assert errs[2] < errs[1] / 5.0
    assert errs[-1] <= errs[-2] * 2.0
    assert errs[-1] < 1.0


def test_gauss_fit_rv_err_converges_at_tiny_ccf_err():
    """Ideal Gaussian CCF: σ_μ scales with ccf_err then hits a small floor."""
    from hispecdrp.primitives.utils.ccf.ccf_gauss_fit_numba import fit_ccf_rv_rows_numba

    v = np.linspace(-20_000.0, 20_000.0, 4001)
    mu0 = 123.0
    sig = 2_500.0
    ccf = np.exp(-0.5 * ((v - mu0) / sig) ** 2)

    def _err_for(ccf_sig: float) -> float:
        er = np.full_like(ccf, float(ccf_sig))
        _rv, rve, ok = fit_ccf_rv_rows_numba(
            v,
            ccf[None, :],
            ccf_err_rows=er[None, :],
            gaussian_seed_mean_m_s=mu0,
            pix_size_m_s=500.0,
            max_cores=1,
        )
        assert int(ok[0]) == 1
        assert np.isfinite(rve[0]) and rve[0] > 0.0
        return float(rve[0])

    e_lo = _err_for(1.0e-2)
    e_mid = _err_for(1.0e-4)
    e_hi = _err_for(1.0e-12)
    assert e_mid < e_lo / 50.0
    assert e_hi <= e_mid * 2.0
    assert e_hi < 1.0
    assert abs(_err_for(1.0e-12) - e_hi) / e_hi < 0.5
