"""Infinite-SNR recovery of one Gaussian absorption line on a single order.

The spectrum is continuum minus a Gaussian in velocity, painted on one
synthetic BSPEC order. ``StellarRVPrimitive`` (``ccf_line_mask``) is given a
one-line delta mask at the rest wavelength. Resampling is off
(``mask_ccf_resample=False``), so the CCF is the pixel fluxes at the native
velocities, and the continuum is a free parameter of the Gaussian fit
(``mask_ccf_fit_baseline_free=True``). The centroid is then the injected RV
at the precision of those pixel samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import datamodels
from ..primitives.stellar_rv import StellarRVPrimitive
from ..utils.physical_constants import C_M_S
from .e2e_gaia import DEFAULT_E2E_MJD_START
from .spectra import quiet_pipeline_logging

# Same echelle scaling as the synthetic stellar-RV tests: order m is centered
# at 147/m um and spans 1/m of that center, sampled on 4096 linear pixels.
ORDER = 136
N_PIX = 4096
LAMBDA_ORDER_COEF_UM = 147.0
DEPTH = 0.6
SIGMA_V_M_S = 4_000.0
# Positive floor so CCF weights stay finite. Continuum SNR per pixel is 1e8.
SPEC_ERR = 1.0e-8
# The search window is still +/-20 km/s. With resampling off the centroid is
# read from the pixels inside that window, so v_step and the kernel order do
# not set it.
V_STEP_M_S = 1.0
V_HALF_WINDOW_M_S = 20_000.0
# Lanczos-8 is what the mask CCF uses when resampling is on. This example turns
# resampling off and fits the continuum as a free parameter.
MASK_CCF_PIXEL_INTERP_ORDER = 8
MASK_CCF_RESAMPLE = False
MASK_CCF_FIT_BASELINE_FREE = True
# Spectral window half-width, in units of the Gaussian sigma.
WINDOW_HALF_WIDTH_SIGMA = 20.0

# -20, -19, ..., 20 km/s. Pixel scale on this order is ~538 m/s, so a 1 km/s
# step walks through different sub-pixel phases.
DEFAULT_INJECTED_RV_M_S = np.linspace(-20_000.0, 20_000.0, 41)


@dataclass(frozen=True)
class GaussianLineRVResult:
    """One StellarRV recovery per injected velocity."""

    injected_m_s: np.ndarray
    recovered_m_s: np.ndarray
    rv_err_m_s: np.ndarray
    lam_rest_um: float
    pixel_dv_m_s: float
    order: int = ORDER
    n_pix: int = N_PIX
    depth: float = DEPTH
    sigma_v_m_s: float = SIGMA_V_M_S
    spec_err: float = SPEC_ERR
    v_step_m_s: float = V_STEP_M_S
    v_half_window_m_s: float = V_HALF_WINDOW_M_S
    mask_ccf_pixel_interp_order: int = MASK_CCF_PIXEL_INTERP_ORDER
    mask_ccf_resample: bool = MASK_CCF_RESAMPLE
    mask_ccf_fit_baseline_free: bool = MASK_CCF_FIT_BASELINE_FREE
    prior_offset_m_s: float = 0.0

    @property
    def bias_m_s(self) -> np.ndarray:
        return self.recovered_m_s - self.injected_m_s


def single_order_wave_grid(
    order: int = ORDER,
    n_pix: int = N_PIX,
) -> tuple[np.ndarray, float]:
    """Linear wavelength grid and rest-frame line center (um) for one order."""
    lam_c = float(LAMBDA_ORDER_COEF_UM) / float(order)
    delta = lam_c / float(order)
    wave = np.linspace(
        lam_c - 0.5 * delta,
        lam_c + 0.5 * delta,
        int(n_pix),
        dtype=np.float64,
    )
    return wave, lam_c


def gaussian_flux(
    wave_um: np.ndarray,
    lam_rest_um: float,
    rv_m_s: float,
    *,
    depth: float = DEPTH,
    sigma_v_m_s: float = SIGMA_V_M_S,
) -> np.ndarray:
    """Continuum minus a Gaussian in velocity, evaluated on ``wave_um``.

    Velocity of a pixel relative to the injected line is
    ``c * (lambda / lambda_rest - 1) - rv``.
    """
    v_pix = C_M_S * (np.asarray(wave_um, dtype=np.float64) / float(lam_rest_um) - 1.0)
    v_pix = v_pix - float(rv_m_s)
    return 1.0 - float(depth) * np.exp(-0.5 * (v_pix / float(sigma_v_m_s)) ** 2)


def build_gaussian_line_l2(
    rv_m_s: float,
    *,
    wave_um: np.ndarray | None = None,
    lam_rest_um: float | None = None,
    order: int = ORDER,
    depth: float = DEPTH,
    sigma_v_m_s: float = SIGMA_V_M_S,
    spec_err: float = SPEC_ERR,
) -> datamodels.L2Model:
    """Noiseless single-order L2 with one Gaussian line at ``rv_m_s``."""
    if wave_um is None or lam_rest_um is None:
        wave_um, lam_c = single_order_wave_grid(order)
        if lam_rest_um is None:
            lam_rest_um = lam_c
    wave = np.asarray(wave_um, dtype=np.float64).ravel()
    flux = gaussian_flux(
        wave,
        float(lam_rest_um),
        float(rv_m_s),
        depth=depth,
        sigma_v_m_s=sigma_v_m_s,
    )
    model = datamodels.L2Model(
        meta={
            "order_start": int(order),
            "order_end": int(order),
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
            "mjd_start": DEFAULT_E2E_MJD_START,
        }
    )
    model.meta.order_start = int(order)
    model.meta.order_end = int(order)
    model.wave_fib1 = wave.reshape(1, -1)
    # Keep float64. The Level-2 product dtype is float32, and that rounding
    # moves this centroid by several hundredths of a mm/s.
    model.spec_fib1 = np.asarray(flux, dtype=np.float64).reshape(1, -1)
    model.spec_err_fib1 = np.full(wave.shape, float(spec_err), dtype=np.float64).reshape(1, -1)
    model.spec_dq_fib1 = np.zeros(wave.shape, dtype=np.uint32).reshape(1, -1)
    return model


def _line_window_um(lam_rest_um: float, sigma_v_m_s: float) -> float:
    """Total observed-frame window wide enough to hold the whole line."""
    sigma_um = float(lam_rest_um) * float(sigma_v_m_s) / C_M_S
    return 2.0 * WINDOW_HALF_WIDTH_SIGMA * sigma_um


def recover_gaussian_line(
    rv_m_s: float,
    *,
    wave_um: np.ndarray,
    lam_rest_um: float,
    prior_m_s: float | None = None,
    order: int = ORDER,
    v_step_m_s: float = V_STEP_M_S,
    v_half_window_m_s: float = V_HALF_WINDOW_M_S,
    mask_ccf_pixel_interp_order: int = MASK_CCF_PIXEL_INTERP_ORDER,
    mask_ccf_resample: bool = MASK_CCF_RESAMPLE,
    mask_ccf_fit_baseline_free: bool = MASK_CCF_FIT_BASELINE_FREE,
    spec_err: float = SPEC_ERR,
) -> tuple[float, float]:
    """Run ``StellarRVPrimitive`` on one injected line.

    Returns ``(recovered_m_s, rv_err_m_s)`` from the fiber average, which for
    one mask line is that line's Gaussian CCF centroid.
    """
    if prior_m_s is None:
        prior_m_s = float(rv_m_s)
    l2 = build_gaussian_line_l2(
        rv_m_s,
        wave_um=wave_um,
        lam_rest_um=lam_rest_um,
        order=order,
        spec_err=spec_err,
    )
    with quiet_pipeline_logging():
        out = StellarRVPrimitive.call(
            l2,
            algorithm="ccf_line_mask",
            mask_wavelength=np.array([float(lam_rest_um)], dtype=np.float64),
            mask_weight=np.array([1.0], dtype=np.float64),
            ccf_input_velocity=float(prior_m_s),
            do_fibers=["1"],
            do_orders=[int(order)],
            window_line_by_line=True,
            window_size_um=_line_window_um(lam_rest_um, SIGMA_V_M_S),
            v_half_window=float(v_half_window_m_s),
            v_step_coarse=float(v_step_m_s),
            mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
            mask_ccf_resample=bool(mask_ccf_resample),
            mask_ccf_fit_baseline_free=bool(mask_ccf_fit_baseline_free),
            ccf_zb=0.0,
            save_result=False,
            save_preview=False,
            show_preview=False,
            max_cores=1,
        )
    line = out.rvccf_fib1
    if len(line) != 1:
        raise RuntimeError(f"expected one mask line, got {len(line)}")
    avg = out.rvccfavg_fib1
    return float(avg["RV"][0]), float(avg["RV_ERR"][0])


def run_gaussian_line_rv_grid(
    injected_rv_m_s: np.ndarray | None = None,
    *,
    prior_offset_m_s: float = 0.0,
    order: int = ORDER,
    v_step_m_s: float = V_STEP_M_S,
    v_half_window_m_s: float = V_HALF_WINDOW_M_S,
    mask_ccf_pixel_interp_order: int = MASK_CCF_PIXEL_INTERP_ORDER,
    mask_ccf_resample: bool = MASK_CCF_RESAMPLE,
    mask_ccf_fit_baseline_free: bool = MASK_CCF_FIT_BASELINE_FREE,
) -> GaussianLineRVResult:
    """Recover every injected RV on the single-line spectrum."""
    if injected_rv_m_s is None:
        injected = np.asarray(DEFAULT_INJECTED_RV_M_S, dtype=np.float64).copy()
    else:
        injected = np.asarray(injected_rv_m_s, dtype=np.float64).ravel()
    wave, lam_rest = single_order_wave_grid(order)
    pix_dv = float(np.median(np.abs(np.diff(C_M_S * np.log(wave)))))
    recovered = np.empty(injected.shape, dtype=np.float64)
    rv_err = np.empty(injected.shape, dtype=np.float64)
    for i, rv in enumerate(injected):
        rec, err = recover_gaussian_line(
            float(rv),
            wave_um=wave,
            lam_rest_um=lam_rest,
            prior_m_s=float(rv) + float(prior_offset_m_s),
            order=order,
            v_step_m_s=v_step_m_s,
            v_half_window_m_s=v_half_window_m_s,
            mask_ccf_pixel_interp_order=mask_ccf_pixel_interp_order,
            mask_ccf_resample=mask_ccf_resample,
            mask_ccf_fit_baseline_free=mask_ccf_fit_baseline_free,
        )
        recovered[i] = rec
        rv_err[i] = err
    return GaussianLineRVResult(
        injected_m_s=injected,
        recovered_m_s=recovered,
        rv_err_m_s=rv_err,
        lam_rest_um=float(lam_rest),
        pixel_dv_m_s=pix_dv,
        order=int(order),
        v_step_m_s=float(v_step_m_s),
        v_half_window_m_s=float(v_half_window_m_s),
        mask_ccf_pixel_interp_order=int(mask_ccf_pixel_interp_order),
        mask_ccf_resample=bool(mask_ccf_resample),
        mask_ccf_fit_baseline_free=bool(mask_ccf_fit_baseline_free),
        prior_offset_m_s=float(prior_offset_m_s),
    )


def plot_gaussian_line_profile(
    out_path: str | Path,
    *,
    order: int = ORDER,
    zoom_km_s: float = 20.0,
) -> None:
    """Injected line at RV = 0, with the analytic Gaussian overplotted."""
    import matplotlib.pyplot as plt

    wave, lam_rest = single_order_wave_grid(order)
    flux = gaussian_flux(wave, lam_rest, 0.0)
    v_km_s = (C_M_S * (wave / lam_rest - 1.0)) / 1000.0
    keep = np.abs(v_km_s) <= float(zoom_km_s)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(v_km_s[keep], flux[keep], color="C0", lw=1.5, label="injected spectrum")
    v_model = np.linspace(-float(zoom_km_s), float(zoom_km_s), 400)
    model = 1.0 - DEPTH * np.exp(-0.5 * (v_model / (SIGMA_V_M_S / 1000.0)) ** 2)
    ax.plot(v_model, model, color="C1", ls="--", lw=1.2, label="analytic Gaussian")
    ax.set_xlabel("Velocity from line center (km/s)")
    ax.set_ylabel("Flux")
    ax.set_title(
        f"Order {int(order)} Gaussian line, depth {DEPTH:g}, "
        f"sigma {SIGMA_V_M_S / 1000.0:g} km/s"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_gaussian_line_profile_fit(
    out_path: str | Path,
    *,
    order: int = ORDER,
    rv_m_s: float = 0.0,
) -> None:
    """Native pixels of the injected line and the Gaussian fit that supplies the RV.

    The fit is the free-continuum Gaussian on the peak-up CCF, evaluated back
    in flux. Resampling is off, matching this example.
    """
    import matplotlib.pyplot as plt

    from ..primitives.utils.ccf.ccf_utils import fit_ccf_rv, rv_per_chunks_line_mask
    from ..utils.line_mask_utils import MaskLines

    wave, lam_rest = single_order_wave_grid(order)
    flux = gaussian_flux(wave, lam_rest, float(rv_m_s))
    mask = MaskLines.from_delta_centers_um(np.array([lam_rest]), np.array([1.0]))
    out = rv_per_chunks_line_mask(
        wave,
        flux,
        np.full(flux.shape, SPEC_ERR),
        np.zeros(flux.shape, dtype=bool),
        mask,
        window_size_um=_line_window_um(lam_rest, SIGMA_V_M_S),
        v_min_m_s=float(rv_m_s) - V_HALF_WINDOW_M_S,
        v_max_m_s=float(rv_m_s) + V_HALF_WINDOW_M_S,
        v_step_m_s=V_STEP_M_S,
        window_line_by_line=True,
        line_mask_line_chunk_v_center_m_s=float(rv_m_s),
        mask_ccf_pixel_interp_order=MASK_CCF_PIXEL_INTERP_ORDER,
        mask_ccf_resample=False,
        mask_ccf_fit_baseline_free=True,
        min_good_pixels=9,
        max_cores=1,
        return_ccf=True,
    )
    _rvs, _errs, _centers, v_grid, ccf, *_rest = out
    verr = np.asarray(_rest[3][0], dtype=np.float64)
    vv = np.asarray(v_grid, dtype=np.float64)
    yy = np.asarray(ccf[0], dtype=np.float64)
    finite = np.isfinite(vv) & np.isfinite(yy) & np.isfinite(verr) & (verr > 0.0)
    vv, yy, verr = vv[finite], yy[finite], verr[finite]
    step = float(np.median(np.diff(vv))) if vv.size > 1 else 1.0
    _rv, _err, params, diag = fit_ccf_rv(
        vv,
        yy,
        step,
        n_good=int(vv.size),
        return_params=True,
        ccf_err=verr,
        ccf_line_mask_stored_negated=True,
        line_mask_line_window_by_line=True,
        fit_ccf_baseline_free=True,
        gaussian_seed_mean_m_s=float(rv_m_s),
    )
    if not bool(diag.get("fit_ok")):
        raise RuntimeError("Gaussian fit failed while plotting the line profile")
    amp = float(params["amplitude"])
    baseline = float(params["baseline"])
    mean = float(params["v0"])
    sigma = float(params["sigma"])

    def flux_model(v_m_s: np.ndarray) -> np.ndarray:
        gauss = np.exp(-0.5 * ((np.asarray(v_m_s) - mean) / sigma) ** 2)
        # The CCF is stored peak-up, so flux is the negation of baseline plus Gaussian.
        return -(baseline + amp * gauss)

    v_km = vv / 1000.0
    flux_pix = -yy
    v_dense = np.linspace(float(vv[0]), float(vv[-1]), 400)
    resid = flux_pix - flux_model(vv)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    ax = axes[0]
    ax.plot(v_km, flux_pix, "o", ms=3.5, color="C0", label="simulated pixels")
    ax.plot(v_dense / 1000.0, flux_model(v_dense), color="C1", lw=1.4, label="Gaussian fit")
    ax.set_ylabel("Flux")
    ax.set_title(f"Simulated line and Gaussian fit, RV = {float(rv_m_s) / 1000.0:g} km/s")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.plot(v_km, resid, "o", ms=3.5, color="C1")
    half = max(float(np.max(np.abs(resid))) * 1.8, 1.0e-18)
    ax.set_ylim(-half, half)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Flux - Gaussian fit")
    ax.set_title(f"max |residual| = {float(np.max(np.abs(resid))):.3e}")
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_gaussian_line_rv_recovery(
    result: GaussianLineRVResult,
    out_path: str | Path,
) -> None:
    """Recovered RV versus injected RV, and the residual in mm/s."""
    import matplotlib.pyplot as plt

    inj_km = result.injected_m_s / 1000.0
    rec_km = result.recovered_m_s / 1000.0
    bias_mm = result.bias_m_s * 1.0e3
    max_abs_mm = float(np.max(np.abs(bias_mm)))
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    ax = axes[0]
    ax.plot(inj_km, inj_km, color="k", ls="--", lw=1, label="recovered = injected")
    ax.plot(inj_km, rec_km, "o", ms=4, color="C0", label="StellarRV")
    ax.set_ylabel("Recovered RV (km/s)")
    ax.legend(frameon=False)
    ax.set_title("Single Gaussian line, infinite SNR")
    ax = axes[1]
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.plot(inj_km, bias_mm, "o", ms=4, color="C1")
    half = max(max_abs_mm * 1.8, 1.0e-12)
    ax.set_ylim(-half, half)
    ax.set_xlabel("Injected RV (km/s)")
    ax.set_ylabel("Recovered - injected (mm/s)")
    bias_txt = f"{max_abs_mm:.3e}" if max_abs_mm < 1.0e-3 else f"{max_abs_mm:.3f}"
    ax.set_title(f"max |bias| = {bias_txt} mm/s")
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_gaussian_line_fit_valley(
    out_path: str | Path,
    *,
    order: int = ORDER,
) -> None:
    """Show why the Gaussian fit stops 0.013 mm/s from the injected RV.

    Chi-squared of the primitive's fixed-baseline Gaussian, in units of its
    stopping tolerance, along the centroid. The wide panel reaches the
    tolerance; the zoom shows the injected RV and the returned RV on the flat
    bottom. The line is placed at a quarter pixel, where the residual is largest.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar

    from ..primitives.utils.ccf.ccf_gauss_fit_numba import _sigma_clipped_median
    from ..primitives.utils.ccf.ccf_utils import rv_per_chunks_line_mask
    from ..utils.line_mask_utils import MaskLines

    wave, lam_rest = single_order_wave_grid(order)
    pix = float(np.median(np.abs(np.diff(C_M_S * np.log(wave)))))
    mask = MaskLines.from_delta_centers_um(np.array([lam_rest]), np.array([1.0]))

    def ccf_at(rv_m_s: float):
        flux = gaussian_flux(wave, lam_rest, rv_m_s)
        out = rv_per_chunks_line_mask(
            wave,
            flux,
            np.full(flux.shape, SPEC_ERR),
            np.zeros(flux.shape, dtype=bool),
            mask,
            window_size_um=_line_window_um(lam_rest, SIGMA_V_M_S),
            v_min_m_s=float(rv_m_s) - V_HALF_WINDOW_M_S,
            v_max_m_s=float(rv_m_s) + V_HALF_WINDOW_M_S,
            v_step_m_s=V_STEP_M_S,
            window_line_by_line=True,
            line_mask_line_chunk_v_center_m_s=float(rv_m_s),
            mask_ccf_pixel_interp_order=MASK_CCF_PIXEL_INTERP_ORDER,
            min_good_pixels=9,
            max_cores=1,
            return_ccf=True,
        )
        rvs, _errs, _centers, v_grid, ccf, *_rest = out
        return (
            float(rvs[0]),
            np.asarray(v_grid, dtype=np.float64),
            np.asarray(ccf[0], dtype=np.float64),
            np.asarray(_rest[3][0], dtype=np.float64),
        )

    # Quarter-pixel phase: the largest residual on this line.
    rv_q = 0.25 * pix
    recovered_q, v_grid, ccf, ccf_err = ccf_at(rv_q)
    # Second pass of the primitive's Gaussian fit: 0.42 of the velocity span.
    half_m_s = 0.42 * float(v_grid[-1] - v_grid[0])
    keep = np.abs(v_grid - rv_q) <= half_m_s
    vv = v_grid[keep]
    yy = ccf[keep]
    ee = ccf_err[keep]
    med = float(_sigma_clipped_median(np.ascontiguousarray(yy), 3.0, 5))
    ys = 1.0 / max(float(np.max(np.abs(yy - med))), 1.0e-30)
    sw = np.sqrt((1.0 / (ee * ee)) / float(np.median(1.0 / (ee * ee))))
    target = sw * (yy - med) * ys

    def chi2_at(mu_m_s: float) -> float:
        def at_sigma(sig_m_s: float) -> float:
            g = np.exp(-0.5 * ((vv - mu_m_s) / sig_m_s) ** 2)
            gg = sw * g
            amp = float(np.dot(gg, target) / np.dot(gg, gg))
            resid = target - amp * gg
            return float(np.dot(resid, resid))

        return float(minimize_scalar(at_sigma, bounds=(3_000.0, 5_000.0), method="bounded").fun)

    chi2_inj = chi2_at(rv_q)
    tolerance = 1.0e-12 * (1.0 + chi2_inj)
    offsets_mm = np.linspace(-0.25, 0.25, 81)
    delta = np.array([chi2_at(rv_q + d * 1.0e-3) - chi2_inj for d in offsets_mm]) / tolerance
    bias_q_mm = (recovered_q - rv_q) * 1.0e3

    def draw(ax, x_half: float, y_lo: float, y_hi: float) -> None:
        ax.plot(offsets_mm, delta, color="C0", lw=1.6, zorder=2)
        ax.axvline(0.0, color="k", ls="--", lw=1, label="injected RV")
        ax.axvline(
            bias_q_mm,
            color="C1",
            lw=1.4,
            label=f"returned RV ({bias_q_mm:+.3f} mm/s)",
        )
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlim(-x_half, x_half)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Centroid minus injected RV (mm/s)")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    draw(axes[0], 0.25, -0.2, 1.55)
    tol = axes[0].axhline(1.0, color="0.35", ls=":", lw=1.1, label="stopping tolerance")
    axes[0].set_ylabel("Change in chi-squared / stopping tolerance")
    axes[0].set_title("Quarter-pixel line")
    axes[0].legend(handles=[tol], frameon=False, fontsize=8, loc="lower right")

    zoom = np.abs(offsets_mm) <= 0.05
    z_peak = float(np.max(np.abs(delta[zoom])))
    draw(axes[1], 0.05, -1.6 * z_peak, max(1.8 * z_peak, 0.05))
    axes[1].set_ylabel("Change in chi-squared / stopping tolerance")
    axes[1].set_title("Bottom of the same curve")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(
        0.97,
        0.96,
        "stopping tolerance is at 1",
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=8,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_gaussian_line_summary(result: GaussianLineRVResult, path: str | Path) -> None:
    """Machine-readable bias summary. Paths are not recorded."""
    bias = result.bias_m_s
    payload = {
        "order": int(result.order),
        "n_pix": int(result.n_pix),
        "lam_rest_um": float(result.lam_rest_um),
        "pixel_dv_m_s": float(result.pixel_dv_m_s),
        "depth": float(result.depth),
        "sigma_v_m_s": float(result.sigma_v_m_s),
        "spec_err": float(result.spec_err),
        "continuum_snr_per_pixel": 1.0 / float(result.spec_err),
        "v_step_m_s": float(result.v_step_m_s),
        "v_half_window_m_s": float(result.v_half_window_m_s),
        "mask_ccf_pixel_interp_order": int(result.mask_ccf_pixel_interp_order),
        "mask_ccf_resample": bool(result.mask_ccf_resample),
        "mask_ccf_fit_baseline_free": bool(result.mask_ccf_fit_baseline_free),
        "prior_offset_m_s": float(result.prior_offset_m_s),
        "n_injected": int(result.injected_m_s.size),
        "injected_rv_min_m_s": float(np.min(result.injected_m_s)),
        "injected_rv_max_m_s": float(np.max(result.injected_m_s)),
        "injected_rv_step_m_s": float(np.median(np.diff(result.injected_m_s))),
        "max_abs_bias_m_s": float(np.max(np.abs(bias))),
        "rms_bias_m_s": float(np.sqrt(np.mean(bias**2))),
        "max_rv_err_m_s": float(np.max(result.rv_err_m_s)),
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
