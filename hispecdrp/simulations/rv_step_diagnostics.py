"""Step-by-step ideal-spectrum RV recovery (truth 1D → L2 → rate map → UTR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import astropy.constants as const
import astropy.units as u
import numpy as np

from ..primitives.utils.ccf import doppler_paint_lanczos
from ..primitives.utils.spectral_utils import (
    get_spectrum_path,
    load_btsettl_spectrum,
    normalize_2mass_band,
)
from .e2e_support import templates_data_dir
from .ideal_spectrum_rv import (
    FIBERFLAT_IDENTITY,
    build_template_rate_map,
    extract_and_blaze_l2,
    geometric_hdf_wavesol,
    prepare_stage1_cals,
    rate_map_to_l0,
    recover_rv_from_ramp,
    recover_rv_from_rate_map,
)
from .simulate_rate_map import SimulateRateMap, _upsample_spectrum
from .simulate_spectra import (
    _resolve_order_bounds,
    build_ccfopt_mask,
    build_l2,
    quiet_pipeline_logging,
    recover_stellar_rv,
)

__all__ = [
    "NO_LINE_REJECTION_SIGMA",
    "STEP_BULK_RV_KM_S",
    "STELLAR_RV_NO_REJECT_KW",
    "load_densified_template",
    "doppler_shift_spectrum",
    "paint_spectrum_onto_l2",
    "recover_rv_keep_all_lines",
    "recover_rv_from_truth_l2",
    "recover_rv_from_truth_rate_map",
    "recover_rv_from_truth_ramp",
    "run_step_grid",
    "summarize_step_grid",
    "compare_step_summaries",
    "verify_zero_rv_recovery",
    "saved_rate_map_path",
    "wavesol_from_line_zeropoint",
    "wavesol_from_extracted_comb",
    "derive_lfc_wavesol",
    "check_lsf_wavesol_fiber_alignment",
    "build_extract_matched_ccfopt_mask",
    "recover_rv_from_saved_rate_map",
]

# Finite stand-in for infinity: every finite line stays in the aggregate.
# ``float("inf")`` is avoided because ``inf * 0`` becomes NaN in the MAD gate.
NO_LINE_REJECTION_SIGMA = 1.0e30
STEP_BULK_RV_KM_S = np.arange(0.0, 51.0, 10.0)
STELLAR_RV_NO_REJECT_KW = {"outlier_sigma": NO_LINE_REJECTION_SIGMA}

DENSIFY_RESOLUTION = 3.0e5


def load_densified_template(
    template: str,
    *,
    mag_ab: float = 3.05,
    filter_band: str = "J",
    resolution: float = DENSIFY_RESOLUTION,
):
    """Load a BT-Settl template, sort it, and densify to ``resolution``."""
    template_path = get_spectrum_path(template, base_dir=templates_data_dir())
    wl, flux = load_btsettl_spectrum(
        template_path,
        mag_ab=float(mag_ab),
        filter_band=normalize_2mass_band(filter_band),
    )
    sort_idx = np.argsort(np.asarray(wl.value, dtype=np.float64))
    wl, flux = wl[sort_idx], flux[sort_idx]
    return _upsample_spectrum(wl, flux, desired_sampling_resolution=float(resolution))


def doppler_shift_spectrum(wavelengths, flux, rv_km_s: float):
    """Shift a 1D spectrum by ``λ' = λ (1 + RV/c)``. Flux is unchanged."""
    factor = (1.0 + (float(rv_km_s) * u.km / u.s) / const.c).to(u.dimensionless_unscaled)
    return wavelengths * factor, flux


def paint_spectrum_onto_l2(
    wavelengths,
    flux,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    rv_m_s: float = 0.0,
    order_start: int | None = None,
    order_end: int | None = None,
    mask_ccf_pixel_interp_order: int = 3,
    spec_err_floor: float = 1.0e-6,
    spec_err_relative: float = 1.0e-6,
):
    """Resample an already-shifted 1D spectrum onto the native echelle WAVE grid."""
    from .. import datamodels
    from .e2e_gaia import resolve_e2e_mjd_start

    ost, oen = _resolve_order_bounds(spectrograph, order_start, order_end)
    ws_all = SimulateRateMap._instrument_wavelength_solution(spectrograph)[int(fiber)]
    orders = [o for o in sorted(int(o) for o in ws_all.keys()) if ost <= o <= oen]
    if not orders:
        raise ValueError(f"No echelle orders in [{ost}, {oen}]")
    n_pix = int(np.asarray(ws_all[orders[0]]["wavelength"]).size)
    wave_2d = np.zeros((len(orders), n_pix), dtype=np.float64)
    spec = np.empty_like(wave_2d)
    w_src = np.asarray(getattr(wavelengths, "value", wavelengths), dtype=np.float64).ravel()
    f_src = np.asarray(getattr(flux, "value", flux), dtype=np.float64).ravel()
    for i, od in enumerate(orders):
        w_obs = np.asarray(ws_all[od]["wavelength"], dtype=np.float64)
        wave_2d[i] = w_obs
        spec[i] = doppler_paint_lanczos(
            w_obs,
            w_src,
            f_src,
            float(rv_m_s),
            order=int(mask_ccf_pixel_interp_order),
        )
    model = datamodels.L2Model(
        meta={
            "order_start": ost,
            "order_end": oen,
            "spectrograph": str(spectrograph).upper(),
            "instrument_name": "HISPEC",
            "mjd_start": resolve_e2e_mjd_start(None),
        }
    )
    model.meta.order_start = ost
    model.meta.order_end = oen
    fiber_s = str(int(fiber))
    setattr(model, f"wave_fib{fiber_s}", wave_2d.copy())
    setattr(model, f"spec_fib{fiber_s}", spec.astype(np.float32))
    err = np.maximum(float(spec_err_relative) * np.abs(spec), float(spec_err_floor)).astype(
        np.float32
    )
    setattr(model, f"spec_err_fib{fiber_s}", err)
    setattr(model, f"spec_dq_fib{fiber_s}", np.zeros(wave_2d.shape, dtype=np.uint32))
    return model


def recover_rv_keep_all_lines(
    l2,
    bulk_rv_km_s: float,
    *,
    template: str,
    mask_path: str,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    quiet: bool = True,
    **stellar_rv_overrides,
) -> tuple[float, float, float]:
    """``recover_stellar_rv`` with every finite line kept in the aggregate."""
    kw = dict(STELLAR_RV_NO_REJECT_KW)
    kw.update(stellar_rv_overrides)
    return recover_stellar_rv(
        l2,
        bulk_rv_km_s,
        template=template,
        mask_path=str(mask_path),
        spectrograph=spectrograph,
        fiber=fiber,
        quiet=quiet,
        **kw,
    )


def recover_rv_from_truth_l2(
    rv_km_s: float,
    *,
    wavelengths,
    flux,
    template: str,
    mask_path: str,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
) -> tuple[float, float, float]:
    """Paint the Doppler-shifted truth 1D onto L2 and recover RV."""
    l2 = paint_spectrum_onto_l2(
        wavelengths,
        flux,
        spectrograph=spectrograph,
        fiber=fiber,
        rv_m_s=0.0,
    )
    return recover_rv_keep_all_lines(
        l2,
        rv_km_s,
        template=template,
        mask_path=mask_path,
        spectrograph=spectrograph,
        fiber=fiber,
    )


def recover_rv_from_truth_rate_map(
    rv_km_s: float,
    *,
    wavelengths,
    flux,
    template: str,
    mask_path: str,
    output_dir: str | Path,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    wavesol=None,
    fiberflat_mode: str = "identity",
) -> tuple[float, float, float]:
    """ZEMAX/native rate map of the truth 1D → extract → blaze → RV.

    Defaults to an identity fiberflat (blaze no-op) so continuum flattening
    does not reweight the CCF; pair with an LFC-derived ``wavesol`` for ≪1 m/s
    bias on this diagnostic grid.
    """
    return recover_rv_from_rate_map(
        rv_km_s,
        template=template,
        mask_path=str(mask_path),
        output_dir=output_dir,
        fiber=fiber,
        spectrograph=spectrograph,
        spectrum_wavelengths=wavelengths,
        spectrum_flux=flux,
        wavesol=wavesol,
        fiberflat_mode=fiberflat_mode,
        **STELLAR_RV_NO_REJECT_KW,
    )


def recover_rv_from_truth_ramp(
    rv_km_s: float,
    *,
    wavelengths,
    flux,
    template: str,
    mask_path: str,
    output_dir: str | Path,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    wavesol=None,
    fiberflat_mode: str = "identity",
) -> tuple[float, float, float]:
    """Truth 1D → UTR + Stage-0 cals → Stage 1–2 → RV.

    Same identity-fiberflat default as :func:`recover_rv_from_truth_rate_map`.
    """
    return recover_rv_from_ramp(
        rv_km_s,
        template=template,
        mask_path=str(mask_path),
        output_dir=output_dir,
        fiber=fiber,
        spectrograph=spectrograph,
        spectrum_wavelengths=wavelengths,
        spectrum_flux=flux,
        wavesol=wavesol,
        fiberflat_mode=fiberflat_mode,
        **STELLAR_RV_NO_REJECT_KW,
    )


def summarize_step_grid(
    injected_m_s: np.ndarray,
    recovered_m_s: np.ndarray,
    recovered_err_m_s: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    """Bias / precision stats for one recovery step."""
    inj = np.asarray(injected_m_s, dtype=float).ravel()
    rec = np.asarray(recovered_m_s, dtype=float).ravel()
    err = np.asarray(recovered_err_m_s, dtype=float).ravel()
    bias = rec - inj
    return {
        "name": name,
        "injected_m_s": inj,
        "recovered_m_s": rec,
        "recovered_err_m_s": err,
        "bias_m_s": bias,
        "mean_bias_m_s": float(np.mean(bias)),
        "std_bias_m_s": float(np.std(bias)),
        "rms_bias_m_s": float(np.sqrt(np.mean(bias**2))),
        "mean_err_m_s": float(np.mean(err)),
        "mean_bias_cm_s": float(np.mean(bias) * 100.0),
        "std_bias_cm_s": float(np.std(bias) * 100.0),
        "mean_err_cm_s": float(np.mean(err) * 100.0),
    }


def run_step_grid(
    recover_one: Callable[[float], tuple[float, float, float]],
    *,
    name: str,
    title: str,
    out_path: str | Path,
    bulk_rv_km_s: np.ndarray | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Sweep injected RVs, print the table, and optionally write the residual plot."""
    from .plotting import plot_ideal_spectrum_rv_recovery

    grid = np.asarray(
        STEP_BULK_RV_KM_S if bulk_rv_km_s is None else bulk_rv_km_s,
        dtype=float,
    )
    hdr = (
        f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
        f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
    )
    inj_m_s, rec_m_s, err_m_s = [], [], []
    print(hdr)
    print("-" * len(hdr))
    for bulk in grid:
        with quiet_pipeline_logging():
            inj, rec, err = recover_one(float(bulk))
        bias = rec - inj
        inj_m_s.append(float(inj))
        rec_m_s.append(float(rec))
        err_m_s.append(float(err))
        print(
            f"{bulk:10.1f}  {inj:14.4f}  {rec:16.6f}  {err:12.6f}  "
            f"{bias:12.6e}  {bias * 1e3:10.4f}"
        )
    summary = summarize_step_grid(inj_m_s, rec_m_s, err_m_s, name=name)
    print()
    print(
        f"{name}: mean bias = {summary['mean_bias_cm_s']:.3f} cm/s  "
        f"std = {summary['std_bias_cm_s']:.3f} cm/s  "
        f"mean RV_ERR = {summary['mean_err_cm_s']:.3f} cm/s"
    )
    if plot:
        plot_summary = plot_ideal_spectrum_rv_recovery(
            summary["injected_m_s"],
            summary["recovered_m_s"],
            summary["recovered_err_m_s"],
            title=title,
            out_path=out_path,
        )
        summary["out_path"] = plot_summary.get("out_path")
        if summary.get("out_path"):
            print("wrote", summary["out_path"])
    return summary


def compare_step_summaries(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Print mean bias / scatter / RV_ERR and the increment between consecutive steps."""
    rows = [s for s in summaries.values() if s]
    if not rows:
        return []
    hdr = (
        f"{'step':<28}  {'mean_bias':>12}  {'std_bias':>12}  "
        f"{'mean_RV_ERR':>12}  {'Δmean_bias':>12}  {'Δstd':>12}  {'ΔRV_ERR':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    print("                        (m/s)         (m/s)         (m/s)         (m/s)         (m/s)         (m/s)")
    prev = None
    increments: list[dict[str, Any]] = []
    for row in rows:
        d_mean = float("nan") if prev is None else row["mean_bias_m_s"] - prev["mean_bias_m_s"]
        d_std = float("nan") if prev is None else row["std_bias_m_s"] - prev["std_bias_m_s"]
        d_err = float("nan") if prev is None else row["mean_err_m_s"] - prev["mean_err_m_s"]
        rec = {
            "name": row["name"],
            "mean_bias_m_s": row["mean_bias_m_s"],
            "std_bias_m_s": row["std_bias_m_s"],
            "mean_err_m_s": row["mean_err_m_s"],
            "delta_mean_bias_m_s": d_mean,
            "delta_std_bias_m_s": d_std,
            "delta_mean_err_m_s": d_err,
        }
        increments.append(rec)
        print(
            f"{row['name']:<28}  {row['mean_bias_m_s']:12.6f}  "
            f"{row['std_bias_m_s']:12.6f}  {row['mean_err_m_s']:12.6f}  "
            f"{d_mean:12.6f}  {d_std:12.6f}  {d_err:12.6f}"
        )
        prev = row
    if len(increments) >= 2:
        worst = max(
            increments[1:],
            key=lambda r: max(
                abs(r["delta_mean_bias_m_s"]),
                abs(r["delta_std_bias_m_s"]),
                abs(r["delta_mean_err_m_s"]),
            ),
        )
        print()
        print(
            f"Largest increment: {worst['name']}  "
            f"Δmean={worst['delta_mean_bias_m_s']:.4f} m/s  "
            f"Δstd={worst['delta_std_bias_m_s']:.4f} m/s  "
            f"ΔRV_ERR={worst['delta_mean_err_m_s']:.4f} m/s"
        )
    return increments


def verify_zero_rv_recovery(
    *,
    template: str,
    mask_path: str,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    max_abs_bias_m_s: float = 1.0,
) -> tuple[float, float, float]:
    """Paint the rest-frame densified template onto L2 and recover RV at 0 km/s."""
    with quiet_pipeline_logging():
        l2 = build_l2(0.0, template=template, fiber=fiber, spectrograph=spectrograph)
        inj, rec, err = recover_rv_keep_all_lines(
            l2,
            0.0,
            template=template,
            mask_path=mask_path,
            spectrograph=spectrograph,
            fiber=fiber,
        )
    bias = rec - inj
    print(
        f"RV=0 check: injected={inj:.6f} m/s  recovered={rec:.6f} m/s  "
        f"RV_ERR={err:.6f} m/s  bias={bias:.6e} m/s"
    )
    if abs(bias) > float(max_abs_bias_m_s):
        print(
            f"WARNING: |bias| = {abs(bias):.4f} m/s exceeds "
            f"{max_abs_bias_m_s:.4f} m/s at injected RV=0"
        )
    else:
        print(f"|bias| = {abs(bias):.4e} m/s  (within {max_abs_bias_m_s:.4f} m/s)")
    return inj, rec, err


def saved_rate_map_path(output_dir: str | Path, rv_km_s: float) -> Path:
    """Path written by :func:`recover_rv_from_rate_map` for one bulk RV."""
    run_dir = Path(output_dir) / f"rate_map_{float(rv_km_s):+.1f}kms".replace(
        "+", "p"
    ).replace("-", "m")
    return run_dir / "rate_map.fits"


def derive_lfc_wavesol(
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    output_dir: str | Path,
    y_min: int = 200,
    y_max: int = 3896,
    photons_per_line: float = 1.0e7,
    derive_lsf: bool = True,
):
    """Paint a simulated LFC on ``fiber`` and run ``ProcessLFCPipeline``.

    Returns the derived ``lfc_wavesol`` model for use as Stage-1 WAVE in the
    rate-map and UTR recovery paths (instead of the geometric HDF solution).

    When ``derive_lsf`` is True (default), also runs
    :class:`~hispecdrp.primitives.lsf.lsf_gauss_primitive.CalculateGaussianLSFPrimitive`
    with the same settings as ``simulated_lfc_example.ipynb``, saves
    ``lfc_lsf.fits``, and stores the model under ``info["lsf"]``.
    """
    import os

    from astropy import units as u

    from .. import BlazeCorrectionPrimitive, datamodels
    from ..pipelines import ProcessLFCPipeline, Stage1Pipeline
    from .e2e_pipeline import (
        build_lfc_line_list,
        create_tracepos_model_from_sim_hdf,
        create_traceprof_model_from_sim_hdf,
    )
    from .ideal_spectrum_rv import geometric_hdf_wavesol
    from .simulate_rate_map import SimulateRateMap

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["KOA_CALIBRATION_CACHE"] = str(out / "cals" / "lfc_local_caldb")

    line_list, lfc_meta = build_lfc_line_list(spectrograph, fiber=int(fiber))
    lfc_wl = np.unique(
        np.concatenate([np.asarray(v, dtype=float) for v in line_list.values()])
    )
    lfc_wl.sort()
    srm = SimulateRateMap()
    srm.set_fiber_spectrum(
        int(fiber),
        lfc_wl * u.micron,
        np.full(lfc_wl.shape, float(photons_per_line)) * (u.ph / u.s),
        continuous=False,
    )
    rate = srm.create_native_rate_map(
        spectrograph=str(spectrograph).upper(),
        method="prf_convolution",
        apply_blaze=False,
        save=False,
    )
    rate = np.asarray(rate, dtype=np.float32)
    rate[: int(y_min)] = 0
    rate[int(y_max) :] = 0

    do_fibers = [str(int(fiber))]
    l0 = datamodels.L0Model(
        rate=rate,
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": str(spectrograph).upper(),
            "datetime_obs": "2027-01-01T00:00:00.000",
            "exposure_type": "SPEC",
            "lfc_nu0": lfc_meta["nu0"],
            "lfc_delta_nu": lfc_meta["delta_nu"],
            "target_name": lfc_meta["fiber_object"],
        },
    )
    l0.rate_err = np.sqrt(np.maximum(l0.rate, 0.0)).astype(np.float32)
    l0.rate_var_poisson = np.zeros_like(l0.rate, dtype=np.float32)
    l0.rate_var_rnoise = np.zeros_like(l0.rate, dtype=np.float32)
    l0.dq = np.zeros_like(l0.rate, dtype=np.uint32)

    with quiet_pipeline_logging():
        l1 = Stage1Pipeline(
            output_dir=out,
            dataset=l0,
            primitives={
                "spec_extract": {
                    "do_fibers": do_fibers,
                    "method": "box",
                    "tracepos": create_tracepos_model_from_sim_hdf(spectrograph),
                    "traceprof": create_traceprof_model_from_sim_hdf(spectrograph),
                },
                "set_wavesol": {"wavesol": geometric_hdf_wavesol(spectrograph)},
            },
        ).apply()
        n_orders, n_pix = getattr(l1, f"spec_uncal_fib{int(fiber)}").shape
        fiberflat = datamodels.FiberflatModel(
            meta={
                "instrument_name": "HISPEC",
                "spectrograph": str(spectrograph).upper(),
            },
        )
        fiberflat[int(fiber)].spec = np.ones((n_orders, n_pix), dtype=np.float32)
        fiberflat[int(fiber)].spec_err = np.full(
            (n_orders, n_pix), 1.0e-2, dtype=np.float32
        )
        fiberflat[int(fiber)].spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
        l2 = BlazeCorrectionPrimitive.call(
            l1,
            do_fibers=do_fibers,
            fiberflat=fiberflat,
            output_dir=out,
            save_result=False,
        )
        l2.meta.lfc_nu0 = l0.meta.lfc_nu0
        l2.meta.lfc_delta_nu = l0.meta.lfc_delta_nu
        lfc_prims = {
            "fit_lfc_lines": {"do_fibers": do_fibers, "fwhm_bounds": (1.0, 7.0)},
            "lfc_wavesol": {"do_fibers": do_fibers},
            "register_lsf_cal": {"run": False},
        }
        if derive_lsf:
            # Match simulated_lfc_example.ipynb (empirical LFC-spot LSF).
            lfc_prims["lsf"] = {"do_fibers": do_fibers}
        else:
            lfc_prims["lsf"] = {"do_fibers": do_fibers, "run": False}
        lfc_out = ProcessLFCPipeline(
            output_dir=out,
            dataset=l2,
            primitives=lfc_prims,
        ).apply()
        wavesol = lfc_out["wavesol"]
        lsf = lfc_out.get("lsf") if derive_lsf else None
    wavesol_path = out / "lfc_wavesol.fits"
    wavesol.save(filepath=str(wavesol_path))
    info = {
        "n_modes": int(lfc_wl.size),
        "lfc_meta": lfc_meta,
        "wavesol_path": str(wavesol_path),
        "rate_peak": float(np.nanmax(rate)),
        "lsf": None,
        "lsf_path": None,
    }
    if lsf is not None:
        lsf_path = out / "lfc_lsf.fits"
        lsf.save(filepath=str(lsf_path))
        info["lsf"] = lsf
        info["lsf_path"] = str(lsf_path)
    return wavesol, info


def check_lsf_wavesol_fiber_alignment(
    *,
    fiber: int,
    wavesol,
    lsf_model=None,
) -> dict[str, Any]:
    """Verify WAVE (and optional LSF) expose the science fiber with matching orders.

    ProcessLFC WAVE/LSF are fiber-scoped (``wave_fibN``, ``lsf_params_fibN``).
    Recovery and ccfopt must use the same ``fiber`` that produced those cals;
    a mismatch silently falls back to empty / wrong-fiber arrays.
    """
    fiber = int(fiber)
    messages: list[str] = []
    ok = True
    wave_attr = f"wave_fib{fiber}"
    wave = getattr(wavesol, wave_attr, None)
    if wave is None:
        ok = False
        messages.append(f"wavesol missing {wave_attr}")
        wave_shape = None
    else:
        wave_arr = np.asarray(wave)
        wave_shape = tuple(wave_arr.shape)
        messages.append(f"wavesol.{wave_attr} shape={wave_shape}")
        if wave_arr.size == 0 or not np.any(np.isfinite(wave_arr)):
            ok = False
            messages.append(f"wavesol.{wave_attr} has no finite samples")

    lsf_shape = None
    if lsf_model is not None:
        lsf_attr = f"lsf_params_fib{fiber}"
        lsf_params = getattr(lsf_model, lsf_attr, None)
        if lsf_params is None:
            ok = False
            messages.append(f"lsf missing {lsf_attr}")
        else:
            lsf_arr = np.asarray(lsf_params)
            lsf_shape = tuple(lsf_arr.shape)
            messages.append(f"lsf.{lsf_attr} shape={lsf_shape}")
            if wave_shape is not None and lsf_shape and wave_shape[0] != lsf_shape[0]:
                ok = False
                messages.append(
                    f"order count mismatch: WAVE n_orders={wave_shape[0]} "
                    f"vs LSF n_orders={lsf_shape[0]}"
                )
        meta_fiber = None
        for key in ("FIBER", "fiber"):
            if hasattr(lsf_model, "meta") and hasattr(lsf_model.meta, key):
                try:
                    meta_fiber = int(getattr(lsf_model.meta, key))
                except (TypeError, ValueError):
                    meta_fiber = None
                if meta_fiber is not None:
                    break
        if meta_fiber is not None and meta_fiber > 0 and meta_fiber != fiber:
            ok = False
            messages.append(
                f"lsf meta FIBER={meta_fiber} != science fiber={fiber}"
            )

    return {
        "ok": bool(ok),
        "fiber": fiber,
        "wave_shape": wave_shape,
        "lsf_shape": lsf_shape,
        "messages": messages,
    }


def build_extract_matched_ccfopt_mask(
    template: str,
    output_dir: str | Path,
    *,
    wavesol,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    fiberflat_mode: str = FIBERFLAT_IDENTITY,
    extraction_aperture: float = 8.0,
    lsf_model=None,
    rebuild_seed: bool = True,
    mag_ab: float = 3.05,
    filter_band: str = "J",
    spectrum_wavelengths=None,
    spectrum_flux=None,
    **ccfopt_overrides,
) -> tuple[str, str, Any, dict[str, Any]]:
    """Build ccfopt from an RV=0 rate map extracted under the science WAVE/FF.

    Paints the template (or provided densified spectrum) with
    ``prf_convolution``, box-extracts with the same Stage-1 traces / aperture
    as recovery, attaches ``wavesol`` (typically LFC WAVE) and the recovery
    fiberflat, then runs line-by-line ccfopt on that L2.

    Prefer this over convolving an ideal geometric-WAVE L2 with ProcessLFC LSF
    (``*.lfc_lsf.ccfopt.delta.mas``): line shapes then match the rate-map path.
    LFC WAVE + identity fiberflat remain the main ≪1 m/s bias fix; this mask
    only aligns ccfopt centroids with extract-matched line profiles.

    Returns ``(seed_mask_path, ccfopt_mask_path, extracted_l2, info)`` where
    ``info`` includes the LSF/WAVE fiber-alignment check.
    """
    fiber = int(fiber)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    align = check_lsf_wavesol_fiber_alignment(
        fiber=fiber,
        wavesol=wavesol,
        lsf_model=lsf_model,
    )
    if not align["ok"]:
        raise ValueError(
            "LSF/WAVE fiber alignment check failed: "
            + "; ".join(align["messages"])
        )

    work = out / "extract_matched_l2_rv0"
    with quiet_pipeline_logging():
        rate = build_template_rate_map(
            0.0,
            template=template,
            fiber=fiber,
            spectrograph=spectrograph,
            output_dir=work,
            mag_ab=mag_ab,
            filter_band=filter_band,
            spectrum_wavelengths=spectrum_wavelengths,
            spectrum_flux=spectrum_flux,
        )
        stage1_cals = prepare_stage1_cals(
            out,
            rate,
            spectrograph=spectrograph,
            fiber=fiber,
            extraction_aperture=float(extraction_aperture),
            wavesol=wavesol,
            fiberflat_mode=fiberflat_mode,
        )
        l0 = rate_map_to_l0(rate, spectrograph=spectrograph)
        l2 = extract_and_blaze_l2(
            l0,
            output_dir=work,
            fiber=fiber,
            extraction_aperture=float(extraction_aperture),
            stage1_cals=stage1_cals,
            save_result=True,
        )

    seed, ccfopt, l2_out = build_ccfopt_mask(
        template,
        out,
        spectrograph=spectrograph,
        fiber=fiber,
        rebuild_seed=rebuild_seed,
        l2=l2,
        mask_suffix=".extract_matched.ccfopt.delta.mas",
        **ccfopt_overrides,
    )
    info = {
        "alignment": align,
        "fiberflat_mode": str(fiberflat_mode),
        "extraction_aperture": float(extraction_aperture),
        "work_dir": str(work),
        "l2_path": getattr(getattr(l2_out, "meta", None), "filename", None),
    }
    return seed, ccfopt, l2_out, info


def wavesol_from_line_zeropoint(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    known_rv_m_s: float = 0.0,
    poly_deg: int = 0,
):
    """Geometric WAVE plus absorption-line ZP measured on ``rate_map``.

    ``poly_deg=0`` absorbs a constant PSF/deposit shift. Use a rate map whose
    injected stellar RV is ``known_rv_m_s`` so that Doppler is not folded in.
    """
    from .e2e_pipeline import refine_sim_hdf_wavesol_line_zeropoint

    wavesol = geometric_hdf_wavesol(spectrograph)
    info = refine_sim_hdf_wavesol_line_zeropoint(
        wavesol,
        np.asarray(rate_map),
        spectrograph=str(spectrograph).upper(),
        fiber=int(fiber),
        known_rv_m_s=float(known_rv_m_s),
        poly_deg=int(poly_deg),
    )
    return wavesol, info


def wavesol_from_extracted_comb(
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    extraction_aperture: float = 8.0,
    n_lines_per_order: int = 60,
    poly_deg: int = 3,
):
    """WAVE from a 1D box-extracted emission comb (same extract as science).

    Geometric deposit-inverse WAVE is exact at paint centers. Box extraction
    measures a PSF-weighted peak, so this fit is the simulated wavecal that
    matches the science extract.
    """
    from .e2e_pipeline import _order_list, calibrate_wavesol_from_injected_comb

    comb = calibrate_wavesol_from_injected_comb(
        str(spectrograph).upper(),
        fiber=int(fiber),
        n_lines_per_order=int(n_lines_per_order),
        poly_deg=int(poly_deg),
        extraction_aperture=float(extraction_aperture),
        centroid_mode="1d",
        paint_mode="native",
    )
    wavesol = geometric_hdf_wavesol(spectrograph)
    attr = f"wave_fib{int(fiber)}"
    wave = np.asarray(getattr(wavesol, attr), dtype=float).copy()
    pix_grid = np.arange(wave.shape[1], dtype=float)
    order_list = _order_list(str(spectrograph).upper())
    meas = comb.get("line_measurements") or []
    per_order: dict[int, dict[str, Any]] = {}
    for oi, od in enumerate(order_list):
        if oi >= wave.shape[0]:
            break
        pix = np.array(
            [
                m["pix_meas"]
                for m in meas
                if int(m.get("order", -1)) == int(od) and np.isfinite(m.get("pix_meas", np.nan))
            ],
            dtype=float,
        )
        lam = np.array(
            [
                m["lam_true_um"]
                for m in meas
                if int(m.get("order", -1)) == int(od) and np.isfinite(m.get("pix_meas", np.nan))
            ],
            dtype=float,
        )
        ok = np.isfinite(pix) & np.isfinite(lam)
        if int(ok.sum()) < int(poly_deg) + 3:
            continue
        coeffs = np.polyfit(pix[ok], lam[ok], int(poly_deg))
        wave[oi] = np.polyval(coeffs, pix_grid)
        per_order[int(od)] = {"n_lines": int(ok.sum()), "n_coeff": int(coeffs.size)}
    setattr(wavesol, attr, wave)
    cents = [
        float(v.get("centroid_residual_mean_m_s", np.nan))
        for v in (comb.get("per_order") or {}).values()
    ]
    return wavesol, {
        "median_centroid_residual_m_s": float(np.nanmedian(cents)) if cents else float("nan"),
        "n_orders_fit": len(per_order),
        "per_order": per_order,
        "comb_rv_budget": comb.get("rv_budget"),
    }


def recover_rv_from_saved_rate_map(
    rv_km_s: float,
    *,
    rate_map: np.ndarray,
    template: str,
    mask_path: str,
    output_dir: str | Path,
    wavesol,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    stage1_cals: dict | None = None,
) -> tuple[float, float, float]:
    """Re-extract a saved rate map with ``wavesol`` and recover RV."""
    parent = Path(output_dir)
    if stage1_cals is None:
        stage1_cals = prepare_stage1_cals(
            parent,
            np.asarray(rate_map),
            spectrograph=spectrograph,
            fiber=fiber,
            wavesol=wavesol,
        )
    else:
        stage1_cals = dict(stage1_cals)
        stage1_cals["wavesol"] = wavesol
    run_dir = parent / f"reextract_{float(rv_km_s):+.1f}kms".replace("+", "p").replace("-", "m")
    l0 = rate_map_to_l0(rate_map, spectrograph=spectrograph)
    l2 = extract_and_blaze_l2(
        l0,
        output_dir=run_dir,
        fiber=fiber,
        stage1_cals=stage1_cals,
        save_result=False,
    )
    return recover_rv_keep_all_lines(
        l2,
        rv_km_s,
        template=template,
        mask_path=str(mask_path),
        spectrograph=spectrograph,
        fiber=fiber,
    )
