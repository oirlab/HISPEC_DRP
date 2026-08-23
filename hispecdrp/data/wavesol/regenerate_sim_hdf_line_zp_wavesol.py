#!/usr/bin/env python3
"""
Build packaged sim-HDF line-ZP WAVESOL from an e2e science rate map / spectrum.

Pipeline:
  1. Deposit-inverse HDF WAVE
  2. Absorption-line poly correction (rate map box extract, or L2 spectrum)
  3. Optional CCF-order scale refine on a blaze-corrected spectrum + mask
     so order-averaged stellar RV at known injected RV is unbiased

Writes FITS (+ JSON) under ``hispecdrp/data/wavesol/``.

Run from the repository root::

    python hispecdrp/data/wavesol/regenerate_sim_hdf_line_zp_wavesol.py \\
        --rate-map test_data/e2e_rv0_hdf_trace_linezp/rate_map.fits \\
        --spectrum test_data/e2e_rv0_hdf_trace_linezp/*_spec-blaze_correct.fits \\
        --mask hispecdrp/data/masks/lte049*.ccfopt.delta.mas \\
        --ccf-refine --spectrograph BSPEC --known-rv-m-s 0
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _load_rate_map(path: Path) -> np.ndarray:
    from hispecdrp import datamodels
    from astropy.io import fits

    try:
        m = datamodels.open(str(path))
        rate = getattr(m, "rate", None)
        if rate is not None:
            return np.asarray(rate, dtype=np.float64)
        data = getattr(m, "data", None)
        if data is not None:
            return np.asarray(data, dtype=np.float64)
    except Exception:
        pass
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def _refine_from_l2_spectrum(
    wavesol,
    spectrum_path: Path,
    *,
    spectrograph: str,
    fiber: int,
    known_rv_m_s: float,
    poly_deg: int,
    n_lines_per_order: int,
) -> dict:
    """Per-order poly WAVE correction measured on an L2 ``spec_fib`` array."""
    import scipy.constants

    from hispecdrp import datamodels
    from hispecdrp.simulations.e2e_pipeline import (
        _order_list,
        _parabolic_min_nearest,
        _polyfit_scaled,
        _polyval_scaled,
    )
    from hispecdrp.simulations import e2e_support
    from hispecdrp.primitives.utils.spectral_utils import (
        delta_wave_to_delta_vel,
        get_spectrum_path,
        load_btsettl_spectrum,
    )
    from hispecdrp.utils.line_mask_utils import (
        _continuum_envelope,
        detect_template_line_centers_parabolic_um,
    )

    m = datamodels.open(str(spectrum_path))
    flux = np.asarray(getattr(m, f"spec_fib{fiber}"), dtype=np.float64)
    wave_attr = f"wave_fib{fiber}"
    wave_arr = np.asarray(getattr(wavesol, wave_attr), dtype=np.float64).copy()
    order_vec = np.asarray(_order_list(spectrograph), dtype=int)

    tw, tf = load_btsettl_spectrum(
        get_spectrum_path(
            e2e_support.DEFAULT_TEMPLATE_FILENAME,
            base_dir=e2e_support.templates_data_dir(),
        )
    )
    tw = np.asarray(tw, dtype=np.float64).ravel()
    tf = np.asarray(tf, dtype=np.float64).ravel()
    order_t = np.argsort(tw)
    tw, tf = tw[order_t], tf[order_t]
    c_mps = float(scipy.constants.c)
    z = float(known_rv_m_s) / c_mps
    per_order: dict[int, dict] = {}

    for oi, od in enumerate(order_vec):
        if oi >= flux.shape[0] or oi >= wave_arr.shape[0]:
            break
        wave = wave_arr[oi].copy()
        spec = flux[oi]
        ok = np.isfinite(wave) & np.isfinite(spec)
        if int(ok.sum()) < 50 or float(np.nanpercentile(spec[ok], 90)) <= 0:
            continue
        w_lo, w_hi = float(np.nanmin(wave[ok])), float(np.nanmax(wave[ok]))
        msk = (tw >= w_lo) & (tw <= w_hi)
        centers = detect_template_line_centers_parabolic_um(
            tw[msk], tf[msk], min_continuum_normalized_depth=0.12
        )
        if centers.size == 0:
            continue
        cont = _continuum_envelope(tf[msk], 181)
        depths = 1.0 - np.interp(centers, tw[msk], tf[msk]) / np.maximum(
            np.interp(centers, tw[msk], cont), 1e-30
        )
        pick = centers[np.argsort(-depths)[: int(n_lines_per_order)]]
        dlam = float(np.nanmedian(np.abs(np.diff(wave[np.isfinite(wave)]))))
        half = 8.0 * max(dlam, 1e-9)
        ideal = np.interp(wave, tw, tf, left=np.nan, right=np.nan)
        pix = np.arange(wave.size, dtype=np.float64)
        pix_lines: list[float] = []
        dvs: list[float] = []
        for c_native in pick:
            lam_rest = _parabolic_min_nearest(wave, ideal, float(c_native), half)
            if not np.isfinite(lam_rest):
                continue
            lam_expect = lam_rest * (1.0 + z)
            lam_m = _parabolic_min_nearest(wave, spec, lam_expect, half)
            if not np.isfinite(lam_m) or abs(lam_m - lam_expect) > 2.5 * dlam:
                continue
            p_exp = float(np.interp(lam_expect, wave[ok], pix[ok]))
            dv = float(delta_wave_to_delta_vel(lam_m - lam_expect, lam_expect))
            pix_lines.append(p_exp)
            dvs.append(dv)
        if len(dvs) < 5:
            continue
        dvs_a = np.asarray(dvs, dtype=np.float64)
        pix_a = np.asarray(pix_lines, dtype=np.float64)
        med = float(np.median(dvs_a))
        mad = float(np.median(np.abs(dvs_a - med)))
        keep = (
            np.abs(dvs_a - med) <= 4.0 * 1.4826 * mad
            if mad > 0
            else np.ones(dvs_a.shape, dtype=bool)
        )
        if int(keep.sum()) < 5:
            keep = np.ones(dvs_a.shape, dtype=bool)
        dvs_k, pix_k = dvs_a[keep], pix_a[keep]
        med = float(np.median(dvs_k))
        mad = float(np.median(np.abs(dvs_k - med)))
        deg_use = min(int(poly_deg), max(0, int(dvs_k.size) - 2))
        if deg_use <= 0:
            wave_arr[oi] = wave * (1.0 - med / c_mps)
            per_order[int(od)] = {
                "dv_m_s": med,
                "mad_m_s": mad,
                "n_lines": float(keep.sum()),
                "poly_deg": 0,
                "poly_coeffs": [med],
            }
        else:
            coeffs, x_mid, x_half = _polyfit_scaled(pix_k, dvs_k, deg=deg_use)
            dv_pix = _polyval_scaled(coeffs, pix, x_mid=x_mid, x_half=x_half)
            wave_arr[oi] = wave * (1.0 - dv_pix / c_mps)
            resid = dvs_k - _polyval_scaled(
                coeffs, pix_k, x_mid=x_mid, x_half=x_half
            )
            per_order[int(od)] = {
                "dv_m_s": float(np.median(resid)),
                "mad_m_s": float(np.median(np.abs(resid - np.median(resid)))),
                "n_lines": float(keep.sum()),
                "poly_deg": int(deg_use),
                "poly_coeffs": [float(c) for c in coeffs],
                "scale_dv_m_s": med,
            }

    setattr(wavesol, wave_attr, wave_arr)
    scales = [
        float(v.get("scale_dv_m_s", v["dv_m_s"]))
        for v in per_order.values()
        if np.isfinite(float(v.get("scale_dv_m_s", v["dv_m_s"])))
    ]
    return {
        "fiber": int(fiber),
        "source": "l2_spectrum",
        "spectrum_path": str(spectrum_path),
        "per_order": per_order,
        "median_dv_m_s": float(np.median(scales)) if scales else float("nan"),
        "n_orders": len(per_order),
        "poly_deg": int(poly_deg),
    }


def _ccf_refine(
    wavesol,
    spectrum_path: Path,
    mask_path: Path,
    *,
    spectrograph: str,
    fiber: int,
    known_rv_m_s: float,
    v_step_m_s: float,
    n_iter: int,
    output_dir: Path,
) -> dict:
    import shutil

    from hispecdrp import datamodels
    from hispecdrp.simulations import e2e_support
    from hispecdrp.simulations.e2e_pipeline import (
        _order_list,
        refine_sim_hdf_wavesol_from_ccf_order_rvs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    order_vec = np.asarray(_order_list(spectrograph), dtype=int)
    # Unity fiberflat sibling if present (Stage2 blaze already applied; unity is a no-op).
    unity = spectrum_path.parent / "cals" / "unity_fiberflat.fits"
    if not unity.is_file():
        cand = list(spectrum_path.parent.glob("**/unity_fiberflat.fits"))
        if not cand:
            # Fall back to any nearby bake dir (e.g. fiber-2 products).
            cand = list(spectrum_path.parent.parent.glob("**/unity_fiberflat.fits"))
        if not cand:
            raise SystemExit(
                f"unity_fiberflat.fits not found near {spectrum_path.parent}"
            )
        unity = cand[0]
        logger = logging.getLogger(__name__)
        logger.info("Using unity fiberflat from %s", unity)

    info_iters: list[dict] = []
    for it in range(int(n_iter)):
        dest = output_dir / f"ccf_refine_iter{it}_{spectrum_path.name}"
        shutil.copy2(spectrum_path, dest)
        mm = datamodels.open(str(dest))
        w = np.asarray(getattr(wavesol, f"wave_fib{fiber}"), dtype=np.float64)
        setattr(mm, f"wave_fib{fiber}", w)
        mm.save(str(dest))
        primitives = {
            "blaze_corr": {"run": False},
            "barycorr": {"run": False},
            "stellar_rv": {
                "mask_path": str(mask_path),
                "ccf_input_velocity": float(known_rv_m_s),
                "ccf_zb": 0.0,
                "ccf_zb_from_barycorr": False,
                "window_line_by_line": False,
                "window_size_um": 0.25,
                "v_half_window": 15000.0,
                "v_step_coarse": float(v_step_m_s),
                "ccf_min_good_pixels": 5,
                "do_fibers": [int(fiber)],
                "outlier_sigma": 3.0,
                "suffix": "stellar_rv",
                "save_preview": False,
                "preview_write_png": False,
                "save_overlay_debug": False,
                "mask_auto_center_scout": False,
            },
        }
        e2e_support.run_stage2_simulation(
            datamodels.open(str(dest)),
            fiberflat_model=datamodels.open(str(unity)),
            primitives=primitives,
            stellar_rv_mask_path=str(mask_path),
            stellar_rv_ccf_input_velocity_m_s=float(known_rv_m_s),
            output_dir=str(output_dir),
            max_cores=1,
        )
        rv_path = sorted(output_dir.glob("*stellar_rv.fits"))[-1]
        rm = datamodels.open(str(rv_path))
        tbl = getattr(rm, f"rvccf_fib{fiber}")
        orders = np.asarray(tbl["ORDER"], dtype=int)
        rvs = np.asarray(tbl["RV"], dtype=np.float64)
        order_rvs = {int(od): float(rv) for od, rv in zip(orders, rvs)}
        info = refine_sim_hdf_wavesol_from_ccf_order_rvs(
            wavesol,
            fiber=int(fiber),
            order_rvs_m_s=order_rvs,
            order_vec=order_vec,
            known_rv_m_s=float(known_rv_m_s),
            spectrograph=spectrograph,
        )
        info["iter"] = it
        info["stellar_rv_path"] = str(rv_path)
        info_iters.append(info)
        logging.info(
            "CCF WAVE refine iter=%d median_dv=%.3f mad=%.3f m/s n=%d",
            it,
            info["median_dv_m_s"],
            info["mad_dv_m_s"],
            info["n_orders"],
        )
    return {"n_iter": int(n_iter), "iterations": info_iters, "mask": str(mask_path)}


def _parse_fiber_path(spec: str) -> tuple[int, Path]:
    """Parse ``FIBER:PATH`` CLI tokens."""
    if ":" not in spec:
        raise SystemExit(f"Expected FIBER:PATH, got {spec!r}")
    fib_s, path_s = spec.split(":", 1)
    return int(fib_s), Path(path_s).expanduser()


def bake_multi_fiber_packaged_wavesol(
    *,
    spectrograph: str,
    rate_by_fiber: dict[int, Path],
    spectrum_by_fiber: dict[int, Path],
    known_rv_m_s: float = 0.0,
    poly_deg: int = 2,
    n_lines_per_order: int = 40,
    mask_glob: str,
    ccf_refine_iters: int = 2,
    ccf_v_step_m_s: float = 25.0,
    ccf_work_dir: Path,
) -> Path:
    """
    Build packaged WAVE with per-fiber rate-map ZP, L2 refine, and CCF refine.

    Each fiber should have its own science ``rate_map`` and blaze-corrected
    spectrum from a seed=0 / RV=0 pipeline_example-style run.
    """
    from hispecdrp.simulations.e2e_pipeline import (
        create_wavelength_solution_model_from_sim_hdf,
        refine_sim_hdf_wavesol_line_zeropoint,
    )
    from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
        save_packaged_sim_hdf_line_zp_wavesol,
    )

    fibers = sorted(int(f) for f in rate_by_fiber)
    if not fibers:
        raise ValueError("rate_by_fiber is empty")
    for fib in fibers:
        if fib not in spectrum_by_fiber:
            raise ValueError(f"Missing blaze spectrum for fiber {fib}")
        if not Path(rate_by_fiber[fib]).is_file():
            raise FileNotFoundError(rate_by_fiber[fib])
        if not Path(spectrum_by_fiber[fib]).is_file():
            raise FileNotFoundError(spectrum_by_fiber[fib])

    mask_matches = sorted(glob.glob(mask_glob))
    if not mask_matches:
        raise FileNotFoundError(f"No mask matches: {mask_glob}")
    mask_path = Path(mask_matches[-1]).resolve()

    wavesol = create_wavelength_solution_model_from_sim_hdf(
        spectrograph, meta={"mjd_start": 0.0}
    )
    zp_info: dict = {
        "fibers": fibers,
        "known_rv_m_s": float(known_rv_m_s),
        "poly_deg": int(poly_deg),
        "per_fiber": {},
        "rate_map_by_fiber": {str(f): str(rate_by_fiber[f]) for f in fibers},
        "spectrum_by_fiber": {str(f): str(spectrum_by_fiber[f]) for f in fibers},
        "ccf_order_refine_by_fiber": {},
    }

    for fib in fibers:
        rate = _load_rate_map(Path(rate_by_fiber[fib]))
        info = refine_sim_hdf_wavesol_line_zeropoint(
            wavesol,
            rate,
            spectrograph=spectrograph,
            fibers=[fib],
            known_rv_m_s=float(known_rv_m_s),
            poly_deg=int(poly_deg),
            n_lines_per_order=int(n_lines_per_order),
        )
        zp_info["per_fiber"][str(fib)] = {"rate_map_line_zp": info}
        logging.info(
            "Rate-map line ZP fiber=%d median_dv=%.2f m/s",
            fib,
            info.get("median_dv_m_s"),
        )

    for fib in fibers:
        spec_path = Path(spectrum_by_fiber[fib]).resolve()
        l2_info = _refine_from_l2_spectrum(
            wavesol,
            spec_path,
            spectrograph=spectrograph,
            fiber=int(fib),
            known_rv_m_s=float(known_rv_m_s),
            poly_deg=int(poly_deg),
            n_lines_per_order=max(int(n_lines_per_order), 60),
        )
        zp_info["per_fiber"].setdefault(str(fib), {})["l2_spectrum_refine"] = l2_info
        logging.info(
            "L2 spectrum line ZP fiber=%d median_dv=%.2f m/s n_orders=%d",
            fib,
            l2_info["median_dv_m_s"],
            l2_info["n_orders"],
        )
        ccf_info = _ccf_refine(
            wavesol,
            spec_path,
            mask_path,
            spectrograph=spectrograph,
            fiber=int(fib),
            known_rv_m_s=float(known_rv_m_s),
            v_step_m_s=float(ccf_v_step_m_s),
            n_iter=int(ccf_refine_iters),
            output_dir=Path(ccf_work_dir).expanduser().resolve() / f"fib{fib}",
        )
        zp_info["ccf_order_refine_by_fiber"][str(fib)] = ccf_info
        last = (ccf_info.get("iterations") or [{}])[-1]
        logging.info(
            "CCF WAVE refine fiber=%d final median_dv=%.4f mad=%.4f m/s",
            fib,
            last.get("median_dv_m_s"),
            last.get("mad_dv_m_s"),
        )

    # Summary median from last CCF iter across fibers (fiber 2 kept as primary).
    medians = []
    for fib in fibers:
        iters = (zp_info["ccf_order_refine_by_fiber"].get(str(fib)) or {}).get(
            "iterations"
        ) or []
        if iters:
            medians.append(float(iters[-1].get("median_dv_m_s")))
    zp_info["median_dv_m_s"] = float(np.median(medians)) if medians else float("nan")
    # Backward-compat top-level keys mirror fiber 2 when present.
    if 2 in fibers:
        zp_info["fiber"] = 2
        zp_info["l2_spectrum_refine"] = zp_info["per_fiber"]["2"].get(
            "l2_spectrum_refine"
        )
        zp_info["ccf_order_refine"] = zp_info["ccf_order_refine_by_fiber"].get("2")

    primary_rate = rate_by_fiber.get(2) or rate_by_fiber[fibers[0]]
    out = save_packaged_sim_hdf_line_zp_wavesol(
        wavesol,
        spectrograph,
        zp_info=zp_info,
        rate_map_path=primary_rate,
        overwrite=True,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rate-map",
        type=Path,
        default=None,
        help="Science rate_map.fits (or Stage-0 rate) used to measure line ZP",
    )
    p.add_argument(
        "--rate-map-by-fiber",
        action="append",
        default=[],
        help="Per-fiber rate map as FIBER:PATH (repeatable; overrides --rate-map)",
    )
    p.add_argument("--spectrograph", default="BSPEC", choices=("BSPEC", "RSPEC"))
    p.add_argument(
        "--known-rv-m-s",
        type=float,
        default=0.0,
        help="Injected stellar RV in the rate map (not absorbed into WAVE)",
    )
    p.add_argument(
        "--fibers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Fibers for rate-map line ZP when using a single --rate-map",
    )
    p.add_argument(
        "--poly-deg",
        type=int,
        default=2,
        help="Per-order WAVE correction degree (0=scale ZP, 1=linear, 2=quadratic)",
    )
    p.add_argument(
        "--n-lines-per-order",
        type=int,
        default=40,
        help="Deepest template lines used per order for the WAVE fit",
    )
    p.add_argument(
        "--spectrum",
        default=None,
        help="Optional blaze-corrected L2 FITS (glob ok) for L2 line ZP + CCF refine",
    )
    p.add_argument(
        "--spectrum-by-fiber",
        action="append",
        default=[],
        help="Per-fiber blaze spectrum as FIBER:PATH (repeatable)",
    )
    p.add_argument(
        "--l2-fiber",
        type=int,
        default=2,
        help="Fiber for L2 / CCF WAVE refine with a single --spectrum (default: 2)",
    )
    p.add_argument(
        "--l2-fibers",
        type=int,
        nargs="+",
        default=None,
        help="Fibers for L2 / CCF refine when one --spectrum contains all of them",
    )
    p.add_argument(
        "--ccf-refine",
        action="store_true",
        help="After line ZP, refine WAVE with per-order CCF residuals",
    )
    p.add_argument(
        "--mask",
        default=None,
        help="Line mask for CCF refine (required with --ccf-refine; glob ok)",
    )
    p.add_argument("--ccf-v-step-m-s", type=float, default=25.0)
    p.add_argument("--ccf-refine-iters", type=int, default=2)
    p.add_argument(
        "--ccf-work-dir",
        type=Path,
        default=Path("test_data/wavesol_ccf_refine"),
        help="Scratch dir for CCF refine stellar_rv products",
    )
    args = p.parse_args(argv)

    from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
        build_packaged_sim_hdf_line_zp_wavesol,
        save_packaged_sim_hdf_line_zp_wavesol,
    )

    rate_by_fiber: dict[int, Path] = {}
    for token in args.rate_map_by_fiber:
        fib, path = _parse_fiber_path(token)
        rate_by_fiber[fib] = path.resolve()
    spectrum_by_fiber: dict[int, Path] = {}
    for token in args.spectrum_by_fiber:
        fib, path = _parse_fiber_path(token)
        spectrum_by_fiber[fib] = path.resolve()

    if rate_by_fiber and spectrum_by_fiber:
        if not args.mask:
            raise SystemExit("--mask is required for multi-fiber bake")
        out = bake_multi_fiber_packaged_wavesol(
            spectrograph=args.spectrograph,
            rate_by_fiber=rate_by_fiber,
            spectrum_by_fiber=spectrum_by_fiber,
            known_rv_m_s=float(args.known_rv_m_s),
            poly_deg=int(args.poly_deg),
            n_lines_per_order=int(args.n_lines_per_order),
            mask_glob=str(args.mask),
            ccf_refine_iters=int(args.ccf_refine_iters),
            ccf_v_step_m_s=float(args.ccf_v_step_m_s),
            ccf_work_dir=args.ccf_work_dir,
        )
        print(f"wrote {out}")
        return 0

    if args.rate_map is None:
        raise SystemExit("Provide --rate-map or --rate-map-by-fiber with --spectrum-by-fiber")

    rate_path = args.rate_map.expanduser().resolve()
    if not rate_path.is_file():
        raise SystemExit(f"rate map not found: {rate_path}")

    rate = _load_rate_map(rate_path)
    wavesol, zp_info = build_packaged_sim_hdf_line_zp_wavesol(
        args.spectrograph,
        rate,
        known_rv_m_s=float(args.known_rv_m_s),
        fibers=list(args.fibers),
        meta={"mjd_start": 0.0},
        poly_deg=int(args.poly_deg),
        n_lines_per_order=int(args.n_lines_per_order),
    )

    refine_fibers = (
        [int(f) for f in args.l2_fibers]
        if args.l2_fibers is not None
        else [int(args.l2_fiber)]
    )
    spec_path = None
    if args.spectrum:
        matches = sorted(glob.glob(args.spectrum))
        if not matches:
            raise SystemExit(f"No spectrum matches: {args.spectrum}")
        spec_path = Path(matches[-1]).resolve()
        zp_info.setdefault("per_fiber", {})
        for fib in refine_fibers:
            l2_info = _refine_from_l2_spectrum(
                wavesol,
                spec_path,
                spectrograph=args.spectrograph,
                fiber=int(fib),
                known_rv_m_s=float(args.known_rv_m_s),
                poly_deg=int(args.poly_deg),
                n_lines_per_order=max(int(args.n_lines_per_order), 60),
            )
            zp_info["per_fiber"][str(int(fib))] = l2_info
            logging.info(
                "L2 spectrum line ZP fiber=%d median_dv=%.2f m/s n_orders=%d",
                int(fib),
                l2_info["median_dv_m_s"],
                l2_info["n_orders"],
            )
        # Keep fiber-2 (or first) as the top-level summary for older readers.
        primary = 2 if 2 in refine_fibers else refine_fibers[0]
        zp_info["l2_spectrum_refine"] = zp_info["per_fiber"][str(primary)]
        zp_info["median_dv_m_s"] = zp_info["l2_spectrum_refine"]["median_dv_m_s"]
        zp_info["fiber"] = int(primary)

    if args.ccf_refine:
        if not args.mask:
            raise SystemExit("--mask is required with --ccf-refine")
        if spec_path is None:
            raise SystemExit("--spectrum is required with --ccf-refine")
        mask_matches = sorted(glob.glob(args.mask))
        if not mask_matches:
            raise SystemExit(f"No mask matches: {args.mask}")
        mask_path = Path(mask_matches[-1]).resolve()
        ccf_by_fiber = {}
        for fib in refine_fibers:
            ccf_info = _ccf_refine(
                wavesol,
                spec_path,
                mask_path,
                spectrograph=args.spectrograph,
                fiber=int(fib),
                known_rv_m_s=float(args.known_rv_m_s),
                v_step_m_s=float(args.ccf_v_step_m_s),
                n_iter=int(args.ccf_refine_iters),
                output_dir=args.ccf_work_dir.expanduser().resolve() / f"fib{fib}",
            )
            ccf_by_fiber[str(int(fib))] = ccf_info
        zp_info["ccf_order_refine_by_fiber"] = ccf_by_fiber
        primary = 2 if 2 in refine_fibers else refine_fibers[0]
        zp_info["ccf_order_refine"] = ccf_by_fiber[str(primary)]

    out = save_packaged_sim_hdf_line_zp_wavesol(
        wavesol,
        args.spectrograph,
        zp_info=zp_info,
        rate_map_path=rate_path,
        overwrite=True,
    )
    print(
        f"wrote {out}  median_dv={zp_info.get('median_dv_m_s')} m/s  "
        f"ccf_refine={bool(args.ccf_refine)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

