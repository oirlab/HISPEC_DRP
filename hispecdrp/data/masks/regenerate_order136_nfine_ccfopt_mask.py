#!/usr/bin/env python3
"""
Regenerate the order-136 fine-paint ccfopt mask used by ``rv_recovery.ipynb``.

Builds an iso15 seed on BSPEC fiber 1 / order 136, paints the BT-Settl lte040
template onto a WAVE grid oversampled by ``N_FINE``, iteratively rezeros each
line so the δ-mask Gaussian CCF centroid is ~0 at injected RV = 0, drops lines
that lock onto a wrong peak, and writes::

    hispecdrp/data/masks/<stem>.order136.iso{N}.nfine{N_FINE}.ccfopt.delta.mas

Example::

    python hispecdrp/data/masks/regenerate_order136_nfine_ccfopt_mask.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
FIBER = 1
SPECTROGRAPH = "BSPEC"
ORDER = 136
N_MASK_LINES = 15
MIN_SEP_KM_S = 50.0
LANCZOS_A = 3
N_FINE_DEFAULT = 512
TEMPLATE_SAMPLES_PER_PIXEL = 8.0
V_HALF = 4_000.0
V_STEP = 2.0
MAX_SEED_ABS_M_S = 50.0
REZERO_TOL_M_S = 1.0e-4
REZERO_MAX_ITER = 6
REZERO_KEEP_ABS_M_S = 1.0e-3


def _oversample_wave(wave_um: np.ndarray, n: int) -> np.ndarray:
    w = np.sort(np.asarray(wave_um, float).ravel())
    if int(n) <= 1:
        return w.copy()
    segs = [
        np.linspace(w[i], w[i + 1], int(n) + 1, endpoint=True)[:-1]
        for i in range(w.size - 1)
    ]
    segs.append(w[-1:])
    return np.concatenate(segs)


def _delta_mask_ccf(wave_um, flux, lam_rest_um, v_grid_m_s, *, c_m_s, order):
    from hispecdrp.primitives.utils.ccf.ccf_utils import _poly_interp_sorted_1d

    w = np.asarray(wave_um, float).ravel()
    f = np.asarray(flux, float).ravel()
    o = np.argsort(w)
    w_s, f_s = w[o], f[o]
    n = int(w_s.size)
    ccf = np.empty(len(v_grid_m_s), dtype=float)
    for i, v in enumerate(v_grid_m_s):
        lam = float(lam_rest_um) * (1.0 + float(v) / float(c_m_s))
        ccf[i] = -_poly_interp_sorted_1d(lam, w_s, f_s, n, int(order))
    return ccf


def _fit_gaussian_ccf(v_grid, ccf, v_seed=0.0):
    from hispecdrp.primitives.utils.ccf.ccf_utils import _gaussian_ccf_model

    v = np.asarray(v_grid, float)
    y_raw = np.asarray(ccf, float)
    y_scale = float(np.nanmax(np.abs(y_raw)))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0
    y = y_raw / y_scale
    amp0 = float(np.nanmax(y) - np.nanmin(y))
    base0 = float(np.nanmin(y))
    p0 = [amp0, float(v_seed), 2000.0, base0]
    bounds = (
        [0.0, float(v.min()), 50.0, -np.inf],
        [5.0 * amp0 + 1e-30, float(v.max()), 50_000.0, np.inf],
    )
    popt, _ = curve_fit(_gaussian_ccf_model, v, y, p0=p0, bounds=bounds, maxfev=20000)
    return float(popt[1])


def _recover_line_rvs(wave_um, flux, line_lams_um, *, v_center_m_s, c_m_s):
    v_grid = np.arange(
        v_center_m_s - V_HALF, v_center_m_s + V_HALF + 0.5 * V_STEP, V_STEP
    )
    out = np.empty(len(line_lams_um), dtype=float)
    for i, lam in enumerate(line_lams_um):
        ccf = _delta_mask_ccf(
            wave_um, flux, float(lam), v_grid, c_m_s=c_m_s, order=LANCZOS_A
        )
        out[i] = _fit_gaussian_ccf(v_grid, ccf, v_seed=float(v_center_m_s))
    return out


def _build_or_load_seed(seed_path: Path):
    from hispecdrp.simulations.simulate_rate_map import SimulateRateMap, read_pyechelle_hdf
    from hispecdrp.utils.line_mask_utils import MaskLines, paired_line_mask_path_for_template

    if seed_path.is_file():
        return MaskLines.load(str(seed_path))

    packaged = paired_line_mask_path_for_template(TEMPLATE, mask_style="ccfopt.delta")
    ml = MaskLines.load(packaged)
    wl_um = np.asarray(ml.line_center, float)
    weight = np.asarray(ml.weight, float)
    srm = SimulateRateMap()
    hdf = read_pyechelle_hdf(srm.hdf_file_directory + srm.bspec_hdf_model)
    ow = np.asarray(hdf[f"fiber_{FIBER}"][str(ORDER)]["xy_wavelengths"], float)
    lo_um, hi_um = float(ow.min()), float(ow.max())
    idx = np.where((wl_um >= lo_um) & (wl_um <= hi_um))[0]
    wsel = wl_um[idx]
    order = np.argsort(wsel)
    wsort = wsel[order]
    dvel = np.diff(wsort) / wsort[:-1] * 3.0e5
    nn = np.minimum(np.r_[np.inf, dvel], np.r_[dvel, np.inf])
    nn_un = np.empty_like(nn)
    nn_un[order] = nn
    iso_idx = idx[nn_un > MIN_SEP_KM_S]
    n_keep = min(N_MASK_LINES, int(iso_idx.size))
    chosen = iso_idx[np.argsort(weight[iso_idx])[::-1][:n_keep]]
    chosen = chosen[np.argsort(wl_um[chosen])]
    wt = weight[chosen].copy()
    wt /= wt.sum()
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    MaskLines.from_delta_centers_um(
        wl_um[chosen],
        wt,
        metadata={"n_lines": n_keep, "order": ORDER, "source": "iso15_seed"},
    ).save(str(seed_path))
    return MaskLines.load(str(seed_path))


def main() -> None:
    from hispecdrp.primitives.utils.ccf.ccf_utils import (
        C_M_S,
        doppler_paint_lanczos,
        prepare_template_for_doppler_paint,
    )
    from hispecdrp.primitives.utils.spectral_utils import get_spectrum_path
    from hispecdrp.simulations.e2e_support import templates_data_dir
    from hispecdrp.simulations.spectra import _instrument_wavelength_solution
    from hispecdrp.utils.line_mask_utils import MaskLines
    from hispecdrp.utils.spectral_templates_utils import load_template_from_path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-fine", type=int, default=N_FINE_DEFAULT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE.parent,
        help="Directory for the ccfopt product (default: packaged masks/)",
    )
    p.add_argument(
        "--seed-path",
        type=Path,
        default=None,
        help="Optional iso15 seed .mas (built automatically if missing)",
    )
    args = p.parse_args()
    n_fine = int(args.n_fine)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_path = args.seed_path or (out_dir / f"lte040.order{ORDER}.iso{N_MASK_LINES}.seed.delta.mas")
    seed = _build_or_load_seed(seed_path)
    seed_lam = np.asarray(seed.line_center, float)
    seed_wt = np.asarray(seed.weight, float)
    seed_wt = seed_wt / seed_wt.sum()
    print(f"seed: {seed_path}  n={seed_lam.size}")

    wave_native = np.asarray(
        _instrument_wavelength_solution(SPECTROGRAPH)[FIBER][ORDER]["wavelength"], float
    ).ravel()
    wave = _oversample_wave(wave_native, n_fine)
    print(f"fine WAVE n={wave.size} (N_FINE={n_fine})")

    tpl = get_spectrum_path(TEMPLATE, base_dir=templates_data_dir())
    w_nat, f_nat = load_template_from_path(tpl)
    o = np.argsort(w_nat)
    w_nat = np.asarray(w_nat, float).ravel()[o]
    f_nat = np.asarray(f_nat, float).ravel()[o]
    w_dense, f_dense = prepare_template_for_doppler_paint(
        w_nat, f_nat, wave, samples_per_pixel=TEMPLATE_SAMPLES_PER_PIXEL
    )
    flux0 = doppler_paint_lanczos(wave, w_dense, f_dense, 0.0, order=LANCZOS_A)

    v0_seed = _recover_line_rvs(
        wave, flux0, seed_lam, v_center_m_s=0.0, c_m_s=C_M_S
    )
    keep = np.abs(v0_seed) < MAX_SEED_ABS_M_S
    print(
        f"seed gate |v|<{MAX_SEED_ABS_M_S} m/s: keep {int(keep.sum())}/{keep.size}; "
        f"drop {np.where(~keep)[0].tolist()}"
    )
    lam = seed_lam[keep].copy()
    wt = seed_wt[keep].copy()
    wt = wt / wt.sum()

    for it in range(REZERO_MAX_ITER):
        rv = _recover_line_rvs(wave, flux0, lam, v_center_m_s=0.0, c_m_s=C_M_S)
        print(f"rezero iter {it}: max|rv|={np.max(np.abs(rv)):.4e} m/s")
        if float(np.max(np.abs(rv))) < REZERO_TOL_M_S:
            break
        lam = lam * (1.0 + rv / C_M_S)

    rv_chk = _recover_line_rvs(wave, flux0, lam, v_center_m_s=0.0, c_m_s=C_M_S)
    ok = np.abs(rv_chk) < REZERO_KEEP_ABS_M_S
    print(
        f"converge |rv|<{REZERO_KEEP_ABS_M_S} m/s: {int(ok.sum())}/{ok.size}; "
        f"drop {np.where(~ok)[0].tolist()}"
    )
    lam = lam[ok]
    wt = wt[ok]
    wt = wt / wt.sum()

    stem = Path(TEMPLATE).stem
    out_path = out_dir / f"{stem}.order{ORDER}.iso{lam.size}.nfine{n_fine}.ccfopt.delta.mas"
    MaskLines.from_delta_centers_um(
        lam,
        wt,
        metadata={
            "n_lines": int(lam.size),
            "order": ORDER,
            "fiber": FIBER,
            "spectrograph": SPECTROGRAPH,
            "n_fine": n_fine,
            "v_half_m_s": V_HALF,
            "v_step_m_s": V_STEP,
            "template": TEMPLATE,
            "source": "regenerate_order136_nfine_ccfopt_mask",
            "mask_style": "ccfopt.delta",
        },
    ).save(str(out_path))
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
