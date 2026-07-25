"""
Diagnose why cross-correlation may produce no valid RV chunks.

Compares an L1 specuncal FITS file (e.g. HB.20240924.00000.00_specuncal.fits) to the
BT-Settl template: wavelength ranges, units, overlap, and per-order/chunk CCF readiness.
Run from repo root or with PYTHONPATH set:

  python -m hispecdrp.scripts.diagnose_rvccf --specuncal /path/to/HB.20240924.00000.00_specuncal.fits --template /path/to/template.h5

If --template is omitted, a default path under hispecdrp/simulations/data is tried.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Diagnose RVCCF: compare specuncal to template.")
    parser.add_argument("--specuncal", required=True, help="Path to L1 specuncal FITS file.")
    parser.add_argument("--template", default=None, help="Path to BT-Settl .h5 template (optional).")
    parser.add_argument("--fiber", type=int, default=1, help="Fiber to inspect (1-based).")
    parser.add_argument("--dry-run", action="store_true", help="Only check paths; do not load data.")
    args = parser.parse_args()

    specuncal_path = args.specuncal
    template_path = args.template

    if args.dry_run:
        import os
        print("Specuncal path:", specuncal_path, "exists:", os.path.isfile(specuncal_path))
        if template_path:
            print("Template path:", template_path, "exists:", os.path.isfile(template_path))
        return 0

    # Load L1 model
    try:
        from hispecdrp import datamodels
        l1 = datamodels.open(specuncal_path)
    except Exception as e:
        print("ERROR: Could not open specuncal file:", specuncal_path, file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    fib = args.fiber
    wave_attr = f"wave_fib{fib}"
    spec_attr = f"spec_uncal_fib{fib}"
    err_attr = f"spec_uncal_err_fib{fib}"
    dq_attr = f"spec_dq_fib{fib}"

    if not hasattr(l1, wave_attr) or getattr(l1, wave_attr) is None:
        print(f"ERROR: No {wave_attr} in specuncal file.", file=sys.stderr)
        return 1
    if not hasattr(l1, spec_attr) or getattr(l1, spec_attr) is None:
        print(f"ERROR: No {spec_attr} in specuncal file.", file=sys.stderr)
        return 1

    wave = np.asarray(getattr(l1, wave_attr), dtype=np.float64)
    spec = np.asarray(getattr(l1, spec_attr), dtype=np.float64)
    err = getattr(l1, err_attr, None)
    if err is not None:
        err = np.asarray(err, dtype=np.float64)
    else:
        err = np.ones_like(spec)
    dq = getattr(l1, dq_attr, None)
    if dq is not None:
        dq = np.asarray(dq, dtype=np.uint32)
    else:
        dq = np.zeros_like(spec, dtype=np.uint32)

    from hispecdrp.datamodels.dqflags import DQ_FLAGS
    do_not_use = DQ_FLAGS.get("DO_NOT_USE", 1)
    mask = (dq & do_not_use) != 0

    print("=== L1 specuncal (input to Stage 2 blaze -> CCF) ===")
    print("File:", specuncal_path)
    print("Fiber:", fib)
    print("Wave shape:", wave.shape)
    print("Spec shape:", spec.shape)

    # Wavelength units check: CCF expects micron. If range is 1e3--1e4, likely Angstrom/nm.
    w_flat = wave[np.isfinite(wave)]
    if len(w_flat) == 0:
        print("WARNING: No finite wavelengths in WAVE_FIB. CCF will have no valid orders.")
    else:
        w_min, w_max = float(np.min(w_flat)), float(np.max(w_flat))
        print("Wavelength range (all orders): min =", w_min, ", max =", w_max)
        if w_min > 100 or w_max > 500:
            print(">>> LIKELY CAUSE: Wavelength appears to be in Angstrom or nm, not micron.")
            print("    CCF uses template in micron (0.8--3.0). If data is in Angstrom, there is")
            print("    no overlap in log(lambda) and every chunk returns NaN.")
            print("    Fix: ensure set_wavesol / WAVE_FIB are in micron.")
        else:
            print("Wavelength range looks like micron (expected for CCF).")

    # Per-order summary
    n_orders = wave.shape[0]
    order_vec = l1.make_order_vec() if hasattr(l1, "make_order_vec") else np.arange(n_orders)
    print("\nPer-order summary (wave range, good pixels, median spec, median err):")
    for oi in range(n_orders):
        w = wave[oi]
        s = spec[oi]
        e = err[oi] if err.ndim >= 2 else err
        m = mask[oi] if mask.ndim >= 2 else mask
        w_ok = w[np.isfinite(w)]
        if len(w_ok) == 0:
            print(f"  order {order_vec[oi]}: no finite wave")
            continue
        wo_min, wo_max = np.min(w_ok), np.max(w_ok)
        good = ~m & np.isfinite(s) & np.isfinite(e) & (e > 0)
        n_good = int(np.sum(good))
        med_s = float(np.nanmedian(s)) if np.any(np.isfinite(s)) else np.nan
        med_e = float(np.nanmedian(e)) if np.any(np.isfinite(e)) else np.nan
        print(f"  order {order_vec[oi]}: wave [{wo_min:.4f}, {wo_max:.4f}] um, good_pix={n_good}, med_spec={med_s:.2e}, med_err={med_e:.2e}")
        if n_good < 10:
            print("    >>> Chunks need >=10 good pixels for CCF; this order may yield no valid RVs.")

    # Load template and overlap
    if template_path:
        try:
            from hispecdrp.utils.spectral_templates_utils import load_btsettl_spectrum
            w_t, f_t = load_btsettl_spectrum(template_path, wl_min_um=0.8, wl_max_um=3.0)
            tw = w_t.to_value("micron")
            tf = np.asarray(f_t.value, dtype=np.float64)
        except Exception as e:
            print("\nWARNING: Could not load template:", template_path, e)
            tw = tf = None
    else:
        try:
            from hispecdrp.utils.spectral_templates_utils import get_spectrum_path, load_btsettl_spectrum
            default_path = get_spectrum_path("lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5")
            w_t, f_t = load_btsettl_spectrum(default_path, wl_min_um=0.8, wl_max_um=3.0)
            tw = w_t.to_value("micron")
            tf = np.asarray(f_t.value, dtype=np.float64)
            template_path = default_path
        except Exception as e:
            print("\nNo template path given and default load failed:", e)
            tw = tf = None

    if tw is not None and len(tw) > 0:
        print("\n=== BT-Settl template ===")
        print("Template path:", template_path)
        print("Template wavelength range: [%.4f, %.4f] micron" % (float(np.min(tw)), float(np.max(tw))))
        ln_t = np.log(tw)
        w_flat = wave[np.isfinite(wave)]
        if len(w_flat) > 0:
            d_min, d_max = np.min(w_flat), np.max(w_flat)
            ln_min, ln_max = np.log(d_min), np.log(d_max)
            in_range = (ln_t >= ln_min - 0.01) & (ln_t <= ln_max + 0.01)
            n_overlap = int(np.sum(in_range))
            print("Template points overlapping data [%.4f, %.4f] um: %d (need >= 20 per order for CCF)." % (d_min, d_max, n_overlap))
            if n_overlap < 20 and len(w_flat) > 0:
                print(">>> LIKELY CAUSE: Template and data wavelength overlap is too small.")
                print("    Check that data WAVE is in micron and covers a range within 0.8--3.0 um.")

    print("\n=== CCF flow reminder ===")
    print("Stage 2 input is L1 specuncal -> blaze correct -> L2 with SPEC_FIB, WAVE_FIB.")
    print("Cross_corr reads SPEC_FIB (blaze-corrected flux) and WAVE_FIB (same as L1).")
    print("Each order needs: (1) WAVE in micron, (2) >= 20 template points in order range,")
    print("(3) per chunk >= 20 pixels, (4) per chunk >= 10 good pixels (unmasked, finite spec/err, err>0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
