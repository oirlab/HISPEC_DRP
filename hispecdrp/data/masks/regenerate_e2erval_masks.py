#!/usr/bin/env python3
"""
Build e2e-validated delta masks (``*.e2erval.delta.mas``).

Starts from a packaged seed line list (prefer ``*.ccfopt.delta.mas``, else
``*.delta.mas`` / existing ``*.e2erval.delta.mas``) and keeps only lines whose
apparent absorption minimum on a blaze-corrected RV=0 spectrum (with the
packaged line-ZP WAVE) lies within ``--max-abs-dv-m-s`` of the seed center.

By default each kept line is **recentered** onto that apparent minimum so every
mask wavelength is consistent with RV = 0 on the validation spectrum. Weights
are renormalized to sum to 1.

This product is the packaged Stage-2 default
(``DEFAULT_PACKAGED_LINE_MASK_BASENAME``).

Run from the repository root after a seed=0 ``pipeline_example``-style blaze
product exists (same path used to bake packaged line-ZP WAVE)::

    python hispecdrp/data/masks/regenerate_e2erval_masks.py \\
        --spectrum test_data/pipeline_example_wave_rebake_seed0/reference_spec-blaze_correct.fits \\
        --fiber 2 --recenter --keep-all-valid

Then re-bake WAVE with ``bake_pipeline_example_wavesol.py`` so mask and WAVE
stay a closed loop on that spectrum. ``--keep-all-valid`` maximizes line
density (formal RV precision) while ``--recenter`` zeros fib2 RV bias on the
validation spectrum.
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
logger = logging.getLogger("regenerate_e2erval_masks")


def _resolve_seed_mask(masks_dir: Path, seed_mask: str | None) -> Path:
    if seed_mask:
        return Path(seed_mask).expanduser()
    for pattern in (
        "lte049*.ccfopt.delta.mas",
        "lte049*.delta.mas",
        "lte049*.e2erval.delta.mas",
    ):
        seeds = sorted(
            p
            for p in masks_dir.glob(pattern)
            if "e2erv" not in p.name and p.suffix == ".mas"
        )
        # Prefer non-e2erval for the delta.mas pattern.
        if pattern.endswith("*.delta.mas") and "ccfopt" not in pattern and "e2erval" not in pattern:
            seeds = [p for p in seeds if ".ccfopt." not in p.name and ".e2erval." not in p.name and ".e2erv." not in p.name]
        if seeds:
            return seeds[0]
    raise SystemExit(f"No usable seed mask under {masks_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spectrum",
        required=True,
        help="Blaze-corrected L2 FITS (glob ok) used to validate / recenter mask lines",
    )
    p.add_argument("--fiber", type=int, default=2)
    p.add_argument("--spectrograph", default="BSPEC", choices=("BSPEC", "RSPEC"))
    p.add_argument(
        "--max-abs-dv-m-s",
        type=float,
        default=80.0,
        help=(
            "Keep seed lines whose apparent minimum is within this |dv| of the seed "
            "center before recentering (default 80 m/s)"
        ),
    )
    p.add_argument(
        "--min-lines-per-order",
        type=int,
        default=None,
        help=(
            "If set, each order keeps at least this many lines by preferring the "
            "smallest |seed→min dv| candidates (still recentered when --recenter). "
            "Lines beyond --max-abs-dv-m-s are used only to fill the per-order floor."
        ),
    )
    p.add_argument(
        "--keep-all-valid",
        action="store_true",
        help=(
            "Maximize line density: keep every seed line with a valid apparent "
            "minimum on the spectrum (sorted by |seed→min dv|), ignoring "
            "--max-abs-dv-m-s / --min-lines-per-order. With --recenter (default) "
            "this still zeros fib2 RV bias on the validation spectrum; re-bake "
            "WAVE afterward so mask and WAVE stay a closed loop."
        ),
    )
    p.add_argument(
        "--seed-mask",
        default=None,
        help="Seed .mas (default: packaged lte049 ccfopt / delta / e2erval)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .mas path (default: packaged lte049*.e2erval.delta.mas)",
    )
    p.add_argument(
        "--recenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write apparent-minimum wavelengths (RV=0 on the spectrum); default on",
    )
    args = p.parse_args(argv)

    from hispecdrp import datamodels
    from hispecdrp.simulations.e2e_pipeline import (
        _order_list,
        _parabolic_min_nearest,
    )
    from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
        load_packaged_sim_hdf_line_zp_wavesol,
    )
    from hispecdrp.primitives.utils.spectral_utils import delta_wave_to_delta_vel
    from hispecdrp.utils.line_mask_utils import MaskLines, package_data_masks_dir

    matches = sorted(glob.glob(args.spectrum))
    if not matches:
        raise SystemExit(f"No spectrum matches: {args.spectrum}")
    spec_path = Path(matches[-1]).resolve()

    masks_dir = package_data_masks_dir()
    seed = _resolve_seed_mask(masks_dir, args.seed_mask)
    if not seed.is_file():
        raise SystemExit(f"Seed mask not found: {seed}")

    if args.output:
        out = Path(args.output).expanduser()
    else:
        # Stable packaged default name for lte049.
        out = masks_dir / "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.e2erval.delta.mas"
        # If seed is another template, mirror its stem.
        if "lte049" not in seed.name:
            out = masks_dir / seed.name.replace(".ccfopt.delta.mas", ".e2erval.delta.mas")
            if out.name == seed.name:
                out = masks_dir / f"{seed.stem}.e2erval.delta.mas"

    ml0 = MaskLines.load(str(seed))
    c_um = np.asarray(ml0.line_lambda_um, dtype=np.float64)
    w_in = np.asarray(ml0.weight, dtype=np.float64)

    ws = load_packaged_sim_hdf_line_zp_wavesol(args.spectrograph)
    wave_arr = np.asarray(getattr(ws, f"wave_fib{int(args.fiber)}"), dtype=np.float64)
    m = datamodels.open(str(spec_path))
    flux = np.asarray(getattr(m, f"spec_fib{int(args.fiber)}"), dtype=np.float64)
    if flux is None or flux.size == 0 or not np.isfinite(flux).any():
        # Blaze product may only expose SPEC_UNCAL before blaze in some checkouts.
        flux = np.asarray(getattr(m, f"spec_uncal_fib{int(args.fiber)}"), dtype=np.float64)
    order_vec = np.asarray(_order_list(args.spectrograph), dtype=int)

    max_abs = float(args.max_abs_dv_m_s)
    keep_all_valid = bool(args.keep_all_valid)
    min_per_order = (
        int(args.min_lines_per_order)
        if args.min_lines_per_order is not None
        else None
    )
    if keep_all_valid and min_per_order is not None:
        logger.warning(
            "--keep-all-valid ignores --min-lines-per-order=%d", min_per_order
        )
        min_per_order = None
    if min_per_order is not None and min_per_order < 1:
        raise SystemExit("--min-lines-per-order must be >= 1 when set")
    recenter = bool(args.recenter)
    val_c: list[float] = []
    val_w: list[float] = []
    dvs_seed: list[float] = []
    n_fill_beyond_cut = 0
    per_order_counts: list[int] = []
    for oi, _od in enumerate(order_vec):
        if oi >= flux.shape[0] or oi >= wave_arr.shape[0]:
            break
        wave = wave_arr[oi]
        spec = flux[oi]
        ok = np.isfinite(wave) & np.isfinite(spec)
        if int(ok.sum()) < 50:
            continue
        w_lo, w_hi = float(np.nanmin(wave[ok])), float(np.nanmax(wave[ok]))
        in_ord = (c_um >= w_lo) & (c_um <= w_hi)
        if not np.any(in_ord):
            per_order_counts.append(0)
            continue
        dlam = float(np.nanmedian(np.abs(np.diff(wave[np.isfinite(wave)]))))
        half = 8.0 * max(dlam, 1e-9)
        # (|dv|, written_center_um, weight, signed_dv)
        cand: list[tuple[float, float, float, float]] = []
        for c, wt in zip(c_um[in_ord], w_in[in_ord]):
            lam_m = _parabolic_min_nearest(wave, spec, float(c), half)
            if not np.isfinite(lam_m):
                continue
            dv = float(delta_wave_to_delta_vel(lam_m - float(c), float(c)))
            if not np.isfinite(dv):
                continue
            written = float(lam_m) if recenter else float(c)
            cand.append((abs(dv), written, float(wt), dv))
        if not cand:
            per_order_counts.append(0)
            continue
        # Prefer smallest |seed→min dv|; keep all within max_abs, then fill to floor
        # (or keep every valid minimum when --keep-all-valid).
        cand.sort(key=lambda t: t[0])
        if keep_all_valid:
            keep = list(cand)
            n_fill_beyond_cut += sum(1 for t in keep if t[0] > max_abs)
        else:
            keep = [t for t in cand if t[0] <= max_abs]
            if min_per_order is not None and len(keep) < min_per_order:
                for t in cand:
                    if len(keep) >= min_per_order:
                        break
                    if t[0] <= max_abs:
                        continue  # already in keep
                    keep.append(t)
                    n_fill_beyond_cut += 1
        for _abs_dv, written, wt, dv in keep:
            val_c.append(written)
            val_w.append(wt)
            dvs_seed.append(dv)
        per_order_counts.append(len(keep))

    if not val_c:
        raise SystemExit("No lines passed validation cut")
    val_c_a = np.asarray(val_c, dtype=np.float64)
    val_w_a = np.asarray(val_w, dtype=np.float64)
    val_w_a = val_w_a / float(np.sum(val_w_a))
    dvs_a = np.asarray(dvs_seed, dtype=np.float64)
    per_a = np.asarray(per_order_counts, dtype=int)

    # After recentering, residual vs the written centers is ~0 by construction;
    # report seed→minimum dv for provenance.
    meta = {
        "schema_version": 2,
        "mask_style": "delta",
        "generator": "regenerate_e2erval_masks",
        "seed_mask": seed.name,
        "spectrum_path": str(spec_path),
        "fiber": int(args.fiber),
        "spectrograph": str(args.spectrograph).upper(),
        "max_abs_dv_m_s": max_abs,
        "min_lines_per_order": min_per_order,
        "keep_all_valid": bool(keep_all_valid),
        "n_fill_beyond_max_abs_dv": int(n_fill_beyond_cut),
        "lines_per_order_min": int(per_a.min()) if per_a.size else 0,
        "lines_per_order_median": float(np.median(per_a)) if per_a.size else 0.0,
        "recenter_to_apparent_minimum": bool(recenter),
        "known_rv_m_s": 0.0,
        "n_lines": int(val_c_a.size),
        "n_seed_lines": int(c_um.size),
        "seed_to_min_dv_median_m_s": float(np.median(dvs_a)),
        "seed_to_min_dv_mad_m_s": float(np.median(np.abs(dvs_a - np.median(dvs_a)))),
        # Backward-compatible keys used by older readers.
        "dv_median_m_s": 0.0 if recenter else float(np.median(dvs_a)),
        "dv_mad_m_s": 0.0 if recenter else float(np.median(np.abs(dvs_a - np.median(dvs_a)))),
        "purpose": "e2e_rv0_validated_on_packaged_wave",
    }
    MaskLines.from_delta_centers_um(val_c_a, val_w_a, metadata=meta).save(
        output_path=str(out)
    )
    logger.info(
        "wrote %s  n=%d/%d  recenter=%s  min/med lines/order=%d/%.1f  "
        "fill_beyond_cut=%d  seed→min dv_med=%+.2f mad=%.2f m/s",
        out,
        val_c_a.size,
        c_um.size,
        recenter,
        meta["lines_per_order_min"],
        meta["lines_per_order_median"],
        n_fill_beyond_cut,
        meta["seed_to_min_dv_median_m_s"],
        meta["seed_to_min_dv_mad_m_s"],
    )
    print(
        f"wrote {out}  n={val_c_a.size}/{c_um.size}  recenter={recenter}  "
        f"lines/order min/med={meta['lines_per_order_min']}/{meta['lines_per_order_median']:.1f}  "
        f"fill_beyond_cut={n_fill_beyond_cut}  "
        f"seed_to_min_dv_med={meta['seed_to_min_dv_median_m_s']:+.2f} "
        f"mad={meta['seed_to_min_dv_mad_m_s']:.2f} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
