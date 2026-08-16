#!/usr/bin/env python3
"""
Bake packaged sim-HDF line-ZP WAVE for fibers 1, 2, and 3.

Matches the fiber-2 ``pipeline_example`` recipe for each fiber:
  * ``create_ramp_from_spectrum(..., seed=0, RV=0)``
  * Stage0 → Stage1 → Stage2 (blaze-corrected spectrum)
  * rate-map line ZP + L2 poly refine + 2× CCF-order refine (ccfopt mask)

Writes ``hispecdrp/data/wavesol/HB.sim_hdf_wavesol.line_zp.fits`` (+ JSON).

Run from the repository root (needs ZEMAX HDF + KOA Stage-0 masters)::

    export KOA_CALIBRATION_CACHE=/path/to/KOA_CALIBRATION_CACHE
    python hispecdrp/data/wavesol/bake_pipeline_example_wavesol.py \\
        --fibers 1 2 3 --reuse-existing
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bake_pipeline_example_wavesol")


def _fiber_out_dir(base: Path, fiber: int) -> Path:
    if int(fiber) == 2:
        # Preserve the historical fiber-2 bake directory name.
        legacy = base / "pipeline_example_wave_rebake_seed0"
        if legacy.is_dir():
            return legacy
    return base / f"pipeline_example_wave_rebake_seed0_fib{int(fiber)}"


def _has_bake_products(out: Path) -> bool:
    return (out / "rate_map.fits").is_file() and (
        (out / "reference_spec-blaze_correct.fits").is_file()
        or any(out.glob("*_spec-blaze_correct.fits"))
        or any(out.glob("*_spec.fits"))
    )


def _has_ramp_products(out: Path) -> bool:
    return (out / "rate_map.fits").is_file() and (out / "simulated_ramp.fits").is_file()


def _blaze_spectrum_path(out: Path) -> Path:
    ref = out / "reference_spec-blaze_correct.fits"
    if ref.is_file():
        return ref
    matches = sorted(out.glob("*_spec-blaze_correct.fits"))
    if matches:
        return matches[-1]
    # Stage2 with stellar_rv skipped saves the blaze product as ``*_spec.fits``.
    matches = sorted(out.glob("*_spec.fits"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No blaze-corrected spectrum under {out}")


def reextract_blaze_with_wavesol(
    out: Path,
    *,
    fiber: int,
    spectrograph: str,
    wavesol,
) -> Path:
    """
    Re-run Stage0→2 blaze on an existing ramp using ``wavesol``.

    Needed after CCF WAVE refine so the blaze product is extracted with the
    same WAVE that will be packaged (closed-loop consistency).
    """
    from hispecdrp import datamodels
    from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline
    from hispecdrp.simulations.simulate_up_the_ramp import (
        register_sim_stage1_calibrations,
    )
    import numpy as np
    from astropy.io import fits

    ramp_path = out / "simulated_ramp.fits"
    rate_path = out / "rate_map.fits"
    if not ramp_path.is_file() or not rate_path.is_file():
        raise FileNotFoundError(f"Need ramp+rate_map under {out} for re-extract")

    ramp = datamodels.open(str(ramp_path))
    with fits.open(rate_path) as hdul:
        rate_map = np.asarray(hdul[0].data, dtype=np.float64)
    fibs = [int(fiber)]
    cals: dict = {"wavesol": wavesol}
    bundle = register_sim_stage1_calibrations(
        str(out),
        rate_map,
        spectrograph=spectrograph,
        fibers=fibs,
        ramp_meta=getattr(ramp, "meta", None),
        connect_remote=False,
        reuse_existing=True,
    )
    for key in ("trace_pos", "trace_prof", "fiberflat"):
        if bundle.get(key) is not None:
            cals[key] = bundle[key]

    do_fibers = [str(f) for f in fibs]
    l0 = Stage0Pipeline(output_dir=str(out)).apply(ramp)
    stage1_args = {
        "spec_extract": {"do_fibers": do_fibers},
        "set_wavesol": {"wavesol": wavesol},
    }
    if "trace_pos" in cals:
        stage1_args["spec_extract"]["tracepos"] = cals["trace_pos"]
    if "trace_prof" in cals:
        stage1_args["spec_extract"]["traceprof"] = cals["trace_prof"]
    l1 = Stage1Pipeline(output_dir=str(out), prim_args=stage1_args).apply(l0)
    stage2_args = {
        "blaze_corr": {"do_fibers": do_fibers},
        "barycorr": {"run": False},
        "stellar_rv": {"run": False},
    }
    if "fiberflat" in cals:
        stage2_args["blaze_corr"]["fiberflat"] = cals["fiberflat"]
    Stage2Pipeline(output_dir=str(out), prim_args=stage2_args).apply(l1)

    blaze = _blaze_spectrum_path(out)
    ref = out / "reference_spec-blaze_correct.fits"
    if blaze.resolve() != ref.resolve():
        shutil.copy2(blaze, ref)
    logger.info("Fiber %d re-extracted blaze with refined WAVE: %s", fiber, ref)
    return ref


def ensure_fiber_pipeline_products(
    out: Path,
    *,
    fiber: int,
    spectrograph: str,
    reuse_existing: bool,
    mag_ab: float,
) -> tuple[Path, Path]:
    """Return ``(rate_map_path, blaze_spectrum_path)`` for one fiber."""
    out.mkdir(parents=True, exist_ok=True)
    if reuse_existing and _has_bake_products(out):
        rate = out / "rate_map.fits"
        spec = _blaze_spectrum_path(out)
        logger.info("Reusing fiber %d products under %s", fiber, out)
        return rate, spec

    from hispecdrp import datamodels
    from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline
    from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum

    fibs = [int(fiber)]
    if reuse_existing and _has_ramp_products(out):
        logger.info("Resuming fiber %d from existing ramp under %s", fiber, out)
        ramp = datamodels.open(str(out / "simulated_ramp.fits"))
        # Rebuild / reuse Stage-1 sim cals for this fiber.
        from hispecdrp.simulations.simulate_up_the_ramp import (
            register_sim_stage1_calibrations,
            _load_pipeline_example_wavesol,
            _wavesol_meta_from_ramp,
        )
        import numpy as np
        from astropy.io import fits

        with fits.open(out / "rate_map.fits") as hdul:
            rate_map = np.asarray(hdul[0].data, dtype=np.float64)
        cals = {}
        bundle = register_sim_stage1_calibrations(
            str(out),
            rate_map,
            spectrograph=spectrograph,
            fibers=fibs,
            ramp_meta=getattr(ramp, "meta", None),
            connect_remote=False,
            reuse_existing=True,
        )
        for key in ("trace_pos", "trace_prof", "fiberflat", "wavesol"):
            if bundle.get(key) is not None:
                cals[key] = bundle[key]
        if "wavesol" not in cals:
            cals["wavesol"] = _load_pipeline_example_wavesol(
                spectrograph,
                meta=_wavesol_meta_from_ramp(getattr(ramp, "meta", None), spectrograph),
            )
    else:
        logger.info("Building fiber %d ramp + Stage0–2 under %s", fiber, out)
        ramp, cals, _rate = create_ramp_from_spectrum(
            str(out),
            spectrograph=spectrograph,
            fibers=fibs,
            template_filename="lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
            mag_ab=float(mag_ab),
            filter_band="J",
            seed=0,
            radial_velocity_km_s=0.0,
            # Match Basic Usage walkthrough: force BERV=0 for the controlled RV=0 path.
            barycentric_velocity_km_s=0.0,
            # Always reuse Stage-1 masters (fiberflat/traces/WAVE registration).
            # ``reuse_existing`` only gates whether science rate_map/blaze are rebuilt.
            reuse_stage1_sims=True,
        )
    do_fibers = [str(f) for f in fibs]
    l0 = Stage0Pipeline(output_dir=str(out)).apply(ramp)
    stage1_args = {"spec_extract": {"do_fibers": do_fibers}}
    # Prefer in-memory sim cals when present (avoids KOA geometric WAVE for bake).
    if "trace_pos" in cals:
        stage1_args["spec_extract"]["tracepos"] = cals["trace_pos"]
    if "trace_prof" in cals:
        stage1_args["spec_extract"]["traceprof"] = cals["trace_prof"]
    if "wavesol" in cals:
        stage1_args["set_wavesol"] = {"wavesol": cals["wavesol"]}
    l1 = Stage1Pipeline(output_dir=str(out), prim_args=stage1_args).apply(l0)
    # Only blaze is required for WAVE bake; skip stellar_rv (and its preview imports).
    stage2_args = {
        "blaze_corr": {"do_fibers": do_fibers},
        "barycorr": {"run": False},
        "stellar_rv": {"run": False},
    }
    if "fiberflat" in cals:
        stage2_args["blaze_corr"]["fiberflat"] = cals["fiberflat"]
    l2 = Stage2Pipeline(output_dir=str(out), prim_args=stage2_args).apply(l1)

    rate = out / "rate_map.fits"
    if not rate.is_file():
        raise FileNotFoundError(f"expected rate_map.fits under {out}")
    blaze = _blaze_spectrum_path(out)
    ref = out / "reference_spec-blaze_correct.fits"
    if blaze.resolve() != ref.resolve():
        shutil.copy2(blaze, ref)
    logger.info("Fiber %d blaze spectrum: %s", fiber, ref)
    return rate, ref


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("test_data"),
        help="Root for per-fiber bake directories (default: test_data)",
    )
    p.add_argument("--spectrograph", default="BSPEC", choices=("BSPEC", "RSPEC"))
    p.add_argument("--fibers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--mag-ab", type=float, default=3.05)
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse rate_map + blaze spectrum when already present",
    )
    p.add_argument("--known-rv-m-s", type=float, default=0.0)
    p.add_argument("--poly-deg", type=int, default=2)
    p.add_argument("--ccf-refine-iters", type=int, default=2)
    p.add_argument("--ccf-v-step-m-s", type=float, default=25.0)
    p.add_argument(
        "--mask",
        default="hispecdrp/data/masks/lte049*.ccfopt.delta.mas",
        help="Line mask glob for CCF refine",
    )
    p.add_argument(
        "--koa-cache",
        default=None,
        help="Set KOA_CALIBRATION_CACHE for this process (optional)",
    )
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only rebake WAVE from existing per-fiber products",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore --reuse-existing science products and rebuild rate_map/blaze",
    )
    p.add_argument(
        "--iterate-extract",
        action="store_true",
        help=(
            "After packaging WAVE, re-extract each fiber blaze with that WAVE and "
            "run a second CCF refine pass (closed-loop consistency)"
        ),
    )
    args = p.parse_args(argv)

    if args.koa_cache:
        os.environ["KOA_CALIBRATION_CACHE"] = str(Path(args.koa_cache).expanduser())
    if not os.environ.get("KOA_CALIBRATION_CACHE"):
        raise SystemExit(
            "KOA_CALIBRATION_CACHE is unset. Pass --koa-cache or export it."
        )

    out_root = args.out_root.expanduser().resolve()
    fibers = [int(f) for f in args.fibers]
    reuse = bool(args.reuse_existing) and not bool(args.force_rebuild)
    rate_by_fiber: dict[int, Path] = {}
    spec_by_fiber: dict[int, Path] = {}
    for fib in fibers:
        out = _fiber_out_dir(out_root, fib)
        if args.force_rebuild and not args.skip_pipeline:
            for name in (
                "rate_map.fits",
                "simulated_ramp.fits",
                "reference_spec-blaze_correct.fits",
            ):
                pth = out / name
                if pth.is_file():
                    pth.unlink()
                    logger.info("Removed stale %s", pth)
            for pat in ("*_spec-blaze_correct.fits", "*_spec.fits", "*_specuncal*.fits"):
                for pth in out.glob(pat):
                    pth.unlink(missing_ok=True)
        if args.skip_pipeline:
            if not _has_bake_products(out):
                raise SystemExit(f"Missing bake products for fiber {fib} under {out}")
            rate_by_fiber[fib] = out / "rate_map.fits"
            spec_by_fiber[fib] = _blaze_spectrum_path(out)
        else:
            rate, spec = ensure_fiber_pipeline_products(
                out,
                fiber=fib,
                spectrograph=args.spectrograph,
                reuse_existing=reuse,
                mag_ab=float(args.mag_ab),
            )
            rate_by_fiber[fib] = rate
            spec_by_fiber[fib] = spec

    # Import regenerator module by path (hispecdrp.data.wavesol is not a package).
    import importlib.util

    regen_path = Path(__file__).resolve().parent / "regenerate_sim_hdf_line_zp_wavesol.py"
    spec = importlib.util.spec_from_file_location(
        "regenerate_sim_hdf_line_zp_wavesol", regen_path
    )
    regen = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(regen)
    bake_multi_fiber_packaged_wavesol = regen.bake_multi_fiber_packaged_wavesol

    out = bake_multi_fiber_packaged_wavesol(
        spectrograph=args.spectrograph,
        rate_by_fiber=rate_by_fiber,
        spectrum_by_fiber=spec_by_fiber,
        known_rv_m_s=float(args.known_rv_m_s),
        poly_deg=int(args.poly_deg),
        mask_glob=str(args.mask),
        ccf_refine_iters=int(args.ccf_refine_iters),
        ccf_v_step_m_s=float(args.ccf_v_step_m_s),
        ccf_work_dir=out_root / "wavesol_ccf_refine_multifiber",
    )
    print(f"wrote {out}")

    if args.iterate_extract:
        from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
            load_packaged_sim_hdf_line_zp_wavesol,
        )

        logger.info(
            "iterate-extract: re-extracting fibers %s with packaged WAVE, then re-bake",
            fibers,
        )
        wavesol = load_packaged_sim_hdf_line_zp_wavesol(args.spectrograph)
        for fib in fibers:
            fib_out = _fiber_out_dir(out_root, fib)
            spec_by_fiber[fib] = reextract_blaze_with_wavesol(
                fib_out,
                fiber=fib,
                spectrograph=args.spectrograph,
                wavesol=wavesol,
            )
            rate_by_fiber[fib] = fib_out / "rate_map.fits"
        out = bake_multi_fiber_packaged_wavesol(
            spectrograph=args.spectrograph,
            rate_by_fiber=rate_by_fiber,
            spectrum_by_fiber=spec_by_fiber,
            known_rv_m_s=float(args.known_rv_m_s),
            poly_deg=int(args.poly_deg),
            mask_glob=str(args.mask),
            ccf_refine_iters=int(args.ccf_refine_iters),
            ccf_v_step_m_s=float(args.ccf_v_step_m_s),
            ccf_work_dir=out_root / "wavesol_ccf_refine_multifiber_iter2",
        )
        print(f"wrote (iterate-extract) {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
