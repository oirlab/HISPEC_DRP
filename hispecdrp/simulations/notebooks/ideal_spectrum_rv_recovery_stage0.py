#!/usr/bin/env python
"""
Ideal-spectrum RV recovery from a simulated ramp (Stage 0 -> Stage 1).

Standalone script version of ``ideal_spectrum_rv_recovery_stage1.py``, extended
to start from raw data instead of a Level-0 product.

Like the Stage-1-only variant, this simulates a native BSPEC rate map (e-/s)
for each injected RV by Doppler-shifting the BT-Settl lte040 template
(``SimulateRateMap.apply_radial_velocity``) and painting it onto the detector
(``SimulateRateMap.create_native_rate_map``). But instead of wrapping that
rate map directly as a Level-0 product, this version treats it as the true
photon flux and synthesizes a raw up-the-ramp readout from it
(``create_ramp_and_calibrations`` -> ``utr.create_ramp``): gain, bias, dark
current, read noise, and Poisson photon noise are all injected into the ramp,
the same way a real detector readout would look. That raw ramp is the input
handed to ``Stage0Pipeline`` (``run_stage0_simulation``), which runs the full
Stage 0 chain -- DQ init, saturation flagging, bias subtraction, refpix
correction, nonlinearity correction, ramp fitting, dark subtraction, gain
correction, detector-flat correction -- to fit it back down to a Level-0
product (RATE/RATE_ERR/DQ). From there, Stage 1 (box extraction +
wavelength solution), blaze correction to Level 2, and RV recovery
(``recover_stellar_rv`` with the packaged full-band ``*.ccfopt.delta.mas``
line mask) proceed exactly as in the Stage-1-only script.

Calibrations all come from your existing local ``KOA_CALIBRATION_CACHE`` --
the same cache, same env var, same relative path as the Stage-1-only script.
Detector calibrations (gain/bias/dark/rn/dq/saturation/detflat) are fetched
via ``get_master_calibrations`` (used both to inject realistic ramp noise
and, passed straight into ``Stage0Pipeline``, to reduce it back out, so the
simulated and subtracted masters always match). Stage-1/blaze calibrations
(tracepos/traceprof/wavesol/fiberflat) are left unset in the primitive
configs below exactly as before, so the pipeline's normal calibration
selectors resolve them from that same cache. ``connect_remote=False``
everywhere, so this never talks to the network -- it does require the cache
to already be populated for BSPEC (detector *and* Stage-1 calibrations).

Any residual injected-vs-recovered RV bias therefore comes from the full
rate-map -> ramp -> Stage 0 -> Stage 1 extraction -> blaze chain (photon and
read noise, ramp fitting, detector calibration mismatch, trace/PSF sampling,
box extraction, wavelength solution, blaze shape) -- unlike the Stage-1-only
script, detector noise/calibration effects are *not* excluded here.

Runtime: each injected RV now also pays for a synthetic ``N_READS``-read
ramp simulation plus the full Stage 0 reduction, on top of the native
rate-map simulation and Stage 1/blaze/CCF cost -- noticeably slower per
point than the Stage-1-only script.

Run from this directory so the relative ``output/`` and
``koa_calibration_cache/`` paths resolve the same way they do in the
Stage-1-only script::

    python ideal_spectrum_rv_recovery_stage0.py
"""

from __future__ import annotations

import os
from pathlib import Path

import astropy.units as u
import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed

import hispecdrp
from hispecdrp import datamodels
from hispecdrp.calibrations import HISPECCalibrationStore
from hispecdrp.calibrations.run_caldb import get_master_calibrations
from hispecdrp.pipelines import Stage1Pipeline
from hispecdrp.primitives import BlazeCorrectionPrimitive
from hispecdrp.primitives.utils.spectral_utils import (
    get_spectrum_path,
    load_btsettl_spectrum,
    normalize_2mass_band,
)
from hispecdrp.simulations.e2e_gaia import resolve_e2e_mjd_start
from hispecdrp.simulations.e2e_support import templates_data_dir
from hispecdrp.simulations.plotting import plot_ideal_spectrum_rv_recovery
from hispecdrp.simulations.simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap
from hispecdrp.simulations.simulate_up_the_ramp import (
    create_ramp_and_calibrations,
    get_default_stage0_primitives,
    run_stage0_simulation,
)
from hispecdrp.simulations.spectra import quiet_pipeline_logging, recover_stellar_rv
from hispecdrp.utils.line_mask_utils import (
    paired_line_mask_path_for_template,
    read_mask_generation_meta,
)

# --- Configuration -----------------------------------------------------
# Same packaged lte040 template and full-band native *.ccfopt.delta.mas line
# mask (default ccfopt.delta pairing) as the Stage-1-only script. No Stage-1
# calibrations are built here -- everything is resolved from
# KOA_CALIBRATION_CACHE when each pipeline step runs.
OUT = Path("output")
SPECTROGRAPH = "BSPEC"
FIBER = 1
#BULK_RV_KM_S = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
BULK_RV_KM_S = np.arange(-50, 51, 1)
TEMPLATE = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
TEMPLATE_MAG_AB = 3.05
TEMPLATE_FILTER_BAND = "J"
EXTRACTION_APERTURE = 8.0
INSTRUMENT_ERA = "0.0.1"

# Stage 0 / ramp-simulation config (E2E-sim defaults: n_reads=10, readtime=1.7 s).
N_READS = 20
READTIME = 1.7
POISSON_NOISE = False
CONVERT_TO_UINT16 = True
# Per-injection Stage0/ramp-sim core count. Kept at 1: the outer RV loop is
# already parallelized across N_JOBS processes below, so each worker's own
# internal parallelism should not also fan out and oversubscribe the machine.
STAGE0_MAX_CORES = 1
N_JOBS = 2

# Resolved in main(): same simulation epoch used across the simulations
# module (e2e_gaia, run_e2e_sim, etc.), so the ramp/L0Model's meta lines up
# with whatever else registered calibrations into your KOA_CALIBRATION_CACHE
# at this epoch, and the packaged line mask paired to TEMPLATE.
MJD_START: float | None = None
MASK_PATH: Path | None = None


def _configure_environment() -> None:
    """Load .env, point at the local calibration cache, and never touch the network."""
    load_dotenv()
    print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

    OUT.mkdir(parents=True, exist_ok=True)
    print("output:", OUT.resolve())

    os.environ["KOA_CALIBRATION_CACHE"] = "koa_calibration_cache/"

    with HISPECCalibrationStore(sync_on_init=True, connect_remote=True):
        pass

    # All calibrations (detector + Stage-1) come from your existing local KOA
    # calibration cache -- set KOA_CALIBRATION_CACHE before running this
    # script (or here) if it is not already exported in your environment.
    print(
        "KOA_CALIBRATION_CACHE:",
        os.environ.get("KOA_CALIBRATION_CACHE") or "(unset -- will fall back to a default, likely-empty cache)",
    )
    os.environ["HISPEC_KOA_CONNECT_REMOTE"] = "0"  # never touch the network


def _resolve_target() -> None:
    """Resolve the simulation epoch and pair the line mask to TEMPLATE."""
    global MJD_START, MASK_PATH

    MJD_START = resolve_e2e_mjd_start(None)
    MASK_PATH = paired_line_mask_path_for_template(TEMPLATE, mask_style="ccfopt.delta")
    meta = read_mask_generation_meta(MASK_PATH) or {}
    print(f"mask: {MASK_PATH}")
    print(
        f"n_lines={meta.get('n_lines')}  n_seed={meta.get('n_seed_lines')}  "
        f"n_drop={meta.get('n_dropped_lines')}"
    )
    print(f"mjd_start={MJD_START}  instrument_era={INSTRUMENT_ERA}")


def _prefetch_stage0_calibrations() -> None:
    """Download Stage 0 master calibrations once, serially, before parallelizing.

    ``run_one_injection`` fetches the same dq/saturation/gain/detflat/bias/
    dark/rn masters with ``connect_remote=False`` (offline, cache-only), but
    ``koa_middleware``'s ``download_calibration_file`` initializes a remote
    connection and downloads whenever a file is missing locally -- regardless
    of ``connect_remote`` (see ``HISPECCalibrationStore.download_calibration_file``).
    If a calibration is recorded in the local DB but its FITS file was never
    actually downloaded to disk, every N_JOBS worker races to download the
    same file to the same path at once, and one worker's cleanup can pull the
    zip out from under another mid-read. Fetching once here -- in the main
    process, before any worker starts -- guarantees everything is already on
    disk by the time the parallel loop runs, so the workers' offline calls
    never hit the network.
    """
    print("Prefetching Stage 0 master calibrations (serial, avoids a download race across workers)...")
    get_master_calibrations(
        instrument_era=INSTRUMENT_ERA,
        spectrograph=SPECTROGRAPH,
        mjd_start=MJD_START,
        connect_remote=True,
        sync_first=True,
    )


def bulk_run_label(bulk_km_s: float) -> str:
    """Stable run-dir label for a bulk RV."""
    sign = "m" if bulk_km_s < 0 else "p"
    return f"bulk_{sign}{abs(bulk_km_s):06.1f}kms".replace(".", "p")


def build_native_rate_map(bulk_km_s: float, *, output_dir: str) -> np.ndarray:
    """Template -> Doppler shift -> native BSPEC rate map (true photon flux)."""
    if not (-500.0 <= bulk_km_s <= 500.0):
        raise ValueError(f"bulk_km_s must be in [-500, 500], got {bulk_km_s}")

    template_path = get_spectrum_path(TEMPLATE, base_dir=templates_data_dir())
    wl, flux = load_btsettl_spectrum(
        template_path,
        mag_ab=TEMPLATE_MAG_AB,
        filter_band=normalize_2mass_band(TEMPLATE_FILTER_BAND),
    )
    sort_idx = np.argsort(np.asarray(wl.value, dtype=np.float64))
    wl, flux = wl[sort_idx], flux[sort_idx]

    srm = SimulateRateMap()
    srm.output_directory = output_dir
    srm.rate_map_outname = DEFAULT_RATE_MAP_FILENAME
    # continuous=False here is intentional: set_fiber_spectrum() auto-detects
    # this is really a flux-density (continuous) spectrum from its units and
    # flips + auto-densifies (R>=3e5) internally, same as the E2E sim path.
    srm.set_fiber_spectrum(
        fiber_number=FIBER,
        spectrum_wavelengths=wl,
        spectrum_flux=flux,
        continuous=False,
    )
    if bulk_km_s != 0.0:
        srm.apply_radial_velocity(fiber_number=FIBER, rv=bulk_km_s * u.km / u.s)

    return srm.create_native_rate_map(
        spectrograph=SPECTROGRAPH,
        method="prf_convolution",
        save=True,
    )


def rate_map_to_level0(rate_map: np.ndarray, *, output_dir: str, mjd_start: float) -> "datamodels.L0Model":
    """Rate map -> synthetic up-the-ramp readout -> Stage0Pipeline -> Level 0.

    Detector calibrations (gain/bias/dark/rn/dq/saturation/detflat) are
    fetched from the local KOA_CALIBRATION_CACHE via ``get_master_calibrations``
    -- the same masters are then passed straight into
    ``create_ramp_and_calibrations``/``run_stage0_simulation`` so the ramp is
    reduced with the exact calibrations used to simulate it. The fetched
    ``rn`` master's read-noise map is zeroed in place before use: combined
    with POISSON_NOISE=False above, this makes the simulated ramp read-noise-
    and photon-noise-free, isolating the rate-map -> ramp -> Stage 0 ->
    Stage 1 chain's own bias from any detector-noise contribution.
    """
    calibration_models = get_master_calibrations(
        instrument_era=INSTRUMENT_ERA,
        spectrograph=SPECTROGRAPH,
        mjd_start=mjd_start,
        connect_remote=False,
        sync_first=False,
    )
    calibration_models["rn"].rn = np.zeros_like(calibration_models["rn"].rn)

    ramp_model, calibration_models, _ = create_ramp_and_calibrations(
        rate_map=rate_map,
        calibration_models=calibration_models,
        output_dir=output_dir,
        readtime=READTIME,
        n_reads=N_READS,
        spectrograph=SPECTROGRAPH,
        max_cores=STAGE0_MAX_CORES,
        poisson_noise=POISSON_NOISE,
        convert_to_uint16=CONVERT_TO_UINT16,
        instrument_era=INSTRUMENT_ERA,
        mjd_start=mjd_start,
        connect_remote=False,
        sync_caldb_first=False,
    )
    return run_stage0_simulation(
        ramp_model,
        calibration_models,
        output_dir=output_dir,
        primitives=get_default_stage0_primitives(max_cores=STAGE0_MAX_CORES),
        max_cores=STAGE0_MAX_CORES,
    )


def run_one_injection(bulk_km_s: float, *, mjd_start: float, mask_path: Path) -> dict:
    """Rate map -> ramp -> Stage 0 -> Stage 1 -> blaze -> RV recovery for one bulk RV.

    ``mjd_start``/``mask_path`` are passed explicitly (rather than read off the
    module globals set by ``_resolve_target()``) so this function works
    unchanged whether called in-process or from a joblib worker process,
    which does not re-run ``main()``/``_resolve_target()`` and so would
    otherwise see them at their unset defaults.
    """
    label = bulk_run_label(bulk_km_s)
    run_dir = str(OUT / label)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== {label}: inject {bulk_km_s:.3f} km/s ===")
    rate_map = build_native_rate_map(bulk_km_s, output_dir=run_dir)
    print(f"rate_map shape={rate_map.shape}")

    level0 = rate_map_to_level0(rate_map, output_dir=run_dir, mjd_start=mjd_start)

    from hispecdrp.calibrations import WavelengthSolutionSelector

    do_fibers = [str(FIBER)]
    with quiet_pipeline_logging():
        l1 = Stage1Pipeline(
            level0,
            output_dir=run_dir,
            primitives={
                "spec_extract": {
                    "do_fibers": do_fibers,
                    "method": "box",
                    "extraction_aperture": float(EXTRACTION_APERTURE),
                },
                'set_wavesol': {
                    'wavesol': {
                        'selector_kwargs' : {
                            'cal_type' : 'sim_wavesol',
                        },
                        'selector' : WavelengthSolutionSelector,
                    }
                }
            },
        ).apply()

        l2 = BlazeCorrectionPrimitive.call(l1, do_fibers=do_fibers)

    inj_m_s, rec_m_s, err_m_s = recover_stellar_rv(
        l2,
        bulk_km_s,
        template=TEMPLATE,
        mask_path=str(mask_path),
        spectrograph=SPECTROGRAPH,
        fiber=FIBER,
    )
    bias_m_s = rec_m_s - inj_m_s
    print(
        f"injected={inj_m_s:12.4f} m/s  recovered={rec_m_s:16.6f} m/s  "
        f"err={err_m_s:12.6f} m/s  bias={bias_m_s:12.6e} m/s"
    )
    return {
        "label": label,
        "inject_km_s": float(bulk_km_s),
        "inject_m_s": inj_m_s,
        "recover_m_s": rec_m_s,
        "recover_err_m_s": err_m_s,
    }


def main() -> None:
    _configure_environment()
    _resolve_target()
    _prefetch_stage0_calibrations()

    # --- Inject each bulk RV and collate --------------------------------
    hdr = (
        f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
        f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    # N_JOBS: process-based (loky) parallelism, so each worker pays its own
    # rate-map-sim + ramp-sim + Stage 0/1/blaze/CCF cost independently across
    # N_JOBS cores. mjd_start/mask_path are passed explicitly since worker
    # processes don't re-run main()/_resolve_target() and so never see those
    # globals set.
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_injection)(bulk, mjd_start=MJD_START, mask_path=MASK_PATH)
        for bulk in BULK_RV_KM_S
    )

    for bulk, r in zip(BULK_RV_KM_S, results):
        bias = r["recover_m_s"] - r["inject_m_s"]
        print(
            f"{bulk:10.1f}  {r['inject_m_s']:14.4f}  {r['recover_m_s']:16.6f}  "
            f"{r['recover_err_m_s']:12.6f}  {bias:12.6e}  {bias * 1e3:10.4f}"
        )

    inj_m_s = np.asarray([r["inject_m_s"] for r in results], dtype=float)
    rec_m_s = np.asarray([r["recover_m_s"] for r in results], dtype=float)
    err_m_s = np.asarray([r["recover_err_m_s"] for r in results], dtype=float)
    bias_m_s = rec_m_s - inj_m_s
    max_abs = float(np.max(np.abs(bias_m_s)))
    rms = float(np.sqrt(np.mean(bias_m_s**2)))
    d_bias = bias_m_s - bias_m_s[0]
    max_d = float(np.max(np.abs(d_bias[1:])))
    print()
    print(f"max|bias| = {max_abs:.6e} m/s ({max_abs * 1e3:.4f} mm/s)")
    print(f"rms bias  = {rms:.6e} m/s ({rms * 1e3:.4f} mm/s)")
    print(f"max|differential bias| (vs RV=0) = {max_d:.6e} m/s ({max_d * 1e3:.4f} mm/s)")

    # --- Plot -------------------------------------------------------------
    summary = plot_ideal_spectrum_rv_recovery(
        inj_m_s,
        rec_m_s,
        err_m_s,
        title="BT-Settl lte040 (rate map -> ramp -> Stage 0 -> Stage 1): injected vs recovered RV",
        out_path=OUT / "lte040_ramp_stage0_stage1_simulated_vs_recovered_rv.png",
    )
    if summary.get("out_path"):
        print("wrote", summary["out_path"])


if __name__ == "__main__":
    main()
