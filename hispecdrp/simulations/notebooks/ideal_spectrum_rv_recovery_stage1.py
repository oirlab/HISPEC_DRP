#!/usr/bin/env python
"""
Ideal-spectrum RV recovery from a simulated rate map (Level 0 -> Stage 1).

Standalone script version of ``ideal_spectrum_rv_recovery_stage1.ipynb``.

This is a physical-path variant of ``ideal_spectrum_rv_recovery.ipynb``. Instead of
painting the Doppler-shifted BT-Settl lte040 template directly onto an L2
wavelength grid with ``build_l2``, this script simulates a native BSPEC rate
map (Level 0) for each injected RV, runs it through Stage 1
(``SpectralExtractionPrimitive`` + ``SetInitialWLSPrimitive``: box extraction +
wavelength solution), blaze-corrects to Level 2, and then recovers the RV with
the same ``recover_stellar_rv`` / packaged full-band ``*.ccfopt.delta.mas`` line
mask used in the ideal-spectrum notebook.

No ramp, no detector calibrations, anywhere. A rate map (e-/s) is a pure
photon/optics simulation: load the template, Doppler-shift it
(``spectral_utils.doppler_shift_wave``), and paint it onto the detector
(``SimulateRateMap.create_native_rate_map``) -- none of that touches gain,
bias, dark, read noise, or flat-field models. Those detector calibrations
only matter if you simulate a synthetic up-the-ramp readout *from* a rate map
and then fit that ramp back down (``Stage0Pipeline``), which this script does
not do at all -- the rate map produced here *is* the Level-0 product,
wrapped directly as an ``L0Model`` and handed straight to Stage 1.

Stage-1 calibrations come from your existing local ``KOA_CALIBRATION_CACHE``.
``tracepos``/``traceprof``/``wavesol`` (Stage 1) and ``fiberflat`` (blaze) are
left unset in the primitive configs below, so the pipeline's normal
calibration selectors resolve them from that cache (matched by spectrograph +
nearest MJD), exactly as real Stage 1/2 processing would --
``connect_remote=False``, so this never talks to the network, but it does
require that cache to already be populated for BSPEC.

Any residual injected-vs-recovered RV bias therefore comes from the rate-map
-> Stage 1 extraction -> blaze chain itself (trace/PSF sampling, box
extraction, wavelength solution, blaze shape), not from photon or detector
noise, and not from any simulated-calibration mismatch.

Runtime: each injected RV pays for a native BSPEC rate-map simulation
(~30-60 s) plus Stage 1/blaze/CCF (a few seconds) -- no ramp, no ramp fit, no
calibration bootstrap. The default 6-point grid (0-50 km/s in 10 km/s steps)
should take on the order of a few minutes total.

Run from this directory (``hispecdrp/simulations/notebooks/``) so the
relative ``output/`` and ``koa_calibration_cache/`` paths resolve the same
way they do in the notebook::

    python ideal_spectrum_rv_recovery_stage1.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed

import hispecdrp
from hispecdrp import datamodels
from hispecdrp.calibrations import HISPECCalibrationStore
from hispecdrp.pipelines import Stage1Pipeline
from hispecdrp.primitives import BlazeCorrectionPrimitive
from hispecdrp.primitives.utils.spectral_utils import (
    doppler_shift_wave,
    load_btsettl_spectrum,
    normalize_2mass_band,
)
from hispecdrp.utils.spectral_template_utils import get_spectral_template_path
from hispecdrp.simulations.e2e_gaia import resolve_e2e_mjd_start
from hispecdrp.simulations.e2e_support import templates_data_dir
from hispecdrp.simulations.plotting import plot_ideal_spectrum_rv_recovery
from hispecdrp.simulations.simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap
from hispecdrp.driver.logging import quiet_pipeline_logging
from hispecdrp.simulations.simulate_spectra import recover_stellar_rv
from hispecdrp.utils.line_mask_utils import (
    paired_line_mask_path_for_template,
    read_mask_generation_meta,
)

# --- Configuration -----------------------------------------------------
# Same packaged lte040 template and full-band native *.ccfopt.delta.mas line
# mask (default ccfopt.delta pairing) as the ideal-spectrum notebook. No
# detector or Stage-1 calibrations are built here -- everything is resolved
# from KOA_CALIBRATION_CACHE when each pipeline step runs.
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

# Resolved in main(): same simulation epoch used across the simulations
# module (e2e_gaia, run_e2e_sim, etc.), so the L0Model's meta lines up with
# whatever else registered calibrations into your KOA_CALIBRATION_CACHE at
# this epoch, and the packaged line mask paired to TEMPLATE.
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


def bulk_run_label(bulk_km_s: float) -> str:
    """Stable run-dir label for a bulk RV."""
    sign = "m" if bulk_km_s < 0 else "p"
    return f"bulk_{sign}{abs(bulk_km_s):06.1f}kms".replace(".", "p")


def build_native_rate_map(bulk_km_s: float, *, output_dir: str) -> np.ndarray:
    """Template -> Doppler shift -> native BSPEC rate map. No ramp, no cals."""
    if not (-500.0 <= bulk_km_s <= 500.0):
        raise ValueError(f"bulk_km_s must be in [-500, 500], got {bulk_km_s}")

    template_path = get_spectral_template_path(TEMPLATE, base_dir=templates_data_dir())
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
        srm.fibers[FIBER]["wavelength"] = doppler_shift_wave(srm.fibers[FIBER]["wavelength"], bulk_km_s * 1000.0)

    return srm.create_native_rate_map(
        spectrograph=SPECTROGRAPH,
        method="prf_convolution",
        save=True,
    )


def rate_map_to_level0(rate_map: np.ndarray, *, spectrograph: str, mjd_start: float) -> "datamodels.L0Model":
    """Wrap a native e-/s rate map directly as a Level-0 product (no Stage 0)."""
    rate = np.asarray(rate_map, dtype=np.float32)
    # Near-noiseless error floor, same convention as build_l2() spec_err.
    rate_err = np.maximum(1.0e-6 * np.abs(rate), 1.0e-6).astype(np.float32)
    return datamodels.L0Model(
        RATE=rate,
        RATE_ERR=rate_err,
        meta={
            "spectrograph": spectrograph,
            "instrument_name": "HISPEC",
            "instrument_era": INSTRUMENT_ERA,
            "mjd_start": mjd_start,
        },
    )


def run_one_injection(bulk_km_s: float, *, mjd_start: float, mask_path: Path) -> dict:
    """Rate map (Level 0) -> Stage 1 -> blaze -> RV recovery for one bulk RV.

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

    level0 = rate_map_to_level0(rate_map, spectrograph=SPECTROGRAPH, mjd_start=mjd_start)

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

    # --- Inject each bulk RV and collate --------------------------------
    hdr = (
        f"{'bulk_km/s':>10}  {'injected_m/s':>14}  {'recovered_m/s':>16}  "
        f"{'rv_err_m/s':>12}  {'bias_m/s':>12}  {'bias_mm/s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    # n_jobs=6: process-based (loky) parallelism, so each worker pays its own
    # rate-map-sim + Stage 1/blaze/CCF cost independently across 6 cores.
    # mjd_start/mask_path are passed explicitly since worker processes don't
    # re-run main()/_resolve_target() and so never see those globals set.
    results = Parallel(n_jobs=6)(
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
        title="BT-Settl lte040 (rate map -> Stage 1): injected vs recovered RV",
        out_path=OUT / "lte040_rate_map_stage1_simulated_vs_recovered_rv.png",
    )
    if summary.get("out_path"):
        print("wrote", summary["out_path"])


if __name__ == "__main__":
    main()
