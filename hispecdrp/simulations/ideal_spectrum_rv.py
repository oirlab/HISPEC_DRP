"""Physical-path ideal-spectrum RV recovery (rate map / ramp → extract → CCF)."""

from __future__ import annotations

import shutil
from pathlib import Path

import astropy.units as u
import numpy as np

from .. import datamodels
from ..pipelines import Stage1Pipeline
from ..primitives import BlazeCorrectionPrimitive
from ..primitives.utils.spectral_utils import (
    get_spectrum_path,
    load_btsettl_spectrum,
    normalize_2mass_band,
)
from .e2e_gaia import resolve_e2e_mjd_start
from .e2e_support import templates_data_dir
from .simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap
from .spectra import quiet_pipeline_logging, recover_stellar_rv

__all__ = [
    "build_template_rate_map",
    "geometric_hdf_wavesol",
    "prepare_stage1_cals",
    "rate_map_to_l0",
    "extract_and_blaze_l2",
    "recover_rv_from_rate_map",
    "recover_rv_from_ramp",
]


def build_template_rate_map(
    rv_km_s: float,
    *,
    template: str,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    output_dir: str | Path,
    mag_ab: float = 3.05,
    filter_band: str = "J",
    method: str = "prf_convolution",
) -> np.ndarray:
    """Doppler-shift a BT-Settl template and paint a native rate map."""
    template_path = get_spectrum_path(template, base_dir=templates_data_dir())
    wl, flux = load_btsettl_spectrum(
        template_path,
        mag_ab=mag_ab,
        filter_band=normalize_2mass_band(filter_band),
    )
    sort_idx = np.argsort(np.asarray(wl.value, dtype=np.float64))
    wl, flux = wl[sort_idx], flux[sort_idx]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    srm = SimulateRateMap()
    srm.output_directory = str(out)
    srm.rate_map_outname = DEFAULT_RATE_MAP_FILENAME
    # continuous=False is intentional: set_fiber_spectrum auto-detects a
    # flux-density spectrum from its units and densifies (R>=3e5) internally.
    srm.set_fiber_spectrum(
        fiber_number=int(fiber),
        spectrum_wavelengths=wl,
        spectrum_flux=flux,
        continuous=False,
    )
    if float(rv_km_s) != 0.0:
        srm.apply_radial_velocity(fiber_number=int(fiber), rv=float(rv_km_s) * u.km / u.s)
    return srm.create_native_rate_map(
        spectrograph=str(spectrograph).upper(),
        method=method,
        save=True,
    )


def rate_map_to_l0(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    mjd_start: float | None = None,
    instrument_era: str = "0.0.1",
) -> datamodels.L0Model:
    """Wrap a noiseless rate map as an L0 (no ramp / Stage 0)."""
    rate = np.asarray(rate_map, dtype=np.float32)
    rate_err = np.maximum(1.0e-6 * np.abs(rate), 1.0e-6).astype(np.float32)
    return datamodels.L0Model(
        rate=rate,
        rate_err=rate_err,
        meta={
            "spectrograph": str(spectrograph).upper(),
            "instrument_name": "HISPEC",
            "instrument_era": instrument_era,
            "mjd_start": resolve_e2e_mjd_start(mjd_start),
        },
    )


def geometric_hdf_wavesol(spectrograph: str = "BSPEC", *, meta: dict | None = None):
    """Deposit-inverse HDF WAVE in the same pixel order as rate-map extraction.

    The packaged ``*.sim_hdf_wavesol.line_zp`` product is reversed along the
    spectral axis relative to box-extracted ``spec_fib*``. Attaching that WAVE
    makes ``recover_stellar_rv`` report a several-hundred m/s offset and
    inflated ``RV_ERR``. This geometric WAVE matches
    :func:`~hispecdrp.simulations.e2e_pipeline.get_sim_hdf_trace_grids`.
    """
    from .e2e_pipeline import create_wavelength_solution_model_from_sim_hdf

    return create_wavelength_solution_model_from_sim_hdf(
        str(spectrograph).upper(),
        meta=meta,
    )


def prepare_stage1_cals(
    output_dir: str | Path,
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    extraction_aperture: float = 8.0,
) -> dict:
    """Build or reuse sim-HDF traces and fiberflat; always attach geometric WAVE."""
    from .e2e_pipeline import prepare_pipeline_example_stage1

    spectrograph = str(spectrograph).upper()
    bundle = prepare_pipeline_example_stage1(
        str(output_dir),
        rate_map,
        spectrograph=spectrograph,
        fibers=[int(fiber)],
        extraction_aperture=float(extraction_aperture),
        reuse_existing=True,
    )
    # Cached masters may still hold the flipped packaged line-ZP WAVE.
    bundle["wavesol"] = geometric_hdf_wavesol(
        spectrograph,
        meta={"mjd_start": resolve_e2e_mjd_start(None)},
    )
    return bundle


def _remove_run_dir(run_dir: str | Path) -> None:
    """Delete a per-RV working directory after the recovered RV is in memory."""
    path = Path(run_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def extract_and_blaze_l2(
    l0: datamodels.L0Model,
    *,
    output_dir: str | Path,
    fiber: int = 1,
    extraction_aperture: float = 8.0,
    stage1_cals: dict | None = None,
    save_result: bool = True,
) -> datamodels.L2Model:
    """Stage 1 box extract + sim WAVE, then blaze-correct to L2.

    Pass ``stage1_cals`` from :func:`prepare_stage1_cals` so traces / WAVE /
    fiberflat are used in-memory and the local KOA schema is never queried.
    """
    if stage1_cals is None:
        stage1_cals = prepare_stage1_cals(
            output_dir,
            np.asarray(l0.rate),
            spectrograph=str(getattr(l0.meta, "spectrograph", "BSPEC")),
            fiber=fiber,
            extraction_aperture=extraction_aperture,
        )
    do_fibers = [str(int(fiber))]
    with quiet_pipeline_logging():
        l1 = Stage1Pipeline(
            l0,
            output_dir=str(output_dir),
            save_result=bool(save_result),
            primitives={
                "spec_extract": {
                    "do_fibers": do_fibers,
                    "method": "box",
                    "extraction_aperture": float(extraction_aperture),
                    "tracepos": stage1_cals["trace_pos"],
                    "traceprof": stage1_cals["trace_prof"],
                },
                "set_wavesol": {"wavesol": stage1_cals["wavesol"]},
            },
        ).apply()
        return BlazeCorrectionPrimitive.call(
            l1,
            do_fibers=do_fibers,
            fiberflat=stage1_cals["fiberflat"],
            save_result=bool(save_result),
        )


def recover_rv_from_rate_map(
    rv_km_s: float,
    *,
    template: str,
    mask_path: str,
    output_dir: str | Path,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    mag_ab: float = 3.05,
    filter_band: str = "J",
    extraction_aperture: float = 8.0,
) -> tuple[float, float, float]:
    """Native rate map → Stage 1 → blaze → ``recover_stellar_rv``."""
    with quiet_pipeline_logging():
        parent = Path(output_dir)
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = parent / f"rate_map_{rv_km_s:+.1f}kms".replace("+", "p").replace("-", "m")
        rate = build_template_rate_map(
            rv_km_s,
            template=template,
            fiber=fiber,
            spectrograph=spectrograph,
            output_dir=run_dir,
            mag_ab=mag_ab,
            filter_band=filter_band,
        )
        stage1_cals = prepare_stage1_cals(
            parent,
            rate,
            spectrograph=spectrograph,
            fiber=fiber,
            extraction_aperture=extraction_aperture,
        )
        wave = np.asarray(stage1_cals["wavesol"].wave_fib1, dtype=float)
        if np.isfinite(wave[0, 0]) and np.isfinite(wave[0, -1]) and wave[0, 0] < wave[0, -1]:
            raise RuntimeError(
                "Stage-1 WAVE increases along the order "
                f"({wave[0, 0]:.6f} -> {wave[0, -1]:.6f} um). "
                "Extracted spectra use decreasing λ. Restart the notebook kernel "
                "so prepare_stage1_cals attaches the geometric HDF WAVE."
            )
        l0 = rate_map_to_l0(rate, spectrograph=spectrograph)
        l2 = extract_and_blaze_l2(
            l0,
            output_dir=run_dir,
            fiber=fiber,
            extraction_aperture=extraction_aperture,
            stage1_cals=stage1_cals,
        )
        return recover_stellar_rv(
            l2,
            rv_km_s,
            template=template,
            mask_path=str(mask_path),
            spectrograph=spectrograph,
            fiber=fiber,
        )


def recover_rv_from_ramp(
    rv_km_s: float,
    *,
    template: str,
    mask_path: str,
    output_dir: str | Path,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    mag_ab: float = 3.05,
    filter_band: str = "J",
    extraction_aperture: float = 8.0,
    n_reads: int = 10,
    seed: int = 0,
    cleanup: bool = True,
) -> tuple[float, float, float]:
    """Rate map → UTR ramp → Stage 0 → Stage 1 → blaze → ``recover_stellar_rv``.

    When ``cleanup`` is True (default), the per-RV ramp / Stage 0 / Stage 1
    working directory is deleted after the recovered RV is in memory. Shared
    Stage-1 masters under ``output_dir`` are kept for reuse.
    """
    from .simulate_up_the_ramp import create_ramp_from_spectrum, run_stage0_simulation

    parent = Path(output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / f"ramp_{rv_km_s:+.1f}kms".replace("+", "p").replace("-", "m")
    try:
        with quiet_pipeline_logging():
            ramp, cals, rate = create_ramp_from_spectrum(
                str(run_dir),
                spectrograph=spectrograph,
                fibers=[int(fiber)],
                template_filename=template,
                mag_ab=mag_ab,
                filter_band=filter_band,
                radial_velocity_km_s=float(rv_km_s),
                n_reads=int(n_reads),
                seed=int(seed),
                poisson_noise=False,
                use_master_calibrations=False,
                register_stage1_sims=False,
                register_wavesol=False,
                save_ramp=False,
                extraction_aperture=float(extraction_aperture),
            )
            stage1_cals = prepare_stage1_cals(
                parent,
                rate,
                spectrograph=spectrograph,
                fiber=fiber,
                extraction_aperture=extraction_aperture,
            )
            l0 = run_stage0_simulation(ramp, cals, str(run_dir))
            l2 = extract_and_blaze_l2(
                l0,
                output_dir=run_dir,
                fiber=fiber,
                extraction_aperture=extraction_aperture,
                stage1_cals=stage1_cals,
                save_result=False,
            )
            return recover_stellar_rv(
                l2,
                rv_km_s,
                template=template,
                mask_path=str(mask_path),
                spectrograph=spectrograph,
                fiber=fiber,
            )
    finally:
        if cleanup:
            _remove_run_dir(run_dir)
