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
    doppler_shift_wave,
    get_spectrum_path,
    load_btsettl_spectrum,
    normalize_2mass_band,
)
from .e2e_gaia import resolve_e2e_mjd_start
from .e2e_support import templates_data_dir
from .simulate_rate_map import DEFAULT_RATE_MAP_FILENAME, SimulateRateMap
from ..driver.logging import quiet_pipeline_logging
from .simulate_spectra import recover_stellar_rv

__all__ = [
    "build_template_rate_map",
    "build_identity_fiberflat",
    "build_paint_matched_fiberflat",
    "geometric_hdf_wavesol",
    "prepare_stage1_cals",
    "rate_map_to_l0",
    "extract_and_blaze_l2",
    "recover_rv_from_rate_map",
    "recover_rv_from_ramp",
]

# Fiberflat modes for ideal-spectrum Stage-1 recovery (see prepare_stage1_cals).
FIBERFLAT_IDENTITY = "identity"
FIBERFLAT_PAINT_MATCHED = "paint_matched"
FIBERFLAT_SIM_HDF = "sim_hdf"


def build_template_rate_map(
    rv_km_s: float,
    *,
    template: str | None = None,
    fiber: int = 1,
    spectrograph: str = "BSPEC",
    output_dir: str | Path,
    mag_ab: float = 3.05,
    filter_band: str = "J",
    method: str = "prf_convolution",
    spectrum_wavelengths=None,
    spectrum_flux=None,
) -> np.ndarray:
    """Doppler-shift a BT-Settl template and paint a native rate map.

    Pass an already Doppler-shifted ``spectrum_wavelengths`` / ``spectrum_flux``
    pair to skip the template load and the extra Doppler shift (``doppler_shift_wave``).
    """
    if spectrum_wavelengths is None or spectrum_flux is None:
        if template is None:
            raise ValueError("template is required when no explicit spectrum is given")
        template_path = get_spectrum_path(template, base_dir=templates_data_dir())
        wl, flux = load_btsettl_spectrum(
            template_path,
            mag_ab=mag_ab,
            filter_band=normalize_2mass_band(filter_band),
        )
        sort_idx = np.argsort(np.asarray(wl.value, dtype=np.float64))
        wl, flux = wl[sort_idx], flux[sort_idx]
        apply_rv = float(rv_km_s) != 0.0
    else:
        wl, flux = spectrum_wavelengths, spectrum_flux
        apply_rv = False
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
    if apply_rv:
        fib = srm.fibers[int(fiber)]
        fib["wavelength"] = doppler_shift_wave(fib["wavelength"], float(rv_km_s) * 1000.0)
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


def build_identity_fiberflat(
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    n_orders: int | None = None,
    n_pix: int = 4096,
    wavesol=None,
):
    """Ones fiberflat so blaze correction is a no-op (per-order nanmax of 1).

    Ideal-spectrum CCF recovery on PRF-painted rate maps is sensitive to
    continuum *shape*. Dividing out a blaze (even a paint-matched continuum
    fiberflat) reweights CCF lines by ~15–20 m/s relative to leaving the
    painted continuum intact. An identity fiberflat keeps Stage-2 blaze in the
    pipeline while preserving the continuum the mask was effectively tuned to
    for this diagnostic.

    Shape is taken from ``wavesol.wave_fib{fiber}`` when provided; otherwise
    ``n_orders`` × ``n_pix`` (default: all sim-HDF orders × 4096).
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    if wavesol is not None:
        wave = np.asarray(getattr(wavesol, f"wave_fib{fiber}"), dtype=np.float64)
        n_orders, n_pix = int(wave.shape[0]), int(wave.shape[1])
    elif n_orders is None:
        from .e2e_pipeline import _order_list

        n_orders = len(_order_list(spectrograph))
    shape = (int(n_orders), int(n_pix))
    ff = datamodels.FiberflatModel(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": spectrograph,
            "mjd_start": resolve_e2e_mjd_start(None),
        }
    )
    ones = np.ones(shape, dtype=np.float32)
    setattr(ff, f"spec_fib{fiber}", ones)
    setattr(ff, f"spec_err_fib{fiber}", np.full(shape, 1.0e-2, dtype=np.float32))
    setattr(ff, f"spec_var_poisson_fib{fiber}", np.zeros(shape, dtype=np.float32))
    setattr(ff, f"spec_var_rnoise_fib{fiber}", np.zeros(shape, dtype=np.float32))
    setattr(ff, f"spec_dq_fib{fiber}", np.zeros(shape, dtype=np.uint32))
    if wavesol is not None:
        setattr(ff, f"wave_fib{fiber}", wave.astype(np.float64))
    setattr(ff.meta, f"object_fiber{fiber}", "WHITELIGHT")
    for f in (1, 2, 3):
        if f != fiber and getattr(ff.meta, f"object_fiber{f}", None) is None:
            setattr(ff.meta, f"object_fiber{f}", "BLOCKED")
    return ff


def build_paint_matched_fiberflat(
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    extraction_aperture: float = 8.0,
    tracepos=None,
    flat_temperature_k: float = 1000.0,
    flat_scale: float = 2e-14,
):
    """Fiberflat from a continuum rate map painted like science (PRF + blaze).

    Paints a smooth blackbody with the same ``prf_convolution`` +
    ``apply_blaze=True`` engine as :func:`build_template_rate_map`, then
    box-extracts with the science traces / aperture.

    **Does not** by itself bring ideal-spectrum CCF bias under ~1 m/s: removing
    the painted continuum still reweights the CCF by ~15–20 m/s even when the
    fiberflat matches the science blaze. Prefer
    :func:`build_identity_fiberflat` for LFC-WAVE recovery diagnostics; keep
    this builder for continuum / blaze experiments.
    """
    import astropy.units as u
    from astropy.modeling.models import BlackBody

    from .. import datamodels
    from .e2e_pipeline import (
        _extract_box_one_order,
        _order_list,
        create_tracepos_model_from_sim_hdf,
        get_sim_hdf_trace_grids,
    )

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    if tracepos is None:
        tracepos = create_tracepos_model_from_sim_hdf(spectrograph)

    n_wl = 10_000
    wl_flat = np.linspace(0.8, 3.0, n_wl) * u.micron
    bb = BlackBody(
        temperature=float(flat_temperature_k) * u.K,
        scale=1.0 * u.erg / (u.cm**2 * u.s * u.AA * u.sr),
    )
    flux_flat = (
        float(flat_scale) * np.asarray(bb(wl_flat.to(u.AA)).value) * 1e4
    ) * (u.erg / u.s / u.cm**2 / u.micron)

    srm = SimulateRateMap()
    srm.set_fiber_spectrum(
        fiber_number=fiber,
        spectrum_wavelengths=wl_flat,
        spectrum_flux=flux_flat,
        continuous=True,
    )
    rate = np.asarray(
        srm.create_native_rate_map(
            spectrograph=spectrograph,
            method="prf_convolution",
            apply_blaze=True,
            save=False,
        ),
        dtype=np.float64,
    )
    rate_err = np.zeros_like(rate)
    rate_dq = np.zeros(rate.shape, dtype=np.uint32)

    order_list = _order_list(spectrograph)
    grids = get_sim_hdf_trace_grids(spectrograph)
    pos = np.asarray(getattr(tracepos, f"pos_fib{fiber}"), dtype=np.float64)
    n_pix = 4096
    spec_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    wave_rows: list[np.ndarray] = []
    for i, order in enumerate(order_list):
        if i >= pos.shape[0]:
            break
        spec, err = _extract_box_one_order(
            rate,
            rate_err,
            rate_dq,
            pos[i],
            extraction_aperture=float(extraction_aperture),
        )
        spec_rows.append(spec.astype(np.float32))
        err_rows.append(err.astype(np.float32))
        wave = grids.get(fiber, {}).get(order, {}).get("wavelength")
        if wave is None:
            wave = np.full(n_pix, np.nan, dtype=np.float64)
        wave_rows.append(np.asarray(wave, dtype=np.float64))

    spec_arr = np.stack(spec_rows, axis=0)
    err_arr = np.stack(err_rows, axis=0)
    wave_arr = np.stack(wave_rows, axis=0)
    ff = datamodels.FiberflatModel(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": spectrograph,
            "mjd_start": resolve_e2e_mjd_start(None),
        }
    )
    setattr(ff, f"spec_fib{fiber}", spec_arr)
    setattr(ff, f"spec_err_fib{fiber}", err_arr)
    setattr(ff, f"spec_var_poisson_fib{fiber}", np.zeros_like(spec_arr))
    setattr(ff, f"spec_var_rnoise_fib{fiber}", np.zeros_like(spec_arr))
    setattr(ff, f"spec_dq_fib{fiber}", np.zeros(spec_arr.shape, dtype=np.uint32))
    setattr(ff, f"wave_fib{fiber}", wave_arr)
    setattr(ff.meta, f"object_fiber{fiber}", "WHITELIGHT")
    for f in (1, 2, 3):
        if f != fiber and getattr(ff.meta, f"object_fiber{f}", None) is None:
            setattr(ff.meta, f"object_fiber{f}", "BLOCKED")
    return ff


def prepare_stage1_cals(
    output_dir: str | Path,
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    extraction_aperture: float = 8.0,
    wavesol=None,
    fiberflat_mode: str = FIBERFLAT_IDENTITY,
    paint_matched_fiberflat: bool | None = None,
) -> dict:
    """Build or reuse sim-HDF traces; attach WAVE and a recovery fiberflat.

    Parameters
    ----------
    fiberflat_mode : {"identity", "paint_matched", "sim_hdf"}, optional
        Default ``"identity"`` attaches :func:`build_identity_fiberflat` so
        blaze correction does not reshape the continuum (needed for ≪1 m/s
        CCF bias with LFC WAVE on PRF-painted rate maps). ``"paint_matched"``
        uses :func:`build_paint_matched_fiberflat` (continuum cancel, still
        ~15–20 m/s CCF bias). ``"sim_hdf"`` keeps the UTR/Stage-0 fiberflat
        from :func:`~hispecdrp.simulations.e2e_pipeline.prepare_pipeline_example_stage1`.
    paint_matched_fiberflat : bool, optional
        Deprecated alias: ``True`` → ``"paint_matched"``, ``False`` →
        ``"sim_hdf"``. Ignored when ``fiberflat_mode`` is passed explicitly
        and this is left as ``None``.
    """
    from .e2e_pipeline import prepare_pipeline_example_stage1

    if paint_matched_fiberflat is not None:
        fiberflat_mode = (
            FIBERFLAT_PAINT_MATCHED if paint_matched_fiberflat else FIBERFLAT_SIM_HDF
        )
    mode = str(fiberflat_mode).strip().lower()
    if mode not in {
        FIBERFLAT_IDENTITY,
        FIBERFLAT_PAINT_MATCHED,
        FIBERFLAT_SIM_HDF,
    }:
        raise ValueError(
            f"fiberflat_mode must be one of "
            f"{FIBERFLAT_IDENTITY!r}, {FIBERFLAT_PAINT_MATCHED!r}, "
            f"{FIBERFLAT_SIM_HDF!r}; got {fiberflat_mode!r}"
        )

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
    bundle["wavesol"] = wavesol if wavesol is not None else geometric_hdf_wavesol(
        spectrograph,
        meta={"mjd_start": resolve_e2e_mjd_start(None)},
    )
    out = Path(output_dir)
    cal_dir = out / "cals"
    cal_dir.mkdir(parents=True, exist_ok=True)
    if mode == FIBERFLAT_IDENTITY:
        ff_path = cal_dir / (
            f"{spectrograph.lower()}.identity_fiberflat_fib{int(fiber)}.fits"
        )
        if ff_path.is_file():
            bundle["fiberflat"] = datamodels.open(str(ff_path))
        else:
            bundle["fiberflat"] = build_identity_fiberflat(
                spectrograph=spectrograph,
                fiber=int(fiber),
                wavesol=bundle["wavesol"],
            )
            bundle["fiberflat"].save(filepath=str(ff_path))
        bundle["fiberflat_source"] = "identity_ones"
    elif mode == FIBERFLAT_PAINT_MATCHED:
        ff_path = cal_dir / (
            f"{spectrograph.lower()}.paint_matched_fiberflat_fib{int(fiber)}"
            f"_ap{float(extraction_aperture):.1f}.fits"
        )
        if ff_path.is_file():
            bundle["fiberflat"] = datamodels.open(str(ff_path))
        else:
            bundle["fiberflat"] = build_paint_matched_fiberflat(
                spectrograph=spectrograph,
                fiber=int(fiber),
                extraction_aperture=float(extraction_aperture),
                tracepos=bundle["trace_pos"],
            )
            bundle["fiberflat"].save(filepath=str(ff_path))
        bundle["fiberflat_source"] = "paint_matched_prf_blaze"
    else:
        bundle["fiberflat_source"] = "sim_hdf_stage1"
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
    spectrum_wavelengths=None,
    spectrum_flux=None,
    wavesol=None,
    fiberflat_mode: str = FIBERFLAT_IDENTITY,
    **stellar_rv_overrides,
) -> tuple[float, float, float]:
    """Native rate map → Stage 1 → blaze → ``recover_stellar_rv``.

    Pass ``wavesol`` to use an externally derived WAVE (e.g. from a simulated
    LFC) instead of the geometric HDF deposit-inverse solution. Stage-1 cals
    default to an identity fiberflat so blaze correction does not reshape the
    continuum (see :func:`prepare_stage1_cals`).
    """
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
            spectrum_wavelengths=spectrum_wavelengths,
            spectrum_flux=spectrum_flux,
        )
        stage1_cals = prepare_stage1_cals(
            parent,
            rate,
            spectrograph=spectrograph,
            fiber=fiber,
            extraction_aperture=extraction_aperture,
            wavesol=wavesol,
            fiberflat_mode=fiberflat_mode,
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
            **stellar_rv_overrides,
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
    spectrum_wavelengths=None,
    spectrum_flux=None,
    wavesol=None,
    fiberflat_mode: str = FIBERFLAT_IDENTITY,
    **stellar_rv_overrides,
) -> tuple[float, float, float]:
    """Rate map → UTR ramp → Stage 0 → Stage 1 → blaze → ``recover_stellar_rv``.

    When ``cleanup`` is True (default), the per-RV ramp / Stage 0 / Stage 1
    working directory is deleted after the recovered RV is in memory. Shared
    Stage-1 masters under ``output_dir`` are kept for reuse.

    Pass ``wavesol`` to use an externally derived WAVE (e.g. from a simulated
    LFC) instead of the geometric HDF deposit-inverse solution. Stage-1 cals
    default to an identity fiberflat (see :func:`prepare_stage1_cals`).
    """
    from .simulate_up_the_ramp import create_ramp_from_spectrum, run_stage0_simulation

    parent = Path(output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / f"ramp_{rv_km_s:+.1f}kms".replace("+", "p").replace("-", "m")
    try:
        with quiet_pipeline_logging():
            if spectrum_wavelengths is not None and spectrum_flux is not None:
                fiber_spectra = {int(fiber): (spectrum_wavelengths, spectrum_flux)}
                paint_rv = 0.0
            else:
                fiber_spectra = None
                paint_rv = float(rv_km_s)
            ramp, cals, rate = create_ramp_from_spectrum(
                str(run_dir),
                spectrograph=spectrograph,
                fibers=[int(fiber)],
                template_filename=template,
                fiber_spectra=fiber_spectra,
                mag_ab=mag_ab,
                filter_band=filter_band,
                radial_velocity_km_s=paint_rv,
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
                wavesol=wavesol,
                fiberflat_mode=fiberflat_mode,
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
                **stellar_rv_overrides,
            )
    finally:
        if cleanup:
            _remove_run_dir(run_dir)
