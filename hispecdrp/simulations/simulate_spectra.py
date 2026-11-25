"""
Simulated L2 spectra from stellar templates on the instrument wavelength grid.

Doppler-shift a rest-frame template onto native echelle WAVE with the same Lanczos
kernel as ``ccf_line_mask``, for noiseless injection / RV validation workflows.

Also spectrum-level effects applied to an input spectrum before it is simulated on the detector:
`apply_radial_velocity`, `apply_tellurics` and `add_sky_background`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.units import Quantity

from .. import datamodels
from ..datamodels import hispec_config
from ..driver.logging import quiet_pipeline_logging
from ..primitives.utils.ccf import doppler_paint_lanczos
from ..primitives.utils.ccf.ccf_utils import prepare_template_for_doppler_paint
from ..primitives.utils.spectral_utils import get_spectrum_path
from ..utils.spectral_templates_utils import load_template_from_path
from ..utils.line_mask_utils import (
    CCFOPT_SINGLE_LINE_STELLAR_RV_KW,
    IDEAL_SPECTRUM_STELLAR_RV_KW,
    RELAXED_TEMPLATE_MASK_BUILD_KW,
    build_ccfopt_line_mask_from_l2_line_by_line_rv,
    build_line_mask_from_template,
    paired_line_mask_path_for_template,
    read_mask_generation_meta,
)
from ..primitives import StellarRVPrimitive
from .e2e_gaia import resolve_e2e_mjd_start
from .e2e_support import templates_data_dir
from .simulate_rate_map import SimulateRateMap, _upsample_spectrum

__all__ = [
    "add_sky_background",
    "apply_radial_velocity",
    "apply_tellurics",
    "build_ccfopt_mask",
    "build_ccfopt_mask_lsf_matched",
    "ensure_lsf_ccfopt_mask",
    "build_l2",
    "build_l2_lsf_matched",
    "build_l2_and_recover_rv",
    "recover_stellar_rv",
    "IDEAL_SPECTRUM_STELLAR_RV_KW",
    "stellar_rv_kwargs_for_ideal_spectrum",
]


def apply_radial_velocity(wavelength: Quantity, rv: Quantity) -> Quantity:
    """
    Doppler-shift wavelengths by rv: wavelength * (1 + rv / c).

    Apply it BEFORE tellurics, sky and throughput, which are in the observer frame.
    """
    return wavelength * (1 + (rv / const.c).to(u.dimensionless_unscaled))


def apply_tellurics(wavelength: Quantity, flux: Quantity) -> Quantity:
    """Placeholder for atmospheric absorption effects."""
    raise NotImplementedError("Telluric application not yet implemented.")


def add_sky_background(
    wavelength: Quantity | None = None,
    flux: Quantity | None = None,
    sky_filename: str = 'mk_skybg_zm_1.0_1.0_ph.dat',
) -> tuple[Quantity, Quantity]:
    """
    Add the sky background to a spectrum, or return the sky alone.

    Apply it AFTER tellurics but BEFORE throughput.

    Parameters
    ----------
    wavelength, flux : Quantity or None
        Spectrum to add the sky to. The sky is interpolated onto wavelength and converted to the
        unit of flux. If None, the sky spectrum itself is returned, upsampled to R >= 3e5.
    sky_filename : str
        Sky background file in the atmosphere data directory.

    Returns
    -------
    wavelength, flux : Quantity
    """
    if sky_filename is None:
        raise ValueError("Sky background filename must be provided.")

    from . import e2e_support
    from ..utils.spectral_templates_utils import get_skybackground_path, load_sky_spectrum

    sky_template_path = get_skybackground_path(sky_filename, base_dir=e2e_support.atmosphere_data_dir())
    sky_wl, sky_flux = load_sky_spectrum(sky_template_path)

    if wavelength is None:
        return _upsample_spectrum(sky_wl, sky_flux, desired_sampling_resolution=3e5)

    sky_flux = sky_flux.to(flux.unit)
    sky_flux_interp = np.interp(wavelength.value, sky_wl.value, sky_flux.value) * sky_flux.unit
    return wavelength, flux + sky_flux_interp


@dataclass
class _PaintContext:
    wave_2d: np.ndarray
    w_dense_per_order: tuple[np.ndarray, ...]
    f_dense_per_order: tuple[np.ndarray, ...]
    orders: tuple[int, ...]


def _spectrograph_wavelength_clip_um(
    spectrograph: str,
    fiber: int,
    *,
    order_start: int | None = None,
    order_end: int | None = None,
    padding_um: float = 0.01,
) -> tuple[float, float]:
    """Vacuum-µm clip spanning echelle orders (± padding) for template line search."""
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph, order_start, order_end)
    ost, oen = orders[0], orders[-1]
    ws_all = SimulateRateMap._instrument_wavelength_solution(spectrograph)[int(fiber)]
    all_orders = sorted(int(o) for o in ws_all.keys())
    orders = [o for o in all_orders if ost <= o <= oen]
    if not orders:
        raise ValueError(
            f"No echelle orders in [{ost}, {oen}] for "
            f"spectrograph={spectrograph!r} fiber={fiber}"
        )
    lam = np.concatenate(
        [np.asarray(ws_all[o]["wavelength"], dtype=np.float64) for o in orders]
    )
    return float(np.nanmin(lam)) - padding_um, float(np.nanmax(lam)) + padding_um


@lru_cache(maxsize=16)
def _paint_context(
    template_path: str,
    spectrograph: str,
    fiber: int,
    order_start: int,
    order_end: int,
) -> _PaintContext:
    ws_all = SimulateRateMap._instrument_wavelength_solution(spectrograph)[fiber]
    all_orders = sorted(int(o) for o in ws_all.keys())
    orders = [o for o in all_orders if order_start <= o <= order_end]
    if not orders:
        raise ValueError(
            f"No echelle orders in [{order_start}, {order_end}] for "
            f"spectrograph={spectrograph!r} fiber={fiber}"
        )

    n_pix = int(np.asarray(ws_all[orders[0]]["wavelength"]).size)
    wave_2d = np.zeros((len(orders), n_pix), dtype=np.float64)
    w_obs_inc_rows: list[np.ndarray] = []
    for i, od in enumerate(orders):
        w = np.asarray(ws_all[od]["wavelength"], dtype=np.float64)
        if w.size != n_pix:
            raise ValueError(f"order {od}: expected {n_pix} pixels, got {w.size}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"order {od}: wavelength grid contains non-finite values")
        w_obs_inc = np.unique(w)
        if w_obs_inc.size < 2:
            raise ValueError(f"order {od}: wavelength grid has no span")
        wave_2d[i] = w
        w_obs_inc_rows.append(w_obs_inc)

    w_native, f_native = load_template_from_path(template_path)
    order = np.argsort(w_native)
    w_native = np.asarray(w_native, dtype=np.float64).ravel()[order]
    f_native = np.asarray(f_native, dtype=np.float64).ravel()[order]

    w_dense: list[np.ndarray] = []
    f_dense: list[np.ndarray] = []
    for i in range(wave_2d.shape[0]):
        wd, fd = prepare_template_for_doppler_paint(
            w_native, f_native, w_obs_inc_rows[i]
        )
        w_dense.append(wd)
        f_dense.append(fd)

    return _PaintContext(
        wave_2d=wave_2d,
        w_dense_per_order=tuple(w_dense),
        f_dense_per_order=tuple(f_dense),
        orders=tuple(orders),
    )


def build_l2(
    rv_km_s: float,
    *,
    spectrograph: str = "BSPEC",
    template: str,
    fiber: int = 1,
    order_start: int | None = None,
    order_end: int | None = None,
    mask_ccf_pixel_interp_order: int | None = None,
    templates_base_dir: str | None = None,
    spec_err_floor: float = 1.0e-6,
    spec_err_relative: float = 1.0e-6,
    apply_lsf: bool = False,
    lsf_model=None,
    mjd_start: float | None = None,
) -> datamodels.L2Model:
    """
    Build a noiseless L2 with a Doppler-shifted template on the instrument grid.

    The first call for a given ``(template, spectrograph, fiber, order range)``
    pays a one-time cost to load the WAVE solution, read the template, and
    densify it per echelle order (``prepare_template_for_doppler_paint``).
    Subsequent calls with different ``rv_km_s`` reuse that cache and only run
    the Lanczos paint.

    Parameters
    ----------
    rv_km_s
        Injected stellar RV in km/s.
    spectrograph
        HISPEC spectrograph name (e.g. ``"BSPEC"``).
    template
        Template basename under ``templates_base_dir`` (e.g. BT-Settl ``*.h5``)
        or an absolute path to a template file.
    fiber
        Fiber index for ``wave_fib*`` / ``spec_fib*`` columns.
    order_start, order_end
        Inclusive echelle order range. When omitted, all orders in the
        wavelength solution are used.
    mask_ccf_pixel_interp_order
        Lanczos-a order passed to :func:`doppler_paint_lanczos` (default 3,
        matching ``ccf_line_mask``).
    templates_base_dir
        Directory for bundled templates; default
        :func:`~hispecdrp.simulations.e2e_support.templates_data_dir`.
    spec_err_floor, spec_err_relative
        Near-noiseless error bars:
        ``max(spec_err_relative * |flux|, spec_err_floor)``.
    apply_lsf
        If True (and ``lsf_model`` is None), convolve each order with the HDF
        PSF→1-D LSF after paint so the L2 matches instrument-apparent line
        shapes (for ccfopt masks and fidelity vs extracted spectra).
    lsf_model
        Optional :class:`~hispecdrp.datamodels.gausslsf.GaussianLSFModel` from
        :class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`.
        When provided, this empirical LFC LSF is used instead of the HDF PSF
        library (``apply_lsf`` is implied).
    mjd_start
        Observation start MJD. Default is the E2E / UTR simulation epoch
        (:data:`~hispecdrp.simulations.e2e_gaia.DEFAULT_E2E_MJD_START`) so
        datamodel init does not warn and invent wall-clock time.

    Returns
    -------
    datamodels.L2Model
        In-memory L2 with one spectrum row per echelle order.
    """
    base_dir = templates_data_dir() if templates_base_dir is None else templates_base_dir
    template_path = get_spectrum_path(template, base_dir=base_dir)

    orders = hispec_config.HISPECConfig.get_order_list(spectrograph, order_start, order_end)
    ost, oen = orders[0], orders[-1]

    if mask_ccf_pixel_interp_order is None:
        mask_ccf_pixel_interp_order = int(
            IDEAL_SPECTRUM_STELLAR_RV_KW["mask_ccf_pixel_interp_order"]
        )

    ctx = _paint_context(template_path, spectrograph, int(fiber), ost, oen)
    inj_m_s = float(rv_km_s) * 1000.0
    spec = np.empty_like(ctx.wave_2d, dtype=np.float64)
    for i in range(ctx.wave_2d.shape[0]):
        spec[i] = doppler_paint_lanczos(
            ctx.wave_2d[i],
            ctx.w_dense_per_order[i],
            ctx.f_dense_per_order[i],
            inj_m_s,
            order=int(mask_ccf_pixel_interp_order),
        )

    if lsf_model is not None:
        from .lsf import convolve_l2_with_gaussian_lsf_model

        spec = convolve_l2_with_gaussian_lsf_model(
            ctx.wave_2d,
            spec,
            lsf_model,
            fiber=int(fiber),
        )
    elif apply_lsf:
        from .lsf import convolve_l2_with_hdf_lsf

        spec = convolve_l2_with_hdf_lsf(
            ctx.wave_2d,
            spec,
            spectrograph=spectrograph,
            fiber=int(fiber),
            order_start=ost,
        )

    model = datamodels.L2Model(
        meta={
            "order_start": ost,
            "order_end": oen,
            "spectrograph": spectrograph,
            "instrument_name": "HISPEC",
            "mjd_start": resolve_e2e_mjd_start(mjd_start),
        }
    )
    model.meta.order_start = ost
    model.meta.order_end = oen
    fiber_s = str(fiber)
    setattr(model, f"wave_fib{fiber_s}", ctx.wave_2d.copy())
    setattr(model, f"spec_fib{fiber_s}", spec.astype(np.float32))
    err = np.maximum(float(spec_err_relative) * np.abs(spec), float(spec_err_floor)).astype(
        np.float32
    )
    setattr(model, f"spec_err_fib{fiber_s}", err)
    setattr(
        model,
        f"spec_dq_fib{fiber_s}",
        np.zeros(ctx.wave_2d.shape, dtype=np.uint32),
    )
    return model


def build_l2_lsf_matched(
    rv_km_s: float,
    *,
    spectrograph: str = "BSPEC",
    template: str,
    fiber: int = 1,
    **kwargs,
) -> datamodels.L2Model:
    """Paint an ideal L2 then apply the HDF detector LSF (see ``apply_lsf=True``)."""
    return build_l2(
        rv_km_s,
        spectrograph=spectrograph,
        template=template,
        fiber=fiber,
        apply_lsf=True,
        **kwargs,
    )


def build_ccfopt_mask(
    template: str,
    output_dir: str | Path,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    order_start: int | None = None,
    order_end: int | None = None,
    rebuild_seed: bool = True,
    templates_base_dir: str | None = None,
    apply_lsf: bool = False,
    lsf_model=None,
    mask_suffix: str | None = None,
    l2: datamodels.L2Model | None = None,
    **ccfopt_overrides,
) -> tuple[str, str, datamodels.L2Model]:
    """
    Build a CCF-opt delta mask from a template via seed detection, ideal L2, and line-by-line RV.

    Runs gradient seed detection on the 1D template, paints an injected-RV=0 L2 with
    :func:`build_l2` (unless ``l2`` is provided), then calls
    :func:`~hispecdrp.utils.line_mask_utils.build_ccfopt_line_mask_from_l2_line_by_line_rv`
    with ccfopt line-by-line defaults.

    When ``lsf_model`` is provided (ProcessLFC ``GaussianLSFModel``), the L2 is
    convolved with that empirical LSF and the default product name is
    ``<stem>.lfc_lsf.ccfopt.delta.mas``. When ``apply_lsf=True`` without
    ``lsf_model``, the HDF PSF→1-D LSF is used and the default name is
    ``<stem>.lsf.ccfopt.delta.mas``. Pass a prebuilt ``l2`` (e.g. extract-matched
    under LFC WAVE) with ``mask_suffix=".extract_matched.ccfopt.delta.mas"`` to
    skip painting; ``apply_lsf`` / ``lsf_model`` are ignored in that case.

    Returns ``(seed_mask_path, ccfopt_mask_path, l2_at_zero_rv)``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = templates_data_dir() if templates_base_dir is None else templates_base_dir
    template_path = get_spectrum_path(template, base_dir=base_dir)
    stem = Path(template).stem

    seed_mask = out_dir / f"{stem}.delta.mas"
    if mask_suffix is None:
        if l2 is not None:
            mask_suffix = ".extract_matched.ccfopt.delta.mas"
        elif lsf_model is not None:
            mask_suffix = ".lfc_lsf.ccfopt.delta.mas"
        elif apply_lsf:
            mask_suffix = ".lsf.ccfopt.delta.mas"
        else:
            mask_suffix = ".ccfopt.delta.mas"
    if not str(mask_suffix).startswith("."):
        mask_suffix = "." + str(mask_suffix)
    ccfopt_mask = out_dir / f"{stem}{mask_suffix}"
    wl_min, wl_max = _spectrograph_wavelength_clip_um(
        spectrograph,
        fiber,
        order_start=order_start,
        order_end=order_end,
    )

    if rebuild_seed or not seed_mask.is_file():
        build_line_mask_from_template(
            str(template_path),
            wl_min_um=wl_min,
            wl_max_um=wl_max,
            output_path=str(seed_mask),
            masks_directory=str(out_dir),
            **RELAXED_TEMPLATE_MASK_BUILD_KW,
        )

    if l2 is None:
        l2 = build_l2(
            0.0,
            spectrograph=spectrograph,
            template=template,
            fiber=fiber,
            order_start=order_start,
            order_end=order_end,
            templates_base_dir=templates_base_dir,
            apply_lsf=bool(apply_lsf) and lsf_model is None,
            lsf_model=lsf_model,
        )
    ccfopt_path, _ = build_ccfopt_line_mask_from_l2_line_by_line_rv(
        str(seed_mask),
        l2,
        fiber=fiber,
        template_path=str(template_path),
        output_path=str(ccfopt_mask),
        masks_directory=str(out_dir),
        **ccfopt_overrides,
    )
    return str(seed_mask), ccfopt_path, l2


def ensure_lsf_ccfopt_mask(
    template: str,
    output_dir: str | Path,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    min_lines: int = 100,
    **build_kwargs,
) -> str:
    """
    Return a full-band ``*.lsf.ccfopt.delta.mas`` path, building it if needed.

    The packaged ``lte040...lsf.ccfopt.delta.mas`` is a two-order smoke file
    (``n_lines=18``). Ideal-spectrum recovery needs the full echelle product
    written by :func:`build_ccfopt_mask` with ``apply_lsf=True``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{Path(template).stem}.lsf.ccfopt.delta.mas"
    meta = read_mask_generation_meta(str(dest)) if dest.is_file() else None
    n_lines = int((meta or {}).get("n_lines") or 0)
    if dest.is_file() and n_lines >= int(min_lines):
        return str(dest)
    _, path, _ = build_ccfopt_mask(
        template,
        out_dir,
        spectrograph=spectrograph,
        fiber=fiber,
        apply_lsf=True,
        **build_kwargs,
    )
    return path


def build_ccfopt_mask_lsf_matched(
    template: str,
    output_dir: str | Path,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    lsf_model=None,
    **kwargs,
) -> tuple[str, str, datamodels.L2Model]:
    """Build an LSF-matched ``*.ccfopt.delta.mas`` from an ideal L2.

    Pass ``lsf_model`` (ProcessLFC ``GaussianLSFModel``) for an empirical LFC
    LSF product (``*.lfc_lsf.ccfopt.delta.mas``). Otherwise uses the HDF PSF
    library (``*.lsf.ccfopt.delta.mas``).
    """
    return build_ccfopt_mask(
        template,
        output_dir,
        spectrograph=spectrograph,
        fiber=fiber,
        apply_lsf=lsf_model is None,
        lsf_model=lsf_model,
        **kwargs,
    )


def stellar_rv_kwargs_for_ideal_spectrum(
    bulk_rv_km_s: float,
    *,
    template: str,
    mask_path: str | None = None,
    mask_style: str = "ccfopt.delta",
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    order_start: int | None = None,
    order_end: int | None = None,
    **overrides,
) -> dict:
    """
    Keyword arguments for :class:`~hispecdrp.primitives.StellarRVPrimitive` on ideal L2 data.

    Uses :data:`~hispecdrp.utils.line_mask_utils.IDEAL_SPECTRUM_STELLAR_RV_KW`
    (ccfopt line-by-line defaults).  With ``window_line_by_line=True``, the primitive
    also upgrades wide default ``window_size_um`` / velocity grid to the same ccfopt
    settings when those kwargs are omitted here.
    """
    kw = dict(IDEAL_SPECTRUM_STELLAR_RV_KW)
    inj_m_s = float(bulk_rv_km_s) * 1000.0
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph, order_start, order_end)
    if mask_path is None:
        mask_path = paired_line_mask_path_for_template(template, mask_style=mask_style)
    pars = {
        "algorithm": "ccf_line_mask",
        "mask_path": str(mask_path),
        "ccf_input_velocity": inj_m_s,
        "do_fibers": [str(fiber)],
        "do_orders": orders,
        "window_line_by_line": bool(kw["window_line_by_line"]),
        "window_size_um": float(kw["window_size_um"]),
        "v_half_window": float(kw["v_half_window_m_s"]),
        "v_step_coarse": float(kw["v_step_m_s"]),
        "mask_ccf_pixel_interp_order": int(kw["mask_ccf_pixel_interp_order"]),
        "ccf_zb": 0.0,
        "save_result": False,
        "save_preview": False,
        "show_preview": False,
    }
    pars.update(overrides)
    return pars


def recover_stellar_rv(
    l2: datamodels.L2Model,
    bulk_rv_km_s: float,
    *,
    template: str,
    mask_path: str | None = None,
    quiet: bool = True,
    **stellar_rv_overrides,
) -> tuple[float, float, float]:
    """
    Recover stellar RV from an ideal L2; return (injected_m_s, recovered_m_s, rv_err_m_s).
    """
    inj_m_s = float(bulk_rv_km_s) * 1000.0
    rv_kw = stellar_rv_kwargs_for_ideal_spectrum(
        bulk_rv_km_s,
        template=template,
        mask_path=mask_path,
        **stellar_rv_overrides,
    )
    if quiet:
        ctx = quiet_pipeline_logging()
    else:
        from contextlib import nullcontext

        ctx = nullcontext()
    with ctx:
        out = StellarRVPrimitive.call(l2, **rv_kw)
    fiber = int(rv_kw["do_fibers"][0])
    avg = getattr(out, f"rvccfavg_fib{fiber}")
    rec_m_s = float(avg["RV"][0])
    err_m_s = float(avg["RV_ERR"][0])
    return inj_m_s, rec_m_s, err_m_s


def build_l2_and_recover_rv(
    bulk_rv_km_s: float,
    *,
    template: str,
    spectrograph: str = "BSPEC",
    fiber: int = 1,
    order_start: int | None = None,
    order_end: int | None = None,
    mask_path: str | None = None,
    quiet: bool = True,
    **stellar_rv_overrides,
) -> tuple[datamodels.L2Model, float, float, float]:
    """
    Paint template onto BSPEC WAVE and recover RV (ideal-spectrum default mode).

    Returns ``(l2, injected_m_s, recovered_m_s, rv_err_m_s)``.
    """
    paint_order = stellar_rv_overrides.pop(
        "mask_ccf_pixel_interp_order",
        IDEAL_SPECTRUM_STELLAR_RV_KW["mask_ccf_pixel_interp_order"],
    )
    l2 = build_l2(
        bulk_rv_km_s,
        spectrograph=spectrograph,
        template=template,
        fiber=fiber,
        order_start=order_start,
        order_end=order_end,
        mask_ccf_pixel_interp_order=int(paint_order),
    )
    inj_m_s, rec_m_s, err_m_s = recover_stellar_rv(
        l2,
        bulk_rv_km_s,
        template=template,
        mask_path=mask_path,
        quiet=quiet,
        spectrograph=spectrograph,
        fiber=fiber,
        order_start=order_start,
        order_end=order_end,
        **stellar_rv_overrides,
    )
    return l2, inj_m_s, rec_m_s, err_m_s
