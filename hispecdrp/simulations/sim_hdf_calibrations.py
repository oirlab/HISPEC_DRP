"""
Build Stage-1 calibration products for infinite-S/N RV validation.

Trace positions are derived from simulated fiber-flat rate maps (same detector
orientation as :class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`),
and wavelength solutions from the pyechelle HDF model. This avoids
ProcessFiberflatsPipeline trace-walker offsets while keeping extraction aligned
with science rate maps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .. import datamodels
from ..datamodels import hispec_config
from ..datamodels.fiberflat import FiberflatModel
from ..datamodels.lfcwavesol import LFCWavesolModel
from ..datamodels.tracepos import TracePosModel
from ..datamodels.traceprof import TraceProfModel
from .simulate_rate_map import SimulateRateMap, read_pyechelle_hdf

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE1_TRACE_SOURCE_SIM_HDF",
    "get_sim_hdf_trace_grids",
    "create_tracepos_model_from_sim_hdf",
    "create_tracepos_model_from_fiberflat_rates",
    "create_traceprof_model_from_sim_hdf",
    "create_wavelength_solution_model_from_sim_hdf",
    "create_fiberflat_model_from_rate_maps",
    "materialize_stage1_bundle_from_sim_hdf",
]

STAGE1_TRACE_SOURCE_SIM_HDF = "sim_hdf"

_DEFAULT_FWHM_PIX = 3.5
_TRACE_POLY_DEG = 2


def _order_list(spectrograph: str) -> list[int]:
    return list(hispec_config.HISPECConfig.get_order_list(spectrograph.upper()))


def _load_hdf_model_data(spectrograph: str) -> tuple[str, dict]:
    srm = SimulateRateMap()
    spectrograph = spectrograph.upper()
    model = srm.bspec_hdf_model if spectrograph == "BSPEC" else srm.rspec_hdf_model
    model_path = srm.hdf_file_directory + model
    return spectrograph, read_pyechelle_hdf(model_path)


def _orient_hdf_pixel_xy(
    ox: np.ndarray,
    oy: np.ndarray,
    *,
    nx: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map native HDF detector ``(x, y)`` to oriented rate-map ``(row, col)``.

    Matches ``SimulateRateMap.flip_rate_map_horizontal`` followed by
    ``rotate_rate_map_vertical`` (``rot90`` ``k=3``) with ``orders_are_vertical=True``
    extraction along spectral rows.
    """
    ox = np.asarray(ox, dtype=np.float64)
    oy = np.asarray(oy, dtype=np.float64)
    row = (nx - 1) - ox
    col = (nx - 1) - oy
    return row, col


def _interp_oriented_spectral_row(
    row: np.ndarray,
    values: np.ndarray,
    *,
    n_pix: int = 4096,
) -> np.ndarray:
    """Interpolate ``values`` onto spectral row indices ``0 .. n_pix-1``."""
    row = np.asarray(row, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(row) & np.isfinite(values)
    row = row[good]
    values = values[good]
    if row.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    sidx = np.argsort(row)
    rows = row[sidx]
    vals = values[sidx]
    ur, inv = np.unique(rows, return_inverse=True)
    if ur.size < 2:
        return np.full(n_pix, np.nan, dtype=np.float64)
    uvals = np.bincount(inv, weights=vals) / np.maximum(np.bincount(inv), 1)
    f = PchipInterpolator(ur, uvals, extrapolate=True)
    return f(np.arange(n_pix, dtype=np.float64))


def get_sim_hdf_trace_grids(
    spectrograph: str,
    *,
    n_pix: int = 4096,
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """
    Per-fiber, per-order detector grids from the HDF model.

    Returns ``{fiber: {order: {pixel, wavelength, y_pos}}}`` with ``pixel`` = ``0..n_pix-1``
    indexing the **spectral row** in the oriented rate map (same convention as
    ``orders_are_vertical=True`` extraction). ``y_pos[pixel]`` is the cross-dispersion
    column at that row.
    """
    spectrograph, model_data = _load_hdf_model_data(spectrograph)
    order_list = _order_list(spectrograph)
    nx = int(model_data.get("nx", n_pix))
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    pixel = np.arange(n_pix, dtype=np.float64)
    for fiber in (1, 2, 3):
        out[fiber] = {}
        fiber_key = f"fiber_{fiber}"
        for od in order_list:
            od_s = str(od)
            if od_s not in model_data.get(fiber_key, {}):
                continue
            block = model_data[fiber_key][od_s]
            ow = np.asarray(block["xy_wavelengths"], dtype=np.float64)
            ox = np.asarray(block["x"], dtype=np.float64)
            oy = np.asarray(block["y"], dtype=np.float64)
            good = np.isfinite(ox) & np.isfinite(oy) & np.isfinite(ow)
            if int(np.sum(good)) < 2:
                continue
            row, col = _orient_hdf_pixel_xy(ox[good], oy[good], nx=nx)
            wavelength = _interp_oriented_spectral_row(row, ow[good], n_pix=n_pix)
            y_pos = _interp_oriented_spectral_row(row, col, n_pix=n_pix)
            out[fiber][od] = {
                "pixel": pixel.copy(),
                "wavelength": wavelength.astype(np.float64),
                "y_pos": y_pos.astype(np.float64),
            }
    return out


def _fit_trace_poly_coeffs(y_row: np.ndarray, deg: int = _TRACE_POLY_DEG) -> np.ndarray:
    x = np.arange(y_row.size, dtype=np.float64)
    ok = np.isfinite(y_row)
    if int(np.sum(ok)) < deg + 1:
        return np.full(deg + 1, np.nan, dtype=np.float64)
    return np.polyfit(x[ok], y_row[ok], deg)


def create_tracepos_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
) -> TracePosModel:
    """TracePosModel with POS/FWHM rows taken from the HDF echelle model."""
    spectrograph = spectrograph.upper()
    grids = get_sim_hdf_trace_grids(spectrograph)
    order_list = _order_list(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    model = TracePosModel(config=hispec_config.HISPECConfig, meta=model_meta)
    n_pix = 4096
    pixel_arr = np.arange(n_pix, dtype=np.float64)
    for fiber in (1, 2, 3):
        rows = []
        coeff_pos = []
        coeff_fwhm = []
        for order in order_list:
            if order not in grids.get(fiber, {}):
                rows.append(np.full(n_pix, np.nan, dtype=np.float64))
                coeff_pos.append(np.full(_TRACE_POLY_DEG + 1, np.nan))
                coeff_fwhm.append(np.full(_TRACE_POLY_DEG + 1, np.nan))
                continue
            y_row = grids[fiber][order]["y_pos"]
            rows.append(y_row)
            coeff_pos.append(_fit_trace_poly_coeffs(y_row))
            coeff_fwhm.append(np.polyfit(pixel_arr, np.full(n_pix, _DEFAULT_FWHM_PIX), 1))
        pos = np.stack(rows, axis=0)
        fwhm = np.full_like(pos, _DEFAULT_FWHM_PIX, dtype=np.float64)
        setattr(model, f"coeff_pos_fib{fiber}", np.stack(coeff_pos, axis=0))
        setattr(model, f"coeff_fwhm_fib{fiber}", np.stack(coeff_fwhm, axis=0))
        setattr(model, f"pos_fib{fiber}", pos)
        setattr(model, f"fwhm_fib{fiber}", fwhm)
    return model


def create_tracepos_model_from_fiberflat_rates(
    fiberflat_rate_paths: list[str],
    *,
    spectrograph: str,
    meta: dict | None = None,
    prominence_factor: float = 0.02,
) -> TracePosModel:
    """
    TracePosModel from peak finding on simulated fiber-flat rate maps.

    Uses the same detector orientation as science simulations (post flip/rotate),
    unlike :func:`create_tracepos_model_from_sim_hdf` which reads raw HDF ``x/y``.
    """
    from ..primitives.trace_fitting.tracepos_primitive import TracePositionPrimitive
    from .simulate_up_the_ramp import (
        _DEFAULT_FIBERFLAT_DISP_RANGE,
        _DEFAULT_FIBERFLAT_XDISP_RANGE,
    )

    spectrograph = spectrograph.upper()
    models_by_fiber: dict[str, datamodels.DataModel] = {}
    for path in fiberflat_rate_paths:
        model = datamodels.open(path)
        base = str(getattr(model.meta, "filename", path)).upper()
        fib = None
        for f in (1, 2, 3):
            if f"FIB{f}" in base or f"FIBERFLAT-FIB{f}" in base:
                fib = str(f)
                break
        if fib is None:
            logger.warning("Skipping fiberflat rate %s (fiber not inferred from name).", path)
            continue
        models_by_fiber[fib] = model
    if not models_by_fiber:
        raise ValueError("No fiberflat rate paths with inferable fiber labels.")
    prim = TracePositionPrimitive()
    prim.init_args(
        deg_pos=4,
        deg_width=2,
        disp_range=_DEFAULT_FIBERFLAT_DISP_RANGE.get(spectrograph, (500, 4000)),
        xdisp_range=_DEFAULT_FIBERFLAT_XDISP_RANGE.get(spectrograph, (100, 4050)),
        prominence_factor=prominence_factor,
        output_model=TracePosModel,
    )
    trace_pos: TracePosModel = prim.apply(models_by_fiber)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    for key, val in model_meta.items():
        setattr(trace_pos.meta, key, val)
    logger.info(
        "Sim HDF trace_pos from fiberflat rates spectrograph=%s n_rate_files=%d",
        spectrograph,
        len(fiberflat_rate_paths),
    )
    return trace_pos


def create_traceprof_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
    detector_shape: tuple[int, int] = (4096, 4160),
) -> TraceProfModel:
    """Placeholder trace profiles (box extraction does not use them)."""
    spectrograph = spectrograph.upper()
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    model = TraceProfModel(config=hispec_config.HISPECConfig, meta=model_meta)
    ny, nx = detector_shape
    for fiber in (1, 2, 3):
        prof = np.ones((ny, nx), dtype=np.float64)
        err = np.zeros((ny, nx), dtype=np.float64)
        setattr(model, f"prof_fib{fiber}", prof)
        setattr(model, f"prof_err_fib{fiber}", err)
    return model


def create_wavelength_solution_model_from_sim_hdf(
    spectrograph: str,
    *,
    meta: dict | None = None,
) -> LFCWavesolModel:
    """LFCWavesolModel from the same HDF grids as :func:`create_tracepos_model_from_sim_hdf`."""
    spectrograph = spectrograph.upper()
    order_list = _order_list(spectrograph)
    grids = get_sim_hdf_trace_grids(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    wavesol = LFCWavesolModel(config=hispec_config.HISPECConfig, meta=model_meta)
    for fiber in (1, 2, 3):
        wave_rows = []
        for order in order_list:
            block = grids.get(fiber, {}).get(order)
            if block is None:
                break
            wave_rows.append(np.asarray(block["wavelength"], dtype=np.float64))
        if wave_rows:
            wave_arr = np.stack(wave_rows, axis=0)
            setattr(wavesol, f"wave_fib{fiber}", wave_arr)
            setattr(
                wavesol,
                f"spec_dq_fib{fiber}",
                np.zeros(wave_arr.shape, dtype=np.uint32),
            )
    return wavesol


def _extract_box_one_order(
    rate: np.ndarray,
    rate_err: np.ndarray,
    rate_dq: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
) -> tuple[np.ndarray, np.ndarray]:
    from ..primitives.spectral_extraction.extract_box import extract_spectrum_box

    rate = np.asarray(rate, dtype=np.float64)
    if rate_err.shape != rate.shape:
        rate_err = np.zeros_like(rate, dtype=np.float64)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
    if rate_dq.shape != rate.shape:
        rate_dq = np.zeros_like(rate, dtype=np.uint32)
    else:
        rate_dq = np.asarray(rate_dq, dtype=np.uint32)
    ny, nx = rate.shape
    spec = np.full(ny, np.nan, dtype=np.float64)
    err = np.full(ny, np.nan, dtype=np.float64)
    var_r = np.zeros_like(rate, dtype=np.float64)
    var_p = np.zeros_like(rate, dtype=np.float64)
    dq = np.zeros_like(rate, dtype=np.uint32)
    mid = np.zeros_like(rate, dtype=np.float64)
    extract_spectrum_box(
        input_data=rate,
        input_err=rate_err,
        input_dq=rate_dq,
        input_var_rnoise=var_r,
        input_var_poisson=var_p,
        input_mid_time=mid,
        trace_positions=trace_y,
        extraction_aperture=extraction_aperture,
        spec_output=spec,
        err_output=err,
        var_rnoise_output=var_r,
        var_poisson_output=var_p,
        dq_output=dq,
        mid_time_output=mid,
        orders_are_vertical=True,
    )
    return spec, err


def create_fiberflat_model_from_rate_maps(
    rate_paths: list[str],
    tracepos_model: TracePosModel,
    *,
    spectrograph: str,
    meta: dict | None = None,
    extraction_aperture: float = 10.0,
) -> FiberflatModel:
    """
    Box-extract simulated fiber-flat rate maps with HDF ``tracepos_model`` rows.

    Produces a ``FiberflatModel`` consistent with the science extraction geometry.
    """
    spectrograph = spectrograph.upper()
    order_list = _order_list(spectrograph)
    hdf_grids = get_sim_hdf_trace_grids(spectrograph)
    model_meta = {"spectrograph": spectrograph}
    if meta:
        model_meta.update(meta)
    ff = FiberflatModel(config=hispec_config.HISPECConfig, meta=model_meta)
    n_orders = len(order_list)
    n_pix = 4096

    for path in rate_paths:
        l0 = datamodels.open(path)
        fib = None
        base = str(getattr(l0.meta, "filename", path)).upper()
        for f in (1, 2, 3):
            if f"FIB{f}" in base or f"FIBERFLAT-FIB{f}" in base:
                fib = f
                break
        if fib is None:
            logger.warning("Skipping fiberflat rate %s (fiber not inferred from name).", path)
            continue
        pos = getattr(tracepos_model, f"pos_fib{fib}")
        rate = l0.rate
        rate_err = getattr(l0, "rate_err", None)
        if rate_err is None:
            rate_err = np.zeros_like(rate, dtype=np.float64)
        rate_dq = getattr(l0, "rate_dq", None)
        if rate_dq is None:
            rate_dq = getattr(l0, "dq", None)
        if rate_dq is None:
            rate_dq = np.zeros_like(rate, dtype=np.uint32)
        spec_rows = []
        err_rows = []
        wave_rows = []
        for i, _order in enumerate(order_list):
            if i >= pos.shape[0]:
                break
            trace_y = pos[i]
            spec, err = _extract_box_one_order(
                rate,
                rate_err,
                rate_dq,
                trace_y,
                extraction_aperture=extraction_aperture,
            )
            spec_rows.append(spec)
            err_rows.append(err)
            wave = hdf_grids.get(fib, {}).get(_order, {}).get("wavelength")
            if wave is None:
                wave = np.full(n_pix, np.nan, dtype=np.float64)
            wave_rows.append(np.asarray(wave, dtype=np.float64))
        if not spec_rows:
            continue
        spec_arr = np.stack(spec_rows, axis=0).astype(np.float32)
        err_arr = np.stack(err_rows, axis=0).astype(np.float32)
        setattr(ff, f"spec_fib{fib}", spec_arr)
        setattr(ff, f"spec_err_fib{fib}", err_arr)
        setattr(ff, f"spec_var_poisson_fib{fib}", np.zeros_like(spec_arr, dtype=np.float32))
        setattr(ff, f"spec_var_rnoise_fib{fib}", np.zeros_like(spec_arr, dtype=np.float32))
        setattr(ff, f"spec_dq_fib{fib}", np.zeros(spec_arr.shape, dtype=np.uint32))
        setattr(ff, f"wave_fib{fib}", np.stack(wave_rows, axis=0).astype(np.float64))

    return ff


def materialize_stage1_bundle_from_sim_hdf(
    *,
    spectrograph: str,
    fiberflat_rate_paths: list[str],
    meta: dict | None = None,
    extraction_aperture: float = 10.0,
) -> dict[str, Any]:
    """Return ``trace_pos``, ``trace_prof``, ``fiberflat``, and ``wavesol`` from HDF truth."""
    spectrograph = spectrograph.upper()
    trace_pos = create_tracepos_model_from_fiberflat_rates(
        fiberflat_rate_paths,
        spectrograph=spectrograph,
        meta=meta,
    )
    trace_prof = create_traceprof_model_from_sim_hdf(spectrograph, meta=meta)
    wavesol = create_wavelength_solution_model_from_sim_hdf(spectrograph, meta=meta)
    fiberflat = create_fiberflat_model_from_rate_maps(
        fiberflat_rate_paths,
        trace_pos,
        spectrograph=spectrograph,
        meta=meta,
        extraction_aperture=extraction_aperture,
    )
    logger.info(
        "Sim HDF Stage-1 bundle ready spectrograph=%s n_fiberflat_rates=%d extraction_aperture=%.1f",
        spectrograph,
        len(fiberflat_rate_paths),
        float(extraction_aperture),
    )
    return {
        "trace_pos": trace_pos,
        "trace_prof": trace_prof,
        "fiberflat": fiberflat,
        "wavesol": wavesol,
    }
