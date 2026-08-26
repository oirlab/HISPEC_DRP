"""1-D spectrum fidelity vs ideal ``build_l2`` (continuum-normalized)."""

from __future__ import annotations

import io
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from ...datamodels import hispec_config
from ...primitives.spectral_extraction.extract_sheared_sp import (
    extract_sheared_banded_ls,
)
from ...primitives.spectral_extraction.iso_lambda_shear import (
    create_trace_shear_from_sim_hdf,
)
from ..e2e_pipeline import (
    create_tracepos_model_from_sim_hdf,
    extract_spectrum_from_rate_map,
    get_sim_hdf_trace_grids,
)
from .stage1_extraction_noise import (
    build_profile_image,
    continuum_rate_from_spectral_smooth,
    extract_order_optimal,
)


def continuum_envelope(spec: np.ndarray, size: int = 151) -> np.ndarray:
    """Robust continuum envelope via median filter (shape-only comparisons)."""
    x = np.asarray(spec, dtype=np.float64)
    valid = np.isfinite(x) & (x > 0)
    if not np.any(valid):
        return np.ones_like(x)
    fill = float(np.nanmedian(x[valid]))
    y = np.where(valid, x, fill)
    size = min(int(size), max(3, (len(y) // 8) | 1))
    if size % 2 == 0:
        size += 1
    cont = median_filter(y, size=size, mode="nearest")
    return np.maximum(cont, np.nanpercentile(y[valid], 1))


def continuum_normalize_orders(spec_2d: np.ndarray, *, size: int = 151) -> np.ndarray:
    """Per-order continuum normalization for shape comparison to ``build_l2``."""
    spec_2d = np.asarray(spec_2d, dtype=np.float64)
    out = np.empty_like(spec_2d)
    for i in range(spec_2d.shape[0]):
        row = spec_2d[i]
        cont = continuum_envelope(row, size=size)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = row / cont
    return out


def extract_sheared_sp_from_rate_map(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    extraction_aperture: float = 8.0,
    orders: list[int] | None = None,
    continuum_smooth_pix: int = 81,
    shear: np.ndarray | None = None,
    rate_err: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Sheared banded-LS extract of all (or selected) orders from a 2-D rate map.

    Continuum ``P`` is built by spectral smoothing of ``rate_map`` inside each
    order aperture. Wavelength rows come from the HDF deposit-inverse grid
    (same as :func:`extract_spectrum_from_rate_map`).
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate_err is None:
        # Near-uniform weights; noiseless truth maps use this.
        rate_err = np.ones_like(rate)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)

    tracepos_model = create_tracepos_model_from_sim_hdf(spectrograph)
    tracepos = np.asarray(getattr(tracepos_model, f"pos_fib{fiber}"), dtype=np.float64)
    if shear is None:
        shear = create_trace_shear_from_sim_hdf(spectrograph, fiber=fiber)
    else:
        shear = np.asarray(shear, dtype=np.float64)

    order_offset = int(
        hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"]
    )
    order_list = list(
        orders
        if orders is not None
        else hispec_config.HISPECConfig.get_order_list(spectrograph)
    )
    grids = get_sim_hdf_trace_grids(spectrograph)
    if fiber not in grids:
        raise KeyError(f"No HDF trace grids for fiber {fiber}")

    wave_rows: list[np.ndarray] = []
    flux_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    kept: list[int] = []
    rate_var = np.maximum(rate_err, 1.0e-30) ** 2

    for od in order_list:
        row = int(od) - order_offset
        if row < 0 or row >= tracepos.shape[0]:
            continue
        g = grids[fiber].get(int(od))
        if g is None:
            continue
        ty = tracepos[row]
        cont = continuum_rate_from_spectral_smooth(
            rate,
            ty,
            extraction_aperture=float(extraction_aperture),
            smooth_pix=int(continuum_smooth_pix),
        )
        profile = build_profile_image(
            cont, ty, extraction_aperture=float(extraction_aperture)
        )
        extracted = extract_sheared_banded_ls(
            rate,
            rate_var,
            profile,
            ty,
            extraction_aperture=float(extraction_aperture),
            shear=shear[row],
        )
        kept.append(int(od))
        wave_rows.append(np.asarray(g["wavelength"], dtype=np.float64))
        flux_rows.append(np.asarray(extracted["spec"], dtype=np.float64))
        err_rows.append(np.asarray(extracted["err"], dtype=np.float64))

    if not kept:
        raise RuntimeError(
            f"No orders extracted for {spectrograph} fiber {fiber}"
        )
    return {
        "spectrograph": spectrograph,
        "fiber": fiber,
        "extraction_aperture": float(extraction_aperture),
        "method": "sheared_sp",
        "orders": kept,
        "wavelength": np.stack(wave_rows, axis=0),
        "flux": np.stack(flux_rows, axis=0),
        "flux_err": np.stack(err_rows, axis=0),
    }


def _profile_image_for_optimal(
    rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    profile_source: str,
    profile_rate_map: np.ndarray | None,
    continuum_smooth_pix: int,
) -> np.ndarray:
    """Resolve the 2-D spatial profile image ``P`` for Horne optimal extract."""
    source = str(profile_source).strip().lower()
    rate = np.asarray(rate, dtype=np.float64)
    if source in ("fiberflat_rate", "fiberflat", "flat"):
        if profile_rate_map is None:
            raise ValueError(
                "profile_rate_map is required when profile_source='fiberflat_rate'"
            )
        prof = np.asarray(profile_rate_map, dtype=np.float64)
        if prof.shape != rate.shape:
            raise ValueError(
                f"profile_rate_map shape {prof.shape} must match rate {rate.shape}"
            )
        return np.maximum(prof, 0.0)
    if source in ("unity", "placeholder"):
        return np.ones_like(rate, dtype=np.float64)
    if source in ("stellar_continuum", "continuum_stellar", "science_continuum"):
        cont = continuum_rate_from_spectral_smooth(
            rate,
            trace_y,
            extraction_aperture=float(extraction_aperture),
            smooth_pix=int(continuum_smooth_pix),
        )
        return build_profile_image(
            cont, trace_y, extraction_aperture=float(extraction_aperture)
        )
    raise ValueError(
        f"Unknown profile_source {profile_source!r}; use 'fiberflat_rate', "
        "'stellar_continuum', or 'unity'"
    )


def extract_optimal_from_rate_map(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    extraction_aperture: float = 8.0,
    orders: list[int] | None = None,
    continuum_smooth_pix: int = 81,
    shear: np.ndarray | None = None,
    rate_err: np.ndarray | None = None,
    optimal_method: str = "horne",
    profile_source: str = "fiberflat_rate",
    profile_rate_map: np.ndarray | None = None,
    use_shear: bool = False,
) -> dict[str, Any]:
    """
    Horne optimal extract of all (or selected) orders from a 2-D rate map.

    Spatial profile ``P`` should come from a **fiberflat rate map**
    (``profile_source='fiberflat_rate'``, default) — the same pattern as
    (``profile_source='fiberflat_rate'``, default) — the same pattern as
    production :class:`~hispecdrp.primitives.trace_fitting.traceprof_primitive.TraceProfilePrimitive`.
    Do **not** use the science spectrum for ``P`` (see ``stellar_continuum`` only
    for legacy/debug).

    Parameters
    ----------
    profile_source : str
        ``'fiberflat_rate'`` (default), ``'stellar_continuum'``, or ``'unity'``.
    profile_rate_map : ndarray or None
        2-D rate image for ``P`` when ``profile_source='fiberflat_rate'``.
    use_shear : bool
        Apply HDF iso-λ shear in sheared-gather Horne (default False for
        like-to-like comparison with box extract).
    optimal_method : str
        ``'horne'`` for classic / sheared-gather Horne (default).
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate_err is None:
        rate_err = np.ones_like(rate)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)

    tracepos_model = create_tracepos_model_from_sim_hdf(spectrograph)
    tracepos = np.asarray(getattr(tracepos_model, f"pos_fib{fiber}"), dtype=np.float64)
    shear_arr = None
    if use_shear:
        if shear is None:
            shear_arr = create_trace_shear_from_sim_hdf(spectrograph, fiber=fiber)
        else:
            shear_arr = np.asarray(shear, dtype=np.float64)

    order_offset = int(
        hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"]
    )
    order_list = list(
        orders
        if orders is not None
        else hispec_config.HISPECConfig.get_order_list(spectrograph)
    )
    grids = get_sim_hdf_trace_grids(spectrograph)
    if fiber not in grids:
        raise KeyError(f"No HDF trace grids for fiber {fiber}")

    wave_rows: list[np.ndarray] = []
    flux_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    kept: list[int] = []
    ny, nx = rate.shape
    dq = np.zeros((ny, nx), dtype=np.uint32)
    var_r = np.ones((ny, nx), dtype=np.float64)
    var_p = np.zeros((ny, nx), dtype=np.float64)

    for od in order_list:
        row = int(od) - order_offset
        if row < 0 or row >= tracepos.shape[0]:
            continue
        g = grids[fiber].get(int(od))
        if g is None:
            continue
        ty = tracepos[row]
        profile = _profile_image_for_optimal(
            rate,
            ty,
            extraction_aperture=float(extraction_aperture),
            profile_source=profile_source,
            profile_rate_map=profile_rate_map,
            continuum_smooth_pix=int(continuum_smooth_pix),
        )
        shear_row = shear_arr[row] if shear_arr is not None else None
        extracted = extract_order_optimal(
            rate,
            rate_err,
            dq,
            var_r,
            var_p,
            ty,
            profile,
            extraction_aperture=float(extraction_aperture),
            shear=shear_row,
            method=optimal_method,
        )
        kept.append(int(od))
        wave_rows.append(np.asarray(g["wavelength"], dtype=np.float64))
        flux_rows.append(np.asarray(extracted["spec"], dtype=np.float64))
        err_rows.append(np.asarray(extracted["err"], dtype=np.float64))

    if not kept:
        raise RuntimeError(
            f"No orders extracted for {spectrograph} fiber {fiber}"
        )
    return {
        "spectrograph": spectrograph,
        "fiber": fiber,
        "extraction_aperture": float(extraction_aperture),
        "method": f"optimal_{optimal_method}",
        "profile_source": str(profile_source),
        "use_shear": bool(use_shear),
        "orders": kept,
        "wavelength": np.stack(wave_rows, axis=0),
        "flux": np.stack(flux_rows, axis=0),
        "flux_err": np.stack(err_rows, axis=0),
    }


def extract_spectroperfection_from_rate_map(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    extraction_aperture: float = 8.0,
    orders: list[int] | None = None,
    profile_rate_map: np.ndarray | None = None,
    shear: np.ndarray | None = None,
    sigma_spec: np.ndarray | None = None,
    rate_err: np.ndarray | None = None,
    spectral_kernel: str = "bilinear",
) -> dict[str, Any]:
    """Calibration-driven spectroperfection extract (fiberflat P + LFC moments)."""
    from ...primitives.spectral_extraction.extract_spectroperfection import (
        extract_spectroperfection,
    )
    from ...primitives.spectral_extraction.iso_lambda_shear import (
        create_trace_shear_from_sim_hdf,
    )
    from ..e2e_pipeline import create_tracepos_model_from_sim_hdf, get_sim_hdf_trace_grids

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate_err is None:
        rate_err = np.ones_like(rate)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
    rate_var = np.maximum(rate_err, 1.0e-30) ** 2
    if profile_rate_map is None:
        raise ValueError("profile_rate_map (fiberflat rate) is required")
    prof = np.asarray(profile_rate_map, dtype=np.float64)
    if prof.shape != rate.shape:
        raise ValueError(
            f"profile_rate_map shape {prof.shape} must match rate {rate.shape}"
        )

    tracepos_model = create_tracepos_model_from_sim_hdf(spectrograph)
    tracepos = np.asarray(getattr(tracepos_model, f"pos_fib{fiber}"), dtype=np.float64)
    if shear is None:
        shear = create_trace_shear_from_sim_hdf(spectrograph, fiber=fiber)
    else:
        shear = np.asarray(shear, dtype=np.float64)

    order_offset = int(
        hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"]
    )
    order_list = list(
        orders
        if orders is not None
        else hispec_config.HISPECConfig.get_order_list(spectrograph)
    )
    grids = get_sim_hdf_trace_grids(spectrograph)
    if fiber not in grids:
        raise KeyError(f"No HDF trace grids for fiber {fiber}")

    wave_rows: list[np.ndarray] = []
    flux_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    kept: list[int] = []
    for od in order_list:
        row = int(od) - order_offset
        if row < 0 or row >= tracepos.shape[0]:
            continue
        g = grids[fiber].get(int(od))
        if g is None:
            continue
        ty = tracepos[row]
        shear_row = None if shear is None else np.asarray(shear[row], dtype=np.float64)
        sig_row = None
        if sigma_spec is not None:
            ss = np.asarray(sigma_spec, dtype=np.float64)
            sig_row = ss[row] if ss.ndim == 2 else ss
        extracted = extract_spectroperfection(
            rate,
            rate_var,
            prof,
            ty,
            extraction_aperture=float(extraction_aperture),
            shear=shear_row,
            sigma_spec=sig_row,
            spectral_kernel=str(spectral_kernel),
        )
        kept.append(int(od))
        wave_rows.append(np.asarray(g["wavelength"], dtype=np.float64))
        flux_rows.append(np.asarray(extracted["spec"], dtype=np.float64))
        err_rows.append(np.asarray(extracted["err"], dtype=np.float64))

    if not kept:
        raise RuntimeError(
            f"No orders extracted for {spectrograph} fiber {fiber}"
        )
    return {
        "spectrograph": spectrograph,
        "fiber": fiber,
        "extraction_aperture": float(extraction_aperture),
        "method": "spectroperfection",
        "orders": kept,
        "wavelength": np.stack(wave_rows, axis=0),
        "flux": np.stack(flux_rows, axis=0),
        "flux_err": np.stack(err_rows, axis=0),
    }


def extract_spectroperfectionism_from_rate_map(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    orders: list[int] | None = None,
    rate_err: np.ndarray | None = None,
    spectral_bandwidth: int | None = None,
    pix_range: tuple[int, int] | None = None,
    psf_rel_threshold: float = 1.0e-2,
) -> dict[str, Any]:
    """PSF-deposit spectroperfectionism extract from a noiseless / low-noise rate map.

    On 16 GB machines pass ``pix_range`` (e.g. ``(1800, 2056)``); full-order SP
    with ``256×256`` HDF stamps can exhaust RAM.
    """
    from ...primitives.spectral_extraction.extract_spectroperfectionism import (
        extract_order_spectroperfectionism_from_hdf,
    )

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    rate = np.asarray(rate_map, dtype=np.float64)
    if rate_err is None:
        rate_err = np.ones_like(rate)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
    rate_var = np.maximum(rate_err, 1.0e-30) ** 2

    order_list = list(
        orders
        if orders is not None
        else hispec_config.HISPECConfig.get_order_list(spectrograph)
    )
    grids = get_sim_hdf_trace_grids(spectrograph)
    if fiber not in grids:
        raise KeyError(f"No HDF trace grids for fiber {fiber}")

    wave_rows: list[np.ndarray] = []
    flux_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    kept: list[int] = []
    for od in order_list:
        g = grids[fiber].get(int(od))
        if g is None:
            continue
        extracted = extract_order_spectroperfectionism_from_hdf(
            rate,
            rate_var,
            spectrograph=spectrograph,
            fiber=fiber,
            order=int(od),
            wave=np.asarray(g["wavelength"], dtype=np.float64),
            spectral_bandwidth=spectral_bandwidth,
            pix_range=pix_range,
            psf_rel_threshold=psf_rel_threshold,
        )
        kept.append(int(od))
        wave_rows.append(np.asarray(g["wavelength"], dtype=np.float64))
        flux_rows.append(np.asarray(extracted["spec"], dtype=np.float64))
        err_rows.append(np.asarray(extracted["err"], dtype=np.float64))

    if not kept:
        raise RuntimeError(
            f"No orders extracted for {spectrograph} fiber {fiber}"
        )
    return {
        "spectrograph": spectrograph,
        "fiber": fiber,
        "method": "spectroperfectionism",
        "orders": kept,
        "pix_range": pix_range,
        "wavelength": np.stack(wave_rows, axis=0),
        "flux": np.stack(flux_rows, axis=0),
        "flux_err": np.stack(err_rows, axis=0),
    }


def extract_spectroperfectionism_order_chunked(
    rate_map: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    fiber: int = 2,
    order: int,
    chunk_pix: int = 128,
    rate_err: np.ndarray | None = None,
    psf_rel_threshold: float = 1.0e-2,
    spectral_bandwidth: int | None = None,
) -> dict[str, np.ndarray]:
    """Chunked SP extract for one order (memory-safe on 16 GB machines).

    ``rate_map`` must be built with the HDF **bilinear deposit** forward model
    (``create_native_rate_map(method='bilinear_weights')``), not ``prf_convolution``.
  """
    from ...primitives.spectral_extraction.extract_spectroperfectionism import (
        extract_order_spectroperfectionism_from_hdf,
    )

    rate = np.asarray(rate_map, dtype=np.float64)
    if rate_err is None:
        rate_err = np.maximum(rate, 1.0e-30)
    else:
        rate_err = np.asarray(rate_err, dtype=np.float64)
    rate_var = np.maximum(rate_err, 1.0e-30) ** 2

    grids = get_sim_hdf_trace_grids(str(spectrograph).upper())
    wave = np.asarray(grids[int(fiber)][int(order)]["wavelength"], dtype=np.float64)
    n_spec = int(wave.size)
    spec = np.full(n_spec, np.nan, dtype=np.float64)
    err = np.full(n_spec, np.nan, dtype=np.float64)
    step = max(16, int(chunk_pix))
    for i0 in range(0, n_spec, step):
        i1 = min(n_spec, i0 + step)
        extracted = extract_order_spectroperfectionism_from_hdf(
            rate,
            rate_var,
            spectrograph=str(spectrograph).upper(),
            fiber=int(fiber),
            order=int(order),
            wave=wave,
            spectral_bandwidth=spectral_bandwidth,
            pix_range=(i0, i1),
            psf_rel_threshold=float(psf_rel_threshold),
        )
        spec[i0:i1] = extracted["spec"][i0:i1]
        err[i0:i1] = extracted["err"][i0:i1]
    return {
        "wavelength": wave,
        "spec": spec,
        "err": err,
        "method": "spectroperfectionism_chunked",
        "chunk_pix": step,
    }


def robust_continuum_sigma(
    shape_residual: np.ndarray,
    ref_norm: np.ndarray,
    *,
    depth_threshold: float = 0.02,
) -> float:
    """MAD-based σ from shallow-continuum pixels (for shape residual significance)."""
    resid = np.asarray(shape_residual, dtype=np.float64)
    ref = np.asarray(ref_norm, dtype=np.float64)
    shallow = (1.0 - ref) < float(depth_threshold)
    good = shallow & np.isfinite(resid)
    if int(np.sum(good)) < 20:
        good = np.isfinite(resid)
    if int(np.sum(good)) < 5:
        return 1.0
    mad = float(np.median(np.abs(resid[good] - np.median(resid[good]))))
    return max(1.4826 * mad, 1.0e-8)


def compare_continuum_normalized_spectra(
    wave_test: np.ndarray,
    flux_test: np.ndarray,
    wave_ref: np.ndarray,
    flux_ref: np.ndarray,
    *,
    label: str = "test",
    continuum_size: int = 151,
    edge_trim: int = 50,
    min_good_pix: int = 100,
) -> dict[str, Any]:
    """
    Compare continuum-normalized 2-D spectra on a common pixel grid.

    Both cubes are continuum-normalized per order. ``flux_test`` is linearly
    interpolated onto ``wave_ref`` per order (assumes matching order rows).

    For short windows (e.g. SP ``pix_range``), pass a smaller ``continuum_size``
    and ``edge_trim``, and lower ``min_good_pix`` (e.g. 40).
    """
    wave_test = np.asarray(wave_test, dtype=np.float64)
    wave_ref = np.asarray(wave_ref, dtype=np.float64)
    flux_test = np.asarray(flux_test, dtype=np.float64)
    flux_ref = np.asarray(flux_ref, dtype=np.float64)
    if wave_test.ndim == 1:
        wave_test = wave_test[None, :]
        flux_test = flux_test[None, :]
    if wave_ref.ndim == 1:
        wave_ref = wave_ref[None, :]
        flux_ref = flux_ref[None, :]
    if wave_test.shape[0] != wave_ref.shape[0]:
        n = min(wave_test.shape[0], wave_ref.shape[0])
        wave_test, flux_test = wave_test[:n], flux_test[:n]
        wave_ref, flux_ref = wave_ref[:n], flux_ref[:n]

    # Auto-shrink continuum kernel for short spectral windows.
    n_pix_in = int(flux_ref.shape[-1])
    cont = int(continuum_size)
    if n_pix_in < 2 * cont:
        cont = max(11, (n_pix_in // 4) | 1)

    test_n = continuum_normalize_orders(flux_test, size=cont)
    ref_n = continuum_normalize_orders(flux_ref, size=cont)

    n_ord, n_pix = ref_n.shape
    test_on_ref = np.full_like(ref_n, np.nan)
    for i in range(n_ord):
        wt = wave_test[i]
        ft = test_n[i]
        wr = wave_ref[i]
        m = np.isfinite(wt) & np.isfinite(ft)
        if int(np.sum(m)) < 2:
            continue
        wt_m = wt[m]
        ft_m = ft[m]
        srt = np.argsort(wt_m)
        test_on_ref[i] = np.interp(
            wr, wt_m[srt], ft_m[srt], left=np.nan, right=np.nan
        )

    trim = max(0, int(edge_trim))
    if trim > 0 and n_pix > 2 * trim:
        sl = slice(trim, n_pix - trim)
    else:
        sl = slice(None)
    t = test_on_ref[:, sl].ravel()
    r = ref_n[:, sl].ravel()
    good = np.isfinite(t) & np.isfinite(r)
    frac_finite = float(np.mean(good)) if good.size else 0.0
    t, r = t[good], r[good]
    empty = {
        "label": label,
        "n_pix": int(t.size),
        "frac_finite": frac_finite,
        "continuum_size": cont,
        "norm_rms": float("nan"),
        "norm_rms_pct": float("nan"),
        "corr": float("nan"),
        "depth_median_ratio": float("nan"),
        "test_norm": test_on_ref,
        "ref_norm": ref_n,
    }
    if t.size < int(min_good_pix):
        return empty

    resid = t - r
    norm_rms = float(np.sqrt(np.mean(resid**2)))
    depth_t = 1.0 - t
    depth_r = 1.0 - r
    deep = (depth_r > 0.02) & np.isfinite(depth_t)
    depth_ratio = (
        float(np.median(depth_t[deep] / depth_r[deep]))
        if int(np.sum(deep)) > 20
        else float("nan")
    )
    return {
        "label": label,
        "n_pix": int(t.size),
        "frac_finite": frac_finite,
        "continuum_size": cont,
        "norm_rms": norm_rms,
        "norm_rms_pct": 100.0 * norm_rms,
        "corr": float(np.corrcoef(t, r)[0, 1]),
        "depth_median_ratio": depth_ratio,
        "test_norm": test_on_ref,
        "ref_norm": ref_n,
        "wave_ref": wave_ref,
    }


def plot_spectrum_fidelity_overlay(
    wave: np.ndarray,
    ref_norm: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    order_row: int,
    order_number: int,
    pix_slice: slice | None = None,
    title: str | None = None,
    savefig: str | os.PathLike[str] | None = None,
    display: bool = True,
) -> None:
    """Overlay continuum-normalized spectra for one order (+ residual panel)."""
    if pix_slice is None:
        pix_slice = slice(1200, 1600)
    w = np.asarray(wave[order_row], dtype=np.float64)[pix_slice]
    ref = np.asarray(ref_norm[order_row], dtype=np.float64)[pix_slice]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    axes[0].plot(w, ref, color="k", lw=1.2, label="build_l2")
    colors = ["C0", "C1", "C2", "C3"]
    for (lab, arr), col in zip(series.items(), colors, strict=False):
        y = np.asarray(arr[order_row], dtype=np.float64)[pix_slice]
        axes[0].plot(w, y, color=col, lw=1.0, alpha=0.9, label=lab)
        axes[1].plot(w, y - ref, color=col, lw=0.9, label=lab)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel("continuum-norm flux")
    axes[1].set_ylabel("test − build_l2")
    axes[1].set_xlabel("wavelength [μm]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    if title is None:
        title = f"Order {order_number} continuum-normalized fidelity"
    fig.suptitle(title)
    fig.tight_layout()

    if savefig is not None:
        out = os.fspath(savefig)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)


def summarize_fidelity_gaps(rows: list[dict[str, Any]]) -> str:
    """Human-readable gap summary for notebook printing."""
    lines = ["Spectrum fidelity vs build_l2 (continuum-normalized):"]
    for r in rows:
        lines.append(
            f"  {r['label']:28s}  "
            f"rms={r.get('norm_rms_pct', float('nan')):6.3f}%  "
            f"corr={r.get('corr', float('nan')):7.4f}  "
            f"depth_ratio={r.get('depth_median_ratio', float('nan')):6.3f}  "
            f"n={r.get('n_pix', 0)}"
        )
    lines.append("")
    lines.append("Expected irreducible / methodological gaps:")
    lines.append(
        "  • Native build_l2 (no LSF) vs extracts: use build_l2(..., apply_lsf=True)"
    )
    lines.append(
        "    so depth_ratio / continuum-norm RMS are like-to-like."
    )
    lines.append(
        "  • Separable sheared-SP: continuum P × shear, not full PSF stamps (~1% floor)."
    )
    lines.append(
        "    method=spectroperfectionism inverts the HDF deposit operator."
    )
    lines.append(
        "  • Box extract: no profile/shear → extra line smearing vs sheared-SP."
    )
    lines.append(
        "  • Continuum / blaze: median continuum norm removes blaze for shape tests;"
    )
    lines.append(
        "    science RV still needs fiberflat blaze + consistent WAVE solution."
    )
    return "\n".join(lines)
