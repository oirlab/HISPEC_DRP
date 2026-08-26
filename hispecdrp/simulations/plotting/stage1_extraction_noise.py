"""Stage-1 optimal-extraction residual pulls (PypeIt-style noise self-check)."""

from __future__ import annotations

import io
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import norm

from ...datamodels import hispec_config
from ...datamodels.dqflags import DQ_FLAGS
from ...primitives.spectral_extraction.extract_optimal import extract_spectrum_optimal
from ...primitives.spectral_extraction.extract_sheared_sp import (
    extract_sheared_banded_ls,
    forward_sheared_separable,
)
from ...primitives.spectral_extraction.extraction_utils import get_extraction_window
from .stage0_rate_noise import DEFAULT_STAGE0_PULL_DQ_EXCLUDE, robust_gaussian_params

DO_NOT_USE = int(DQ_FLAGS["DO_NOT_USE"])


def continuum_rate_from_spectral_smooth(
    rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    smooth_pix: int = 81,
) -> np.ndarray:
    """
    Build a continuum-like 2-D map by spectral smoothing within the aperture.

    Optimal extraction requires a slowly varying spatial profile ``P``; using a
    stellar rate map (absorption lines) as ``P`` biases Horne and residuals.
    """
    rate = np.asarray(rate, dtype=np.float64)
    trace_y = np.asarray(trace_y, dtype=np.float64)
    ny, nx = rate.shape
    strip = np.zeros((ny, nx), dtype=np.float64)
    for x in range(ny):
        yc = float(trace_y[x])
        if not np.isfinite(yc):
            continue
        window_inds, _W = get_extraction_window(yc, float(extraction_aperture), nx)
        if window_inds.size == 0:
            continue
        strip[x, window_inds] = np.maximum(rate[x, window_inds], 0.0)

    half = max(1, int(smooth_pix) // 2)
    size = 2 * half + 1
    continuum = np.zeros_like(strip)
    # Smooth only columns that have aperture flux (avoids mixing other orders).
    active = np.any(strip > 0.0, axis=0)
    for y in np.where(active)[0]:
        continuum[:, y] = uniform_filter1d(strip[:, y], size=size, mode="nearest")
    return continuum


def build_profile_image(
    profile_rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
) -> np.ndarray:
    """2-D raw profile image (same shape as rate) for optimal extraction."""
    profile_rate = np.asarray(profile_rate, dtype=np.float64)
    trace_y = np.asarray(trace_y, dtype=np.float64)
    ny, nx = profile_rate.shape
    profile = np.zeros((ny, nx), dtype=np.float64)
    for x in range(ny):
        yc = float(trace_y[x])
        if not np.isfinite(yc):
            continue
        window_inds, _W = get_extraction_window(yc, float(extraction_aperture), nx)
        if window_inds.size == 0:
            continue
        profile[x, window_inds] = np.maximum(profile_rate[x, window_inds], 0.0)
    return profile


# Backward-compatible alias (name historically implied stellar truth).
build_truth_profile_image = build_profile_image


def truth_profile_row(
    truth_row: np.ndarray,
    window_inds: np.ndarray,
    W: np.ndarray,
    *,
    floor: float = 1.0e-30,
) -> np.ndarray:
    """Normalized cross-dispersion profile from a noiseless rate row."""
    p = np.maximum(np.asarray(truth_row, dtype=np.float64)[window_inds], 0.0)
    denom = float(np.sum(p * W))
    if denom <= floor:
        return np.zeros(window_inds.size, dtype=np.float64)
    return p / denom


def extract_order_optimal(
    rate: np.ndarray,
    rate_err: np.ndarray,
    rate_dq: np.ndarray,
    var_rnoise: np.ndarray,
    var_poisson: np.ndarray,
    trace_y: np.ndarray,
    profile: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    method: str = "auto",
) -> dict[str, np.ndarray]:
    """Optimal-extract one echelle order from a full 2-D L0 rate frame.

    ``method``:

    - ``'horne'``: classic / sheared-gather Horne
    - ``'sheared_sp'``: banded LS inverse of the sheared scatter operator
    - ``'auto'``: ``sheared_sp`` when ``shear`` is active, else ``horne``
    """
    method_l = str(method).strip().lower()
    shear_arr = None if shear is None else np.asarray(shear, dtype=np.float64)
    use_sp = method_l == "sheared_sp" or (
        method_l == "auto"
        and shear_arr is not None
        and bool(np.any(np.isfinite(shear_arr) & (np.abs(shear_arr) > 1.0e-12)))
    )
    if use_sp:
        rate_var = np.asarray(rate_err, dtype=np.float64) ** 2
        return extract_sheared_banded_ls(
            np.asarray(rate, dtype=np.float64),
            rate_var,
            np.asarray(profile, dtype=np.float64),
            np.asarray(trace_y, dtype=np.float64),
            extraction_aperture=float(extraction_aperture),
            shear=shear_arr,
            rate_dq=np.asarray(rate_dq, dtype=np.uint32),
            do_not_use=DO_NOT_USE,
        )

    ny, nx = rate.shape
    mid = np.zeros((ny, nx), dtype=np.float64)
    return extract_spectrum_optimal(
        input_data=np.asarray(rate, dtype=np.float64),
        input_err=np.asarray(rate_err, dtype=np.float64),
        input_dq=np.asarray(rate_dq, dtype=np.uint32),
        input_var_rnoise=np.asarray(var_rnoise, dtype=np.float64),
        input_var_poisson=np.asarray(var_poisson, dtype=np.float64),
        input_mid_time=mid,
        trace_positions=np.asarray(trace_y, dtype=np.float64),
        profile=np.asarray(profile, dtype=np.float64),
        extraction_aperture=float(extraction_aperture),
        orders_are_vertical=True,
        shear=shear_arr,
    )


def model_and_sigma_from_extract(
    spec: np.ndarray,
    spec_err: np.ndarray,
    trace_y: np.ndarray,
    profile_img: np.ndarray,
    *,
    extraction_aperture: float,
    n_cross: int,
    shear: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2-D model, per-pixel σ, and aperture mask from a 1-D extraction.

    Uses the same sheared scatter operator as :func:`forward_sheared_separable`
    so extract+rebuild is an exact round-trip when extraction is the LS inverse.
    """
    del n_cross  # shape taken from profile_img
    model, mask = forward_sheared_separable(
        spec,
        profile_img,
        trace_y,
        extraction_aperture=extraction_aperture,
        shear=shear,
    )
    # Propagate 1-D formal error through the same linear operator (same weights).
    sigma, _ = forward_sheared_separable(
        np.asarray(spec_err, dtype=np.float64),
        profile_img,
        trace_y,
        extraction_aperture=extraction_aperture,
        shear=shear,
    )
    return model, sigma, mask


def project_rate_onto_separable_model(
    rate: np.ndarray,
    profile_rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    continuum_smooth_pix: int | None = 81,
) -> dict[str, Any]:
    """
    Project a 2-D rate map onto the sheared separable model ``A f``.

    Returns continuum ``profile``, extracted ``spec``, and noiseless model
    ``model = A f``.  ``rms(rate - model)`` is the irreducible Horne/SP model
    floor against a full PSF-deposited scene.
    """
    rate = np.asarray(rate, dtype=np.float64)
    profile_rate = np.asarray(profile_rate, dtype=np.float64)
    if continuum_smooth_pix is not None and int(continuum_smooth_pix) > 1:
        profile_rate = continuum_rate_from_spectral_smooth(
            profile_rate,
            trace_y,
            extraction_aperture=extraction_aperture,
            smooth_pix=int(continuum_smooth_pix),
        )
    profile_img = build_profile_image(
        profile_rate, trace_y, extraction_aperture=extraction_aperture
    )
    ny, nx = rate.shape
    err = np.ones((ny, nx), dtype=np.float64)
    dq = np.zeros((ny, nx), dtype=np.uint32)
    vr = np.ones((ny, nx), dtype=np.float64)
    vp = np.zeros((ny, nx), dtype=np.float64)
    extracted = extract_order_optimal(
        rate,
        err,
        dq,
        vr,
        vp,
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        shear=shear,
        method="auto",
    )
    model, _mask = forward_sheared_separable(
        extracted["spec"],
        profile_img,
        trace_y,
        extraction_aperture=extraction_aperture,
        shear=shear,
    )
    aperture = _mask
    diff = rate - model
    with np.errstate(invalid="ignore"):
        floor_rms = float(np.sqrt(np.mean(diff[aperture] ** 2))) if np.any(aperture) else np.nan
        truth_rms = (
            float(np.sqrt(np.mean(rate[aperture] ** 2))) if np.any(aperture) else np.nan
        )
    return {
        "profile": profile_img,
        "spec": np.asarray(extracted["spec"], dtype=np.float64),
        "spec_err": np.asarray(extracted["err"], dtype=np.float64),
        "model": model,
        "aperture_mask": aperture,
        "floor_rms": floor_rms,
        "floor_frac": floor_rms / truth_rms if truth_rms and truth_rms > 0 else np.nan,
    }


def order_extraction_residuals(
    rate: np.ndarray,
    rate_err: np.ndarray,
    rate_dq: np.ndarray,
    var_rnoise: np.ndarray,
    var_poisson: np.ndarray,
    profile_rate: np.ndarray,
    trace_y: np.ndarray,
    *,
    extraction_aperture: float,
    shear: np.ndarray | None = None,
    continuum_smooth_pix: int | None = None,
    method: str = "auto",
) -> dict[str, Any]:
    """
    Optimal extract one order, subtract the profile model, return 2-D maps and pulls.

    ``profile_rate`` should be a **continuum-like** 2-D map (fiberflat rate or
    spectrally smoothed science rate). If ``continuum_smooth_pix`` is set, the
    map is re-smoothed along the spectral axis inside the aperture before use.
    Optional ``shear`` enables iso-λ sheared extraction; with ``method='auto'``
    this uses banded LS (``sheared_sp``) so extract+rebuild is an exact inverse
    of the separable forward model.
    """
    rate = np.asarray(rate, dtype=np.float64)
    profile_rate = np.asarray(profile_rate, dtype=np.float64)
    ny, nx = rate.shape

    if continuum_smooth_pix is not None and int(continuum_smooth_pix) > 1:
        profile_rate = continuum_rate_from_spectral_smooth(
            profile_rate,
            trace_y,
            extraction_aperture=extraction_aperture,
            smooth_pix=int(continuum_smooth_pix),
        )

    profile_img = build_profile_image(
        profile_rate, trace_y, extraction_aperture=extraction_aperture
    )
    shear_arr = None if shear is None else np.asarray(shear, dtype=np.float64)
    extracted = extract_order_optimal(
        rate,
        rate_err,
        rate_dq,
        var_rnoise,
        var_poisson,
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        shear=shear_arr,
        method=method,
    )
    spec = np.asarray(extracted["spec"], dtype=np.float64)
    spec_err = np.asarray(extracted["err"], dtype=np.float64)
    # Formal err can be NaN at empty spectral bins; treat as 0 for sigma map.
    spec_err_safe = np.where(np.isfinite(spec_err), spec_err, 0.0)

    model, sigma, aperture_mask = model_and_sigma_from_extract(
        spec,
        spec_err_safe,
        trace_y,
        profile_img,
        extraction_aperture=extraction_aperture,
        n_cross=nx,
        shear=shear_arr,
    )
    residual = rate - model
    rate_err_arr = np.asarray(rate_err, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls_pixel = residual / rate_err_arr
        pulls_spec = residual / sigma

    dq_ok = (np.asarray(rate_dq, dtype=np.uint32) & np.uint32(DO_NOT_USE)) == 0
    good_pixel = (
        aperture_mask
        & dq_ok
        & np.isfinite(rate)
        & np.isfinite(model)
        & np.isfinite(rate_err_arr)
        & (rate_err_arr > 0.0)
        & np.isfinite(pulls_pixel)
    )
    good_spec = (
        aperture_mask
        & dq_ok
        & np.isfinite(rate)
        & np.isfinite(model)
        & np.isfinite(sigma)
        & (sigma > 0.0)
        & np.isfinite(pulls_spec)
    )

    return {
        "spec": spec,
        "spec_err": spec_err,
        "profile": profile_img,
        "model": model,
        "residual": residual,
        "sigma": sigma,
        "pulls": pulls_pixel,
        "pulls_pixel": pulls_pixel,
        "pulls_spec": pulls_spec,
        "good_mask": good_pixel,
        "good_mask_pixel": good_pixel,
        "good_mask_spec": good_spec,
        "aperture_mask": aperture_mask,
        "shear": shear_arr,
        "method": method,
    }


def extraction_pull_samples(
    pulls: np.ndarray,
    good_mask: np.ndarray,
    *,
    dq: np.ndarray | None = None,
    dq_exclude_flags: int | None = DEFAULT_STAGE0_PULL_DQ_EXCLUDE,
) -> np.ndarray:
    """1-D pull samples on extraction-aperture pixels."""
    good = np.asarray(good_mask, dtype=bool)
    if dq is not None:
        dq_arr = np.asarray(dq, dtype=np.uint32)
        exclude = (
            DEFAULT_STAGE0_PULL_DQ_EXCLUDE
            if dq_exclude_flags is None
            else int(dq_exclude_flags)
        )
        if exclude != 0:
            good &= (dq_arr & np.uint32(exclude)) == 0
    return np.asarray(pulls, dtype=np.float64)[good]


def collect_fiber_extraction_pulls(
    rate: np.ndarray,
    rate_err: np.ndarray,
    rate_dq: np.ndarray,
    var_rnoise: np.ndarray,
    var_poisson: np.ndarray,
    profile_rate: np.ndarray,
    tracepos: np.ndarray,
    *,
    spectrograph: str = "BSPEC",
    orders: list[int] | None = None,
    order_offset: int = 110,
    extraction_aperture: float = 8.0,
    error_mode: str = "both",
    store_maps_orders: list[int] | None = None,
    shear: np.ndarray | None = None,
    continuum_smooth_pix: int | None = 81,
) -> dict[str, Any]:
    """Run per-order optimal extraction residuals and concatenate pulls.

    Each order is extracted **once**. Both pull normalizations are always
    computed; ``error_mode`` only selects which array is aliased as ``pulls``
    for backward compatibility.

    ``profile_rate``:
        Continuum-like 2-D map (fiberflat / spectrally smoothed rate).
    ``shear``:
        Optional ``(n_orders, n_pix)`` iso-λ shear matching ``tracepos``.
    ``continuum_smooth_pix``:
        If set (>1), spectrally smooth ``profile_rate`` per order before use.
    ``store_maps_orders``:
        If set, keep full 2-D residual maps only for those echelle orders.
    """
    mode = str(error_mode).strip().lower()
    if mode not in ("pixel", "spec_profile", "both"):
        raise ValueError(
            f"error_mode must be 'pixel', 'spec_profile', or 'both', got {error_mode!r}"
        )

    tracepos = np.asarray(tracepos, dtype=np.float64)
    if orders is None:
        orders = hispec_config.HISPECConfig.get_order_list(str(spectrograph).upper())
    keep_maps = (
        None
        if store_maps_orders is None
        else {int(o) for o in store_maps_orders}
    )
    shear_arr = None if shear is None else np.asarray(shear, dtype=np.float64)

    all_pulls_pixel: list[np.ndarray] = []
    all_pulls_spec: list[np.ndarray] = []
    per_order: dict[int, dict[str, Any]] = {}
    n_orders = 0
    shear_abs = []
    for od in orders:
        row = int(od) - int(order_offset)
        if row < 0 or row >= tracepos.shape[0]:
            continue
        trace_y = tracepos[row]
        order_shear = None
        if shear_arr is not None and row < shear_arr.shape[0]:
            order_shear = shear_arr[row]
            ok = np.isfinite(order_shear)
            if np.any(ok):
                shear_abs.append(float(np.mean(np.abs(order_shear[ok]))))
        result = order_extraction_residuals(
            rate,
            rate_err,
            rate_dq,
            var_rnoise,
            var_poisson,
            profile_rate,
            trace_y,
            extraction_aperture=extraction_aperture,
            shear=order_shear,
            continuum_smooth_pix=continuum_smooth_pix,
            method="auto",
        )
        n_orders += 1
        pulls_pix = extraction_pull_samples(
            result["pulls_pixel"], result["good_mask_pixel"], dq=rate_dq
        )
        pulls_sp = extraction_pull_samples(
            result["pulls_spec"], result["good_mask_spec"], dq=rate_dq
        )
        if pulls_pix.size:
            all_pulls_pixel.append(pulls_pix)
        if pulls_sp.size:
            all_pulls_spec.append(pulls_sp)

        od_i = int(od)
        if keep_maps is None or od_i in keep_maps:
            per_order[od_i] = result
        else:
            per_order[od_i] = {
                "spec": result["spec"],
                "spec_err": result["spec_err"],
                "n_pix_pixel": int(pulls_pix.size),
                "n_pix_spec": int(pulls_sp.size),
            }

    pulls_pixel = (
        np.concatenate(all_pulls_pixel)
        if all_pulls_pixel
        else np.array([], dtype=np.float64)
    )
    pulls_spec = (
        np.concatenate(all_pulls_spec)
        if all_pulls_spec
        else np.array([], dtype=np.float64)
    )
    if mode == "spec_profile":
        combined = pulls_spec
    else:
        combined = pulls_pixel

    return {
        "pulls": combined,
        "pulls_pixel": pulls_pixel,
        "pulls_spec": pulls_spec,
        "error_mode": mode,
        "per_order": per_order,
        "n_orders": n_orders,
        "n_pix": int(combined.size),
        "n_pix_pixel": int(pulls_pixel.size),
        "n_pix_spec": int(pulls_spec.size),
        "mean_abs_shear": float(np.mean(shear_abs)) if shear_abs else 0.0,
    }


def order_zoom_slice(
    trace_y: np.ndarray,
    *,
    row_start: int = 1200,
    row_stop: int = 1400,
    col_margin: float = 12.0,
) -> tuple[slice, slice]:
    """Row/column slices for a zoomed 2-D cut around an echelle trace."""
    trace_y = np.asarray(trace_y, dtype=np.float64)
    ny = trace_y.size
    r0 = max(0, int(row_start))
    r1 = min(ny, int(row_stop))
    seg = trace_y[r0:r1]
    ok = np.isfinite(seg)
    if not np.any(ok):
        return slice(r0, r1), slice(0, 1)
    yc = seg[ok]
    c0 = int(np.floor(np.min(yc) - float(col_margin)))
    c1 = int(np.ceil(np.max(yc) + float(col_margin)))
    return slice(r0, r1), slice(max(0, c0), c1)


def plot_order_extraction_zoom_triptych(
    rate: np.ndarray,
    residual: np.ndarray,
    pulls: np.ndarray,
    good_mask: np.ndarray,
    *,
    row_slice: slice,
    col_slice: slice,
    title: str,
    pull_label: str = r"$(D - M) / {\rm rate\_err}$",
    savefig: str | os.PathLike[str] | None = None,
    display: bool = True,
) -> None:
    """Three-panel zoom: rate, residual, σ-normalized residual."""
    rate_z = np.asarray(rate, dtype=np.float64)[row_slice, col_slice]
    resid_z = np.asarray(residual, dtype=np.float64)[row_slice, col_slice]
    pull_z = np.where(
        np.asarray(good_mask, dtype=bool)[row_slice, col_slice],
        np.asarray(pulls, dtype=np.float64)[row_slice, col_slice],
        np.nan,
    )

    vmax_rate = float(np.nanpercentile(rate_z, 99.0))
    vmax_res = float(np.nanpercentile(np.abs(resid_z), 99.0))
    vmax_pull = 4.0

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    im0 = axes[0].imshow(
        rate_z,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(vmax_rate, 1e-12),
    )
    axes[0].set_title("Rate map")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        resid_z,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax_res,
        vmax=vmax_res,
    )
    axes[1].set_title("Profile-subtracted residual")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        pull_z,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax_pull,
        vmax=vmax_pull,
    )
    axes[2].set_title(pull_label)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("cross-dispersion [pix]")
    axes[0].set_ylabel("spectral row [pix]")
    fig.suptitle(title)
    fig.tight_layout()

    if savefig is not None:
        out = os.fspath(savefig)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out, format="png", dpi=150, bbox_inches="tight")

    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)


def plot_extraction_pull_histogram(
    pulls: np.ndarray,
    *,
    bins: int = 80,
    x_lim: float = 5.0,
    title: str | None = None,
    xlabel: str | None = None,
    display: bool = True,
    print_stats: bool = True,
    savefig: str | os.PathLike[str] | None = None,
    robust_clip_sigma: float = 3.0,
) -> dict[str, Any]:
    """Histogram of extraction residual pulls with Gaussian overlay (like Stage 0)."""
    pulls = np.asarray(pulls, dtype=np.float64).ravel()
    pulls = pulls[np.isfinite(pulls)]
    if pulls.size < 10:
        raise ValueError(f"too few extraction pull pixels: n={pulls.size}")

    mu, sigma, n_core = robust_gaussian_params(
        pulls, n_sigma=float(robust_clip_sigma)
    )
    frac_1sig = float(np.mean(np.abs(pulls) < 1.0))

    if title is None:
        title = "Optimal extraction residual pulls"
    if xlabel is None:
        xlabel = r"$(D - M) / \sigma$"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        pulls,
        bins=bins,
        range=(-float(x_lim), float(x_lim)),
        density=True,
        histtype="stepfilled",
        alpha=0.55,
        color="C0",
        edgecolor="C0",
        label=f"aperture pixels (n={pulls.size})",
    )
    x = np.linspace(-float(x_lim), float(x_lim), 400)
    ax.plot(
        x,
        norm.pdf(x, mu, sigma),
        color="C1",
        lw=2.0,
        label=f"Gaussian fit N({mu:.3f}, {sigma:.3f})",
    )
    ax.plot(
        x,
        norm.pdf(x, 0.0, 1.0),
        color="0.35",
        lw=1.5,
        ls="--",
        label="N(0, 1)",
    )
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(-float(x_lim), float(x_lim))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savefig is not None:
        out = os.fspath(savefig)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out, format="png", dpi=150, bbox_inches="tight")

    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)

    summary = {
        "pulls": pulls,
        "n_pix": int(pulls.size),
        "n_core": int(n_core),
        "mu": float(mu),
        "sigma": float(sigma),
        "frac_within_1sig": frac_1sig,
        "mean_pull": float(np.mean(pulls)),
        "std_pull": float(np.std(pulls)),
    }
    if print_stats:
        print(f"n_pix={summary['n_pix']}  n_core={summary['n_core']}")
        print(f"Gaussian fit: mu={mu:.4f}  sigma={sigma:.4f}")
        print(
            f"fraction |z|<1: {frac_1sig:.3f}  "
            f"(N(0,1) ≈ {2.0 * norm.cdf(1.0) - 1.0:.3f})"
        )
    return summary
