"""Plots for the extracted-LFC / FitLFCLines / Gaussian-LSF documentation example."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def _display_or_save(
    fig,
    *,
    out_path: str | Path | None,
    display: bool,
) -> None:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)


def _orders_from_model(model) -> np.ndarray:
    if hasattr(model, "make_order_vec"):
        return np.asarray(model.make_order_vec(), dtype=int)
    start = int(getattr(getattr(model, "meta", None), "order_start", 0))
    arr = getattr(model, "wave_fib1", None)
    if arr is None:
        return np.array([], dtype=int)
    return start + np.arange(np.asarray(arr).shape[0], dtype=int)


def _fwhm_wave_array(lsf_params: np.ndarray) -> np.ndarray:
    """Return per-pixel FWHM in wavelength; accept 2-D FWHM or 3-D (…, 3)."""
    arr = np.asarray(lsf_params, dtype=float)
    if arr.ndim >= 2 and arr.shape[-1] == 3 and arr.ndim == 3:
        # Schema leftover (mu, sigma, offset); sigma → FWHM if that layout appears.
        return arr[..., 1] * 2.35482004503
    return arr


def plot_extracted_lfc_order(
    model,
    *,
    fiber: int,
    order: int | None = None,
    order_index: int = 0,
    xlim: tuple[float, float] | None = None,
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """Plot one extracted LFC order versus pixel and versus WAVE."""
    spec_attr = f"spec_fib{int(fiber)}"
    if not hasattr(model, spec_attr) and hasattr(model, f"spec_uncal_fib{int(fiber)}"):
        spec_attr = f"spec_uncal_fib{int(fiber)}"
    spec = np.asarray(getattr(model, spec_attr), dtype=float)
    wave = getattr(model, f"wave_fib{int(fiber)}", None)
    orders = _orders_from_model(model)
    if order is not None:
        hits = np.where(orders == int(order))[0]
        if hits.size == 0:
            raise ValueError(f"order {order} not in model order vector {orders}")
        idx = int(hits[0])
    else:
        idx = int(order_index)
    od = int(orders[idx]) if orders.size else idx
    flux = spec[idx]
    pix = np.arange(flux.size, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=False)
    axes[0].plot(pix, flux, color="C0", lw=0.8)
    axes[0].set_xlabel("pixel")
    axes[0].set_ylabel("flux")
    axes[0].set_title(f"Extracted LFC  fiber {int(fiber)}  order {od}")
    if xlim is not None:
        axes[0].set_xlim(*xlim)
    axes[0].grid(True, alpha=0.3)
    if wave is not None:
        w = np.asarray(wave, dtype=float)[idx]
        axes[1].plot(w, flux, color="C1", lw=0.8)
        axes[1].set_xlabel("wavelength (µm)")
    else:
        axes[1].plot(pix, flux, color="C1", lw=0.8)
        axes[1].set_xlabel("pixel")
    axes[1].set_ylabel("flux")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    return {"fiber": int(fiber), "order": od, "order_index": idx, "out_path": out_path}


def _style_fwhm_wavelength_axes(ax, *, xlabel, ylabel, title, n_legend: int) -> None:
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    if n_legend:
        ax.legend(ncols=3, fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_lfc_line_fwhm(
    line_fits_model,
    *,
    fiber: int,
    orders: Sequence[int] | None = None,
    y: str = "fwhm_pix",
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """
    Per-order LFC line FWHM versus assigned mode wavelength.

    ``y`` is ``\"fwhm_pix\"``, ``\"fwhm_wave\"``, or ``\"resolving_power\"``
    (``WAVE / FWHM_WAVE``).
    """
    table = getattr(line_fits_model, f"lfc_fitparams_fib{int(fiber)}")
    y_key = str(y).lower()
    if y_key == "fwhm_pix":
        ycol, ylabel = "FWHM_PIX", "FWHM (pixels)"
    elif y_key == "fwhm_wave":
        ycol, ylabel = "FWHM_WAVE", "FWHM (µm)"
    elif y_key in {"resolving_power", "r", "resolution"}:
        ycol, ylabel = None, r"spectral resolution [$R=\lambda/\Delta\lambda$(FWHM)]"
    else:
        raise ValueError("y must be 'fwhm_pix', 'fwhm_wave', or 'resolving_power'")

    all_orders = np.asarray(table["ORDER"], dtype=int)
    uniq = np.unique(all_orders)
    if orders is None:
        use_orders = uniq
    else:
        use_orders = np.asarray([int(o) for o in orders if int(o) in set(uniq)], dtype=int)

    fig, ax = plt.subplots(figsize=(16, 9))
    n_plotted = 0
    for od in use_orders:
        sel = all_orders == int(od)
        wave = np.asarray(table["WAVE"][sel], dtype=float)
        if ycol is None:
            yy = wave / np.asarray(table["FWHM_WAVE"][sel], dtype=float)
        else:
            yy = np.asarray(table[ycol][sel], dtype=float)
        ok = np.isfinite(wave) & np.isfinite(yy)
        if int(ok.sum()) == 0:
            continue
        ax.plot(wave[ok], yy[ok], ".", ms=3, label=f"order {int(od)}")
        n_plotted += 1
    _style_fwhm_wavelength_axes(
        ax,
        xlabel="wavelength (µm)",
        ylabel=ylabel,
        title=f"LFC line fits  fiber {int(fiber)}",
        n_legend=n_plotted,
    )
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    return {
        "fiber": int(fiber),
        "n_orders": int(n_plotted),
        "y": y_key,
        "out_path": str(out_path) if out_path is not None else None,
    }


def plot_lsf_resolving_power(
    lsf_model,
    *,
    fiber: int,
    orders: Sequence[int] | None = None,
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """Plot Chebyshev-smoothed LSF resolving power ``λ / FWHM_λ`` per order."""
    wave_2d = np.asarray(getattr(lsf_model, f"wave_fib{int(fiber)}"), dtype=float)
    fwhm_2d = _fwhm_wave_array(getattr(lsf_model, f"lsf_params_fib{int(fiber)}"))
    order_vec = _orders_from_model(lsf_model)
    n_ord = min(wave_2d.shape[0], fwhm_2d.shape[0])
    if order_vec.size < n_ord:
        order_vec = np.arange(n_ord, dtype=int)
    if orders is None:
        indices = range(n_ord)
    else:
        want = {int(o) for o in orders}
        indices = [i for i in range(n_ord) if int(order_vec[i]) in want]

    fig, ax = plt.subplots(figsize=(16, 9))
    n_plotted = 0
    for i in indices:
        w = wave_2d[i]
        r = w / fwhm_2d[i]
        ok = np.isfinite(w) & np.isfinite(r) & (fwhm_2d[i] > 0)
        if int(ok.sum()) == 0:
            continue
        ax.plot(w[ok], r[ok], lw=0.9, label=f"order {int(order_vec[i])}")
        n_plotted += 1
    _style_fwhm_wavelength_axes(
        ax,
        xlabel="wavelength (µm)",
        ylabel=r"spectral resolution [$R=\lambda/\Delta\lambda$(FWHM)]",
        title=f"Gaussian LSF  fiber {int(fiber)}",
        n_legend=n_plotted,
    )
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    return {
        "fiber": int(fiber),
        "n_orders": int(n_plotted),
        "out_path": str(out_path) if out_path is not None else None,
    }


def _pick_zoom_orders(available: Sequence[int], orders: Sequence[int] | None) -> list[int]:
    have = [int(o) for o in available]
    if orders is not None:
        want = [int(o) for o in orders]
        missing = [o for o in want if o not in have]
        if missing:
            raise ValueError(f"orders {missing} not in line_list {have}")
        return want
    if len(have) <= 3:
        return have
    idx = np.linspace(0, len(have) - 1, 3)
    return [have[int(round(i))] for i in idx]


def _nice_scale_length(span_pix: float) -> float:
    target = 0.28 * float(span_pix)
    if not np.isfinite(target) or target <= 0:
        return 20.0
    exp = 10.0 ** np.floor(np.log10(target))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if mult * exp >= 0.6 * target:
            return float(mult * exp)
    return float(10.0 * exp)


def _format_delta_lambda(dlam_um: float) -> str:
    dlam_um = abs(float(dlam_um))
    nm = dlam_um * 1e3
    if nm >= 100.0:
        return f"{dlam_um:.3f} µm"
    if nm >= 0.1:
        return f"{nm:.2f} nm"
    return f"{dlam_um * 1e4:.1f} Å"


def _dlam_per_row(rows: np.ndarray, waves_um: np.ndarray) -> float:
    rows = np.asarray(rows, dtype=float)
    waves = np.asarray(waves_um, dtype=float)
    ok = np.isfinite(rows) & np.isfinite(waves)
    rows, waves = rows[ok], waves[ok]
    if rows.size < 2:
        return float("nan")
    order = np.argsort(rows)
    drow = np.diff(rows[order])
    dwave = np.diff(waves[order])
    good = np.isfinite(drow) & np.isfinite(dwave) & (np.abs(drow) > 1e-6)
    if int(good.sum()) == 0:
        return float("nan")
    return float(np.median(np.abs(dwave[good] / drow[good])))


def _deposit_xy_for_orders(
    line_list: Mapping[int, np.ndarray],
    *,
    spectrograph: str,
    fiber: int,
    deposit_xy: Mapping[int, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if deposit_xy is not None:
        return {
            int(od): (np.asarray(rc[0], dtype=float), np.asarray(rc[1], dtype=float))
            for od, rc in deposit_xy.items()
        }
    from hispecdrp.simulations.e2e_pipeline import expected_feature_xy_oriented

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for od, lams in line_list.items():
        rows, cols = expected_feature_xy_oriented(
            spectrograph, int(fiber), int(od), np.asarray(lams, dtype=float)
        )
        out[int(od)] = (np.asarray(rows, dtype=float), np.asarray(cols, dtype=float))
    return out


def _zoom_window(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    shape: tuple[int, int],
    n_modes: int = 2,
    row_pad: float = 16.0,
    col_half: float = 20.0,
) -> tuple[int, int, int, int]:
    """Return ``r0, r1, c0, c1`` covering at least ``n_modes`` painted spots."""
    ny, nx = int(shape[0]), int(shape[1])
    ok = np.isfinite(rows) & np.isfinite(cols)
    if int(ok.sum()) == 0:
        return 0, min(ny, 80), 0, min(nx, 40)
    rows = np.asarray(rows[ok], dtype=float)
    cols = np.asarray(cols[ok], dtype=float)
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    mid = rows.size // 2
    n_keep = max(1, min(int(n_modes), rows.size))
    i1 = min(rows.size - 1, mid + max(0, n_keep - 1))
    i0 = max(0, i1 - (n_keep - 1))
    if i1 == i0 and rows.size >= 2:
        if mid + 1 < rows.size:
            i1 = mid + 1
        else:
            i0 = max(0, mid - 1)
    r_lo = float(min(rows[i0], rows[i1]))
    r_hi = float(max(rows[i0], rows[i1]))
    if r_hi <= r_lo:
        r_lo -= 20.0
        r_hi += 20.0
    pad = max(float(row_pad), 0.15 * (r_hi - r_lo))
    r0 = max(0, int(np.floor(r_lo - pad)))
    r1 = min(ny, int(np.ceil(r_hi + pad)))
    c_mid = float(np.median(cols[i0 : i1 + 1]))
    c_span = float(np.nanmax(np.abs(cols[i0 : i1 + 1] - c_mid)))
    half_c = max(float(col_half), c_span + 6.0)
    c0 = max(0, int(np.floor(c_mid - half_c)))
    c1 = min(nx, int(np.ceil(c_mid + half_c)))
    if r1 <= r0 + 4:
        r1 = min(ny, r0 + 40)
    if c1 <= c0 + 4:
        c1 = min(nx, c0 + 20)
    return r0, r1, c0, c1


def _max_pool2d(img: np.ndarray, block: int) -> np.ndarray:
    """Block max-pool so isolated LFC cores survive display downsampling."""
    img = np.asarray(img, dtype=float)
    block = max(1, int(block))
    if block == 1:
        return img
    ny, nx = img.shape
    sy = (ny // block) * block
    sx = (nx // block) * block
    if sy == 0 or sx == 0:
        return img
    return img[:sy, :sx].reshape(sy // block, block, sx // block, block).max(axis=(1, 3))


def _log_norm(rate: np.ndarray) -> tuple[np.ndarray, LogNorm]:
    """Log stretch over three decades from the peak.

    The floor is 1e-3 of the peak (not lower) so that blaze-tapered
    order ends -- a few percent of the peak flux -- read as faint
    rather than blooming like blaze-center spots.
    """
    finite = np.asarray(rate, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    peak = float(np.nanmax(finite)) if finite.size else 1.0
    peak = max(peak, 1e-12)
    vmin = max(peak * 1.0e-3, 1.0e-6)
    disp = np.asarray(rate, dtype=float)
    disp = np.where(np.isfinite(disp) & (disp > 0), disp, vmin)
    return disp, LogNorm(vmin=vmin, vmax=peak)


def _draw_spectral_scale_bar(
    ax,
    *,
    r0: float,
    r1: float,
    c0: float,
    c1: float,
    dlam_per_row: float,
) -> dict[str, float]:
    height = float(r1 - r0)
    width = float(c1 - c0)
    length = _nice_scale_length(height)
    if length >= 0.85 * height:
        length = max(5.0, 0.4 * height)
    x = c0 + 0.12 * width
    y = r0 + 0.10 * height
    y1 = y + length
    ax.plot([x, x], [y, y1], color="white", lw=2.2, solid_capstyle="butt", zorder=5)
    ax.plot([x, x], [y, y1], color="black", lw=0.8, solid_capstyle="butt", zorder=6)
    tick = 0.035 * width
    ax.plot([x - tick, x + tick], [y, y], color="white", lw=2.0, zorder=5)
    ax.plot([x - tick, x + tick], [y1, y1], color="white", lw=2.0, zorder=5)
    ax.plot([x - tick, x + tick], [y, y], color="black", lw=0.7, zorder=6)
    ax.plot([x - tick, x + tick], [y1, y1], color="black", lw=0.7, zorder=6)
    dlam = length * dlam_per_row if np.isfinite(dlam_per_row) else float("nan")
    label = f"{length:.0f} pix"
    if np.isfinite(dlam) and dlam > 0:
        label = f"{length:.0f} pix\n≈ {_format_delta_lambda(dlam)}"
    ax.text(
        x + 1.8 * tick,
        0.5 * (y + y1),
        label,
        color="white",
        fontsize=8,
        va="center",
        ha="left",
        zorder=6,
        path_effects=[
            patheffects.Stroke(linewidth=2.2, foreground="black"),
            patheffects.Normal(),
        ],
    )
    return {"length_pix": float(length), "dlam_um": float(dlam)}


def plot_lfc_rate_map_order_zooms(
    rate: np.ndarray,
    line_list: Mapping[int, np.ndarray],
    *,
    spectrograph: str,
    fiber: int,
    orders: Sequence[int] | None = None,
    deposit_xy: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    n_modes: int = 2,
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """
    Full 2-D LFC rate map plus zooms of three orders with a spectral scale bar.

    Each zoom covers at least ``n_modes`` painted spots around the middle of
    that order. The full map and every zoom use square pixels
    (``aspect='equal'``): equal pixel counts have equal on-screen size. The
    scale bar is along the spectral (row) axis and is labeled in both pixels
    and wavelength using the local ``dλ/drow``. The full map is max-pooled and
    log-stretched so isolated cores survive display downsampling.
    """
    rate = np.asarray(rate, dtype=float)
    if rate.ndim != 2:
        raise ValueError(f"rate must be 2-D, got shape {rate.shape}")
    pick = _pick_zoom_orders(sorted(int(o) for o in line_list), orders)
    xy = _deposit_xy_for_orders(
        line_list, spectrograph=spectrograph, fiber=fiber, deposit_xy=deposit_xy
    )

    disp, norm = _log_norm(rate)
    n_zoom = max(len(pick), 1)
    label_fs, title_fs, tick_fs = 16, 20, 13
    fig = plt.figure(figsize=(15.5, 18.0), layout="constrained")
    gs = fig.add_gridspec(2, n_zoom, height_ratios=[3.0, 1.0], hspace=0.10, wspace=0.18)
    ax_full = fig.add_subplot(gs[0, :])
    ny, nx = rate.shape
    block = 8 if max(ny, nx) >= 1024 else 1
    overview = _max_pool2d(disp, block)
    im = ax_full.imshow(
        overview,
        origin="lower",
        aspect="equal",
        cmap="magma",
        norm=norm,
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
        interpolation="nearest",
    )
    ax_full.set_aspect("equal", adjustable="box")
    ax_full.set_box_aspect(ny / nx)
    ax_full.set_xlabel("cross-dispersion [pix]", fontsize=label_fs)
    ax_full.set_ylabel("spectral row [pix]", fontsize=label_fs)
    ax_full.set_title(f"Simulated LFC rate map  fiber {int(fiber)}", fontsize=title_fs)
    ax_full.tick_params(labelsize=tick_fs)
    ax_full.set_xlim(-0.5, nx - 0.5)
    ax_full.set_ylim(-0.5, ny - 0.5)
    cbar = fig.colorbar(im, ax=ax_full, fraction=0.035, pad=0.01, label="e-/s")
    cbar.set_label("e-/s", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)
    # NOTE: no marker overlay at the deposit positions. Markers drawn
    # independent of flux would show comb modes even where the blaze
    # has tapered the painted flux to ~zero, fabricating "lines" at
    # the order ends that are not in the rate map.

    zoom_info: list[dict[str, Any]] = []
    colors = [f"C{i}" for i in range(n_zoom)]
    for i, od in enumerate(pick):
        ax = fig.add_subplot(gs[1, i])
        rows, cols = xy[int(od)]
        lams = np.asarray(line_list[int(od)], dtype=float)
        r0, r1, c0, c1 = _zoom_window(rows, cols, shape=rate.shape, n_modes=n_modes)
        # Zooms get their own deeper log stretch (5 decades from the
        # crop peak, vs 3 for the full map) so the faint PSF wings
        # around each spot are visible.
        raw_crop = rate[r0:r1, c0:c1]
        crop_peak = float(np.nanmax(raw_crop)) if raw_crop.size else 1.0
        crop_peak = max(crop_peak, 1.0e-12)
        zoom_vmin = crop_peak * 1.0e-5
        crop = np.where(
            np.isfinite(raw_crop) & (raw_crop > 0), raw_crop, zoom_vmin
        )
        im_zoom = ax.imshow(
            crop,
            origin="lower",
            aspect="equal",
            cmap="magma",
            norm=LogNorm(vmin=zoom_vmin, vmax=crop_peak),
            extent=(c0 - 0.5, c1 - 0.5, r0 - 0.5, r1 - 0.5),
            interpolation="nearest",
        )
        cbar_zoom = fig.colorbar(im_zoom, ax=ax, fraction=0.046, pad=0.04)
        cbar_zoom.ax.tick_params(labelsize=tick_fs - 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect((r1 - r0) / (c1 - c0))
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(r0 - 0.5, r1 - 0.5)
        ax.set_title(f"order {int(od)}", color=colors[i], fontsize=title_fs)
        ax.set_xlabel("cross-dispersion [pix]", fontsize=label_fs)
        ax.set_ylabel("spectral row [pix]", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        bar = _draw_spectral_scale_bar(
            ax,
            r0=r0 - 0.5,
            r1=r1 - 0.5,
            c0=c0 - 0.5,
            c1=c1 - 0.5,
            dlam_per_row=_dlam_per_row(rows, lams),
        )
        rect = Rectangle(
            (c0, r0),
            c1 - c0,
            r1 - r0,
            fill=False,
            edgecolor=colors[i],
            lw=1.4,
        )
        ax_full.add_patch(rect)
        ax_full.annotate(
            f"{int(od)}",
            (c1, r1),
            color=colors[i],
            fontsize=14,
            ha="left",
            va="bottom",
        )
        zoom_info.append(
            {
                "order": int(od),
                "row_slice": (r0, r1),
                "col_slice": (c0, c1),
                "scale_bar_pix": bar["length_pix"],
                "scale_bar_dlam_um": bar["dlam_um"],
            }
        )

    ax_full.set_xlim(-0.5, nx - 0.5)
    ax_full.set_ylim(-0.5, ny - 0.5)
    fig.suptitle(
        "LFC rate map with order zooms (scale bar: spectral pixels and Δλ)",
        fontsize=title_fs,
    )
    _display_or_save(fig, out_path=out_path, display=display)
    return {
        "fiber": int(fiber),
        "orders": pick,
        "zooms": zoom_info,
        "out_path": str(out_path) if out_path is not None else None,
    }


def lfc_vs_hdf_wavesol_shift(
    lfc_wave: np.ndarray,
    hdf_wave: np.ndarray,
    *,
    orders: Sequence[int] | None = None,
    max_abs_dvel_m_s: float | None = None,
) -> dict[str, Any]:
    """Pixel-wise ``LFC − HDF`` shift from :meth:`SimulateRateMap.get_wavelength_solution`.

    ``max_abs_dvel_m_s`` drops unphysical edge residuals (single-pixel
    ``|Δv|`` spikes of hundreds of m/s) from the good mask, plots, and
    summary statistics.
    """
    from ...primitives.utils.spectral_utils import delta_wave_to_delta_vel

    lfc = np.asarray(lfc_wave, dtype=float)
    hdf = np.asarray(hdf_wave, dtype=float)
    if lfc.shape != hdf.shape:
        raise ValueError(f"WAVE shapes differ: LFC {lfc.shape} vs HDF {hdf.shape}")
    if lfc.ndim != 2:
        raise ValueError(f"WAVE must be (n_orders, n_pix), got {lfc.shape}")
    n_ord, n_pix = lfc.shape
    if orders is None:
        order_vec = np.arange(n_ord, dtype=int)
    else:
        order_vec = np.asarray(list(orders), dtype=int)
        if order_vec.size != n_ord:
            raise ValueError(
                f"orders length {order_vec.size} != LFC WAVE rows {n_ord}"
            )
    dlam = lfc - hdf
    dvel = np.asarray(delta_wave_to_delta_vel(dlam, hdf), dtype=float)
    ok = np.isfinite(dlam) & np.isfinite(dvel) & np.isfinite(hdf) & (np.abs(hdf) > 0)
    if max_abs_dvel_m_s is not None:
        ok = ok & (np.abs(dvel) <= float(max_abs_dvel_m_s))
    per_order: list[dict[str, float]] = []
    for i, od in enumerate(order_vec):
        sel = ok[i]
        dv = dvel[i, sel]
        dl = dlam[i, sel]
        if dv.size == 0:
            per_order.append(
                {
                    "order": int(od),
                    "n": 0,
                    "median_dlam_um": float("nan"),
                    "median_dvel_m_s": float("nan"),
                    "mad_dvel_m_s": float("nan"),
                }
            )
            continue
        med_v = float(np.median(dv))
        per_order.append(
            {
                "order": int(od),
                "n": int(dv.size),
                "median_dlam_um": float(np.median(dl)),
                "median_dvel_m_s": med_v,
                "mad_dvel_m_s": float(np.median(np.abs(dv - med_v))),
            }
        )
    dv_all = dvel[ok]
    med_all = float(np.median(dv_all)) if dv_all.size else float("nan")
    finite = np.isfinite(dlam) & np.isfinite(dvel) & np.isfinite(hdf) & (np.abs(hdf) > 0)
    return {
        "dlam_um": dlam,
        "dvel_m_s": dvel,
        "good": ok,
        "orders": order_vec,
        "n_pix": int(n_pix),
        "n_clipped": int(np.count_nonzero(finite & ~ok)),
        "max_abs_dvel_m_s": (
            None if max_abs_dvel_m_s is None else float(max_abs_dvel_m_s)
        ),
        "median_dvel_m_s": med_all,
        "mad_dvel_m_s": (
            float(np.median(np.abs(dv_all - med_all))) if dv_all.size else float("nan")
        ),
        "per_order": per_order,
    }


def plot_lfc_vs_hdf_wavesol_shift(
    lfc_wave: np.ndarray,
    hdf_wave: np.ndarray,
    *,
    orders: Sequence[int] | None = None,
    xlim: tuple[float, float] | None = None,
    max_abs_dvel_m_s: float | None = None,
    resid_percentile: tuple[float, float] | None = None,
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """Order-by-order LFC minus HDF WAVE, in wavelength and velocity.

    If ``resid_percentile`` is given (e.g. ``(1, 99)``), both panels'
    y-limits are set to those percentiles of the plotted residuals
    (plus a 10% pad) instead of autoscaling to the full range.
    """
    stats = lfc_vs_hdf_wavesol_shift(
        lfc_wave, hdf_wave, orders=orders, max_abs_dvel_m_s=max_abs_dvel_m_s
    )
    dlam = stats["dlam_um"]
    dvel = stats["dvel_m_s"]
    good = stats["good"]
    order_vec = stats["orders"]
    pix = np.arange(stats["n_pix"], dtype=float)
    title = "ProcessLFCPipeline WAVE vs get_wavelength_solution"
    if xlim is not None:
        title = f"{title}  (pixels {int(xlim[0])}–{int(xlim[1])})"
    fig, (ax_l, ax_v) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for i, od in enumerate(order_vec):
        sel = good[i]
        if xlim is not None:
            sel = sel & (pix >= float(xlim[0])) & (pix <= float(xlim[1]))
        if int(sel.sum()) == 0:
            continue
        ax_l.plot(pix[sel], dlam[i, sel] * 1.0e7, lw=0.9, label=f"order {int(od)}")
        ax_v.plot(pix[sel], dvel[i, sel], lw=0.9, label=f"order {int(od)}")
    _style_fwhm_wavelength_axes(
        ax_l,
        xlabel="",
        ylabel=r"$\Delta\lambda$ (mÅ)  [LFC − HDF]",
        title=title,
        n_legend=0,
    )
    _style_fwhm_wavelength_axes(
        ax_v,
        xlabel="spectral pixel",
        ylabel=r"$\Delta v$ (m/s)  [LFC − HDF]",
        title="",
        n_legend=len(order_vec),
    )
    if xlim is not None:
        ax_l.set_xlim(float(xlim[0]), float(xlim[1]))
        ax_v.set_xlim(float(xlim[0]), float(xlim[1]))
    if resid_percentile is not None:
        sel_all = good
        if xlim is not None:
            in_x = (pix >= float(xlim[0])) & (pix <= float(xlim[1]))
            sel_all = good & in_x[None, :]
        for ax, arr, scale in ((ax_l, dlam, 1.0e7), (ax_v, dvel, 1.0)):
            vals = np.asarray(arr, dtype=float)[sel_all] * scale
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo_y, hi_y = np.percentile(
                    vals, [float(resid_percentile[0]), float(resid_percentile[1])]
                )
                pad = 0.1 * (hi_y - lo_y)
                ax.set_ylim(lo_y - pad, hi_y + pad)
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    stats["out_path"] = str(out_path) if out_path is not None else None
    stats["xlim"] = tuple(float(v) for v in xlim) if xlim is not None else None
    return stats


def plot_lfc_vs_hdf_wavesol_vs_wavelength(
    lfc_wave: np.ndarray,
    hdf_wave: np.ndarray,
    *,
    orders: Sequence[int] | None = None,
    lam_xlim: tuple[float, float] | None = None,
    max_abs_dvel_m_s: float | None = None,
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """``LFC − HDF`` velocity versus ``λ_LFC``, one trace per order.

    Same pixel-wise comparison as :func:`plot_lfc_vs_hdf_wavesol_shift`,
    remapped onto wavelength so the axes match the usual
    ``λ_LFC − λ_SIM`` vs ``λ_LFC`` diagnostic.
    """
    stats = lfc_vs_hdf_wavesol_shift(
        lfc_wave, hdf_wave, orders=orders, max_abs_dvel_m_s=max_abs_dvel_m_s
    )
    lfc = np.asarray(lfc_wave, dtype=float)
    dvel = stats["dvel_m_s"]
    good = stats["good"]
    order_vec = stats["orders"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    n_drawn = 0
    for i, od in enumerate(order_vec):
        sel = good[i]
        if lam_xlim is not None:
            sel = sel & (lfc[i] >= float(lam_xlim[0])) & (lfc[i] <= float(lam_xlim[1]))
        if int(sel.sum()) == 0:
            continue
        lam = lfc[i, sel]
        dv = dvel[i, sel]
        order = np.argsort(lam)
        ax.plot(lam[order], dv[order], lw=0.9, label=f"order {int(od)}")
        n_drawn += 1
    ax.set_xlabel(r"$\lambda_{\rm LFC}$ (micron)")
    ax.set_ylabel(r"$\lambda_{\rm LFC}-\lambda_{\rm SIM}$ (m/s)")
    ax.grid(True, alpha=0.3)
    if n_drawn and n_drawn <= 8:
        ax.legend(fontsize=8, ncols=1)
    if lam_xlim is not None:
        ax.set_xlim(float(lam_xlim[0]), float(lam_xlim[1]))
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    stats["out_path"] = str(out_path) if out_path is not None else None
    stats["lam_xlim"] = (
        tuple(float(v) for v in lam_xlim) if lam_xlim is not None else None
    )
    return stats
