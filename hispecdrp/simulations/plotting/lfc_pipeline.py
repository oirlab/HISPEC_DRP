"""Plots for the extracted-LFC / FitLFCLines / Gaussian-LSF documentation example."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
    spec = np.asarray(getattr(model, f"spec_fib{int(fiber)}"), dtype=float)
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
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"LFC line fits  fiber {int(fiber)}")
    if n_plotted:
        ax.legend(ncols=3, fontsize=8)
    ax.grid(True, alpha=0.3)
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
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel(r"spectral resolution [$R=\lambda/\Delta\lambda$(FWHM)]")
    ax.set_title(f"Gaussian LSF  fiber {int(fiber)}")
    if n_plotted:
        ax.legend(ncols=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _display_or_save(fig, out_path=out_path, display=display)
    return {
        "fiber": int(fiber),
        "n_orders": int(n_plotted),
        "out_path": str(out_path) if out_path is not None else None,
    }
