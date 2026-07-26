#!/usr/bin/env python
"""
Simulate up-the-ramp data and run Stage 0 and Stage 1 pipelines.

Uses hispecdrp.simulations to create a rate map (with spectra only on requested
fibers), ramp, and Stage 0 result, then (if fiberflat files are available) runs
the fiberflat, Stage 1, and Stage 2 pipelines. Stage 2 runs blaze correction
using the same fiber flat and cross correlation using the same spectral template
as the rate map; it is only run when a valid --spectrum template file exists. Use --fibers to restrict to fibers 1, 2, 3, and/or 4. Use --spectrum with
a BT-Settl .h5 file path to use that spectrum for the rate map instead of a flat spectrum.

Calibration cache uses the default (see hispecdrp.calibrations.run_caldb) if KOA_CALIBRATION_CACHE is not set.
Dummy fiber flat rate files are created with a blackbody spectrum (0.8--3.0 µm, optional temperature and scale)
using the same create_ramp / Stage 0 path, then run through the fiber flat pipeline.

Usage:
    python -m hispecdrp.scripts.run_simulated_pipeline [--datadir DIR] [--fibers 1,2,3] [options]
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import threading
import time

# Suppress OpenMP deprecation info when using --max-cores
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "0")

import numpy as np
from scipy.interpolate import interp1d

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hispecdrp import datamodels
from hispecdrp.calibrations.run_caldb import get_master_calibrations
from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline, ProcessFiberflatsPipeline
from hispecdrp.simulations import simulate_up_the_ramp as simulations
from hispecdrp.utils.logging import LogConfig, ROOT_LOGGER
from hispecdrp.primitives.utils.spectral_utils import get_spectrum_path, load_btsettl_spectrum

# Use package logger name so records propagate to "hispecdrp" (which has the handler).
# When run as __main__, __name__ is "__main__", so they would go to root and be dropped (root is WARNING, no handler).
logger = logging.getLogger("hispecdrp.scripts.run_simulated_pipeline")


def _ensure_script_logging():
    """Ensure the hispecdrp logger has a live handler so messages after pipeline runs are emitted."""
    log = logging.getLogger(ROOT_LOGGER)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        log.addHandler(h)
    if log.level > logging.INFO:
        log.setLevel(logging.INFO)


def _plot_spectrum_fiber(
    spec: np.ndarray,
    err: np.ndarray,
    wave: np.ndarray | None,
    do_orders,
    fiber: int,
    method: str,
    output_dir: str,
) -> str | None:
    """
    Plot 1D spectra (spec and err) vs wavelength (micron) if provided, else vs array index.
    Save one PNG per fiber with orders in a grid of 6 columns for easier viewing.

    Returns
    -------
    str or None
        Path to the saved PNG file, or None if plotting failed.
    """
    n_orders = spec.shape[0]
    if n_orders == 0:
        return None
    orders = list(do_orders) if hasattr(do_orders, "__iter__") and not isinstance(do_orders, (str, bytes)) else list(range(n_orders))
    if len(orders) != n_orders:
        orders = list(range(n_orders))

    ncol = 6
    nrow = (n_orders + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.4 * nrow), squeeze=False)
    # Flatten so we can index by order; hide unused subplots in last row
    axes_flat = axes.ravel()
    for i in range(n_orders):
        ax = axes_flat[i]
        order = orders[i]
        wave_arr = None
        if wave is not None:
            wave_arr = np.asarray(getattr(wave, "value", wave))
        if wave_arr is not None and getattr(wave_arr, "shape", None) == spec.shape:
            x = wave_arr[i, :]
            xlabel = "wavelength (microns)"
        else:
            x = np.arange(spec.shape[1])
            xlabel = "pixel index"
        ax.plot(x, spec[i, :], color="C0", label="spec", linewidth=0.8)
        ax.plot(x, err[i, :], color="C1", alpha=0.8, label="err", linewidth=0.8)
        ax.set_ylabel("flux (uncal)")
        ax.set_title(f"Order {order}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    for idx in range(n_orders, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    axes_flat[(nrow - 1) * ncol].set_xlabel(xlabel)
    fig.suptitle(f"Fiber {fiber} — {method.upper()} extraction", y=1.002)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"spectrum_FIB{fiber}_{method}.png")
    try:
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outpath


def _plot_rvccf_chunks(
    l2_model,
    input_radial_velocity_km_s: float,
    output_dir: str,
) -> str | None:
    """
    Plot per-chunk radial velocity vs wavelength from L2 RVCCF_* tables and overplot input RV.

    Reads RVCCF_[BSPEC/RSPEC]_FIB# tables (WAVELENGTH, RV, RV_ERR in m/s), plots wavelength
    (micron) vs RV with error bars for each fiber, and adds a dashed horizontal line at the
    input radial velocity (converted from km/s to m/s). Saves the figure in output_dir.

    Parameters
    ----------
    l2_model : DataModel
        Level 2 model with RVCCF_{spectrograph}_FIB# table extensions.
    input_radial_velocity_km_s : float
        Input radial velocity used in the simulated pipeline (km/s).
    output_dir : str
        Directory to save the PNG file.

    Returns
    -------
    str or None
        Path to the saved figure, or None if no RVCCF data or plot failed.
    """
    try:
        spectrograph = getattr(
            getattr(l2_model, "meta", None),
            "spectrograph",
            None,
        )
    except Exception:
        spectrograph = None
    if spectrograph not in ("BSPEC", "RSPEC"):
        spectrograph = "BSPEC"
    ext_prefix = f"rvccf_{spectrograph}_fib"
    fibers_with_data = []
    data_per_fiber = {}
    for f in (1, 2, 3, 4):
        attr = f"{ext_prefix}{f}"
        tbl = getattr(l2_model, attr, None)
        if tbl is None and hasattr(l2_model, attr):
            tbl = getattr(l2_model, attr)
        if tbl is not None and hasattr(tbl, "columns") and len(tbl) > 0:
            if "WAVELENGTH" not in tbl.colnames or "RV" not in tbl.colnames:
                continue
            wl = np.asarray(tbl["WAVELENGTH"])
            rv = np.asarray(tbl["RV"])
            rv_err = np.asarray(tbl["RV_ERR"]) if "RV_ERR" in tbl.colnames else None
            fibers_with_data.append(f)
            data_per_fiber[f] = (wl, rv, rv_err)
    if not fibers_with_data:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    input_rv_m_s = input_radial_velocity_km_s * 1000.0
    no_signal_plotted = False
    for fib in fibers_with_data:
        wl, rv, rv_err = data_per_fiber[fib]
        valid = np.isfinite(rv)
        invalid = ~valid
        if np.any(valid):
            wl_v, rv_v = wl[valid], rv[valid]
            err_v = rv_err[valid] if rv_err is not None and len(rv_err) == len(rv) else None
            if err_v is not None and np.any(np.isfinite(err_v)):
                ax.errorbar(wl_v, rv_v, yerr=err_v, fmt="o", capsize=2, label=f"Fiber {fib}", alpha=0.8)
            else:
                ax.plot(wl_v, rv_v, "o", label=f"Fiber {fib}", alpha=0.8)
        if np.any(invalid):
            ax.plot(wl[invalid], np.zeros(np.sum(invalid)), "x", color="gray", markersize=6, markeredgewidth=1.5, label="No RV signal" if not no_signal_plotted else None, alpha=0.7)
            no_signal_plotted = True
    ax.axhline(input_rv_m_s, color="k", linestyle="--", linewidth=1.5, label=f"Input RV ({input_radial_velocity_km_s} km/s)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_title(f"Per-chunk CCF RV vs wavelength ({spectrograph})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "rvccf_chunks_vs_wavelength.png")
    try:
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outpath


# Speed of light in m/s for template RV shift
_C_M_S = 299792458.0


def _plot_rvccf_spectra(
    l2_model,
    template_path: str,
    output_dir: str,
    window_size_um: float = 0.3,
    max_panels: int = 48,
) -> str | None:
    """
    Plot blaze-corrected spectrum vs wavelength for each CCF chunk with template shifted by chunk RV.

    For each chunk in the RVCCF table, extracts the blaze spectrum in that wavelength range,
    shifts the spectral template by the chunk's measured RV, and plots both in a panel.
    Output is one figure with a grid of panels (one per chunk), wavelength on x-axis.

    Parameters
    ----------
    l2_model : DataModel
        Level 2 model with SPEC_FIB, WAVE_FIB and RVCCF_* table extensions.
    template_path : str
        Path to BT-Settl (or similar) template .h5 file.
    output_dir : str
        Directory to save the PNG file.
    window_size_um : float
        Half-width in micron for chunk wavelength range (default 0.3).
    max_panels : int
        Maximum number of panels per figure (default 48). Extra chunks are skipped.

    Returns
    -------
    str or None
        Path to the saved figure, or None if no data or plot failed.
    """
    try:
        spectrograph = getattr(
            getattr(l2_model, "meta", None),
            "spectrograph",
            None,
        )
    except Exception:
        spectrograph = None
    if spectrograph not in ("BSPEC", "RSPEC"):
        spectrograph = "BSPEC"

    try:
        w_t, f_t = load_btsettl_spectrum(template_path, wl_min_um=0.8, wl_max_um=3.0)
        template_wl = np.asarray(w_t.to_value("micron"), dtype=np.float64)
        template_flux = np.asarray(f_t.value, dtype=np.float64)
        ok_t = np.isfinite(template_wl) & np.isfinite(template_flux) & (template_wl > 0)
        template_wl = template_wl[ok_t]
        template_flux = template_flux[ok_t]
        if len(template_wl) < 10:
            return None
    except Exception:
        return None

    half = window_size_um / 2.0
    ext_prefix = f"rvccf_{spectrograph}_fib"
    chunks_to_plot = []

    for fib in (1, 2, 3, 4):
        tbl = getattr(l2_model, f"{ext_prefix}{fib}", None)
        if tbl is None or not hasattr(tbl, "columns") or len(tbl) == 0:
            continue
        if "WAVELENGTH" not in tbl.colnames or "RV" not in tbl.colnames:
            continue
        spec_attr = f"spec_fib{fib}"
        wave_attr = f"wave_fib{fib}"
        if not hasattr(l2_model, spec_attr) or getattr(l2_model, spec_attr) is None:
            continue
        if not hasattr(l2_model, wave_attr) or getattr(l2_model, wave_attr) is None:
            continue
        wave_all = np.asarray(getattr(l2_model, wave_attr), dtype=np.float64)
        spec_all = np.asarray(getattr(l2_model, spec_attr), dtype=np.float64)
        n_orders = wave_all.shape[0]
        wl_centers = np.asarray(tbl["WAVELENGTH"])
        rvs = np.asarray(tbl["RV"])
        for idx in range(len(tbl)):
            wl_c = wl_centers[idx]
            rv = rvs[idx]
            if not np.isfinite(wl_c):
                continue
            wl_lo, wl_hi = wl_c - half, wl_c + half
            for oi in range(n_orders):
                w = wave_all[oi]
                if not np.any(np.isfinite(w)):
                    continue
                in_chunk = (w >= wl_lo) & (w <= wl_hi) & np.isfinite(w)
                if np.sum(in_chunk) < 5:
                    continue
                wave_chunk = np.asarray(w[in_chunk], dtype=np.float64)
                spec_chunk = np.asarray(spec_all[oi][in_chunk], dtype=np.float64)
                rv_m_s = float(rv) if np.isfinite(rv) else 0.0
                chunks_to_plot.append((fib, idx, wl_c, rv_m_s, wave_chunk, spec_chunk))
                break

    if not chunks_to_plot:
        return None

    n_show = min(len(chunks_to_plot), max_panels)
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for k in range(n_show):
        fib, idx, wl_c, rv_m_s, wave_chunk, spec_chunk = chunks_to_plot[k]
        ax = axes_flat[k]
        ax.plot(wave_chunk, spec_chunk, "b-", lw=0.8, alpha=0.9, label="Blaze-corrected")
        wl_rest = wave_chunk / (1.0 + rv_m_s / _C_M_S)
        t_interp = interp1d(
            template_wl,
            template_flux,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        t_flux = t_interp(wl_rest)
        valid = np.isfinite(t_flux)
        if np.any(valid):
            scale = np.nanmedian(spec_chunk[valid]) / np.nanmedian(t_flux[valid]) if np.any(np.isfinite(spec_chunk) & valid) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            ax.plot(wave_chunk, t_flux * scale, "r-", lw=0.8, alpha=0.8, label="Template (shifted)")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Flux")
        ax.set_title(f"F{fib} chunk {idx + 1} λ={wl_c:.3f} µm RV={rv_m_s:.0f} m/s")
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

    for k in range(n_show, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(f"Per-chunk spectra and shifted template ({spectrograph})", y=1.002)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "rvccf_spectra_per_chunk.png")
    try:
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outpath


def _attach_stage0_calibrations(pipe: Stage0Pipeline, calibration_models: dict) -> None:
    """Attach calibration models to Stage0Pipeline primitives (for simulated or master cals)."""
    pipe.dq_init.dq = calibration_models["dq"]
    pipe.saturation_check.saturation = calibration_models["saturation"]
    pipe.bias_sub.bias = calibration_models["bias"]
    pipe.ramp_fit.gain = calibration_models["gain"]
    pipe.ramp_fit.rn = calibration_models["rn"]
    pipe.dark_sub.dark = calibration_models["dark"]
    pipe.gain_corr.gain = calibration_models["gain"]
    pipe.detflat_corr.detflat = calibration_models["detflat"]


def _attach_stage2_calibrations(pipe: Stage2Pipeline, ff_results: dict) -> None:
    """Attach fiber flat from the simulated fiber flat pipeline to Stage2Pipeline blaze step."""
    pipe.blaze_corr.fiberflat = ff_results["fiberflat"]


def _get_trace_positions(trace_pos_model, fiber: int):
    """Get trace position array for a fiber from TracePosModel (supports FITS and in-memory names)."""
    for name in (f"POS_FIB{fiber}", f"pos_fib{fiber}"):
        val = getattr(trace_pos_model, name, None)
        if val is not None:
            return val
    return None


def _plot_ramp_traces(
    ramp_model,
    trace_pos_model,
    output_dir: str,
) -> str | None:
    """
    Create a diagnostic plot: greyscale image from the ramp data with trace positions
    overplotted as red lines (one line per order per fiber). Saves to output_dir/ramp_traces.png.

    Returns
    -------
    str or None
        Path to the saved PNG, or None if plotting failed or no trace data.
    """
    if not hasattr(ramp_model, "data") or ramp_model.data is None:
        return None
    img = np.mean(ramp_model.data, axis=0)
    ny, nx = img.shape

    orders_are_vertical = getattr(
        getattr(trace_pos_model, "meta", None),
        "orders_are_vertical",
        True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    # Inverted grey: larger values darker (gray_r)
    ax.imshow(img, aspect="auto", cmap="gray_r", vmin=vmin, vmax=vmax, origin="upper")

    # Trace position convention (from tracepos_primitive): when orders_are_vertical, pixel_arr = arange(ny),
    # so pos has shape (n_orders, ny) and pos[order, row_idx] = column position at that row. So (x, y) = (col, row) = (pos, row_idx).
    # When orders_are_vertical is False, pixel_arr = arange(nx), so pos has shape (n_orders, nx) and pos[order, col_idx] = row position; (x, y) = (col_idx, pos).
    n_plotted = 0
    for fiber in range(1, 20):
        pos = _get_trace_positions(trace_pos_model, fiber)
        if pos is None:
            continue
        pos = np.asarray(pos)
        if pos.ndim != 2:
            continue
        n_orders, n_pix = pos.shape
        for ord_idx in range(n_orders):
            trace_vals = np.asarray(pos[ord_idx, :], dtype=np.float64)
            valid = np.isfinite(trace_vals)
            if not np.any(valid):
                continue
            x_old = np.arange(n_pix, dtype=np.float64)
            if orders_are_vertical:
                # pos = column position as function of row index; need (x, y) = (col_pos, row_idx) with length ny
                if n_pix != ny:
                    x_new = np.linspace(0, n_pix - 1, ny)
                    trace_interp = np.interp(x_new, x_old[valid], trace_vals[valid])
                else:
                    trace_interp = trace_vals
                x_plot = trace_interp
                y_plot = np.arange(ny)
            else:
                # pos = row position as function of column index; (x, y) = (col_idx, row_pos) with length nx
                if n_pix != nx:
                    x_new = np.linspace(0, n_pix - 1, nx)
                    trace_interp = np.interp(x_new, x_old[valid], trace_vals[valid])
                else:
                    trace_interp = trace_vals
                x_plot = np.arange(nx)
                y_plot = trace_interp
            # Trace positions are in array index convention (col, row); imshow(origin="upper") matches, so no flip
            ax.plot(x_plot, y_plot, "r-", linewidth=0.6, alpha=0.9)
            n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        return None
    ax.set_xlabel("column (pixel index)")
    ax.set_ylabel("row (pixel index)")
    ax.set_title("Ramp image with trace positions overplotted")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "ramp_traces.png")
    try:
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outpath


try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


@contextlib.contextmanager
def _memory_sampler(interval_s: float = 0.3):
    """Context manager that samples process RSS in a background thread. Sets .peak_gb and .avg_gb on exit."""
    class Sampler:
        peak_gb: float | None = None
        avg_gb: float | None = None

    sampler = Sampler()
    samples: list[int] = []
    stop = threading.Event()

    def _sample():
        proc = psutil.Process()
        while True:
            try:
                samples.append(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if stop.wait(interval_s):
                break

    if psutil is None:
        yield sampler
        return

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield sampler
    finally:
        stop.set()
        t.join(timeout=interval_s * 4)
    if samples:
        sampler.peak_gb = max(samples) / (1024**3)
        sampler.avg_gb = sum(samples) / len(samples) / (1024**3)


ALLOWED_FIBERS = (1, 2, 3, 4)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate up-the-ramp and run Stage 0 and Stage 1.")
    parser.add_argument(
        "--datadir",
        "-d",
        default=".",
        help="Output directory for all simulated files (default: current directory).",
    )
    parser.add_argument(
        "--fibers",
        "-f",
        default="1",
        metavar="N",
        help="Comma-separated fiber numbers (1, 2, 3, and/or 4). Default: 1",
    )
    parser.add_argument(
        "--spectrograph",
        "-s",
        default="BSPEC",
        choices=("BSPEC", "RSPEC"),
        help="Spectrograph: BSPEC (HB frames) or RSPEC (HR frames). Default: BSPEC",
    )
    parser.add_argument(
        "--spectrum",
        default="lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
        metavar="PATH",
        help="BT-Settl spectrum .h5 filename in hispecdrp/data/templates/ or absolute path. Default: lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
    )
    parser.add_argument(
        "--radial-velocity",
        "--rv",
        type=float,
        default=0.0,
        metavar="KM_S",
        dest="radial_velocity",
        help="Radial velocity in km/s to apply to the input spectrum when building the rate map (range -500 to 500). Default: 0.0",
    )
    parser.add_argument(
        "--J",
        type=float,
        default=None,
        metavar="MAG",
        help="Normalize spectrum to this J-band AB magnitude (fetches 2MASS J from SVO).",
    )
    parser.add_argument(
        "--H",
        type=float,
        default=None,
        metavar="MAG",
        help="Normalize spectrum to this H-band AB magnitude (fetches 2MASS H from SVO).",
    )
    parser.add_argument(
        "--K",
        type=float,
        default=None,
        metavar="MAG",
        help="Normalize spectrum to this K-band AB magnitude (fetches 2MASS K from SVO).",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=1,
        metavar="N",
        help="Max cores for simulation (create_ramp) and Stage 0/Stage 1 pipelines (default: 1).",
    )
    parser.add_argument(
        "--use-master-calibrations",
        action="store_true",
        help="Fetch master calibrations from calDB. Default: use simulated calibrations.",
    )
    parser.add_argument(
        "--fiberflat-scale",
        type=float,
        default=1e-14,
        metavar="F",
        help="Scale factor for dummy fiber flat blackbody spectrum (0.8--3.0 µm). Default: 1e-14",
    )
    parser.add_argument(
        "--fiberflat-temperature",
        type=float,
        default=1000.0,
        metavar="T",
        help="Blackbody temperature (K) for dummy fiber flat spectrum. Default: 1000.0",
    )
    return parser.parse_args()


def _parse_fibers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise ValueError(f"Invalid fiber '{part}'; must be integers in {ALLOWED_FIBERS}")
        if n not in ALLOWED_FIBERS:
            raise ValueError(f"Fiber must be one of {ALLOWED_FIBERS}, got {n}")
        out.append(n)
    if not out:
        raise ValueError("At least one fiber required")
    return sorted(set(out))


def main():
    args = parse_args()
    config = LogConfig(level=logging.INFO, enable_stdout=True, enable_file=False)
    config.apply([ROOT_LOGGER])

    datadir = args.datadir
    fibers = _parse_fibers(args.fibers)
    fibers_str = [str(f) for f in fibers]
    spectrograph = args.spectrograph
    if not (-500.0 <= args.radial_velocity <= 500.0):
        raise ValueError(
            f"--radial-velocity must be in the range -500 to 500 km/s, got {args.radial_velocity}"
        )
    logger.info("Output directory: %s; fibers: %s; spectrograph: %s", datadir, fibers, spectrograph)
    if args.radial_velocity != 0.0:
        logger.info("Radial velocity for rate map spectrum: %s km/s", args.radial_velocity)

    spectrum_wavelengths = None
    spectrum_flux = None
    mag_ab = None
    filter_band = None
    for band, mag in [("J", args.J), ("H", args.H), ("K", args.K)]:
        if mag is not None:
            mag_ab = mag
            filter_band = band
            break
    spectrum_path = get_spectrum_path(args.spectrum)
    try:
        spectrum_wavelengths, spectrum_flux = load_btsettl_spectrum(
            spectrum_path, mag_ab=mag_ab, filter_band=filter_band
        )
        logger.info("Using spectrum: %s", spectrum_path)
    except FileNotFoundError:
        if args.spectrum:
            logger.warning("Spectrum file not found: %s; using flat spectrum.", spectrum_path)

    # Ramp and calibrations: create or load if files already exist
    if simulations.ramp_file_exists(datadir):
        ramp_path = simulations.get_expected_ramp_path(datadir)
        logger.info("Ramp file exists, loading from %s", ramp_path)
        ramp_model = datamodels.open(ramp_path)
        if args.use_master_calibrations:
            logger.info("Loading master calibrations from calDB...")
            calibration_models = get_master_calibrations(
                meta=None,
                instrument_name=None,
                instrument_era="0.0.1",
                spectrograph=spectrograph,
                mjd_start=None,
                datetime_obs=None,
                connect_remote=True,
                sync_first=True,
            )
        else:
            calibration_models = simulations.create_simulated_calibration_models(
                shape=(4096, 4096),
                dq_bad_rows=[5],
            )
    else:
        if args.use_master_calibrations:
            logger.info("Creating ramp and calibrations from calDB (simulation only)...")
            ramp_model, calibration_models = simulations.create_ramp_and_calibrations(
                rate_map=None,
                shape=(4096, 4096),
                fibers=fibers,
                spectrum_wavelengths=spectrum_wavelengths,
                spectrum_flux=spectrum_flux,
                radial_velocity_km_s=args.radial_velocity,
                output_dir=datadir,
                calibration_models=None,
                use_master_calibrations=True,
                readtime=1.7,
                n_reads=10,
                spectrograph=spectrograph,
                max_cores=args.max_cores,
                instrument_name=None,
                instrument_era="0.0.1",
                mjd_start=None,
                datetime_obs=None,
                connect_remote=True,
                sync_caldb_first=True,
            )
        else:
            logger.info("Creating ramp and calibrations (simulation only)...")
            ramp_model, calibration_models = simulations.create_ramp_and_calibrations(
                rate_map=None,
                shape=(4096, 4096),
                fibers=fibers,
                spectrum_wavelengths=spectrum_wavelengths,
                spectrum_flux=spectrum_flux,
                radial_velocity_km_s=args.radial_velocity,
                output_dir=datadir,
                calibration_models=None,
                readtime=1.7,
                n_reads=10,
                spectrograph=spectrograph,
                max_cores=args.max_cores,
                dq_bad_rows=[5],
            )
        raw_path = ramp_model.save(
            filepath=os.path.join(os.path.abspath(datadir), ramp_model.meta.filename or "simulated_ramp_raw.fits")
        )
        logger.info("Saved raw ramp: %s", raw_path)

    stage0_prim_args = simulations.get_default_stage0_prim_args(max_cores=args.max_cores)
    fiberflat_fibers = [f for f in fibers if f in (1, 2, 3)]
    if simulations.fiberflat_rate_files_exist(datadir, spectrograph, fiberflat_fibers):
        fiberflat_filepaths = simulations.get_expected_fiberflat_rate_paths(
            datadir, spectrograph, fiberflat_fibers
        )
        logger.info(
            "Using existing dummy fiber flat rate file(s) in %s",
            os.path.join(os.path.abspath(datadir), "cals"),
        )
    else:
        fiberflat_filepaths = simulations.create_dummy_fiberflat_rate_files(
            fiberflat_fibers,
            spectrograph,
            datadir,
            calibration_models,
            flat_scale=args.fiberflat_scale,
            flat_temperature_k=args.fiberflat_temperature,
            readtime=1.7,
            n_reads=10,
            max_cores=args.max_cores,
            stage0_prim_args=stage0_prim_args,
        )
        if fiberflat_filepaths:
            logger.info(
                "Created %d dummy fiber flat rate file(s) in %s",
                len(fiberflat_filepaths),
                os.path.join(os.path.abspath(datadir), "cals"),
            )

    logger.info("Running Stage 0 pipeline...")
    start_time = time.time()
    mem = None
    try:
        with _memory_sampler(interval_s=0.3) as mem:
            stage0_pipe = Stage0Pipeline(prim_args=stage0_prim_args)
            _attach_stage0_calibrations(stage0_pipe, calibration_models)
            model_result = stage0_pipe.apply(ramp_model, output_dir=datadir)
            stage0_end_time = time.time()
            logger.info("Stage 0 complete. Elapsed: %.2fs", stage0_end_time - start_time)
            rate_filename = getattr(model_result, "meta", None) and getattr(model_result.meta, "filename", None) or "in memory"
            logger.info("Rate/slope product: %s", rate_filename)

            if fiberflat_filepaths and all(p for p in fiberflat_filepaths):
                ff_models = simulations.prepare_fiberflat_models(fiberflat_filepaths)
                ff_prim_args = simulations.get_fiberflat_prim_args(
                    ff_models,
                    yrange=(500, 4000),
                    xrange=(100, 4050),
                    deg_pos=4,
                    deg_width=2,
                    extraction_aperture=10,
                )
                ff_pipe = ProcessFiberflatsPipeline(prim_args=ff_prim_args)
                ff_results = ff_pipe.apply(ff_models)
                trace_pos_model = ff_results["trace_pos"]
                trace_prof_model = ff_results["trace_prof"]
                do_orders = model_result.make_order_vec() if hasattr(model_result, "make_order_vec") else []

                # Wavelength solution from simulation HDF for set_wavesol (Stage 1)
                try:
                    wavesol_model = simulations.create_wavelength_solution_model_from_sim(spectrograph)
                except Exception as ws_err:
                    logger.warning("Could not build wavelength solution from sim: %s; set_wavesol will be skipped.", ws_err)
                    wavesol_model = None

                try:
                    ramp_traces_path = _plot_ramp_traces(ramp_model, trace_pos_model, datadir)
                    if ramp_traces_path:
                        logger.info("Wrote ramp+trace diagnostic: %s", ramp_traces_path)
                except Exception as e:
                    logger.warning("Failed to write ramp+trace diagnostic: %s", e)

                if len(do_orders) == 0:
                    logger.warning("No orders in model (do_orders is empty); skipping per-order SNR reporting.")

                set_wavesol_args = {"run": True if wavesol_model is not None else False, "do_orders": do_orders}
                if wavesol_model is not None:
                    set_wavesol_args["wavesol"] = wavesol_model
                for method in ("box",):  # box extraction only; optimal skipped
                    stage1_prim_args = {
                        "spec_extract": {
                            "method": method,
                            "extraction_aperture": 10,
                            "do_orders": do_orders,
                            "do_fibers": fibers_str,
                            "tracepos": trace_pos_model,
                            "traceprof": trace_prof_model,
                        },
                        "set_wavesol": set_wavesol_args,
                    }
                    stage1_pipe = Stage1Pipeline(prim_args=stage1_prim_args)
                    spec_model = stage1_pipe.apply(model_result)
                    # Ensure script logger has a handler (pipeline logging_context can leave it broken)
                    _ensure_script_logging()
                    config.apply([ROOT_LOGGER])
                    logger.info("Stage 1 (%s) complete, reporting SNR.", method)
                    if spec_model is None:
                        logger.warning("Stage 1 returned None; cannot report SNR for %s.", method)
                        continue
                    try:
                        spec_path = spec_model.save(output_dir=datadir)
                        logger.info("Saved L1 spec model to %s", spec_path)
                    except Exception as save_err:
                        logger.warning("Failed to save L1 spec model to %s: %s", datadir, save_err)
                    try:
                        for fib in fibers:
                            spec = getattr(spec_model, f"SPEC_UNCAL_FIB{fib}", None)
                            if spec is None:
                                spec = getattr(spec_model, f"spec_uncal_fib{fib}", None)
                            err = getattr(spec_model, f"SPEC_UNCAL_ERR_FIB{fib}", None)
                            if err is None:
                                err = getattr(spec_model, f"spec_uncal_err_fib{fib}", None)
                            wave = getattr(spec_model, f"WAVE_FIB{fib}", None)
                            if wave is None:
                                wave = getattr(spec_model, f"wave_fib{fib}", None)
                            if spec is not None and err is not None:
                                for i, order in enumerate(do_orders):
                                    snr = np.nanmean(spec[i, :] / err[i, :])
                                    logger.info("Fiber %s Order %s: %s SNR = %.2f", fib, order, method.upper(), snr)
                                try:
                                    plot_path = _plot_spectrum_fiber(
                                        spec, err, wave, do_orders, fib, method, datadir
                                    )
                                    if plot_path:
                                        logger.info("Wrote spectrum figure: %s", plot_path)
                                except Exception as plot_err:
                                    logger.warning("Failed to write spectrum figure for fiber %s %s: %s", fib, method, plot_err)
                    except Exception as snr_err:
                        logger.exception("SNR reporting failed for %s: %s", method, snr_err)

                # Stage 2: blaze correction (same fiber flat) + stellar RV (template CCF)
                if spec_model is not None and spectrum_path is not None and os.path.isfile(spectrum_path):
                    try:
                        stage2_prim_args = {
                            "blaze_corr": {"do_fibers": fibers_str},
                            "stellar_rv": {
                                "algorithm": "ccf_stellar_template",
                                "template_path": spectrum_path,
                                "do_fibers": fibers_str,
                                "save_preview": False,
                            },
                        }
                        stage2_pipe = Stage2Pipeline(prim_args=stage2_prim_args)
                        _attach_stage2_calibrations(stage2_pipe, ff_results)
                        l2_model = stage2_pipe.apply(spec_model, output_dir=datadir)
                        _ensure_script_logging()
                        config.apply([ROOT_LOGGER])
                        logger.info("Stage 2 complete (blaze + stellar RV); L2 output: %s", getattr(l2_model, "meta", None) and getattr(l2_model.meta, "filename", None) or "in memory")
                        rvccf_plot_path = _plot_rvccf_chunks(l2_model, args.radial_velocity, datadir)
                        if rvccf_plot_path:
                            logger.info("Wrote RVCCF diagnostic: %s", rvccf_plot_path)
                        rvccf_spectra_path = _plot_rvccf_spectra(l2_model, spectrum_path, datadir)
                        if rvccf_spectra_path:
                            logger.info("Wrote RVCCF spectra diagnostic: %s", rvccf_spectra_path)
                    except Exception as stage2_err:
                        logger.warning("Stage 2 failed: %s", stage2_err, exc_info=True)
                elif spec_model is not None and (spectrum_path is None or not os.path.isfile(spectrum_path)):
                    logger.info("Skipping Stage 2: no spectral template file for stellar RV (use --spectrum with a valid path to enable Stage 2).")
    except Exception as e:
        logger.exception("Pipeline step failed: %s", e)
        raise
    finally:
        _ensure_script_logging()
        end_time = time.time()
        logger.info("Total time for pipeline L0 and L1, elapsed: %.2fs", end_time - start_time)
        if mem is not None and mem.peak_gb is not None and mem.avg_gb is not None:
            logger.info("Pipeline memory: peak %.2f GB, average %.2f GB", mem.peak_gb, mem.avg_gb)
        else:
            logger.info("Pipeline memory: not available (install psutil for memory profiling)")
        for _h in logging.root.handlers:
            _h.flush()


if __name__ == "__main__":
    main()
