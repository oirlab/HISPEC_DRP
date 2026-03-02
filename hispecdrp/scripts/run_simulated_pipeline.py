#!/usr/bin/env python
"""
Simulate up-the-ramp data and run Stage 0 and Stage 1 pipelines.

Uses hispecdrp.simulations to create a rate map (with spectra only on requested
fibers), ramp, and Stage 0 result, then (if fiberflat files are available) runs
the fiberflat and Stage 1 pipelines and prints per-order SNR for each requested
fiber. Use --fibers to restrict to fibers 1, 2, 3, and/or 4. Use --spectrum with
a BT-Settl .h5 file path to use that spectrum for the rate map instead of a flat spectrum.

Calibration cache uses the default (see run_caldb) if KOA_CALIBRATION_CACHE is not set.
Fiberflat files are downloaded via hispecdrp.utils.download_gdrive_file only for
requested fibers (FIB1, FIB2, FIB3 in testing_data): HB frames for BSPEC, HR for RSPEC.

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

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hispecdrp import simulations
from hispecdrp.utils import download_gdrive_file
from hispecdrp.utils.logging import LogConfig, ROOT_LOGGER

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
    do_orders,
    fiber: int,
    method: str,
    output_dir: str,
) -> str | None:
    """
    Plot 1D spectra (spec and err) vs array index, one subplot per order.
    Save one PNG per fiber with all orders in a vertical stack of subplots.

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

    ncol = 1
    nrow = n_orders
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 1.5 * n_orders), squeeze=True)
    if n_orders == 1:
        axes = [axes]
    for i, (ax, order) in enumerate(zip(axes, orders)):
        x = np.arange(spec.shape[1])
        ax.plot(x, spec[i, :], color="C0", label="spec", linewidth=0.8)
        ax.plot(x, err[i, :], color="C1", alpha=0.8, label="err", linewidth=0.8)
        ax.set_ylabel("flux / err")
        ax.set_title(f"Order {order}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("pixel index")
    fig.suptitle(f"Fiber {fiber} — {method.upper()} extraction", y=1.002)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"spectrum_FIB{fiber}_{method}.png")
    try:
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return outpath


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
            # Flip 180° so trace coords match image (e.g. if trace coords use opposite origin)
            x_plot = (nx - 1) - np.asarray(x_plot, dtype=np.float64)
            y_plot = (ny - 1) - np.asarray(y_plot, dtype=np.float64)
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


def _fiberflat_filename(spectrograph: str, fiber: int) -> str:
    prefix = "HB" if spectrograph.upper() == "BSPEC" else "HR"
    return f"{prefix}.20231006.00000.00_rate-FIBERFLAT-FIB{fiber}.fits"


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
        help="BT-Settl spectrum .h5 filename in simulations/data/ or absolute path. Default: lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
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
    logger.info("Output directory: %s; fibers: %s; spectrograph: %s", datadir, fibers, spectrograph)

    spectrum_wavelengths = None
    spectrum_flux = None
    mag_ab = None
    filter_band = None
    for band, mag in [("J", args.J), ("H", args.H), ("K", args.K)]:
        if mag is not None:
            mag_ab = mag
            filter_band = band
            break
    spectrum_path = simulations.get_spectrum_path(args.spectrum)
    try:
        spectrum_wavelengths, spectrum_flux = simulations.load_btsettl_spectrum(
            spectrum_path, mag_ab=mag_ab, filter_band=filter_band
        )
        logger.info("Using spectrum: %s", spectrum_path)
    except FileNotFoundError:
        if args.spectrum:
            logger.warning("Spectrum file not found: %s; using flat spectrum.", spectrum_path)

    if args.use_master_calibrations:
        logger.info("Creating ramp and calibrations from calDB (simulation only)...")
        ramp_model, calibration_models = simulations.create_ramp_and_calibrations(
            rate_map=None,
            shape=(4096, 4096),
            fibers=fibers,
            spectrum_wavelengths=spectrum_wavelengths,
            spectrum_flux=spectrum_flux,
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

    fiberflat_filepaths = []
    for f in fibers:
        if f in (1, 2, 3):
            path = download_gdrive_file(_fiberflat_filename(spectrograph, f), output_dir=datadir)
            fiberflat_filepaths.append(path)

    logger.info("Running Stage 0 pipeline...")
    start_time = time.time()
    mem = None
    try:
        with _memory_sampler(interval_s=0.3) as mem:
            stage0_prim_args = simulations.get_default_stage0_prim_args(max_cores=args.max_cores)
            model_result = simulations.run_stage0_simulation(
                ramp_model,
                calibration_models,
                output_dir=datadir,
                prim_args=stage0_prim_args,
            )
            stage0_end_time = time.time()
            logger.info("Stage 0 complete. Elapsed: %.2fs", stage0_end_time - start_time)
            rate_filename = getattr(model_result, "meta", None) and getattr(model_result.meta, "filename", None) or "in memory"
            logger.info("Rate/slope product: %s", rate_filename)

            if fiberflat_filepaths and all(p for p in fiberflat_filepaths):
                ff_results = simulations.run_fiberflat_pipeline(
                    fiberflat_filepaths,
                    yrange=(500, 4000),
                    xrange=(100, 4050),
                    deg_pos=4,
                    deg_width=2,
                    extraction_aperture=10,
                )
                trace_pos_model = ff_results["trace_pos"]
                trace_prof_model = ff_results["trace_prof"]
                do_orders = model_result.make_order_vec() if hasattr(model_result, "make_order_vec") else []

                try:
                    ramp_traces_path = _plot_ramp_traces(ramp_model, trace_pos_model, datadir)
                    if ramp_traces_path:
                        logger.info("Wrote ramp+trace diagnostic: %s", ramp_traces_path)
                except Exception as e:
                    logger.warning("Failed to write ramp+trace diagnostic: %s", e)

                if len(do_orders) == 0:
                    logger.warning("No orders in model (do_orders is empty); skipping per-order SNR reporting.")

                for method in ("box", "optimal"):
                    spec_model = simulations.run_stage1_simulation(
                        model_result,
                        trace_pos_model,
                        trace_prof_model,
                        extraction_aperture=10,
                        do_orders=do_orders,
                        do_fibers=fibers_str,
                        prim_args={
                            "spec_extract": {
                                "method": method,
                                "extraction_aperture": 10,
                                "do_orders": do_orders,
                                "do_fibers": fibers_str,
                                "tracepos": trace_pos_model,
                                "traceprof": trace_prof_model,
                            },
                            "set_wavesol": {"run": False},
                        },
                    )
                    # Ensure script logger has a handler (pipeline logging_context can leave it broken)
                    _ensure_script_logging()
                    config.apply([ROOT_LOGGER])
                    logger.info("Stage 1 (%s) complete, reporting SNR.", method)
                    if spec_model is None:
                        logger.warning("Stage 1 returned None; cannot report SNR for %s.", method)
                        continue
                    try:
                        for fib in fibers:
                            spec = getattr(spec_model, f"SPEC_UNCAL_FIB{fib}", None)
                            if spec is None:
                                spec = getattr(spec_model, f"spec_uncal_fib{fib}", None)
                            err = getattr(spec_model, f"SPEC_UNCAL_ERR_FIB{fib}", None)
                            if err is None:
                                err = getattr(spec_model, f"spec_uncal_err_fib{fib}", None)
                            if spec is not None and err is not None:
                                for i, order in enumerate(do_orders):
                                    snr = np.nanmean(spec[i, :] / err[i, :])
                                    logger.info("Fiber %s Order %s: %s SNR = %.2f", fib, order, method.upper(), snr)
                                try:
                                    plot_path = _plot_spectrum_fiber(
                                        spec, err, do_orders, fib, method, datadir
                                    )
                                    if plot_path:
                                        logger.info("Wrote spectrum figure: %s", plot_path)
                                except Exception as plot_err:
                                    logger.warning("Failed to write spectrum figure for fiber %s %s: %s", fib, method, plot_err)
                    except Exception as snr_err:
                        logger.exception("SNR reporting failed for %s: %s", method, snr_err)
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
