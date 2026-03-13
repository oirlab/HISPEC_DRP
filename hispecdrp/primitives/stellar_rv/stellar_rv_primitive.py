"""
Stellar RV primitive: masking-based CCF with template, windowed RVs,
median/MAD outlier rejection, and inverse-variance weighted mean.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import astropy.table

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base_primitive import BasePrimitive
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS

from ..utils.ccf_utils import (
    _HAS_NUMBA,
    _gaussian_ccf_model,
    combine_rvs_median_mad,
    continuum_normalize,
    fit_gaussian_to_ccf,
    masked_ccf_rv_diagnostics,
    masked_ccf_rv_full,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    rv_per_chunks_with_ccf,
    weighted_mean_rv,
)

__all__ = ["StellarRVPrimitive"]

logger = logging.getLogger(__name__)

def _load_template_from_path(filepath: str, wl_min_um: float, wl_max_um: float):
    """
    Load wavelength and flux from a BT-Settl-style HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 spectrum file.
    wl_min_um : float
        Minimum wavelength to load (micron).
    wl_max_um : float
        Maximum wavelength to load (micron).

    Returns
    -------
    tuple of np.ndarray
        (wave_um, flux) as numpy arrays in micron and flux units.
    """
    from ...utils.spectral_templates_utils import load_btsettl_spectrum
    w, f = load_btsettl_spectrum(filepath, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    return w.to_value("micron"), f.value


class StellarRVPrimitive(BasePrimitive):
    """
    Compute radial velocity via masking-based cross correlation with a spectral template.

    All radial velocities (per-chunk RV and aggregated RV) are in meters per second (m/s).
    The spectrum is split into wavelength chunks
    (window_size_um). Each chunk yields one RV from a CCF in log(lambda) space with
    the template; bad pixels are masked using DQ. Chunk RVs are combined by:
    (1) unweighted median and MAD, (2) flagging outliers at outlier_sigma * MAD,
    (3) inverse-variance weighted mean and standard deviation over the remaining chunks.
    """

    ARGUMENTS = {
        "suffix": "stellar_rv",
        "do_fibers": None,
        "do_orders": None,
        "template_path": None,
        "template_wavelength": None,
        "template_flux": None,
        "window_size_um": 0.3,
        "outlier_sigma": 4.0,
        "output_model": "L2Model",
        "max_cores": 1,
        "continuum_normalize_spec": False,
        "continuum_normalize_template": False,
        "continuum_lowpass_window": 51,
        "continuum_poly_order": 3,
        "debug_ccf": False,
    }

    CALIBRATIONS = {}

    def _perform(self):
        """
        Run stellar RV on the input model(s) and produce L2 output.

        Loads the template, then for each input model and each fiber runs
        CCF (serial or Numba-parallel), combines chunk RVs, and writes
        RVCCF_FIB# and RVCCFAVG_FIB# tables to the output L2 model.

        Returns
        -------
        dict
            With key "output" containing the output L2 model (or list of models).
        """
        if isinstance(self.input, (datamodels.DataModel, str)):
            output = None
            inp_list = [self.input]
        elif isinstance(self.input, Sequence):
            output = []
            inp_list = self.input
        else:
            raise TypeError("Input must be a DataModel, path, or sequence of these.")

        template_ln, template_flux = self._get_template()
        if template_ln is None:
            raise ValueError(
                "Template not set: provide template_path or (template_wavelength, template_flux)."
            )

        for _input in inp_list:
            input_model = datamodels.open(_input)
            output_model = self.get_output_model(
                input_model, output_model=self.output_model, include_data=True
            )
            spectrograph = self._get_spectrograph(input_model)
            do_fibers, do_orders = self.get_do_traces(
                input_model, self.do_fibers, self.do_orders
            )
            order_vec = input_model.make_order_vec()

            for fiber in do_fibers:
                self._process_fiber(
                    input_model,
                    output_model,
                    fiber,
                    spectrograph,
                    do_orders,
                    order_vec,
                    template_ln,
                    template_flux,
                )

            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        return dict(output=output)

    def _get_template(self):
        """
        Resolve template to (log wavelength, flux) numpy arrays.

        Loads from template_path if set, or uses template_wavelength and
        template_flux. Applies continuum normalization to the template if
        configured. Per-fiber wavelength clipping happens in _process_fiber.

        Returns
        -------
        tuple of np.ndarray or (None, None)
            (template_ln, template_flux) in log(micron) and flux units,
            or (None, None) if template could not be resolved.
        """
        tw = getattr(self, "template_wavelength", None)
        tf = getattr(self, "template_flux", None)
        path = getattr(self, "template_path", None)
        if path is not None and (tw is None or tf is None):
            # Load from file; use a broad range; per-fiber clipping happens in _process_fiber
            wl_min, wl_max = 0.8, 3.0
            try:
                tw, tf = _load_template_from_path(path, wl_min, wl_max)
            except Exception as e:
                logger.exception("Failed to load template from %s: %s", path, e)
                return None, None
        if tw is None or tf is None:
            return None, None
        tw = np.asarray(tw, dtype=np.float64)
        tf = np.asarray(tf, dtype=np.float64)
        ok = np.isfinite(tw) & np.isfinite(tf) & (tw > 0)
        if not np.any(ok):
            return None, None
        tw = tw[ok]
        tf = tf[ok]
        # Sort by wavelength
        idx = np.argsort(tw)
        tw = tw[idx]
        tf = tf[idx]
        if getattr(self, "continuum_normalize_template", False):
            lw = getattr(self, "continuum_lowpass_window", None)
            po = getattr(self, "continuum_poly_order", None)
            tf = continuum_normalize(tf, wavelength=tw, lowpass_window=lw, poly_order=po)
        ln_lam = np.log(tw)
        return ln_lam, tf

    @staticmethod
    def _get_spectrograph(input_model) -> str:
        """
        Return spectrograph name from model metadata.

        Parameters
        ----------
        input_model : DataModel
            Input data model with meta.spectrograph.

        Returns
        -------
        str
            "BSPEC" or "RSPEC" if present in metadata; otherwise "BSPEC".
        """
        try:
            sg = getattr(
                getattr(input_model, "meta", None),
                "spectrograph",
                None,
            )
        except Exception:
            sg = None
        if sg in ("BSPEC", "RSPEC"):
            return sg
        return "BSPEC"

    def _process_fiber(
        self,
        input_model,
        output_model,
        fiber: str,
        spectrograph: str,
        do_orders: list,
        order_vec: list,
        template_ln: np.ndarray,
        template_flux: np.ndarray,
    ):
        """
        Run CCF for one fiber and write RVCCF/RVCCFAVG tables to the output model.

        Reads SPEC_FIB{fiber}, SPEC_ERR_FIB{fiber}, SPEC_DQ_FIB{fiber}, WAVE_FIB{fiber}
        from the input. Processes each order (serial or Numba-parallel depending on
        debug_ccf and max_cores), combines chunk RVs with median/MAD outlier
        rejection and inverse-variance weighting, then writes RVCCF_FIB{fiber}
        and RVCCFAVG_FIB{fiber} extensions to the output model.

        Parameters
        ----------
        input_model : DataModel
            Input model with spectrum and wavelength arrays.
        output_model : DataModel
            Output L2 model to which RV tables are written.
        fiber : str
            Fiber number (e.g. "1").
        spectrograph : str
            "BSPEC" or "RSPEC".
        do_orders : list
            Order numbers to process.
        order_vec : list
            Full order vector (order number per row index).
        template_ln : np.ndarray
            Template log(wavelength) in log(micron).
        template_flux : np.ndarray
            Template flux.
        """
        spec_attr = f"SPEC_FIB{fiber}"
        err_attr = f"SPEC_ERR_FIB{fiber}"
        dq_attr = f"SPEC_DQ_FIB{fiber}"
        wave_attr = f"WAVE_FIB{fiber}"

        if not hasattr(input_model, spec_attr) or getattr(input_model, spec_attr) is None:
            logger.warning("No %s on input; skipping CCF for fiber %s.", spec_attr, fiber)
            return
        if not hasattr(input_model, wave_attr) or getattr(input_model, wave_attr) is None:
            logger.warning("No %s on input; skipping CCF for fiber %s.", wave_attr, fiber)
            return

        spec = getattr(input_model, spec_attr)
        err = getattr(input_model, err_attr) if hasattr(input_model, err_attr) else np.ones_like(spec)
        dq = getattr(input_model, dq_attr) if hasattr(input_model, dq_attr) else np.zeros_like(spec, dtype=np.uint32)
        wave = getattr(input_model, wave_attr)

        if err is None:
            err = np.ones_like(spec, dtype=np.float64)
        if dq is None:
            dq = np.zeros_like(spec, dtype=np.uint32)
        do_not_use = DQ_FLAGS.get("DO_NOT_USE", 1)
        debug_ccf = getattr(self, "debug_ccf", False)
        max_cores = max(1, int(getattr(self, "max_cores", 1) or 1))
        do_continuum = getattr(self, "continuum_normalize_spec", False)
        cont_lw = getattr(self, "continuum_lowpass_window", None)
        cont_po = getattr(self, "continuum_poly_order", None)

        order_tasks = [
            (oi, order)
            for oi, order in enumerate(order_vec)
            if order in do_orders
        ]

        all_rvs = []
        all_rv_errs = []
        all_wavelengths = []

        if debug_ccf:
            self._process_fiber_orders_serial_debug(
                order_tasks, wave, spec, err, dq, do_not_use,
                template_ln, template_flux, fiber,
                all_rvs, all_rv_errs, all_wavelengths,
            )
        elif _HAS_NUMBA:
            self._process_fiber_orders_parallel(
                order_tasks, wave, spec, err, dq, do_not_use,
                template_ln, template_flux, max_cores,
                do_continuum, cont_lw, cont_po,
                all_rvs, all_rv_errs, all_wavelengths,
            )
        else:
            self._process_fiber_orders_serial(
                order_tasks, wave, spec, err, dq, do_not_use,
                template_ln, template_flux,
                do_continuum, cont_lw, cont_po,
                all_rvs, all_rv_errs, all_wavelengths,
            )

        all_rvs = np.array(all_rvs)
        all_rv_errs = np.array(all_rv_errs)
        all_wavelengths = np.array(all_wavelengths)

        valid = np.isfinite(all_rvs) & np.isfinite(all_rv_errs)
        if not np.any(valid):
            logger.warning("No valid chunk RVs for fiber %s.", fiber)
            inlier = np.zeros(len(all_rvs), dtype=bool)
            rv_mean, rv_std = np.nan, np.nan
        else:
            inlier, _, _ = combine_rvs_median_mad(
                all_rvs, all_rv_errs, self.outlier_sigma
            )
            rv_mean, rv_std = weighted_mean_rv(all_rvs, all_rv_errs, inlier)

        if getattr(self, "debug_ccf", False) and len(all_wavelengths) > 0:
            half_win = self.window_size_um / 2.0
            residuals = all_rvs - rv_mean if np.isfinite(rv_mean) else np.full_like(all_rvs, np.nan)
            logger.info(
                "CCF fiber=%s inferred RV per chunk and residuals vs wavelength "
                "(rv_mean=%.1f m/s rv_std=%.1f m/s)",
                fiber, rv_mean, rv_std,
            )
            for i in range(len(all_wavelengths)):
                wl_c = all_wavelengths[i]
                wl_lo = wl_c - half_win
                wl_hi = wl_c + half_win
                rv = all_rvs[i]
                res = residuals[i]
                logger.info(
                    "  wl_range=[%.4f, %.4f] um wl_center=%.4f um rv=%.1f m/s rv_err=%.1f residual=%.1f m/s",
                    wl_lo, wl_hi, wl_c, rv, all_rv_errs[i], res,
                )

        self._write_rv_tables(
            output_model,
            fiber,
            spectrograph,
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            rv_mean,
            rv_std,
            inlier=inlier,
        )
        # Store one representative CCF curve (first order, middle chunk) for preview power spectrum
        self._store_ccf_curve_for_preview(
            fiber,
            order_tasks,
            wave,
            spec,
            err,
            dq,
            do_not_use,
            template_ln,
            template_flux,
            do_continuum=do_continuum,
            cont_lw=cont_lw,
            cont_po=cont_po,
        )

    def _store_ccf_curve_for_preview(
        self,
        fiber: str,
        order_tasks: list,
        wave: np.ndarray,
        spec: np.ndarray,
        err: np.ndarray,
        dq: np.ndarray,
        do_not_use: int,
        template_ln: np.ndarray,
        template_flux: np.ndarray,
        do_continuum: bool = False,
        cont_lw: int | None = None,
        cont_po: int | None = None,
    ):
        """
        Compute CCF vs velocity for every chunk, average across all chunks and orders,
        and store on the primitive (self._ccf_curves) for the preview. In-memory only.
        """
        if not order_tasks:
            return
        all_ccf = []
        v_grid_ref = None
        for oi, _ in order_tasks:
            w = np.asarray(wave[oi], dtype=np.float64)
            s = np.asarray(spec[oi], dtype=np.float64).copy()
            e = np.asarray(err[oi] if err.ndim >= 2 else err, dtype=np.float64)
            d = np.asarray(dq[oi] if dq.ndim >= 2 else dq, dtype=np.uint32)
            mask = (d & do_not_use) != 0
            if do_continuum and (cont_lw is not None or cont_po is not None):
                s = continuum_normalize(s, wavelength=w, lowpass_window=cont_lw, poly_order=cont_po)
            w_min, w_max = np.nanmin(w), np.nanmax(w)
            ln_min, ln_max = np.log(w_min), np.log(w_max)
            in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
            if np.sum(in_range) < 20:
                continue
            t_ln = template_ln[in_range]
            t_flux = template_flux[in_range]
            _rvs, _rv_errs, _centers, v_grid, ccf_2d = rv_per_chunks_with_ccf(
                w, s, e, mask,
                t_ln, t_flux,
                self.window_size_um,
            )
            if v_grid_ref is None:
                v_grid_ref = v_grid
            all_ccf.append(ccf_2d)
        if not all_ccf or v_grid_ref is None:
            return
        stacked = np.vstack(all_ccf)
        combined_ccf = np.nanmean(stacked, axis=0)
        tbl = astropy.table.Table({
            "VELOCITY": np.asarray(v_grid_ref, dtype=np.float64),
            "CCF": np.asarray(combined_ccf, dtype=np.float64),
        })
        if not hasattr(self, "_ccf_curves"):
            self._ccf_curves = {}
        self._ccf_curves[fiber] = tbl

    def _process_fiber_orders_serial_debug(
        self,
        order_tasks: list,
        wave: np.ndarray,
        spec: np.ndarray,
        err: np.ndarray,
        dq: np.ndarray,
        do_not_use: int,
        template_ln: np.ndarray,
        template_flux: np.ndarray,
        fiber: str,
        all_rvs: list,
        all_rv_errs: list,
        all_wavelengths: list,
    ):
        """
        Process each order serially with per-chunk logging (debug_ccf=True).

        Runs CCF chunk-by-chunk and logs each chunk's diagnostics (wl_center,
        n_good, ccf_max, rv, rv_err, etc.) via the logger. Appends results
        to all_rvs, all_rv_errs, all_wavelengths in place.

        Parameters
        ----------
        order_tasks : list of tuple (int, int)
            (order_index, order_number) for each order to process.
        wave, spec, err, dq : np.ndarray
            2D arrays (n_orders x n_pix) for this fiber.
        do_not_use : int
            DQ flag value for bad pixels.
        template_ln, template_flux : np.ndarray
            Template in log(wavelength) and flux.
        fiber : str
            Fiber identifier for log messages.
        all_rvs, all_rv_errs, all_wavelengths : list
            Lists to append per-chunk RV (m/s), RV_ERR (m/s), and wavelength (µm).
        """
        for oi, order in order_tasks:
            w = np.asarray(wave[oi], dtype=np.float64)
            s = np.asarray(spec[oi], dtype=np.float64)
            e = err[oi] if err.ndim >= 2 else err
            d = dq[oi] if dq.ndim >= 2 else dq
            mask = (np.asarray(d, dtype=np.uint32) & do_not_use) != 0
            e = np.asarray(e, dtype=np.float64)
            if getattr(self, "continuum_normalize_spec", False):
                lw = getattr(self, "continuum_lowpass_window", None)
                po = getattr(self, "continuum_poly_order", None)
                s = continuum_normalize(s, wavelength=w, lowpass_window=lw, poly_order=po)
            w_min, w_max = np.nanmin(w), np.nanmax(w)
            ln_min, ln_max = np.log(w_min), np.log(w_max)
            in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
            if np.sum(in_range) < 20:
                continue
            t_ln = template_ln[in_range]
            t_flux = template_flux[in_range]
            n_chunk = max(1, int((w_max - w_min) / self.window_size_um))
            for k in range(n_chunk):
                lam_lo = w_min + k * (w_max - w_min) / n_chunk
                lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
                in_chunk = (w >= lam_lo) & (w < lam_hi)
                if np.sum(in_chunk) < 20:
                    continue
                wl_center = 0.5 * (lam_lo + lam_hi)
                rv, rv_err, diag = masked_ccf_rv_diagnostics(
                    w[in_chunk], s[in_chunk], e[in_chunk], mask[in_chunk],
                    t_ln, t_flux,
                )
                logger.info(
                    "CCF fiber=%s order=%s chunk=%s wl_center=%.4f um wl_range=[%.4f, %.4f] "
                    "n_good=%s n_use_at_peak=%s ccf_max=%.6f ccf_min=%.6f "
                    "spec_mean=%.4e spec_std=%.4e tpl_mean=%.4e tpl_std=%.4e rv=%.1f m/s rv_err=%.1f",
                    fiber, order, k, wl_center, diag["wl_min"], diag["wl_max"],
                    diag["n_good"], diag["n_use_at_peak"],
                    diag["ccf_max"], diag["ccf_min"],
                    diag["spec_mean"], diag["spec_std"],
                    diag["template_mean_at_peak"], diag["template_std_at_peak"],
                    rv, rv_err,
                )
                all_rvs.append(rv)
                all_rv_errs.append(rv_err)
                all_wavelengths.append(wl_center)

    def _process_fiber_orders_serial(
        self,
        order_tasks: list,
        wave: np.ndarray,
        spec: np.ndarray,
        err: np.ndarray,
        dq: np.ndarray,
        do_not_use: int,
        template_ln: np.ndarray,
        template_flux: np.ndarray,
        do_continuum: bool,
        cont_lw: int | None,
        cont_po: int | None,
        all_rvs: list,
        all_rv_errs: list,
        all_wavelengths: list,
    ):
        """
        Process each order serially using process_one_order_rvs.

        Calls process_one_order_rvs for each (order_index, order_number) and
        appends the returned chunk RVs, errors, and wavelength centers to
        all_rvs, all_rv_errs, all_wavelengths in place.

        Parameters
        ----------
        order_tasks : list of tuple (int, int)
            (order_index, order_number) for each order to process.
        wave, spec, err, dq : np.ndarray
            2D arrays (n_orders x n_pix) for this fiber.
        do_not_use : int
            DQ flag value for bad pixels.
        template_ln, template_flux : np.ndarray
            Template in log(wavelength) and flux.
        do_continuum : bool
            Whether to continuum-normalize the spectrum before CCF.
        cont_lw, cont_po : int or None
            Continuum lowpass window and polynomial order (if do_continuum).
        all_rvs, all_rv_errs, all_wavelengths : list
            Lists to append per-chunk RV (m/s), RV_ERR (m/s), wavelength (µm).
        """
        for oi, _order in order_tasks:
            w = wave[oi]
            s = spec[oi]
            e = err[oi] if err.ndim >= 2 else err
            d = dq[oi] if dq.ndim >= 2 else dq
            rvs, rv_errs, chunk_centers = process_one_order_rvs(
                w, s, e, d, do_not_use,
                template_ln, template_flux,
                self.window_size_um,
                do_continuum_normalize=do_continuum,
                continuum_lowpass_window=cont_lw,
                continuum_poly_order=cont_po,
            )
            for i in range(len(rvs)):
                all_rvs.append(rvs[i])
                all_rv_errs.append(rv_errs[i])
                all_wavelengths.append(chunk_centers[i])

    def _process_fiber_orders_parallel(
        self,
        order_tasks: list,
        wave: np.ndarray,
        spec: np.ndarray,
        err: np.ndarray,
        dq: np.ndarray,
        do_not_use: int,
        template_ln: np.ndarray,
        template_flux: np.ndarray,
        max_cores: int,
        do_continuum: bool,
        cont_lw: int | None,
        cont_po: int | None,
        all_rvs: list,
        all_rv_errs: list,
        all_wavelengths: list,
    ):
        """
        Process orders in parallel using Numba prange.

        Optionally continuum-normalizes the spectrum in Python, then calls
        process_orders_rvs_numba_parallel to run CCF over orders in parallel
        (prange). Appends the returned chunk RVs, errors, and wavelengths
        to all_rvs, all_rv_errs, all_wavelengths in place.

        Parameters
        ----------
        order_tasks : list of tuple (int, int)
            (order_index, order_number) for each order to process.
        wave, spec, err, dq : np.ndarray
            2D arrays (n_orders x n_pix) for this fiber.
        do_not_use : int
            DQ flag value for bad pixels.
        template_ln, template_flux : np.ndarray
            Template in log(wavelength) and flux.
        max_cores : int
            Maximum number of threads for Numba parallel execution.
        do_continuum : bool
            Whether to continuum-normalize the spectrum before CCF.
        cont_lw, cont_po : int or None
            Continuum lowpass window and polynomial order (if do_continuum).
        all_rvs, all_rv_errs, all_wavelengths : list
            Lists to append per-chunk RV (m/s), RV_ERR (m/s), wavelength (µm).
        """
        spec_2d = np.asarray(spec, dtype=np.float64)
        if do_continuum and (cont_lw is not None or cont_po is not None):
            spec_2d = spec_2d.copy()
            for oi, _ in order_tasks:
                w = wave[oi]
                spec_2d[oi] = continuum_normalize(
                    spec_2d[oi],
                    wavelength=w,
                    lowpass_window=cont_lw,
                    poly_order=cont_po,
                )
        err_2d = np.asarray(err, dtype=np.float64)
        if err.ndim == 1:
            err_2d = np.broadcast_to(err, spec.shape).copy()
        dq_2d = np.asarray(dq, dtype=np.uint32)
        if dq.ndim == 1:
            dq_2d = np.broadcast_to(dq, spec.shape).copy()
        wave_2d = np.asarray(wave, dtype=np.float64)

        all_rvs_arr, all_rv_errs_arr, all_wl_arr = process_orders_rvs_numba_parallel(
            wave_2d, spec_2d, err_2d, dq_2d, do_not_use,
            template_ln, template_flux, self.window_size_um,
            order_tasks, max_cores=max_cores,
        )
        for i in range(len(all_rvs_arr)):
            all_rvs.append(all_rvs_arr[i])
            all_rv_errs.append(all_rv_errs_arr[i])
            all_wavelengths.append(all_wl_arr[i])

    def _write_rv_tables(
        self,
        output_model,
        fiber: str,
        spectrograph: str,
        wavelengths: np.ndarray,
        rvs: np.ndarray,
        rv_errs: np.ndarray,
        rv_agg: float,
        rv_agg_err: float,
        inlier: np.ndarray | None = None,
    ):
        """
        Write RVCCF_FIB{fiber} and RVCCFAVG_FIB{fiber} extensions.

        Creates table extensions on the output L2 model with per-chunk RVs and
        the aggregated RV (mean and std). All velocities are in m/s.

        Parameters
        ----------
        output_model : DataModel
            L2 output model to update.
        fiber : str
            Fiber number (e.g. "1").
        spectrograph : str
            "BSPEC" or "RSPEC".
        wavelengths : np.ndarray
            Chunk center wavelengths (µm).
        rvs : np.ndarray
            Per-chunk radial velocity (m/s).
        rv_errs : np.ndarray
            Per-chunk RV uncertainty (m/s).
        rv_agg : float
            Aggregated RV (e.g. weighted mean) in m/s.
        rv_agg_err : float
            Uncertainty of the aggregated RV in m/s.
        inlier : np.ndarray, optional
            Boolean mask of chunks used for the aggregate; used for table only.
        """
        n = len(rvs)
        if inlier is None:
            inlier = np.ones(n, dtype=bool) if n else np.array([], dtype=bool)
        wl = np.asarray(wavelengths, dtype=np.float64) if len(wavelengths) else np.array([], dtype=np.float64)
        mbjd = np.zeros(n, dtype=np.float64)
        berv = np.zeros(n, dtype=np.float64)
        res = np.full(n, np.nan, dtype=np.float64)
        tell_abs = np.zeros(n, dtype=np.float64)
        bis = np.zeros(n, dtype=np.float64)
        rv_abs = np.zeros(n, dtype=np.float64)
        rv_abs_err = np.zeros(n, dtype=np.float64)
        inlier_col = np.asarray(inlier, dtype=np.uint8)
        tbl = astropy.table.Table({
            "MBJDW": mbjd,
            "BERVW": berv,
            "RV": np.asarray(rvs, dtype=np.float64),
            "RV_ERR": np.asarray(rv_errs, dtype=np.float64),
            "WAVELENGTH": wl,
            "RESOLUTION": res,
            "TELL_ABS": tell_abs,
            "BIS": bis,
            "RV_ABS": rv_abs,
            "RV_ABS_ERR": rv_abs_err,
            "INLIER": inlier_col,
        })
        def _ext(name: str) -> str:
            return (f"{name}_FIB{fiber}").lower()

        setattr(output_model, _ext("RVCCF"), tbl)
        avg_tbl = astropy.table.Table({
            "MBJDW": [0.0],
            "BERVW": [0.0],
            "RV": [rv_agg],
            "RV_ERR": [rv_agg_err],
            "WAVELENGTH": [np.mean(wl) if len(wl) else np.nan],
            "RESOLUTION": [np.nan],
            "TELL_ABS": [0.0],
            "RV_ABS": [rv_agg],
            "RV_ABS_ERR": [rv_agg_err],
        })
        setattr(output_model, _ext("RVCCFAVG"), avg_tbl)

    def _create_preview(self):
        """
        Create preview figure(s) for stellar RV results.

        Plots per-chunk RV vs wavelength with error bars for each fiber that has
        CCF results, and shows the aggregated RV (weighted mean) as a reference line.
        """
        out = self._output.get("output") if self._output else None
        if out is None:
            logger.warning("No output model for stellar RV preview.")
            return
        # Handle list of outputs (multi-input) by taking first
        if isinstance(out, (list, tuple)):
            if not out:
                logger.warning("Empty output list for stellar RV preview.")
                return
            output_model = out[0]
        else:
            output_model = out

        # Find fibers that have RVCCF tables with plottable data (skip empty or all-NaN)
        fibers_with_ccf = []
        for f in ["1", "2", "3"]:
            tbl = getattr(output_model, f"rvccf_fib{f}", None)
            if tbl is None or not hasattr(tbl, "columns") or "RV" not in tbl.colnames:
                continue
            wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
            rv = np.asarray(tbl["RV"], dtype=np.float64)
            valid = np.isfinite(wl) & np.isfinite(rv)
            if not np.any(valid):
                continue
            fibers_with_ccf.append(f)

        if not fibers_with_ccf:
            logger.warning("No RVCCF tables found on output for stellar RV preview.")
            return

        # Build row layout: one RV row per fiber; one CCF-vs-velocity row per fiber that has CCF curve
        row_specs = []  # (fiber, "rv" | "ccf")
        rv_row = {}
        ccf_row = {}
        r = 0
        for f in fibers_with_ccf:
            row_specs.append((f, "rv"))
            r += 1
            rv_row[f] = r
        ccf_curves = getattr(self, "_ccf_curves", {})
        for f in fibers_with_ccf:
            curve = ccf_curves.get(f)
            if curve is not None and hasattr(curve, "colnames") and "VELOCITY" in curve.colnames and "CCF" in curve.colnames:
                vel = np.asarray(curve["VELOCITY"], dtype=np.float64)
                ccf = np.asarray(curve["CCF"], dtype=np.float64)
                valid = np.isfinite(vel) & np.isfinite(ccf)
                if np.sum(valid) > 10:
                    row_specs.append((f, "ccf"))
                    r += 1
                    ccf_row[f] = r
        subplot_titles = []
        for f, kind in row_specs:
            if kind == "rv":
                subplot_titles.append(f"Fiber {f} — per-chunk RV")
            else:
                subplot_titles.append(f"Fiber {f} — CCF vs velocity shift (mean over all chunks)")
        n_rows = len(row_specs)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.18,
        )

        v_step_ccf = 500.0  # m/s, for Gaussian fit to mean CCF
        for fiber in fibers_with_ccf:
            tbl = getattr(output_model, f"rvccf_fib{fiber}", None)
            avg_tbl = getattr(output_model, f"rvccfavg_fib{fiber}", None)
            wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
            rv = np.asarray(tbl["RV"], dtype=np.float64)
            rv_err = np.asarray(tbl["RV_ERR"], dtype=np.float64)
            inlier = np.asarray(tbl["INLIER"], dtype=bool) if "INLIER" in tbl.colnames else np.ones(len(rv), dtype=bool)
            valid = np.isfinite(wl) & np.isfinite(rv)
            if not np.any(valid):
                continue
            wl, rv, rv_err = wl[valid], rv[valid], rv_err[valid]
            inlier = inlier[valid]
            i = rv_row[fiber]
            # Per-chunk RV: inliers in blue, outliers in grey
            in_ok = inlier & np.isfinite(rv_err) & (rv_err > 0)
            out_ok = ~inlier & np.isfinite(rv_err) & (rv_err > 0)
            if np.any(in_ok):
                fig.add_trace(
                    go.Scatter(
                        x=wl[in_ok],
                        y=rv[in_ok],
                        error_y=dict(type="data", array=rv_err[in_ok], visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} (inlier)",
                        marker=dict(size=6, color="royalblue"),
                    ),
                    row=i,
                    col=1,
                )
            if np.any(out_ok):
                fig.add_trace(
                    go.Scatter(
                        x=wl[out_ok],
                        y=rv[out_ok],
                        error_y=dict(type="data", array=rv_err[out_ok], visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} (outlier)",
                        marker=dict(size=5, color="grey", symbol="circle-open"),
                    ),
                    row=i,
                    col=1,
                )
            if not np.any(in_ok) and np.any(valid):
                fig.add_trace(
                    go.Scatter(
                        x=wl,
                        y=rv,
                        error_y=dict(type="data", array=rv_err, visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} chunks",
                        marker=dict(size=6, color="royalblue"),
                    ),
                    row=i,
                    col=1,
                )
            # Aggregated RV as horizontal line with ±1σ band
            if avg_tbl is not None and len(avg_tbl) > 0:
                rv_agg = float(avg_tbl["RV"][0])
                rv_agg_err = float(avg_tbl["RV_ERR"][0])
                x_range = [np.nanmin(wl), np.nanmax(wl)] if len(wl) else [0, 1]
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[rv_agg, rv_agg],
                        mode="lines",
                        line=dict(dash="dash", color="crimson", width=2),
                        name=f"RV mean = {rv_agg:.3f} ± {rv_agg_err:.3f} m/s",
                    ),
                    row=i,
                    col=1,
                )
                if rv_agg_err > 0:
                    for y_val, label in [(rv_agg + rv_agg_err, "+1σ"), (rv_agg - rv_agg_err, "-1σ")]:
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=[y_val, y_val],
                                mode="lines",
                                line=dict(dash="dot", color="crimson", width=1),
                                name=label,
                            ),
                            row=i,
                            col=1,
                        )

        # CCF vs velocity shift; add Gaussian fit over fit window (like combined fit in test preview)
        for fiber in fibers_with_ccf:
            if fiber not in ccf_row:
                continue
            curve = ccf_curves.get(fiber)
            vel = np.asarray(curve["VELOCITY"], dtype=np.float64)
            ccf = np.asarray(curve["CCF"], dtype=np.float64)
            valid = np.isfinite(vel) & np.isfinite(ccf)
            vel_plot = vel[valid]
            ccf_plot = ccf[valid]
            fig.add_trace(
                go.Scatter(
                    x=vel_plot,
                    y=ccf_plot,
                    mode="lines",
                    name=f"Fiber {fiber} CCF (mean)",
                    line=dict(width=1.5, color="darkgreen"),
                ),
                row=ccf_row[fiber],
                col=1,
            )
            if len(vel_plot) > 10:
                rv_fit, rv_err_fit, params, diag = fit_gaussian_to_ccf(
                    vel_plot, ccf_plot, v_step_ccf, return_params=True
                )
                if params and diag.get("fit_ok"):
                    v_lo = diag.get("fit_v_lo", np.nanmin(vel_plot))
                    v_hi = diag.get("fit_v_hi", np.nanmax(vel_plot))
                    v_fit = np.linspace(v_lo, v_hi, 200)
                    ccf_fit = _gaussian_ccf_model(
                        v_fit,
                        params["amplitude"],
                        params["v0"],
                        params["sigma"],
                        params["baseline"],
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=v_fit,
                            y=ccf_fit,
                            mode="lines",
                            name=f"Fiber {fiber} Gaussian fit",
                            line=dict(width=2, color="crimson", dash="dash"),
                        ),
                        row=ccf_row[fiber],
                        col=1,
                    )

        title = "Stellar RV"
        if hasattr(output_model, "meta") and getattr(output_model.meta, "filename", None):
            title += f" — {output_model.meta.filename}"
        fig.update_layout(
            height=340 * max(1, n_rows),
            width=900,
            title_text=title,
            showlegend=True,
            margin=dict(l=70, r=50, t=70, b=60),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        # Axis labels: RV rows get wavelength/RV; CCF rows get velocity shift / CCF
        # CCF is not normalized: it is weighted covariance (data − mean)×(template − mean), units of flux².
        for idx, (f, kind) in enumerate(row_specs):
            if kind == "rv":
                fig.update_xaxes(title_text="Wavelength (µm)", row=idx + 1, col=1)
                fig.update_yaxes(title_text="RV (m/s)", row=idx + 1, col=1)
            else:
                fig.update_xaxes(title_text="Velocity shift (m/s)", row=idx + 1, col=1)
                fig.update_yaxes(
                    title_text="CCF (weighted cov., not norm.; flux²)",
                    row=idx + 1,
                    col=1,
                )

        # get_output_path requires output_dir or filename; use cwd when only showing preview
        output_dir = getattr(self, "output_dir", None) or os.getcwd()
        base_path = output_model.get_output_path(
            output_dir=output_dir, suffix=self.suffix
        )
        output_path_html = base_path.replace(".fits", "_preview.html")
        output_path_png = base_path.replace(".fits", "_preview.png")

        if self.save_preview:
            fig.write_html(output_path_html, include_plotlyjs="cdn")
            try:
                fig.write_image(output_path_png, scale=2)
            except Exception as e:
                logger.debug("Could not write preview PNG (kaleido may be missing): %s", e)
        if self.show_preview:
            logger.info("Showing stellar RV preview...")
            fig.show()

        self._output["preview"] = {
            "html": output_path_html,
            "png": output_path_png,
            "fig": fig,
        }
