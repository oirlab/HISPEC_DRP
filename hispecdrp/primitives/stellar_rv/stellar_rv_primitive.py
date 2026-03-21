"""
Stellar RV primitive: masking-based CCF with template, windowed RVs,
median/MAD outlier rejection, and inverse-variance weighted mean.
"""
from __future__ import annotations

# Provenance: ccf_median_mad uses hispecdrp.primitives.utils.ccf_utils (template CCF).
# ccf_mask_kpf uses ccf_mask_kpf_utils (SPIROU .mas masks; KPF-inspired mask-shift CCF).
# Default mask: hispecdrp/data/masks/M2_filt_allRV.mas; templates: hispecdrp/data/templates/.

import logging
import os
import time
from typing import Sequence

import numpy as np
import astropy.table

from ..base_primitive import BasePrimitive
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS

from ..utils.ccf_utils import (
    _HAS_NUMBA,
    _V_MAX,
    _V_MIN,
    _V_STEP,
    _resolve_stellar_rv_numba_threads,
    ccf_preview_dict_from_numba_flat,
    ccf_preview_dict_from_spectra_orders,
    combine_rvs_median_mad,
    finalize_ccf_preview_dict,
    masked_ccf_rv_diagnostics,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    weighted_mean_rv,
)
from ..utils.ccf_mask_kpf_utils import (
    load_mask,
    resolve_mask_path,
    restrict_mask_to_chunk,
    run_ccf_mask_kpf_for_fiber,
)

__all__ = ["StellarRVPrimitive", "STELLAR_RV_ALGORITHMS"]

logger = logging.getLogger(__name__)


def _echelle_order_per_flat_chunk(
    order_tasks: list,
    wave_2d: np.ndarray,
    window_size_um: float,
    n_chunks_total: int,
) -> np.ndarray:
    """
    Map flattened chunk index to echelle order number (same packing as Numba CCF).

    ``order_tasks`` is ``(order_row_index, order_number)`` per order; chunk layout
    matches :func:`process_orders_rvs_numba_parallel`.
    """

    if n_chunks_total <= 0:
        return np.array([], dtype=np.int64)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    n_chunks_per_order = np.array(
        [
            max(
                1,
                int(
                    (np.nanmax(wave_2d[oi]) - np.nanmin(wave_2d[oi]))
                    / float(window_size_um)
                ),
            )
            for oi, _ in order_tasks
        ],
        dtype=np.int64,
    )
    chunk_start = np.zeros(len(order_tasks), dtype=np.int64)
    for i in range(1, len(order_tasks)):
        chunk_start[i] = chunk_start[i - 1] + n_chunks_per_order[i - 1]
    out = np.empty(n_chunks_total, dtype=np.int64)
    for i in range(n_chunks_total):
        idx = int(np.searchsorted(chunk_start, i, side="right") - 1)
        if 0 <= idx < len(order_tasks):
            out[i] = int(order_tasks[idx][1])
        else:
            out[i] = -1
    return out


# Registry of stellar RV algorithm names -> method name on StellarRVPrimitive.
# Each algorithm method has signature (input_model, output_model, fiber, spectrograph,
# do_orders, order_vec, template_ln, template_flux) and returns
# (all_wavelengths, all_rvs, all_rv_errs, inlier, rv_mean, rv_std) for _write_rv_tables.
STELLAR_RV_ALGORITHMS = {
    "ccf_median_mad": "_run_algorithm_ccf_median_mad",
    "ccf_mask_kpf": "_run_algorithm_ccf_mask_kpf",
}

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
        (wave_um, flux) as numpy arrays: wavelength in **vacuum** micron (BT-Settl HDF5
        grid is converted from air via :func:`hispecdrp.utils.spectral_templates_utils.load_btsettl_spectrum`).
    """
    from ...utils.spectral_templates_utils import load_btsettl_spectrum
    w, f = load_btsettl_spectrum(filepath, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    return w.to_value("micron"), f.value


class StellarRVPrimitive(BasePrimitive):
    """
    Compute radial velocity via masking-based cross correlation with a spectral template.

    All radial velocities (per-chunk RV and aggregated RV) are in meters per second (m/s).
    The spectrum is split into wavelength chunks (window_size_um). Each chunk yields one RV;
    chunk RVs are combined according to the chosen algorithm and written to RVCCF/RVCCFAVG
    extensions on the output L2 model.

    Algorithm
    ---------
    algorithm : str, optional
        Stellar RV algorithm to use. Options are registered in STELLAR_RV_ALGORITHMS.

        - **ccf_median_mad** (default): Template CCF in log(λ); per-chunk RVs;
          median/MAD outlier rejection and inverse-variance weighted mean.

    max_cores : int, optional
        Numba parallelizes the CCF across echelle orders (``prange``). Default
        ``None`` uses all logical CPUs; set ``1`` for single-threaded runs (e.g.
        debugging). This does not change the mathematical result.
    """

    ARGUMENTS = {
        "suffix": "stellar_rv",
        "algorithm": "ccf_median_mad",
        "do_fibers": None,
        "do_orders": None,
        "template_path": None,
        "template_wavelength": None,
        "template_flux": None,
        # Mask-only algorithm inputs (used when algorithm="ccf_mask_kpf").
        # If mask_path is None, ccf_mask_kpf_utils uses its packaged SPIROU default.
        "mask_path": None,
        # Optional wavelength segment half-width for KPF-style 2-col masks.
        # Ignored for SPIROU-style .mas files which already provide start/end.
        "mask_width_um": None,
        "window_size_um": 0.3,
        "outlier_sigma": 4.0,
        "output_model": "L2Model",
        # None: use all CPUs for Numba order-parallel CCF; set to 1 to force single-thread.
        "max_cores": None,
        "debug_ccf": False,
        # Preview export: Kaleido PNG is slow. Env HISPEC_DRP_SHOW_CCF_PREVIEW
        # forces HTML-only (no PNG). Otherwise preview_save_png=False skips PNG;
        # preview_png_scale lowers pixel count (e.g. 1.0) for faster PNGs.
        "preview_save_png": False,
        "preview_png_scale": 2.0,
        # Debug-only ccf_mask_kpf overlay (separate from standard preview):
        # order dropdown + velocity slider/number input in standalone HTML.
        "show_mask_overlay_debug": False,
        "save_mask_overlay_debug": False,
        "mask_overlay_velocity_m_s": 0.0,
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

        algo_key = (getattr(self, "algorithm", None) or "ccf_median_mad").lower().strip()
        template_ln, template_flux = None, None
        if algo_key != "ccf_mask_kpf":
            t_tpl = time.perf_counter()
            template_ln, template_flux = self._get_template()
            n_tpl = int(len(template_ln)) if template_ln is not None else 0
            logger.info(
                "STELLAR_RV_TIMING step=resolve_template algorithm=%s wall_s=%.3f n_ln_points=%d",
                algo_key,
                time.perf_counter() - t_tpl,
                n_tpl,
            )
            if template_ln is None:
                raise ValueError(
                    "Template not set: provide template_path or (template_wavelength, template_flux)."
                )
        else:
            # Mask-based CCF does not require a stellar template; we keep the
            # variables to satisfy the algorithm dispatch signature.
            template_ln = np.array([], dtype=np.float64)
            template_flux = np.array([], dtype=np.float64)

        # Per-fiber CCF preview slots (mean + per-order); filled when save/show preview.
        self._ccf_preview_by_fiber: dict = {}
        # Per-fiber payload for mask overlay debug HTML (ccf_mask_kpf only).
        self._mask_overlay_by_fiber: dict = {}

        for _input in inp_list:
            t_open = time.perf_counter()
            input_model = datamodels.open(_input)
            logger.info(
                "STELLAR_RV_TIMING step=open_input wall_s=%.3f", time.perf_counter() - t_open
            )
            t_clone = time.perf_counter()
            output_model = self.get_output_model(
                input_model, output_model=self.output_model, include_data=True
            )
            logger.info(
                "STELLAR_RV_TIMING step=clone_output_L2 wall_s=%.3f",
                time.perf_counter() - t_clone,
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

            if algo_key == "ccf_mask_kpf" and bool(
                getattr(self, "show_mask_overlay_debug", False)
                or getattr(self, "save_mask_overlay_debug", False)
            ):
                from .stellar_rv_mask_overlay_debug import create_stellar_rv_mask_overlay_debug

                t_dbg = time.perf_counter()
                paths = create_stellar_rv_mask_overlay_debug(self, output_model)
                if paths:
                    if self._output is None:
                        self._output = {}
                    self._output["mask_overlay_debug"] = paths
                logger.info(
                    "STELLAR_RV_TIMING step=create_mask_overlay_debug_html wall_s=%.3f",
                    time.perf_counter() - t_dbg,
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
        template_flux. Per-fiber wavelength clipping happens in _process_fiber.

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

    def _run_algorithm_ccf_median_mad(
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
        CCF + median/MAD outlier rejection + inverse-variance weighted mean.

        Computes per-chunk RVs via CCF in log(lambda) with the template (serial or
        parallel), combines with median/MAD and weighted mean, and optionally stores
        CCF curves for the preview. Returns the standard 6-tuple for _write_rv_tables.

        Returns
        -------
        all_wavelengths : np.ndarray
            Chunk center wavelengths (µm).
        all_rvs : np.ndarray
            Per-chunk RV (m/s).
        all_rv_errs : np.ndarray
            Per-chunk RV uncertainty (m/s).
        inlier : np.ndarray
            Boolean mask of chunks used for the aggregate.
        rv_mean : float
            Aggregated RV (m/s).
        rv_std : float
            Uncertainty of the aggregated RV (m/s).
        """
        spec_attr = f"SPEC_FIB{fiber}"
        err_attr = f"SPEC_ERR_FIB{fiber}"
        dq_attr = f"SPEC_DQ_FIB{fiber}"
        wave_attr = f"WAVE_FIB{fiber}"
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
        _mc = getattr(self, "max_cores", None)
        max_cores = _resolve_stellar_rv_numba_threads(
            None if _mc is None else int(_mc)
        )

        order_tasks = [
            (oi, order)
            for oi, order in enumerate(order_vec)
            if order in do_orders
        ]
        want_preview = bool(
            getattr(self, "save_preview", False) or getattr(self, "show_preview", False)
        )
        if debug_ccf:
            mode = "debug"
        elif _HAS_NUMBA:
            mode = "parallel"
        else:
            mode = "serial"

        all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail = self._process_fiber_orders(
            order_tasks,
            wave,
            spec,
            err,
            dq,
            do_not_use,
            template_ln,
            template_flux,
            fiber,
            mode=mode,
            max_cores=max_cores,
            want_preview=want_preview,
        )
        if preview_detail is not None:
            self._ccf_preview_by_fiber[fiber] = preview_detail

        all_rvs = np.array(all_rvs)
        all_rv_errs = np.array(all_rv_errs)
        all_wavelengths = np.array(all_wavelengths)
        all_orders = np.asarray(all_orders, dtype=np.int64)

        valid = np.isfinite(all_rvs) & np.isfinite(all_rv_errs)
        if not np.any(valid):
            logger.warning("No valid chunk RVs for fiber %s.", fiber)
            inlier = np.zeros(len(all_rvs), dtype=bool)
            rv_mean, rv_std = np.nan, np.nan
        else:
            t_combo = time.perf_counter()
            inlier, _, _ = combine_rvs_median_mad(
                all_rvs, all_rv_errs, self.outlier_sigma
            )
            rv_mean, rv_std = weighted_mean_rv(all_rvs, all_rv_errs, inlier)
            logger.info(
                "STELLAR_RV_TIMING step=combine_median_mad_weighted_mean fiber=%s "
                "n_chunks=%d wall_s=%.3f",
                fiber,
                int(np.sum(valid)),
                time.perf_counter() - t_combo,
            )

        if debug_ccf and len(all_wavelengths) > 0:
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

        return (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            inlier,
            rv_mean,
            rv_std,
            all_orders,
        )

    def _run_algorithm_ccf_mask_kpf(
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
        Mask-shifted KPF-style CCF + Gaussian RV fit.

        Minimal delegation to :func:`ccf_mask_kpf_utils.run_ccf_mask_kpf_for_fiber`.
        """

        want_preview = bool(
            getattr(self, "save_preview", False) or getattr(self, "show_preview", False)
        )
        preview_detail = None
        (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            inlier,
            rv_mean,
            rv_std,
            all_orders,
            preview_detail,
        ) = run_ccf_mask_kpf_for_fiber(
            input_model=input_model,
            fiber=fiber,
            do_orders=do_orders,
            order_vec=order_vec,
            window_size_um=float(getattr(self, "window_size_um", 0.3)),
            outlier_sigma=float(getattr(self, "outlier_sigma", 4.0)),
            mask_path=getattr(self, "mask_path", None),
            mask_width_um=getattr(self, "mask_width_um", None),
            v_min_m_s=_V_MIN,
            v_max_m_s=_V_MAX,
            v_step_m_s=_V_STEP,
            zb=0.0,
            want_preview=want_preview,
            debug_ccf=bool(getattr(self, "debug_ccf", False)),
        )
        if preview_detail is not None:
            self._ccf_preview_by_fiber[fiber] = preview_detail
        if bool(
            getattr(self, "show_mask_overlay_debug", False)
            or getattr(self, "save_mask_overlay_debug", False)
        ):
            payload = self._collect_mask_overlay_payload(
                input_model=input_model,
                fiber=fiber,
                order_vec=order_vec,
            )
            if payload is not None:
                self._mask_overlay_by_fiber[fiber] = payload
        return all_wavelengths, all_rvs, all_rv_errs, inlier, rv_mean, rv_std, all_orders

    def _collect_mask_overlay_payload(
        self,
        *,
        input_model,
        fiber: str,
        order_vec: list,
    ) -> dict | None:
        """
        Build per-order spectrum + mask payload for debug overlay HTML.
        """
        try:
            resolved_mask = resolve_mask_path(getattr(self, "mask_path", None))
            mask_lines = load_mask(
                resolved_mask, mask_width_um=getattr(self, "mask_width_um", None)
            )
        except Exception as exc:
            logger.warning("MASK_OVERLAY_DEBUG cannot load mask: %s", exc)
            return None

        wave = getattr(input_model, f"WAVE_FIB{fiber}", None)
        spec = getattr(input_model, f"SPEC_FIB{fiber}", None)
        if wave is None or spec is None:
            return None

        entries: list[dict] = []
        # For debug visualization, include every order present on the input model so
        # the HTML dropdown can inspect the full BSPEC/RSPEC order set, even if
        # RV fitting itself was restricted to a subset via do_orders.
        for oi, order in enumerate(order_vec):
            w = np.asarray(wave[oi], dtype=np.float64)
            s = np.asarray(spec[oi], dtype=np.float64)
            ok = np.isfinite(w) & np.isfinite(s)
            if int(np.sum(ok)) < 10:
                continue
            w = w[ok]
            s = s[ok]
            med = float(np.nanmedian(s))
            if not np.isfinite(med) or med == 0.0:
                med = 1.0
            s_norm = s / med

            mask_o = restrict_mask_to_chunk(
                mask_lines,
                float(np.nanmin(w)),
                float(np.nanmax(w)),
                v_min_m_s=_V_MIN,
                v_max_m_s=_V_MAX,
                zb=0.0,
            )
            if mask_o.start_um.size > 0:
                mstart = np.asarray(mask_o.start_um, dtype=np.float64)
                mend = np.asarray(mask_o.end_um, dtype=np.float64)
                # Raw file weights for debug HTML: rectangles span mask weight 1 → (1 − w).
                mw = np.asarray(mask_o.weight, dtype=np.float64)
            else:
                mstart = np.array([], dtype=np.float64)
                mend = np.array([], dtype=np.float64)
                mw = np.array([], dtype=np.float64)

            entries.append(
                {
                    "order": int(order),
                    "wave": np.asarray(w, dtype=np.float64),
                    "spec": np.asarray(s_norm, dtype=np.float64),
                    "mask_start_um": np.asarray(mstart, dtype=np.float64),
                    "mask_end_um": np.asarray(mend, dtype=np.float64),
                    "mask_weight": np.asarray(mw, dtype=np.float64),
                }
            )

        if not entries:
            return None
        return {"fiber": str(fiber), "orders": entries}

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
        Run the configured stellar RV algorithm for one fiber and write RVCCF/RVCCFAVG.

        Dispatches to the algorithm named in self.algorithm (see STELLAR_RV_ALGORITHMS).
        The algorithm returns (all_wavelengths, all_rvs, all_rv_errs, inlier, rv_mean, rv_std);
        this method writes RVCCF_FIB{fiber} and RVCCFAVG_FIB{fiber} to the output model.

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

        algo_key = (getattr(self, "algorithm", None) or "ccf_median_mad").lower().strip()
        method_name = STELLAR_RV_ALGORITHMS.get(algo_key)
        if method_name is None:
            raise ValueError(
                f"Unknown stellar RV algorithm: {self.algorithm}. "
                f"Valid: {list(STELLAR_RV_ALGORITHMS.keys())}"
            )
        runner = getattr(self, method_name)
        t_run = time.perf_counter()
        (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            inlier,
            rv_mean,
            rv_std,
            all_orders,
        ) = runner(
            input_model,
            output_model,
            fiber,
            spectrograph,
            do_orders,
            order_vec,
            template_ln,
            template_flux,
        )
        logger.info(
            "STELLAR_RV_TIMING step=rv_algorithm fiber=%s algo=%s wall_s=%.3f",
            fiber,
            algo_key,
            time.perf_counter() - t_run,
        )
        t_w = time.perf_counter()
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
            orders=all_orders,
        )
        logger.info(
            "STELLAR_RV_TIMING step=write_rv_tables fiber=%s wall_s=%.3f",
            fiber,
            time.perf_counter() - t_w,
        )

    def _process_fiber_orders(
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
        *,
        mode: str,
        max_cores: int | None = None,
        want_preview: bool = False,
    ) -> tuple[list, list, list, list, dict | None]:
        """
        Single driver for per-order chunk RVs: ``mode`` is ``debug``, ``serial``, or
        ``parallel``. When ``want_preview`` is True, also returns a CCF preview
        dict (Numba path reuses the CCF stack; serial path uses one Python pass;
        debug + preview uses a separate Python CCF pass).

        Returns per-chunk echelle order numbers in the fourth list (parallel to wavelengths).
        """
        all_rvs: list = []
        all_rv_errs: list = []
        all_wavelengths: list = []
        all_orders: list = []
        preview_detail: dict | None = None

        if mode == "debug":
            for oi, order in order_tasks:
                w = np.asarray(wave[oi], dtype=np.float64)
                s = np.asarray(spec[oi], dtype=np.float64)
                e = err[oi] if err.ndim >= 2 else err
                d = dq[oi] if dq.ndim >= 2 else dq
                mask = (np.asarray(d, dtype=np.uint32) & do_not_use) != 0
                e = np.asarray(e, dtype=np.float64)
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
                        w[in_chunk],
                        s[in_chunk],
                        e[in_chunk],
                        mask[in_chunk],
                        t_ln,
                        t_flux,
                    )
                    logger.info(
                        "CCF fiber=%s order=%s chunk=%s wl_center=%.4f um wl_range=[%.4f, %.4f] "
                        "n_good=%s n_use_at_peak=%s ccf_max=%.6f ccf_min=%.6f "
                        "spec_mean=%.4e spec_std=%.4e tpl_mean=%.4e tpl_std=%.4e rv=%.1f m/s rv_err=%.1f",
                        fiber,
                        order,
                        k,
                        wl_center,
                        diag["wl_min"],
                        diag["wl_max"],
                        diag["n_good"],
                        diag["n_use_at_peak"],
                        diag["ccf_max"],
                        diag["ccf_min"],
                        diag["spec_mean"],
                        diag["spec_std"],
                        diag["template_mean_at_peak"],
                        diag["template_std_at_peak"],
                        rv,
                        rv_err,
                    )
                    all_rvs.append(rv)
                    all_rv_errs.append(rv_err)
                    all_wavelengths.append(wl_center)
                    all_orders.append(int(order))
            if want_preview:
                t_pv = time.perf_counter()
                preview_detail = ccf_preview_dict_from_spectra_orders(
                    order_tasks,
                    wave,
                    spec,
                    err,
                    dq,
                    do_not_use,
                    template_ln,
                    template_flux,
                    self.window_size_um,
                    _V_MIN,
                    _V_MAX,
                    _V_STEP,
                )
                logger.info(
                    "STELLAR_RV_TIMING fiber=%s ccf_preview_dict_from_spectra_orders "
                    "(debug path) wall_s=%.3f",
                    fiber,
                    time.perf_counter() - t_pv,
                )
            return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail

        if mode == "parallel":
            spec_2d = np.asarray(spec, dtype=np.float64)
            err_2d = np.asarray(err, dtype=np.float64)
            if err.ndim == 1:
                err_2d = np.broadcast_to(err, spec.shape).copy()
            dq_2d = np.asarray(dq, dtype=np.uint32)
            if dq.ndim == 1:
                dq_2d = np.broadcast_to(dq, spec.shape).copy()
            wave_2d = np.asarray(wave, dtype=np.float64)

            if want_preview and _HAS_NUMBA:
                t_nb = time.perf_counter()
                (
                    all_rvs_arr,
                    all_rv_errs_arr,
                    all_wl_arr,
                    all_ccf,
                    v_grid,
                    n_chunks_per_order,
                    chunk_start,
                    order_indices,
                ) = process_orders_rvs_numba_parallel(
                    wave_2d,
                    spec_2d,
                    err_2d,
                    dq_2d,
                    do_not_use,
                    template_ln,
                    template_flux,
                    self.window_size_um,
                    order_tasks,
                    max_cores=max_cores,
                    return_ccf=True,
                )
                logger.info(
                    "STELLAR_RV_TIMING fiber=%s process_orders_rvs_numba_parallel"
                    "(return_ccf=True) wall_s=%.3f",
                    fiber,
                    time.perf_counter() - t_nb,
                )
                ao_flat = _echelle_order_per_flat_chunk(
                    order_tasks, wave_2d, self.window_size_um, len(all_rvs_arr)
                )
                for i in range(len(all_rvs_arr)):
                    all_rvs.append(all_rvs_arr[i])
                    all_rv_errs.append(all_rv_errs_arr[i])
                    all_wavelengths.append(all_wl_arr[i])
                    all_orders.append(int(ao_flat[i]))
                t_pv = time.perf_counter()
                preview_detail = ccf_preview_dict_from_numba_flat(
                    order_tasks,
                    order_indices,
                    n_chunks_per_order,
                    chunk_start,
                    all_ccf,
                    v_grid,
                    wave_2d,
                    _V_MIN,
                    _V_MAX,
                    _V_STEP,
                )
                logger.info(
                    "STELLAR_RV_TIMING fiber=%s ccf_preview_dict_from_numba_flat "
                    "wall_s=%.3f",
                    fiber,
                    time.perf_counter() - t_pv,
                )
            else:
                all_rvs_arr, all_rv_errs_arr, all_wl_arr = process_orders_rvs_numba_parallel(
                    wave_2d,
                    spec_2d,
                    err_2d,
                    dq_2d,
                    do_not_use,
                    template_ln,
                    template_flux,
                    self.window_size_um,
                    order_tasks,
                    max_cores=max_cores,
                    return_ccf=False,
                )
                ao_flat = _echelle_order_per_flat_chunk(
                    order_tasks, wave_2d, self.window_size_um, len(all_rvs_arr)
                )
                for i in range(len(all_rvs_arr)):
                    all_rvs.append(all_rvs_arr[i])
                    all_rv_errs.append(all_rv_errs_arr[i])
                    all_wavelengths.append(all_wl_arr[i])
                    all_orders.append(int(ao_flat[i]))
            return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail

        # serial (no Numba)
        if want_preview:
            order_slots: list[dict] = []
            all_rows: list[np.ndarray] = []
            v_ref: np.ndarray | None = None
            for oi, order in order_tasks:
                w = wave[oi]
                s = spec[oi]
                e = err[oi] if err.ndim >= 2 else err
                d = dq[oi] if dq.ndim >= 2 else dq
                rvs, rv_errs, chunk_centers, v_grid, ccf_2d = process_one_order_rvs(
                    w,
                    s,
                    e,
                    d,
                    do_not_use,
                    template_ln,
                    template_flux,
                    self.window_size_um,
                    return_ccf=True,
                )
                if len(rvs) == 0:
                    continue
                for i in range(len(rvs)):
                    all_rvs.append(rvs[i])
                    all_rv_errs.append(rv_errs[i])
                    all_wavelengths.append(chunk_centers[i])
                    all_orders.append(int(order))
                if v_ref is None:
                    v_ref = v_grid
                elif v_ref.shape != v_grid.shape or not np.allclose(v_ref, v_grid):
                    continue
                ccf_m = np.nanmean(ccf_2d, axis=0)
                w_arr = np.asarray(w, dtype=np.float64)
                order_slots.append(
                    {
                        "kind": "order",
                        "order": int(order),
                        "v": np.asarray(v_grid, dtype=np.float64),
                        "ccf": np.asarray(ccf_m, dtype=np.float64),
                        "wl_lo": float(np.nanmin(w_arr)),
                        "wl_hi": float(np.nanmax(w_arr)),
                    }
                )
                all_rows.append(ccf_2d)
            if order_slots and v_ref is not None:
                t_fin = time.perf_counter()
                preview_detail = finalize_ccf_preview_dict(
                    order_slots,
                    all_rows,
                    v_ref,
                    _V_MIN,
                    _V_MAX,
                    _V_STEP,
                )
                logger.info(
                    "STELLAR_RV_TIMING fiber=%s finalize_ccf_preview_dict "
                    "(serial+preview path) wall_s=%.3f",
                    fiber,
                    time.perf_counter() - t_fin,
                )
        else:
            for oi, order in order_tasks:
                w = wave[oi]
                s = spec[oi]
                e = err[oi] if err.ndim >= 2 else err
                d = dq[oi] if dq.ndim >= 2 else dq
                rvs, rv_errs, chunk_centers = process_one_order_rvs(
                    w,
                    s,
                    e,
                    d,
                    do_not_use,
                    template_ln,
                    template_flux,
                    self.window_size_um,
                )
                for i in range(len(rvs)):
                    all_rvs.append(rvs[i])
                    all_rv_errs.append(rv_errs[i])
                    all_wavelengths.append(chunk_centers[i])
                    all_orders.append(int(order))

        return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail

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
        orders: np.ndarray | None = None,
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
        orders : np.ndarray, optional
            Echelle order number per chunk (same length as ``rvs``). -1 if unknown.
        """
        n = len(rvs)
        if inlier is None:
            inlier = np.ones(n, dtype=bool) if n else np.array([], dtype=bool)
        wl = np.asarray(wavelengths, dtype=np.float64) if len(wavelengths) else np.array([], dtype=np.float64)
        if orders is not None and len(orders) == n:
            order_col = np.asarray(orders, dtype=np.int64).ravel()
        else:
            order_col = np.full(n, -1, dtype=np.int64)
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
            "ORDER": order_col,
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
        from .stellar_rv_preview import create_stellar_rv_preview

        t0 = time.perf_counter()
        create_stellar_rv_preview(self)
        logger.info(
            "STELLAR_RV_TIMING step=create_stellar_rv_preview total wall_s=%.3f",
            time.perf_counter() - t0,
        )
