"""
Stellar radial-velocity primitive (template and mask-line CCF).

Computes RVs by cross-correlating the observed L2 spectrum with either a **stellar
template** (``algorithm="ccf_stellar_template"``) or a **line mask**
(``algorithm="ccf_mask_kpf"``). Per-wavelength-chunk RVs are combined with
median/MAD outlier rejection and an inverse-variance weighted mean where applicable.

Notes
-----
- **Template path** uses :mod:`hispecdrp.primitives.utils.ccf_utils` and
  :mod:`hispecdrp.utils.spectral_templates_utils`.
- **Mask path** uses :mod:`hispecdrp.primitives.utils.ccf_mask_kpf_utils` with
  SPIROU-style ``.mas`` files (default ``hispecdrp/data/masks/M2_filt_allRV.mas``).
  Mask-shift CCF is **inspired by** Keck KPF’s public mask-line design; packaged
  masks trace to **spirou-ccf** — see ``hispecdrp/data/masks/README.md``.

References
----------
.. [1] KPF ``alg_rv_mask_line.py`` (mask-line reference):
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg_rv_mask_line.py
.. [2] KPF radial_velocity ``src``:
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
.. [3] spirou-ccf ``spiroulib.py``:
       https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py
.. [4] spirou-ccf ``ccf_masks``:
       https://github.com/edermartioli/spirou-ccf/tree/main/ccf_masks
"""
from __future__ import annotations

import logging
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
    flat_chunk_echelle_orders,
    masked_ccf_rv_diagnostics,
    process_one_order_rvs,
    process_orders_rvs_numba_parallel,
    weighted_mean_rv,
)
from ..utils.ccf_mask_kpf_utils import run_ccf_mask_kpf_for_fiber
from ...utils.spectral_templates_utils import resolve_stellar_template_to_log_micron

__all__ = ["StellarRVPrimitive", "STELLAR_RV_ALGORITHMS"]

logger = logging.getLogger(__name__)


def _fiber_l2_bundle(input_model, fiber: str):
    """
    Return (spec, err, dq, wave) for a fiber, or None if SPEC/WAVE missing.

    Logs a warning when returning None (same messages as former _process_fiber checks).
    """
    sa, wa = f"SPEC_FIB{fiber}", f"WAVE_FIB{fiber}"
    if not hasattr(input_model, sa) or getattr(input_model, sa) is None:
        logger.warning("No %s on input; skipping CCF for fiber %s.", sa, fiber)
        return None
    if not hasattr(input_model, wa) or getattr(input_model, wa) is None:
        logger.warning("No %s on input; skipping CCF for fiber %s.", wa, fiber)
        return None
    spec = getattr(input_model, sa)
    ea, da = f"SPEC_ERR_FIB{fiber}", f"SPEC_DQ_FIB{fiber}"
    err = getattr(input_model, ea) if hasattr(input_model, ea) else None
    dq = getattr(input_model, da) if hasattr(input_model, da) else None
    if err is None:
        err = np.ones_like(spec, dtype=np.float64)
    else:
        err = np.asarray(err, dtype=np.float64)
    if dq is None:
        dq = np.zeros_like(spec, dtype=np.uint32)
    else:
        dq = np.asarray(dq, dtype=np.uint32)
    return spec, err, dq, getattr(input_model, wa)


# Registry of stellar RV algorithm names -> method name on StellarRVPrimitive.
# Each algorithm method has signature (input_model, output_model, fiber, spectrograph,
# do_orders, order_vec, template_ln, template_flux) and returns
# (all_wavelengths, all_rvs, all_rv_errs, inlier, rv_mean, rv_std) for _write_rv_tables.
STELLAR_RV_ALGORITHMS = {
    "ccf_stellar_template": "_run_algorithm_ccf_stellar_template",
    "ccf_mask_kpf": "_run_algorithm_ccf_mask_kpf",
}


class StellarRVPrimitive(BasePrimitive):
    """
    Cross-correlate L2 spectra against a template or a line mask to measure RV.

    **ccf_stellar_template** (default): weighted CCF in :math:`\\ln\\lambda` against a
    stellar template; per-chunk RVs; median/MAD rejection; inverse-variance mean
    (:mod:`~hispecdrp.primitives.utils.ccf_utils`).

    **ccf_mask_kpf**: mask-shifted CCF using SPIROU ``.mas`` or KPF-style two-column
    masks (:mod:`~hispecdrp.primitives.utils.ccf_mask_kpf_utils`). Default packaged
    mask and citations: ``hispecdrp/data/masks/README.md``.

    All RVs are in **m s\\ :sup:`−1`**. Chunks use ``window_size_um``; results go to
    RVCCF / RVCCFAVG on the output L2 model.

    Parameters (primitive arguments)
    --------------------------------
    The full argument schema is in ``ARGUMENTS``; key fields include:

    algorithm : str, optional
        One of ``STELLAR_RV_ALGORITHMS`` keys: ``ccf_stellar_template`` or
        ``ccf_mask_kpf``.
    template_path, template_wavelength, template_flux : optional
        Template inputs for ``ccf_stellar_template``.
    mask_path, mask_width_um : optional
        Mask file and (for two-column masks) half-width in µm for ``ccf_mask_kpf``.
    window_size_um, outlier_sigma, max_cores : optional
        Chunk width, MAD threshold, and Numba thread count for template path.

    Notes
    -----
    ``max_cores`` affects only parallel echelle processing (``prange``); set ``1`` for
    deterministic single-thread debugging.

    See Also
    --------
    :mod:`hispecdrp.primitives.utils.ccf_utils`
    :mod:`hispecdrp.primitives.utils.ccf_mask_kpf_utils`
    :mod:`hispecdrp.utils.spectral_templates_utils`
    """

    ARGUMENTS = {
        "suffix": "stellar_rv",
        "algorithm": "ccf_stellar_template",
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

        For template mode, loads the stellar template, then for each input model and
        each fiber runs CCF (serial or Numba-parallel), combines chunk RVs, and writes
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

        algo_key = (getattr(self, "algorithm", None) or "ccf_stellar_template").lower().strip()
        logger.info(
            "Stellar RV: starting _perform algorithm=%s n_inputs=%d suffix=%s",
            algo_key,
            len(inp_list),
            getattr(self, "suffix", ""),
        )
        template_ln, template_flux = None, None
        if algo_key != "ccf_mask_kpf":
            logger.info("Stellar RV: resolving stellar template (path or inline arrays)")
            template_ln, template_flux = resolve_stellar_template_to_log_micron(
                getattr(self, "template_path", None),
                getattr(self, "template_wavelength", None),
                getattr(self, "template_flux", None),
            )
            n_tpl = int(len(template_ln)) if template_ln is not None else 0
            logger.info(
                "Stellar RV: template ready algorithm=%s n_ln_points=%d",
                algo_key,
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
            logger.info("Stellar RV: ccf_mask_kpf mode — empty template arrays (mask CCF only)")

        # Per-fiber CCF preview slots (mean + per-order); filled when save/show preview.
        self._ccf_preview_by_fiber: dict = {}
        # Per-fiber payload for mask overlay debug HTML (ccf_mask_kpf only).
        self._mask_overlay_by_fiber: dict = {}

        for _input in inp_list:
            logger.info("Stellar RV: opening input (path or in-memory model)")
            input_model = datamodels.open(_input)
            logger.info("Stellar RV: input open complete")
            logger.info(
                "Stellar RV: cloning output L2 (output_model=%s include_data=True)",
                self.output_model,
            )
            output_model = self.get_output_model(
                input_model, output_model=self.output_model, include_data=True
            )
            logger.info("Stellar RV: output L2 clone complete")
            spectrograph = self._get_spectrograph(input_model)
            do_fibers, do_orders = self.get_do_traces(
                input_model, self.do_fibers, self.do_orders
            )
            order_vec = input_model.make_order_vec()
            logger.info(
                "Stellar RV: spectrograph=%s do_fibers=%s n_do_orders=%d n_order_vec=%d",
                spectrograph,
                do_fibers,
                len(do_orders) if do_orders is not None else 0,
                len(order_vec),
            )

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
                from .stellar_rv_preview import write_mask_overlay_debug_pages

                logger.info("Stellar RV: writing mask overlay debug HTML pages")
                paths = write_mask_overlay_debug_pages(self, output_model)
                if paths:
                    if self._output is None:
                        self._output = {}
                    self._output["mask_overlay_debug"] = paths
                logger.info(
                    "Stellar RV: mask overlay debug HTML done n_files=%d",
                    len(paths),
                )

            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        logger.info("Stellar RV: _perform finished algorithm=%s", algo_key)
        return dict(output=output)

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

    def _run_algorithm_ccf_stellar_template(
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
        Stellar-template cross-correlation + median/MAD + inverse-variance weighted mean.

        Cross-correlates the stellar template spectrum to the input spectrum in
        log(wavelength), producing per-chunk RVs (serial or Numba-parallel). Chunk RVs
        are combined with median/MAD outlier rejection and a weighted mean; CCF curves
        may be stored for the preview. Returns the standard 6-tuple for _write_rv_tables.

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
        bundle = _fiber_l2_bundle(input_model, fiber)
        if bundle is None:
            raise RuntimeError("internal: expected SPEC/WAVE after _process_fiber guard")
        spec, err, dq, wave = bundle
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

        logger.info(
            "Stellar RV: ccf_stellar_template fiber=%s spectrograph=%s mode=%s "
            "n_order_tasks=%d want_preview=%s max_cores=%s window_um=%.4f",
            fiber,
            spectrograph,
            mode,
            len(order_tasks),
            want_preview,
            max_cores,
            float(getattr(self, "window_size_um", 0.3)),
        )

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
            logger.info(
                "Stellar RV: combining chunk RVs (median/MAD + weighted mean) fiber=%s "
                "n_valid_chunks=%d outlier_sigma=%.2f",
                fiber,
                int(np.sum(valid)),
                float(self.outlier_sigma),
            )
            inlier, _, _ = combine_rvs_median_mad(
                all_rvs, all_rv_errs, self.outlier_sigma
            )
            rv_mean, rv_std = weighted_mean_rv(all_rvs, all_rv_errs, inlier)
            logger.info(
                "Stellar RV: aggregate RV fiber=%s rv_mean=%.2f m/s rv_std=%.2f m/s "
                "n_inlier=%d/%d",
                fiber,
                float(rv_mean),
                float(rv_std),
                int(np.sum(inlier)),
                len(inlier),
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
        logger.info(
            "Stellar RV: dispatching ccf_mask_kpf fiber=%s spectrograph=%s "
            "n_do_orders=%d want_preview=%s",
            fiber,
            spectrograph,
            len(do_orders) if do_orders is not None else 0,
            want_preview,
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
            from .stellar_rv_preview import collect_mask_overlay_payload_for_primitive

            payload = collect_mask_overlay_payload_for_primitive(
                self,
                input_model=input_model,
                fiber=fiber,
                order_vec=order_vec,
            )
            if payload is not None:
                self._mask_overlay_by_fiber[fiber] = payload
                logger.debug(
                    "Stellar RV: mask overlay payload stored fiber=%s n_orders_in_payload=%d",
                    fiber,
                    len(payload.get("orders", [])),
                )
        return all_wavelengths, all_rvs, all_rv_errs, inlier, rv_mean, rv_std, all_orders

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
        if _fiber_l2_bundle(input_model, fiber) is None:
            return

        algo_key = (getattr(self, "algorithm", None) or "ccf_stellar_template").lower().strip()
        method_name = STELLAR_RV_ALGORITHMS.get(algo_key)
        if method_name is None:
            raise ValueError(
                f"Unknown stellar RV algorithm: {self.algorithm}. "
                f"Valid: {list(STELLAR_RV_ALGORITHMS.keys())}"
            )
        logger.info(
            "Stellar RV: _process_fiber fiber=%s spectrograph=%s algorithm=%s -> %s",
            fiber,
            spectrograph,
            algo_key,
            method_name,
        )
        runner = getattr(self, method_name)
        logger.info("Stellar RV: running RV algorithm fiber=%s algo=%s", fiber, algo_key)
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
            "Stellar RV: RV algorithm returned fiber=%s algo=%s n_chunks=%d",
            fiber,
            algo_key,
            len(all_wavelengths),
        )
        logger.info("Stellar RV: writing RV tables fiber=%s", fiber)
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
            "Stellar RV: RV tables written fiber=%s (RVCCF + RVCCFAVG)",
            fiber,
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

        logger.info(
            "Stellar RV: _process_fiber_orders fiber=%s mode=%s n_order_tasks=%d "
            "want_preview=%s max_cores=%s",
            fiber,
            mode,
            len(order_tasks),
            want_preview,
            max_cores,
        )

        if mode == "debug":
            logger.info(
                "Stellar RV: debug mode — per-chunk masked_ccf_rv_diagnostics fiber=%s",
                fiber,
            )
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
                logger.info(
                    "Stellar RV: building CCF preview dict (spectra_orders path) fiber=%s",
                    fiber,
                )
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
                    "Stellar RV: CCF preview dict ready (debug path) fiber=%s slots=%s",
                    fiber,
                    len(preview_detail.get("slots", [])) if preview_detail else 0,
                )
            return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail

        if mode == "parallel":
            logger.info(
                "Stellar RV: Numba parallel path fiber=%s want_preview=%s has_numba=%s",
                fiber,
                want_preview,
                _HAS_NUMBA,
            )
            spec_2d = np.asarray(spec, dtype=np.float64)
            err_2d = np.asarray(err, dtype=np.float64)
            if err.ndim == 1:
                err_2d = np.broadcast_to(err, spec.shape).copy()
            dq_2d = np.asarray(dq, dtype=np.uint32)
            if dq.ndim == 1:
                dq_2d = np.broadcast_to(dq, spec.shape).copy()
            wave_2d = np.asarray(wave, dtype=np.float64)

            preview_detail = None
            if want_preview and _HAS_NUMBA:
                logger.info(
                    "Stellar RV: process_orders_rvs_numba_parallel (return_ccf=True) fiber=%s",
                    fiber,
                )
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
                    "Stellar RV: Numba parallel CCF+RV done fiber=%s n_flat_chunks=%d",
                    fiber,
                    len(all_rvs_arr),
                )
                logger.info(
                    "Stellar RV: building preview from Numba flat CCF fiber=%s",
                    fiber,
                )
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
                    "Stellar RV: preview from Numba flat complete fiber=%s",
                    fiber,
                )
            else:
                logger.info(
                    "Stellar RV: process_orders_rvs_numba_parallel (return_ccf=False) fiber=%s",
                    fiber,
                )
                (
                    all_rvs_arr,
                    all_rv_errs_arr,
                    all_wl_arr,
                    n_chunks_per_order,
                    chunk_start,
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
                    return_ccf=False,
                )
                logger.info(
                    "Stellar RV: Numba parallel RV only done fiber=%s n_flat_chunks=%d",
                    fiber,
                    len(all_rvs_arr),
                )

            ao_flat = flat_chunk_echelle_orders(order_tasks, chunk_start, len(all_rvs_arr))
            all_rvs.extend(all_rvs_arr)
            all_rv_errs.extend(all_rv_errs_arr)
            all_wavelengths.extend(all_wl_arr)
            all_orders.extend(int(x) for x in ao_flat)
            logger.info(
                "Stellar RV: parallel path collected fiber=%s n_chunks=%d preview=%s",
                fiber,
                len(all_rvs),
                preview_detail is not None,
            )
            return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail

        # serial (no Numba)
        logger.info(
            "Stellar RV: serial Python path (per-order process_one_order_rvs) fiber=%s",
            fiber,
        )
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
                logger.info(
                    "Stellar RV: finalize_ccf_preview_dict (serial+preview) fiber=%s "
                    "n_order_slots=%d",
                    fiber,
                    len(order_slots),
                )
                preview_detail = finalize_ccf_preview_dict(
                    order_slots,
                    all_rows,
                    v_ref,
                    _V_MIN,
                    _V_MAX,
                    _V_STEP,
                )
                logger.info(
                    "Stellar RV: serial preview dict complete fiber=%s total_slots=%s",
                    fiber,
                    len(preview_detail.get("slots", [])) if preview_detail else 0,
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

        logger.info(
            "Stellar RV: serial path done fiber=%s n_chunks=%d preview=%s",
            fiber,
            len(all_rvs),
            preview_detail is not None,
        )
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
        logger.info(
            "Stellar RV: _write_rv_tables fiber=%s spectrograph=%s n_per_chunk_rows=%d "
            "rv_agg=%.2f m/s",
            fiber,
            spectrograph,
            n,
            float(rv_agg) if np.isfinite(rv_agg) else float("nan"),
        )
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
        logger.debug(
            "Stellar RV: RVCCF/AVG attributes set on model (%s, %s)",
            _ext("RVCCF"),
            _ext("RVCCFAVG"),
        )

    def _create_preview(self):
        from .stellar_rv_preview import create_stellar_rv_preview

        logger.info("Stellar RV: _create_preview starting (Plotly stellar RV preview)")
        create_stellar_rv_preview(self)
        logger.info("Stellar RV: _create_preview finished")
