"""Stellar radial-velocity primitive (template and mask-line CCF).

Algorithm overview and references live in ``docs/source/algorithms/stellar_rv.rst``.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import astropy.table

from ..base_primitive import BasePrimitive
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
from ...datamodels.extensions import RVCCF_CCF_ELEM_DTYPE
from ..utils.ccf.ccf_common import science_centroid_spec_err

from .rv_combine_utils import (
    DEFAULT_CHUNK_RV_MAX_ABS_M_S,
    DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S,
    combine_rv_estimates_median_mad_weighted,
    combine_rvs_median_mad,
    combine_stellar_rv_chunk_estimates,
    weighted_mean_rv,
)

from ...utils.parallelization_utils import resolve_numba_threads
from ..utils.ccf.ccf_common import (
    V_MAX_DEFAULT_M_S,
    V_MIN_DEFAULT_M_S,
    V_STEP_DEFAULT_M_S,
)
from ..utils.ccf.ccf_utils import (
    assemble_rvccf_fib_table,
    build_centered_uniform_velocity_grid,
    ccf_order_n_chunks_for_span,
    validate_line_mask_line_window_chunking_config,
    build_rvccf_chunk_supplement,
    effective_ccf_v_step_for_fit,
    resolve_ccf_broad_grid_m_s,
    run_ccf_line_mask_for_fiber,
    STELLAR_RV_LINE_MASK_ALGORITHMS,
    build_ccf_preview_dict,
    flat_chunk_echelle_orders,
    masked_ccf_rv,
    process_orders_rvs_numba_parallel,
)
from ..utils.spectral_utils import resolve_stellar_template_to_log_micron

__all__ = [
    "StellarRVPrimitive",
    "STELLAR_RV_ALGORITHMS",
    "combine_rvs_median_mad",
    "weighted_mean_rv",
    "combine_rv_estimates_median_mad_weighted",
    "combine_stellar_rv_chunk_estimates",
    "DEFAULT_CHUNK_RV_MAX_ABS_M_S",
    "DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S",
]

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
    uncal = getattr(input_model, f"SPEC_UNCAL_FIB{fiber}", None)
    uncal_err = getattr(input_model, f"SPEC_UNCAL_ERR_FIB{fiber}", None)
    err = science_centroid_spec_err(spec, err, uncal, uncal_err)
    return spec, err, dq, getattr(input_model, wa)


STELLAR_RV_ALGORITHMS = {
    "ccf_stellar_template": "_run_algorithm_ccf_stellar_template",
    "ccf_line_mask": "_run_algorithm_ccf_line_mask",
}


class StellarRVPrimitive(BasePrimitive):
    """
    Cross-correlate L2 spectra against a template or a line mask to measure RV.

    **Algorithms**

    * ``ccf_line_mask`` (default) — mask-shifted CCF using a **line mask** (SPIROU ``.mas`` or
      two-column center+weight tables; see :mod:`~hispecdrp.utils.line_mask_utils`). Per-chunk
      CCFs use Numba Lanczos / linear spectrum sampling
      (:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask` /
      ``mask_ccf_pixel_interp_order``; default Lanczos-3); final RVs are Gaussian centroids from
      :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`.
    * ``ccf_stellar_template`` — weighted CCF in :math:`\\ln\\lambda` against a stellar
      template; per-chunk RVs; median/MAD rejection; inverse-variance mean
      (:mod:`~hispecdrp.primitives.utils.ccf.ccf_utils`).

    All constructor arguments list their **default** from :attr:`PARAMETERS`; override per run as
    needed. Order matches the grouped layout in :attr:`PARAMETERS`.

    Parameters
    ----------
    suffix : str
        **Default:** ``"stellar_rv"``. Short label used by the framework for this primitive’s
        derived products, logging, and file naming together with ``output_dir``.
    algorithm : str
        **Default:** ``"ccf_line_mask"``. Must be a key of :data:`STELLAR_RV_ALGORITHMS`.
        Chooses ``_run_algorithm_ccf_line_mask`` (line-mask CCF, no PHOENIX template) or
        ``_run_algorithm_ccf_stellar_template`` (log-λ template CCF).
    do_fibers : sequence of str or None
        **Default:** ``None``. Which fiber indices to process (e.g. ``["1","2"]``). ``None``
        uses :meth:`~hispecdrp.processing_component.ProcessingComponent.get_do_traces` /
        recipe defaults for the input model.
    do_orders : sequence of int or None
        **Default:** ``None``. Echelle **order numbers** (not row indices) to include. ``None``
        selects all orders handled by the same L2 helper as sibling primitives (subject to
        config / model).
    template_path : str or None
        **Default:** ``None``. Path to a stellar template spectrum on disk for
        ``ccf_stellar_template``. Passed to
        :func:`~hispecdrp.primitives.utils.spectral_utils.resolve_stellar_template_to_log_micron`
        with ``template_wavelength`` / ``template_flux`` when those are set.
    template_wavelength : array-like or None
        **Default:** ``None``. In-memory template wavelength grid (µm). Used with
        ``template_flux`` when you do not load from ``template_path``.
    template_flux : array-like or None
        **Default:** ``None``. Template flux samples matching ``template_wavelength`` (same length).
    mask_path : str or None
        **Default:** ``None``. Path to a SPIROU ``.mas`` or KPF-style two-column mask file for
        ``ccf_line_mask`` (delta-line model). ``None`` resolves to the packaged lte049
        **ccfopt** delta mask (see
        :func:`~hispecdrp.utils.line_mask_utils.packaged_default_line_mask_path`) via
        :func:`~hispecdrp.utils.line_mask_utils.resolve_mask_path` when you are **not**
        passing ``mask_wavelength`` / ``mask_weight``.
    mask_wavelength : array-like or None
        **Default:** ``None``. In-memory mask rest wavelengths (µm) for ``ccf_line_mask``. Must be
        paired with ``mask_weight``; when both are set, ``mask_path`` is not required for loading.
    mask_weight : array-like or None
        **Default:** ``None``. Weights matching ``mask_wavelength`` (same length).
    mask_ccf_pixel_interp_order : int
        **Default:** ``3``. **Mask algorithm only:** Lanczos parameter *a* (``1`` = linear;
        ``3`` = Lanczos-3 windowed-sinc in pixel-index space) for sampling flux and the
        per-pixel gate onto Doppler-shifted mask-line wavelengths in
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils._calc_mask_ccf_chunks_numba`. Values are clamped
        to ``[1, ccf_utils.MAX_MASK_CCF_PIXEL_INTERP_ORDER]``. Values **greater than 1** use
        Lanczos-*a* (not historical Lagrange polynomials).
    window_size_um : float
        **Default:** ``0.25``. Approximate wavelength span (µm) of each **chunk** along an order:
        spectra are split into contiguous bins of roughly this width for per-chunk RVs (template
        algorithm) or mask CCF chunking (mask algorithm).
    outlier_sigma : float
        **Default:** ``3.0``. Multiplier for error-aware chunk rejection before the
        inverse-variance weighted mean: keep chunks with
        ``|RV − center| <= outlier_sigma * max(MAD, σ_i)`` (see
        :func:`~hispecdrp.primitives.stellar_rv.rv_combine_utils.combine_stellar_rv_chunk_estimates`).
    chunk_rv_max_abs_m_s : float
        **Default:** ``120000.0`` (120 km s\\ :sup:`−1`). Per-chunk RVs with ``|RV|`` above this limit
        are excluded before aggregation (coarse-grid CCF alias peaks). Set very large to disable.
        Ignored when ``ccf_input_velocity`` is set (prior gate used instead).
    chunk_rv_prior_half_width_m_s : float or None
        **Default:** ``None``. When ``ccf_input_velocity`` is set, keep only chunks with
        ``|RV − ccf_input_velocity| <= half_width`` before aggregation. ``None`` uses
        ``max(2000, 0.5 × v_half_window)`` m/s.
    window_line_by_line : bool
        **Default:** ``True``. **Line-mask algorithm only** (``ccf_line_mask``). Ignored for
        ``ccf_stellar_template`` (that path always chunks on ``window_size_um``). When ``False``,
        chunk boundaries follow ``window_size_um`` via
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`.
        When ``True``, each mask line defines its own wavelength chunk (Voronoi bins in observed λ
        at ``ccf_input_velocity``, which must be finite — grid center). Spectral pixels
        per chunk use the intersection of that Voronoi interval with a symmetric ``window_size_um``
        band; see :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.build_line_mask_chunk_wavelength_intervals_line_by_line`.
    v_min_coarse : float or None
        **Default:** ``None``. Broad uniform CCF lower bound (m/s) when no center is set (scout /
        full-range search). ``None`` → :data:`~hispecdrp.primitives.utils.ccf.ccf_common.V_MIN_DEFAULT_M_S`.
    v_max_coarse : float or None
        **Default:** ``None``. Broad uniform CCF upper bound (m/s). ``None`` → ``V_MAX_DEFAULT_M_S``.
    v_step_coarse : float or None
        **Default:** ``25.0``. Uniform velocity sampling (m/s) for **both** the broad and centered
        grids. ``None`` → :data:`~hispecdrp.primitives.utils.ccf.ccf_common.V_STEP_DEFAULT_M_S`.
    ccf_input_velocity : float or None
        **Default:** ``None``. If set (m/s), the CCF uses a **uniform** window centered here:
        ``[ccf_input_velocity − v_half_window, ccf_input_velocity + v_half_window]`` at
        ``v_step_coarse``. If ``None``, a scout pass on the broad uniform grid estimates the
        peak, then the fiber is evaluated on a centered window (``v_half_window`` at
        ``v_step_coarse``).
    v_half_window : float
        **Default:** ``15000.0``. Half-width (m/s) of the centered uniform velocity window
        (±this value around ``ccf_input_velocity`` or the auto-scout center).
    ccf_zb : float
        **Default:** ``0.0``. **Both algorithms:** barycentric factor :math:`z_b`. For ``ccf_line_mask``
        it enters :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}(1+v/c)/(1+z_b)` (``zb`` in
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber` /
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`). For ``ccf_stellar_template``
        it applies the same multiplicative shift in :math:`\\ln\\lambda` template evaluation as in
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils._ccf_chunk_ccf_numba`.
        Set by the caller or pipeline (e.g. :class:`~hispecdrp.pipelines.stage2.stage2_pipeline.Stage2Pipeline`
        populates this from ``BARYCORR`` ``BERV`` when not overridden).
    ccf_min_good_pixels : int
        **Default:** ``9``. **Both algorithms:** minimum unmasked finite spectral pixels per
        **wavelength chunk** (chunking from ``window_size_um`` via
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`, or from
        ``window_line_by_line`` Voronoi bins) required to form
        a CCF and fit an RV. For ``ccf_line_mask`` this is ``min_good_pixels`` in
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`; for ``ccf_stellar_template``
        it gates the Numba per-chunk CCF and the Python per-chunk fit. Overlay debug HTML uses the
        same threshold for chunk inclusion.
    output_model : str
        **Default:** ``"L2Model"``. Datamodel type name passed to
        :meth:`~hispecdrp.primitives.base_primitive.BasePrimitive.get_output_model` when cloning
        the input to attach RVCCF tables.
    max_cores : int or None
        **Default:** ``None``. Upper bound on parallel **Numba** threads: template path uses it for
        echelle-order parallelism; mask path passes it to
        :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber` / chunk loops.
        ``None`` lets :func:`~hispecdrp.utils.parallelization_utils.resolve_numba_threads` pick a
        host-appropriate count; ``1`` forces single-threaded, reproducible runs.
    debug_ccf : bool
        **Default:** ``False``. Extra logging and diagnostics. For ``ccf_stellar_template``, forces
        the slower non-Numba CCF path instead of the parallel driver.
    preview_write_png : bool
        **Default:** ``True``. When ``save_preview`` is True, also write a stacked Kaleido PNG
        alongside the HTML preview. Set False for large line-mask runs (many CCF slots) where
        PNG export can take minutes or hang in headless Chrome.
    preview_png_scale : float
        **Default:** ``2.0``. Pixel scale multiplier for Kaleido PNG exports when ``save_preview`` and
        ``preview_write_png`` are True (see :mod:`~hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`).
    show_overlay_debug : bool
        **Default:** ``False``. For ``ccf_line_mask``, open mask-overlay debug HTML in a browser; for
        ``ccf_stellar_template``, open template-overlay debug HTML. Meaning follows ``algorithm``.
    save_overlay_debug : bool
        **Default:** ``False``. For ``ccf_line_mask``, write mask-overlay HTML (``_output["mask_overlay_debug"]``);
        for ``ccf_stellar_template``, write template-overlay HTML (``_output["template_overlay_debug"]``).
    """

    # Default primitive arguments (merged with ProcessingComponent at class init).
    # Keys and semantics: see class docstring **Parameters** (same order as below).
    PARAMETERS = {
        # --- Dispatch ---
        "suffix": "stellar_rv",
        "algorithm": "ccf_line_mask",
        "do_fibers": None,
        "do_orders": None,
        # --- Stellar template (ccf_stellar_template): template_* ---
        "template_path": None,
        "template_wavelength": None,
        "template_flux": None,
        # --- Mask line list & mask-only analysis (ccf_line_mask): mask_* ---
        # None → hispecdrp/data/masks packaged default (ccfopt) via resolve_mask_path.
        "mask_path": None,
        "mask_wavelength": None,
        "mask_weight": None,
        "mask_ccf_pixel_interp_order": 3,
        # --- Shared chunking & chunk RV aggregation (both algorithms) ---
        "window_size_um": 0.25,
        "outlier_sigma": 3.0,
        "chunk_rv_max_abs_m_s": DEFAULT_CHUNK_RV_MAX_ABS_M_S,
        "chunk_rv_prior_half_width_m_s": None,
        "window_line_by_line": True,  # ccf_line_mask only; False -> wavelength window_size_um chunks
        "ccfopt_line_by_line_defaults": True,  # when line-by-line: use CCFOPT_SINGLE_LINE_STELLAR_RV_KW if wide defaults
        # --- CCF velocity grid & shared CCF analysis (both algorithms) ---
        "v_min_coarse": None,  # m/s; None -> ccf_common.V_MIN_DEFAULT_M_S
        "v_max_coarse": None,  # m/s; None -> V_MAX_DEFAULT_M_S
        "v_step_coarse": 25.0,  # m/s; None -> ccf_common.V_STEP_DEFAULT_M_S
        "ccf_input_velocity": None,  # m/s; centered window when set; else auto-center scout
        "v_half_window": 15000.0,  # m/s; ± half-width of centered uniform grid
        "ccf_zb": 0.0,
        "ccf_min_good_pixels": 9,
        # --- Output & compute ---
        "output_model": "L2Model",
        "max_cores": None,
        "debug_ccf": False,
        # --- Previews & overlay debug HTML ---
        "preview_write_png": False,
        "preview_png_scale": 2.0,
        "show_overlay_debug": False,
        "save_overlay_debug": False,
    }

    def _perform(self):
        """
        Run stellar RV on the input model(s) and produce L2 output.

        For template mode, loads the stellar template, then for each input model and
        each fiber runs CCF (debug Python or Numba-parallel), combines chunk RVs, and writes
        RVCCF_FIB# and RVCCFAVG_FIB# tables to the output L2 model.

        Returns
        -------
        dict
            With key "output" containing the output L2 model (or list of models).
        """
        if isinstance(self.dataset, (datamodels.DataModel, str)):
            output = None
            inp_list = [self.dataset]
        elif isinstance(self.dataset, Sequence):
            output = []
            inp_list = self.dataset
        else:
            raise TypeError("Input must be a DataModel, path, or sequence of these.")

        algo_key = (getattr(self, "algorithm", None) or "ccf_line_mask").lower().strip()
        logger.info(
            "Stellar RV: starting _perform algorithm=%s n_inputs=%d suffix=%s",
            algo_key,
            len(inp_list),
            getattr(self, "suffix", ""),
        )
        # Default window_line_by_line=True is for ccf_line_mask. Template CCF still
        # chunks on window_size_um; ignore the line-mask flag so callers can switch
        # algorithm without also flipping chunking.
        if (
            algo_key not in STELLAR_RV_LINE_MASK_ALGORITHMS
            and bool(getattr(self, "window_line_by_line", False))
        ):
            logger.info(
                "Stellar RV: ignoring window_line_by_line for algorithm=%s "
                "(line-mask only)",
                algo_key,
            )
            self.window_line_by_line = False
        validate_line_mask_line_window_chunking_config(
            algorithm=algo_key,
            window_line_by_line=bool(getattr(self, "window_line_by_line", False)),
            ccf_input_velocity_m_s=getattr(self, "ccf_input_velocity", None),
        )
        if bool(getattr(self, "window_line_by_line", False)) and bool(
            getattr(self, "ccfopt_line_by_line_defaults", True)
        ):
            from ...utils.line_mask_utils import apply_ccfopt_defaults_for_line_by_line

            ws, vh, vs = apply_ccfopt_defaults_for_line_by_line(
                window_size_um=float(getattr(self, "window_size_um", 0.25)),
                v_half_window=float(getattr(self, "v_half_window", 15000.0)),
                v_step_coarse=float(getattr(self, "v_step_coarse", 25.0)),
            )
            self.window_size_um = ws
            self.v_half_window = vh
            self.v_step_coarse = vs
        template_ln, template_flux = None, None
        if (algo_key or "").strip().lower() not in STELLAR_RV_LINE_MASK_ALGORITHMS:
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
            self._stellar_template_wave_um = np.exp(
                np.asarray(template_ln, dtype=np.float64)
            )
            self._stellar_template_flux = np.asarray(template_flux, dtype=np.float64)
        else:
            # Mask-based CCF does not require a stellar template; we keep the
            # variables to satisfy the algorithm dispatch signature.
            template_ln = np.array([], dtype=np.float64)
            template_flux = np.array([], dtype=np.float64)
            self._stellar_template_wave_um = None
            self._stellar_template_flux = None
            logger.info("Stellar RV: ccf_line_mask mode — empty template arrays (line-mask CCF only)")
            _mw = getattr(self, "mask_wavelength", None)
            _mwt = getattr(self, "mask_weight", None)
            if (_mw is None) ^ (_mwt is None):
                raise ValueError(
                    "mask_wavelength and mask_weight must both be set when supplying an in-memory mask"
                )

        # Per-fiber CCF preview slots (mean + per-order); filled when save/show preview.
        self._ccf_preview_by_fiber: dict = {}
        # Per-fiber payload for mask overlay debug HTML (ccf_line_mask only).
        self._mask_overlay_by_fiber: dict = {}
        # Per-fiber payload for template overlay debug HTML (ccf_stellar_template only).
        self._template_overlay_by_fiber: dict = {}
        # Keys merged into the return dict so ``apply()`` keeps them on ``primitive._output``
        # (alongside ``output``); do not stash on ``self._output`` during _perform — ``apply``
        # replaces ``self._output`` with the return value.
        output_extras: dict = {}
        from .previews.stellar_rv_preview import maybe_write_ccf_overlay_debug_html

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

            maybe_write_ccf_overlay_debug_html(self, algo_key, output_model, output_extras)

            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        logger.info("Stellar RV: _perform finished algorithm=%s", algo_key)
        out = dict(output=output)
        out.update(output_extras)
        return out

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

    def _resolve_chunk_rv_prior_half_width_m_s(self) -> float | None:
        """Half-width (m/s) for prior-gated chunk aggregation when a CCF center is set."""
        prior = getattr(self, "ccf_input_velocity", None)
        if prior is None or not np.isfinite(float(prior)):
            return None
        half_w = getattr(self, "chunk_rv_prior_half_width_m_s", None)
        if half_w is not None and np.isfinite(float(half_w)) and float(half_w) > 0:
            return float(half_w)
        v_half_window = float(getattr(self, "v_half_window", 20000.0))
        return max(2000.0, 0.5 * v_half_window)

    def _combine_chunk_rvs(
        self,
        all_rvs: np.ndarray,
        all_rv_errs: np.ndarray,
        *,
        fiber: str,
    ) -> tuple[np.ndarray, float, float]:
        """
        Robust per-fiber aggregate from chunk RV arrays (sanity gate + MAD + weighted mean).

        Returns ``(inlier_mask, rv_mean_m_s, rv_std_m_s)``.
        """
        all_rvs = np.asarray(all_rvs, dtype=np.float64)
        all_rv_errs = np.asarray(all_rv_errs, dtype=np.float64)
        valid = np.isfinite(all_rvs) & np.isfinite(all_rv_errs)
        if not np.any(valid):
            logger.warning("No valid chunk RVs for fiber %s.", fiber)
            return np.zeros(len(all_rvs), dtype=bool), np.nan, np.nan

        max_abs = float(getattr(self, "chunk_rv_max_abs_m_s", DEFAULT_CHUNK_RV_MAX_ABS_M_S))
        osig = float(getattr(self, "outlier_sigma", 4.0))
        prior_m_s = getattr(self, "ccf_input_velocity", None)
        prior_half_w = self._resolve_chunk_rv_prior_half_width_m_s()
        use_prior = prior_m_s is not None and np.isfinite(float(prior_m_s)) and prior_half_w is not None
        if use_prior:
            logger.info(
                "Stellar RV: combining chunk RVs fiber=%s n_valid=%d outlier_sigma=%.2f "
                "prior=%.2f m/s prior_half_width=%.0f m/s",
                fiber,
                int(np.sum(valid & (all_rv_errs > 0))),
                osig,
                float(prior_m_s),
                float(prior_half_w),
            )
        else:
            logger.info(
                "Stellar RV: combining chunk RVs fiber=%s n_valid=%d outlier_sigma=%.2f "
                "chunk_rv_max_abs_m_s=%.0f",
                fiber,
                int(np.sum(valid & (all_rv_errs > 0))),
                osig,
                max_abs,
            )
        rv_mean, rv_std, inlier, _, _ = combine_stellar_rv_chunk_estimates(
            all_rvs,
            all_rv_errs,
            osig,
            chunk_rv_max_abs_m_s=max_abs,
            chunk_rv_prior_m_s=float(prior_m_s) if use_prior else None,
            chunk_rv_prior_half_width_m_s=float(prior_half_w) if use_prior else None,
        )
        if use_prior:
            n_sane = int(
                np.sum(
                    valid
                    & (all_rv_errs > 0)
                    & (np.abs(all_rvs - float(prior_m_s)) <= float(prior_half_w))
                )
            )
        else:
            n_sane = int(np.sum(valid & (all_rv_errs > 0) & (np.abs(all_rvs) <= max_abs)))
        logger.info(
            "Stellar RV: aggregate RV fiber=%s rv_mean=%.2f m/s rv_std=%.2f m/s "
            "n_inlier=%d n_sane=%d/%d",
            fiber,
            float(rv_mean) if np.isfinite(rv_mean) else float("nan"),
            float(rv_std) if np.isfinite(rv_std) else float("nan"),
            int(np.sum(inlier)),
            n_sane,
            len(all_rvs),
        )
        return inlier, float(rv_mean), float(rv_std)

    @staticmethod
    def resolve_ccf_zb_for_fiber(
        primitive,
        input_model,
        fiber: str | int,
    ) -> float:
        """
        Return the uniform CCF :math:`z_b` for previews and overlays.

        ``fiber`` is accepted for API compatibility; all fibers share
        ``primitive.ccf_zb``.
        """
        del input_model, fiber
        return float(getattr(primitive, "ccf_zb", 0.0))

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
        log(wavelength), producing per-chunk RVs (debug Python or Numba-parallel). Chunk RVs
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
        max_cores = resolve_numba_threads(
            None if _mc is None else int(_mc)
        )

        order_tasks = [
            (oi, order)
            for oi, order in enumerate(order_vec)
            if order in do_orders
        ]
        want_preview = bool(
            getattr(self, "save_preview", False)
            or getattr(self, "show_preview", False)
            or getattr(self, "save_overlay_debug", False)
            or getattr(self, "show_overlay_debug", False)
        )
        if debug_ccf:
            mode = "debug"
        else:
            mode = "parallel"

        vm, vx, vs = resolve_ccf_broad_grid_m_s(
            getattr(self, "v_min_coarse", None),
            getattr(self, "v_max_coarse", None),
            getattr(self, "v_step_coarse", None),
        )
        logger.info(
            "Stellar RV: ccf_stellar_template fiber=%s spectrograph=%s mode=%s "
            "n_order_tasks=%d want_preview=%s max_cores=%s window_um=%.4f "
            "ccf_v_m_s=[%.2f, %.2f] step=%.4f refined=%s",
            fiber,
            spectrograph,
            mode,
            len(order_tasks),
            want_preview,
            max_cores,
            float(getattr(self, "window_size_um", 0.3)),
            vm,
            vx,
            vs,
            getattr(self, "ccf_input_velocity", None) is not None,
        )

        (
            all_rvs,
            all_rv_errs,
            all_wavelengths,
            all_orders,
            preview_detail,
            rvccf_chunk_supplement,
        ) = self._process_fiber_orders(
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
            v_min_m_s=vm,
            v_max_m_s=vx,
            v_step_m_s=vs,
            ccf_center_m_s=(
                float(getattr(self, "ccf_input_velocity"))
                if getattr(self, "ccf_input_velocity", None) is not None
                else None
            ),
            ccf_half_window_m_s=float(getattr(self, "v_half_window", 20000.0)),
        )
        if preview_detail is not None:
            # Preview lookup always uses string fiber ids ("1"/"2"/"3").
            self._ccf_preview_by_fiber[str(fiber)] = preview_detail

        all_rvs = np.array(all_rvs)
        all_rv_errs = np.array(all_rv_errs)
        all_wavelengths = np.array(all_wavelengths)
        all_orders = np.asarray(all_orders, dtype=np.int64)

        valid = np.isfinite(all_rvs) & np.isfinite(all_rv_errs)
        if not np.any(valid):
            inlier = np.zeros(len(all_rvs), dtype=bool)
            rv_mean, rv_std = np.nan, np.nan
        else:
            inlier, rv_mean, rv_std = self._combine_chunk_rvs(
                all_rvs, all_rv_errs, fiber=fiber
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

        if bool(
            getattr(self, "show_overlay_debug", False)
            or getattr(self, "save_overlay_debug", False)
        ):
            from .previews.stellar_rv_preview import collect_template_overlay_payload_for_primitive

            payload = collect_template_overlay_payload_for_primitive(
                self,
                input_model=input_model,
                fiber=fiber,
                order_vec=order_vec,
                chunk_rv_m_s=all_rvs,
                chunk_rv_err_m_s=all_rv_errs,
                chunk_echelle_order=all_orders,
                chunk_wavelength_um=np.asarray(all_wavelengths, dtype=np.float64),
            )
            if payload is not None:
                self._template_overlay_by_fiber[fiber] = payload
                logger.debug(
                    "Stellar RV: template overlay payload stored fiber=%s n_chunks=%d",
                    fiber,
                    len(payload.get("orders", [])),
                )

        return (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            inlier,
            rv_mean,
            rv_std,
            all_orders,
            rvccf_chunk_supplement,
        )

    def _run_algorithm_ccf_line_mask(
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
        Line-mask CCF (SPIROU/KPF-inspired) + Gaussian RV fit.

        Minimal delegation to :func:`run_ccf_line_mask_for_fiber`.
        """
        want_preview = bool(
            getattr(self, "save_preview", False)
            or getattr(self, "show_preview", False)
            or getattr(self, "show_overlay_debug", False)
            or getattr(self, "save_overlay_debug", False)
        )
        vm, vx, vs = resolve_ccf_broad_grid_m_s(
            getattr(self, "v_min_coarse", None),
            getattr(self, "v_max_coarse", None),
            getattr(self, "v_step_coarse", None),
        )
        logger.info(
            "Stellar RV: dispatching ccf_line_mask fiber=%s spectrograph=%s "
            "n_do_orders=%d want_preview=%s ccf_v_m_s=[%.2f, %.2f] step=%.4f refined=%s",
            fiber,
            spectrograph,
            len(do_orders) if do_orders is not None else 0,
            want_preview,
            vm,
            vx,
            vs,
            getattr(self, "ccf_input_velocity", None) is not None,
        )
        preview_detail = None
        h_center = getattr(self, "ccf_input_velocity", None)
        (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            all_orders,
            preview_detail,
            rvccf_chunk_supplement,
        ) = run_ccf_line_mask_for_fiber(
            input_model=input_model,
            fiber=fiber,
            do_orders=do_orders,
            order_vec=order_vec,
            window_size_um=float(getattr(self, "window_size_um", 0.3)),
            mask_path=getattr(self, "mask_path", None),
            mask_wavelength=getattr(self, "mask_wavelength", None),
            mask_weight=getattr(self, "mask_weight", None),
            v_min_m_s=vm,
            v_max_m_s=vx,
            v_step_m_s=vs,
            ccf_center_m_s=(float(h_center) if h_center is not None else None),
            ccf_half_window_m_s=float(getattr(self, "v_half_window", 20000.0)),
            zb=float(self.ccf_zb),
            max_cores=(None if getattr(self, "max_cores", None) is None else int(getattr(self, "max_cores"))),
            want_preview=want_preview,
            debug_ccf=bool(getattr(self, "debug_ccf", False)),
            min_good_pixels=int(getattr(self, "ccf_min_good_pixels", 9)),
            mask_ccf_pixel_interp_order=int(
                getattr(self, "mask_ccf_pixel_interp_order", 3)
            ),
            window_line_by_line=bool(getattr(self, "window_line_by_line", False)),
        )

        # Fiber-level aggregation belongs to stellar_rv (not ccf_utils) so that stellar_rv depends
        # on ccf_utils, but ccf_utils does not depend on stellar_rv.
        valid = np.isfinite(all_rvs) & np.isfinite(all_rv_errs) & (np.asarray(all_rv_errs) > 0)
        if not np.any(valid):
            inlier = np.zeros(len(all_rvs), dtype=bool)
            rv_mean, rv_std = np.nan, np.nan
        else:
            inlier, rv_mean, rv_std = self._combine_chunk_rvs(
                np.asarray(all_rvs, dtype=np.float64),
                np.asarray(all_rv_errs, dtype=np.float64),
                fiber=fiber,
            )
        if preview_detail is not None:
            # Preview lookup always uses string fiber ids ("1"/"2"/"3").
            self._ccf_preview_by_fiber[str(fiber)] = preview_detail
        if bool(
            getattr(self, "show_overlay_debug", False)
            or getattr(self, "save_overlay_debug", False)
        ):
            from .previews.stellar_rv_preview import collect_mask_overlay_payload_for_primitive

            payload = collect_mask_overlay_payload_for_primitive(
                self,
                input_model=input_model,
                fiber=fiber,
                order_vec=order_vec,
                chunk_rv_m_s=all_rvs,
                chunk_rv_err_m_s=all_rv_errs,
                chunk_echelle_order=all_orders,
                chunk_wavelength_um=np.asarray(all_wavelengths, dtype=np.float64),
            )
            if payload is not None:
                self._mask_overlay_by_fiber[fiber] = payload
                logger.debug(
                    "Stellar RV: mask overlay payload stored fiber=%s n_chunks=%d",
                    fiber,
                    len(payload.get("orders", [])),
                )
        return (
            all_wavelengths,
            all_rvs,
            all_rv_errs,
            inlier,
            rv_mean,
            rv_std,
            all_orders,
            rvccf_chunk_supplement,
        )

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

        Dispatches to the algorithm named in ``self.algorithm`` (see ``STELLAR_RV_ALGORITHMS``).
        The algorithm returns per-chunk arrays, aggregate RV, and optional per-chunk CCF
        supplement columns; this method writes ``RVCCF_FIB{fiber}`` (merged RV + CCF)
        and ``RVCCFAVG_FIB{fiber}`` to the output model.

        Parameters
        ----------
        input_model : DataModel
            Open L2 (or compatible) model with per-fiber ``SPEC_FIB*`` / ``WAVE_FIB*`` (and
            optional ERR/DQ). Used to read science data for the chosen fiber.
        output_model : DataModel
            Cloned L2 output; receives Astropy tables on attributes ``rvccf_fib*`` /
            ``rvccfavg_fib*`` (framework naming).
        fiber : str
            Fiber id string, e.g. ``"1"``, matching attribute suffixes ``_FIB1``.
        spectrograph : str
            ``"BSPEC"`` or ``"RSPEC"`` from metadata; stored in RV tables for downstream.
        do_orders : list of int
            Echelle order numbers to include in the CCF (subset of ``order_vec``).
        order_vec : list of int
            Map from echelle row index to physical order number (one entry per spectral row).
        template_ln : np.ndarray
            Natural log of template wavelength in **microns**; used by template algorithm.
            For ``ccf_line_mask`` this is an empty placeholder array.
        template_flux : np.ndarray
            Template flux aligned with ``template_ln``; empty for mask-only algorithm.
        """
        if _fiber_l2_bundle(input_model, fiber) is None:
            return

        fiber_s = str(fiber)
        ccf_zb = float(getattr(self, "ccf_zb", 0.0))
        if ccf_zb != 0.0:
            logger.info(
                "Stellar RV: fiber=%s ccf_zb=%.9e",
                fiber,
                ccf_zb,
            )

        algo_key = (getattr(self, "algorithm", None) or "ccf_line_mask").lower().strip()
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
            rvccf_chunk_supplement,
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
            rvccf_chunk_supplement=rvccf_chunk_supplement,
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
        v_min_m_s: float = V_MIN_DEFAULT_M_S,
        v_max_m_s: float = V_MAX_DEFAULT_M_S,
        v_step_m_s: float = V_STEP_DEFAULT_M_S,
        ccf_center_m_s: float | None = None,
        ccf_half_window_m_s: float = 20000.0,
    ) -> tuple[list, list, list, list, dict | None, astropy.table.Table | None]:
        """
        Single driver for per-order chunk RVs: ``mode`` is ``debug`` (Python
        ``masked_ccf_rv`` per chunk) or ``parallel`` (Numba). When ``want_preview`` is
        True, the parallel path reuses the flat CCF stack for the preview dict; debug
        + preview uses :func:`build_ccf_preview_dict` with ``kind="spectra_orders"``.

        Parameters
        ----------
        order_tasks : list of tuple
            ``(row_index, order_number)`` for each echelle row to process (already filtered
            by ``do_orders``).
        wave, spec, err, dq : np.ndarray
            2D echelle data for the fiber: wavelength, flux, error, and DQ (same shape).
        do_not_use : int
            DQ bitmask for pixels to exclude from CCF (typically ``DQ_FLAGS["DO_NOT_USE"]``).
        template_ln, template_flux : np.ndarray
            Template in log(µm) and flux for CCF (same length).
        fiber : str
            Fiber id for logging.
        mode : str
            ``"debug"`` (verbose diagnostics) or ``"parallel"`` (Numba).
        max_cores : int or None, optional
            Effective thread count for parallel mode (already resolved/capped).
        want_preview : bool, optional
            Whether to build the mean + per-order CCF preview structure for Plotly.
        v_min_m_s, v_max_m_s, v_step_m_s : float, optional
            Uniform CCF velocity axis (m/s): broad search from the primitive’s
            ``v_min_coarse`` / ``v_max_coarse`` / ``v_step_coarse`` (after defaults).
            When ``ccf_center_m_s`` is set, ``v_step_m_s`` samples the centered window.
        ccf_center_m_s, ccf_half_window_m_s
            When ``ccf_center_m_s`` is not ``None``, build a uniform window via
            :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.build_centered_uniform_velocity_grid`
            (``ccf_input_velocity``, ``v_half_window``, ``v_step_coarse`` on the primitive).

        Returns
        -------
        tuple
            ``(all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail, rvccf_chunk_supplement)`` —
            lists of per-chunk quantities (same length), optional preview dict, and
            optional CCF supplement table (merged into ``RVCCF_FIB*`` by ``_write_rv_tables``).
        """
        all_rvs: list = []
        all_rv_errs: list = []
        all_wavelengths: list = []
        all_orders: list = []
        preview_detail: dict | None = None
        rvccf_chunk_supplement: astropy.table.Table | None = None

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
                "Stellar RV: debug mode — per-chunk masked_ccf_rv fiber=%s",
                fiber,
            )
            dbg_chunk_rows: list[np.ndarray] = []
            dbg_n_good: list[int] = []
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
                span_o = float(w_max - w_min)
                n_chunk = ccf_order_n_chunks_for_span(span_o, float(self.window_size_um))
                for k in range(n_chunk):
                    lam_lo = w_min + k * span_o / n_chunk
                    lam_hi = w_min + (k + 1) * span_o / n_chunk
                    if k == n_chunk - 1:
                        in_chunk = (w >= lam_lo) & (w <= lam_hi)
                    else:
                        in_chunk = (w >= lam_lo) & (w < lam_hi)
                    if np.sum(in_chunk) < 20:
                        continue
                    wl_center = 0.5 * (lam_lo + lam_hi)
                    if ccf_center_m_s is not None:
                        v_merged = build_centered_uniform_velocity_grid(
                            float(ccf_center_m_s),
                            float(ccf_half_window_m_s),
                            float(v_step_m_s),
                        )
                        rv, rv_err, v_grid_c, ccf_c = masked_ccf_rv(
                            w[in_chunk],
                            s[in_chunk],
                            e[in_chunk],
                            mask[in_chunk],
                            t_ln,
                            t_flux,
                            v_grid=v_merged,
                            zb=float(self.ccf_zb),
                            min_good_pixels=int(
                                getattr(self, "ccf_min_good_pixels", 9)
                            ),
                            return_ccf=True,
                        )
                    else:
                        rv, rv_err, v_grid_c, ccf_c = masked_ccf_rv(
                            w[in_chunk],
                            s[in_chunk],
                            e[in_chunk],
                            mask[in_chunk],
                            t_ln,
                            t_flux,
                            v_min=v_min_m_s,
                            v_max=v_max_m_s,
                            v_step=v_step_m_s,
                            zb=float(self.ccf_zb),
                            min_good_pixels=int(
                                getattr(self, "ccf_min_good_pixels", 9)
                            ),
                            return_ccf=True,
                        )
                    n_good = int(np.sum(~mask[in_chunk]))
                    logger.info(
                        "CCF fiber=%s order=%s chunk=%s wl_center=%.4f um wl_range=[%.4f, %.4f] "
                        "n_good=%s ccf_max=%.6f ccf_min=%.6f rv=%.1f m/s rv_err=%.1f",
                        fiber,
                        order,
                        k,
                        wl_center,
                        lam_lo,
                        lam_hi,
                        n_good,
                        float(np.nanmax(ccf_c)),
                        float(np.nanmin(ccf_c)),
                        rv,
                        rv_err,
                    )
                    all_rvs.append(rv)
                    all_rv_errs.append(rv_err)
                    all_wavelengths.append(wl_center)
                    all_orders.append(int(order))
                    dbg_chunk_rows.append(np.asarray(ccf_c, dtype=RVCCF_CCF_ELEM_DTYPE))
                    dbg_n_good.append(n_good)
            if dbg_chunk_rows:
                if ccf_center_m_s is not None:
                    _vg_dbg = build_centered_uniform_velocity_grid(
                        float(ccf_center_m_s),
                        float(ccf_half_window_m_s),
                        float(v_step_m_s),
                    )
                    _pv0 = float(_vg_dbg[0])
                    _pv1 = float(_vg_dbg[-1])
                    _pvs = float(effective_ccf_v_step_for_fit(_vg_dbg))
                else:
                    _pv0 = float(v_min_m_s)
                    _pv1 = float(v_max_m_s)
                    _pvs = float(v_step_m_s)
                rvccf_chunk_supplement = build_rvccf_chunk_supplement(
                    chunk_id=np.arange(len(dbg_chunk_rows), dtype=np.int32),
                    n_good=np.asarray(dbg_n_good, dtype=np.int32),
                    v_min=_pv0,
                    v_max=_pv1,
                    v_step=_pvs,
                    ccf_2d=np.stack(dbg_chunk_rows, axis=0),
                )
            if want_preview:
                logger.info(
                    "Stellar RV: building CCF preview dict (spectra_orders path) fiber=%s",
                    fiber,
                )
                if ccf_center_m_s is not None:
                    _vg_pr = build_centered_uniform_velocity_grid(
                        float(ccf_center_m_s),
                        float(ccf_half_window_m_s),
                        float(v_step_m_s),
                    )
                    _pmin, _pmax = float(_vg_pr[0]), float(_vg_pr[-1])
                    _pst = float(effective_ccf_v_step_for_fit(_vg_pr))
                else:
                    _pmin, _pmax, _pst = v_min_m_s, v_max_m_s, v_step_m_s
                preview_detail = build_ccf_preview_dict(
                    "spectra_orders",
                    order_tasks,
                    _pmin,
                    _pmax,
                    _pst,
                    wave=wave,
                    spec=spec,
                    err=err,
                    dq=dq,
                    do_not_use=do_not_use,
                    template_ln=template_ln,
                    template_flux=template_flux,
                    window_size_um=float(self.window_size_um),
                )
                logger.info(
                    "Stellar RV: CCF preview dict ready (debug path) fiber=%s slots=%s",
                    fiber,
                    len(preview_detail.get("slots", [])) if preview_detail else 0,
                )
            return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail, rvccf_chunk_supplement

        if mode != "parallel":
            raise ValueError(
                f"unknown stellar RV _process_fiber_orders mode: {mode!r} "
                "(expected 'debug' or 'parallel')"
            )

        logger.info(
            "Stellar RV: Numba parallel path fiber=%s want_preview=%s",
            fiber,
            want_preview,
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
            all_n_good,
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
            min_good_pixels=int(getattr(self, "ccf_min_good_pixels", 9)),
            zb=float(self.ccf_zb),
            **(
                dict(
                    ccf_center_m_s=ccf_center_m_s,
                    ccf_half_window_m_s=ccf_half_window_m_s,
                    ccf_v_step=v_step_m_s,
                )
                if ccf_center_m_s is not None
                else dict(
                    ccf_v_min=v_min_m_s,
                    ccf_v_max=v_max_m_s,
                    ccf_v_step=v_step_m_s,
                )
            ),
        )
        logger.info(
            "Stellar RV: Numba parallel CCF+RV done fiber=%s n_flat_chunks=%d",
            fiber,
            len(all_rvs_arr),
        )
        v_grid_np = np.asarray(v_grid, dtype=np.float64)
        v_step_meta = (
            float(v_grid_np[1] - v_grid_np[0])
            if v_grid_np.size > 1
            else float(v_step_m_s)
        )
        if want_preview:
            logger.info(
                "Stellar RV: building preview from Numba flat CCF fiber=%s",
                fiber,
            )
            preview_detail = build_ccf_preview_dict(
                "numba_flat",
                order_tasks,
                float(v_grid_np[0]),
                float(v_grid_np[-1]),
                v_step_meta,
                order_indices=order_indices,
                n_chunks_per_order=n_chunks_per_order,
                chunk_start=chunk_start,
                all_ccf=all_ccf,
                v_grid=v_grid_np,
                wave_2d=wave_2d,
            )
            logger.info(
                "Stellar RV: preview from Numba flat complete fiber=%s",
                fiber,
            )
        ao_flat = flat_chunk_echelle_orders(order_tasks, chunk_start, len(all_rvs_arr))
        if len(all_rvs_arr) > 0:
            rvccf_chunk_supplement = build_rvccf_chunk_supplement(
                chunk_id=np.arange(len(all_rvs_arr), dtype=np.int32),
                n_good=np.asarray(all_n_good, dtype=np.int32),
                v_min=float(v_grid_np[0]),
                v_max=float(v_grid_np[-1]),
                v_step=v_step_meta,
                ccf_2d=all_ccf,
            )
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
        return all_rvs, all_rv_errs, all_wavelengths, all_orders, preview_detail, rvccf_chunk_supplement

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
        *,
        rvccf_chunk_supplement=None,
    ):
        """
        Write RVCCF_FIB{fiber} (per-chunk RV plus optional CCF columns) and RVCCFAVG_FIB{fiber}.

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
        # TODO: Replace placeholder MBJDW, BERVW, RESOLUTION, and TELL_ABS (per-chunk
        # and RVCCFAVG row) with values from the pipeline once other Stage 2 methods
        # supply barycentric/MJD metadata, spectrograph resolution, and telluric
        # absorption context.
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
        inlier_col = np.asarray(inlier, dtype=np.uint8)
        supp = (
            rvccf_chunk_supplement
            if rvccf_chunk_supplement is not None and len(rvccf_chunk_supplement) > 0
            else None
        )
        tbl = assemble_rvccf_fib_table(
            np.asarray(rvs, dtype=np.float64),
            np.asarray(rv_errs, dtype=np.float64),
            wl,
            order_col,
            inlier_col,
            supplement=supp,
        )
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
        from .previews.stellar_rv_preview import create_stellar_rv_preview

        logger.info("Stellar RV: _create_preview starting (Plotly stellar RV preview)")
        create_stellar_rv_preview(self)
        logger.info("Stellar RV: _create_preview finished")
