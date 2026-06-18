Pipeline Stages
===============

Primitives executed in the E2E chain, in order. Pipeline **alias** keys match
``PRIMITIVES`` in each pipeline class and the ``prim_args`` dictionaries used by
:mod:`hispecdrp.simulations.e2e_support`.

Stage 0 (``Stage0Pipeline``)
----------------------------

:func:`~hispecdrp.simulations.simulate_up_the_ramp.run_stage0_simulation` runs on the
science ``simulated_ramp.fits``:

``dq_init`` → ``saturation_check`` → ``bias_sub`` → ``refpix_corr`` → ``nonlin_corr`` →
``jump_detec`` → ``ramp_fit`` → ``dark_sub`` → ``gain_corr`` → ``detflat_corr``

.. csv-table::
   :header: "Step", "Alias", "Primitive", "Role"
   :widths: 5, 18, 32, 40

   "1", "``dq_init``", "``DQInitPrimitive``", "Initialize DQ from calibration"
   "2", "``saturation_check``", "``CheckSaturationPrimitive``", "Flag saturated pixels"
   "3", "``bias_sub``", "``BiasSubtractionPrimitive``", "Subtract bias (E2E default: ``run=False``)"
   "4", "``refpix_corr``", "``RefPixelCorrectionPrimitive``", "Reference-pixel correction"
   "5", "``nonlin_corr``", "``NonlinearCorrectionPrimitive``", "Nonlinearity (E2E default: ``run=False``)"
   "6", "``jump_detec``", "``JumpDetectionPrimitive``", "Jump detection (E2E default: ``run=False``)"
   "7", "``ramp_fit``", "``FitRampPrimitive``", "Fit UTR ramp → rate map (``method=jwst_likely``)"
   "8", "``dark_sub``", "``DarkSubtractionPrimitive``", "Subtract dark"
   "9", "``gain_corr``", "``GainCorrectionPrimitive``", "Gain correction"
   "10", "``detflat_corr``", "``DetectorFlatCorrectionPrimitive``", "Detector flat"

**Output:** L0 rate/slope ``DataModel`` plus ``*_raw-*.fits`` and ``*_rate-*.fits`` products
under ``output_dir``. E2E sets ``save_result=True`` and ``save_preview=True`` on every
Stage 0 primitive.

Dummy fiberflat rates repeat the same ten Stage 0 steps once per fiber before the
fiberflat pipeline runs.

Fiberflat pipeline (``ProcessFiberflatsPipeline``)
--------------------------------------------------

:func:`~hispecdrp.simulations.simulate_up_the_ramp.run_fiberflat_pipeline` processes
per-fiber fiberflat rate FITS files (dummy blackbody sims by default).

.. csv-table::
   :header: "Step", "Alias", "Primitive", "Role"
   :widths: 5, 22, 32, 36

   "1", "``coadd_frames``", "``CoaddFramesPrimitive``", "Coadd frames per fiber (single-frame pass-through)"
   "2", "``trace_pos``", "``TracePositionPrimitive``", "Fit trace positions → ``TracePosModel``"
   "3", "``trace_prof``", "``TraceProfilePrimitive``", "Fit trace profiles → ``TraceProfModel``"
   "4", "``register_tracepos_cal``", "``RegisterCalibrationPrimitive``", "Register trace-pos (E2E: ``run=False``)"
   "5", "``register_traceprof_cal``", "``RegisterCalibrationPrimitive``", "Register trace-prof (E2E: ``run=False``)"
   "6", "``spec_extract``", "``SpectralExtractionPrimitive``", "Extract 1D spectra from fiberflats"
   "7", "``set_wavesol``", "``SetInitialWLSPrimitive``", "Attach WLS to fiberflat product (E2E: ``run=False``)"
   "8", "``make_fiberflat``", "``MakeFiberflatPrimitive``", "Build ``FiberflatModel`` for blaze correction"
   "9", "``register_fiberflat_cal``", "``RegisterCalibrationPrimitive``", "Register fiberflat (E2E: ``run=False``)"

Default trace-fitting ranges (``get_fiberflat_prim_args``):

.. csv-table::
   :header: "Parameter", "BSPEC default", "Meaning"
   :widths: 22, 18, 40

   "``disp_range``", "(300, 4000)", "Dispersion-direction pixel range for trace walk"
   "``xdisp_range``", "(100, 4050)", "Cross-dispersion range for initial peak finding"
   "``deg_pos``", "4", "Trace position polynomial degree"
   "``deg_width``", "2", "Trace width polynomial degree"
   "``extraction_aperture``", "10", "Box extraction aperture (pixels)"

**Outputs** under ``output_dir/cals/``: ``mastercal-tracepos.fits``,
``mastercal-traceprof.fits``, ``mastercal-fiberflat.fits``.

See :doc:`../pipelines/process_fiberflats` for the calibration pipeline class reference.

Stage 1 (``Stage1Pipeline``)
----------------------------

:func:`~hispecdrp.simulations.e2e_support.run_stage1_simulation` on the science Stage 0
rate product:

.. csv-table::
   :header: "Step", "Alias", "Primitive", "Role"
   :widths: 5, 18, 32, 40

   "1", "``spec_extract``", "``SpectralExtractionPrimitive``", "Box extraction using trace pos/prof (all orders/fibers)"
   "2", "``set_wavesol``", "``SetInitialWLSPrimitive``", "E2E: ``run=False``; WLS from HDF sim model via ``prim_args``"

**Output:** ``*_specuncal-specextract.fits`` with ``SPEC_UNCAL_FIB#``, errors, DQ per fiber.

Stage 2 (``Stage2Pipeline``)
----------------------------

:func:`~hispecdrp.simulations.e2e_support.run_stage2_simulation`:

.. csv-table::
   :header: "Step", "Alias", "Primitive", "Role"
   :widths: 5, 18, 32, 40

   "1", "``blaze_corr``", "``BlazeCorrectionPrimitive``", "Divide by ``FiberflatModel`` from fiberflat pipeline"
   "2", "``barycorr``", "``BarycenterCorrectionPrimitive``", "Heliocentric/barycentric velocity correction (Gaia target metadata)"
   "3", "``stellar_rv``", "``StellarRVPrimitive``", "Line-mask CCF RV (``algorithm=ccf_line_mask`` default)"

Stellar RV defaults:

* Mask: ``stellar_rv_mask_path`` or packaged
  ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas``
* Chunked CCF per echelle order (~0.3 µm windows), Gaussian/parabolic refinement
* Products: ``RVCCF_FIB#``, ``RVCCFAVG_FIB#`` on L2 model; CCF and mask-overlay HTML/PNG previews

See :doc:`../algorithms/stellar_rv` and :doc:`../algorithms/line_mask`.

Previews
--------

E2E enables ``save_result=True``, ``save_preview=True`` on all pipeline primitives and
``save_overlay_debug=True`` on ``stellar_rv``. See :doc:`outputs_and_previews`.
