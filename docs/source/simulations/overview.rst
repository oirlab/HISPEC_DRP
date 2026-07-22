Overview
========

Purpose
-------

The E2E simulation is a realistic forward model for assessing pipeline performance,
recovery of injected radial velocity, exposure-time estimates, and calibration
consistency. It combines:

* A **stellar template** (BT-Settl HDF5 or user-supplied spectrum)
* **pyechelle / HDF instrument geometry** (trace, PSF, embedded wavelength solution)
* **Detector noise and UTR readout** (bias, dark, gain, flat, Poisson, read noise, optional ``uint16``)
* **Optional KOA stage-0 calibrations** (bias, dark, gain, DQ, …)
* **Simulated fiber flats and trace calibrations** matched to the same HDF model (default)
* **Stage 1 spectral extraction** and **Stage 2 blaze + barycenter correction + stellar-RV CCF**

Entry points
------------

.. csv-table::
   :header: "Module", "Role"
   :widths: 40, 60

   ":mod:`hispecdrp.simulations.e2e_pipeline`", "Orchestrates cal resolution → ramp → Stage 0 → fiberflats → Stage 1 → Stage 2"
   ":mod:`hispecdrp.simulations.run_e2e_sim`", "CLI entry point: ``python -m hispecdrp.simulations.run_e2e_sim``"
   ":mod:`hispecdrp.simulations.simulate_up_the_ramp`", "Rate-map builders, UTR ramp, Stage 0 / fiberflat helpers"
   ":mod:`hispecdrp.simulations.e2e_calibrations`", "KOA / YAML / simulated calibration resolution"
   ":mod:`hispecdrp.simulations.e2e_gaia`", "Gaia target metadata, barycentric correction, 2MASS normalization"
   ":mod:`hispecdrp.simulations.e2e_support`", "Stage 1/2 runners, KOA cache paths, default ``prim_args``"
   ":mod:`hispecdrp.simulations.e2e_previews`", "PNG previews and HTML report (``index.html``)"
   ":mod:`hispecdrp.simulations.run_caldb`", "KOA calibration store sync/download CLI"
   ":mod:`hispecdrp.simulations.simulate_rate_map`", "``SimulateRateMap`` — pyechelle HDF → 2D rate map"
   ":mod:`hispecdrp.simulations.utr`", "``create_ramp`` — UTR ramp from rate map (Poisson, read noise, uint16)"

Pipeline flow
-------------

Execution order in :func:`~hispecdrp.simulations.e2e_pipeline.run_e2e_simulation`:

#. **Resolve calibrations** — explicit YAML paths → KOA (stage 0 only by default) → simulated fallback
#. **Gaia target** (optional) — source metadata, injected RV, 2MASS magnitude for template normalization
#. **Science rate map + UTR ramp** — :func:`~hispecdrp.simulations.simulate_up_the_ramp.create_ramp_and_calibrations`
#. **Stage 0** — raw ramp → L0 rate/slope product
#. **Fiberflat path** (when ``run_stage1=True``) — per-fiber dummy fiberflat rates → :class:`~hispecdrp.pipelines.calibration.process_fiberflats_pipeline.ProcessFiberflatsPipeline` → trace pos/prof + :class:`~hispecdrp.datamodels.fiberflat.FiberflatModel`
#. **Stage 1** — spectral extraction + HDF-derived wavelength solution
#. **Stage 2** — blaze correction + barycenter correction + :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive` (line-mask CCF)
#. **Preview report** (optional) — :func:`~hispecdrp.simulations.e2e_previews.generate_e2e_preview_report`

.. code-block:: text

   Template + Gaia metadata
        │
        ▼
   SimulateRateMap (PRF convolution) ──► rate_map.fits
        │
        ▼
   simulate_ramp_from_rate_map ──► simulated_ramp.fits
        │
        ▼
   Stage0Pipeline ──► *_rate*.fits (science L0)
        │
        ├──► create_dummy_fiberflat_rate_files (×3 fibers)
        │         │
        │         ▼
        │    ProcessFiberflatsPipeline ──► cals/mastercal-{tracepos,traceprof,fiberflat}.fits
        │
        ▼
   Stage1Pipeline ──► *_specuncal*.fits
        │
        ▼
   Stage2Pipeline ──► *_spec-stellar_rv.fits + previews
        (blaze_corr → barycorr → stellar_rv)

Default trace calibration policy
--------------------------------

By default ``E2EConfig.derive_stage1_trace_from_sim=True``. Stage 1 trace products
(``trace_pos``, ``trace_prof``, ``fiberflat``) are **not** loaded from KOA, local
``cals/`` caches, or bundled test-data masters. They are derived from **dummy fiberflat
rate maps** built with the same :class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`
/ HDF path as the science exposure.

Legacy KOA ``mastercal-tracepos`` products can disagree with the current pyechelle HDF
model (different cross-dispersion order spacing). Using them caused most echelle orders
to extract near-zero flux. See :doc:`calibrations` and :doc:`limitations`.

See also
--------

* :doc:`configuration` — full ``E2EConfig`` reference
* :doc:`running` — how to launch a run
* Example calibration YAML: ``hispecdrp/simulations/e2e_calibrations.example.yaml``
