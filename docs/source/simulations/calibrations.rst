Calibrations
============

The E2E pipeline resolves calibrations in a fixed order before and during the run.
Understanding this order is important when mixing KOA archive products with pyechelle
simulation outputs.

Resolution order
----------------

:func:`~hispecdrp.simulations.e2e_calibrations.resolve_e2e_calibrations` applies:

#. **Explicit paths** in ``calibration_config`` (YAML file or dict, key ``calibrations``)
#. **KOA calibration store** when ``use_master_calibrations=True`` — selectors in
   :mod:`hispecdrp.simulations.e2e_calibrations`
#. **Simulated fallback** for missing stage-0 keys after the rate map shape is known
   (:func:`~hispecdrp.simulations.e2e_calibrations.complete_stage0_calibrations`)

Stage 1 trace products
----------------------

Keys ``trace_pos``, ``trace_prof``, ``fiberflat``, and optionally ``wavesol`` are stage-1
calibrations. Their resolution depends on ``derive_stage1_trace_from_sim``:

.. csv-table::
   :header: "``derive_stage1_trace_from_sim``", "KOA fetch", "Local / test-data masters", "E2E behavior"
   :widths: 28, 12, 22, 38

   "``True`` (default)", "Skipped", "Skipped", "Dummy fiberflats + ``ProcessFiberflatsPipeline``"
   "``False``", "Attempted", "Attempted if still missing", "Use masters when complete; else fiberflat pipeline"

When all three trace keys are supplied explicitly in ``calibration_config``, the fiberflat
pipeline is skipped and Stage 1 uses those masters directly.

**Important:** KOA ``mastercal-fiberflat`` / ``mastercal-tracepos`` products from an older
instrument era may **not** match the pyechelle HDF model used by
:class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`. The default
``derive_stage1_trace_from_sim=True`` avoids this mismatch.

YAML configuration
------------------

Example: ``docs/source/simulations/e2e_calibrations.example.yaml``

.. code-block:: yaml

   koa:
     instrument_era: "0.0.1"
     connect_remote: true
     sync_first: true
   calibrations:
     # dq: /path/to/dq.fits
     # trace_pos: /path/to/mastercal-tracepos.fits
   fiberflat_rates:
     1: /path/to/HB....FIB1.fits

* ``calibrations`` — explicit FITS paths per key
* ``fiberflat_rates`` — per-fiber rate FITS; used when trace masters are incomplete but
  rate files are supplied (fiberflat pipeline still runs)
* ``koa`` — overrides for ``instrument_era``, ``mjd_start``, ``connect_remote``, ``sync_first``

KOA stage-0 products
--------------------

When ``use_master_calibrations=True``, these keys are fetched via KOA selectors (if not
already set in YAML):

.. csv-table::
   :header: "Key", "Selector", "Used in"
   :widths: 12, 28, 30

   "``dq``", "``DQSelector``", "Stage 0 ``dq_init``"
   "``saturation``", "``SaturationSelector``", "Stage 0 ``saturation_check``"
   "``gain``", "``GainSelector``", "Stage 0 ``ramp_fit``, ``gain_corr``"
   "``detflat``", "``DetectorFlatSelector``", "Stage 0 ``detflat_corr``"
   "``bias``", "``BiasSelector``", "Stage 0 ``bias_sub``"
   "``dark``", "``DarkSelector``", "Stage 0 ``dark_sub``"
   "``rn``", "``ReadNoiseSelector``", "Stage 0 ``ramp_fit``"
   "``nonlin``", "``NonlinearCorrectionSelector``", "Stage 0 ``nonlin_corr`` (if enabled)"

When ``derive_stage1_trace_from_sim=False``, KOA may also supply ``trace_pos``,
``trace_prof``, ``fiberflat``, and ``wavesol``.

KOA does **not** provide pyechelle HDF instrument models or BT-Settl templates.

Populating the KOA cache
------------------------

1. Authenticate — Keck Observer account + KOA middleware (:doc:`../installation`)
2. Set ``KOA_CALIBRATION_CACHE`` (E2E also creates a cache under ``output_dir``)
3. Sync and download:

   .. code-block:: bash

      python -m hispecdrp.calibrations.run_caldb --sync --download

4. Match ``instrument_era``, ``spectrograph``, and ``mjd_start`` to your simulated epoch

Fiberflat rate generation (simulation-matched)
----------------------------------------------

When the fiberflat pipeline runs, per-fiber rate FITS files are created by
:func:`~hispecdrp.simulations.simulate_up_the_ramp.create_dummy_fiberflat_rate_files`:

* **1000 K blackbody** SED (0.8–3.0 µm), one illuminated fiber per exposure
* Same ``SimulateRateMap.create_native_rate_map(..., method="prf_convolution")`` call as
  science data — **same HDF trace geometry**, different spectrum
* Each file is run through Stage 0 again (same detector cals as science)
* Saved under ``output_dir/cals/`` as ``HB.*_rate-FIBERFLAT-FIB*.fits``

Critical code location (keep synchronized with science simulation):

.. code-block:: python

   # hispecdrp/simulations/simulate_up_the_ramp.py — create_dummy_fiberflat_rate_files
   srm = SimulateRateMap()
   srm.set_fiber_spectrum(fiber_number=f, spectrum_wavelengths=wl_flat, spectrum_flux=flux_flat, ...)
   rate_map = srm.create_native_rate_map(spectrograph=spectrograph, method="prf_convolution")

Orchestration in :func:`~hispecdrp.simulations.e2e_pipeline.run_e2e_simulation` (lines
257–272): ``create_dummy_fiberflat_rate_files`` → ``run_fiberflat_pipeline``.

Coadd note
----------

:class:`~hispecdrp.primitives.coadd.coadd_frames_primitive.CoaddFramesPrimitive` passes
through a **single input frame unchanged**. Previously, coadding one frame NaN'd
DQ-flagged pixels and broke trace finding; all fibers also overwrote one shared
``*_rate-coadded.fits`` path.

Wavelength solution
-------------------

Default E2E uses :func:`~hispecdrp.simulations.simulate_up_the_ramp.create_wavelength_solution_model_from_sim`,
which copies the HDF trace solution into an in-memory
:class:`~hispecdrp.datamodels.lfcwavesol.LFCWavesolModel`. Stage 1 ``set_wavesol`` is
configured with ``run=False``; wavelengths are attached via ``prim_args`` instead of a
separate FITS product.

If ``wavesol`` is resolved from KOA or YAML, that model is used instead.

See also
--------

* :doc:`../calibrations/calibration_database` — calibration store internals
* :doc:`basic_usage` — short E2E worked example
* :doc:`limitations` — known simulation limits
