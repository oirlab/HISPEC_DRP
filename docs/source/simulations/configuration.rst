Configuration
=============

``E2EConfig`` dataclass
-----------------------

Defined in :mod:`hispecdrp.simulations.e2e_pipeline`. Defaults below match
:mod:`hispecdrp.simulations.run_e2e_sim` unless noted.

Output and instrument
~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Field", "Default", "Description"
   :widths: 28, 22, 45

   "``output_dir``", "``test_data/`` (via ``repository_root()``)", "All products and KOA cache subdirectory"
   "``spectrograph``", "``BSPEC``", "``BSPEC`` or ``RSPEC``"
   "``fibers``", "``[1, 2, 3]``", "Fibers illuminated in science rate map"
   "``fiberflat_fibers``", "``[1, 2, 3]``", "Fibers for dummy fiberflat rate generation"
   "``instrument_name``", "``None``", "KOA instrument name override"
   "``instrument_era``", "``0.0.1``", "KOA calibration era"

Ramp and detector
~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Field", "Default", "Description"
   :widths: 28, 22, 45

   "``n_reads``", "10", "UTR read count"
   "``readtime``", "1.7", "Seconds between reads"
   "``poisson_noise``", "``True``", "Poisson noise in ramp simulation"
   "``convert_to_uint16``", "``True``", "Quantize ramp to uint16 DN"
   "``max_cores``", "1", "Parallelism for ramp and pipeline primitives"
   "``overwrite_ramp``", "``True``", "Replace existing ramp/rate-map products"

Calibrations
~~~~~~~~~~~~

.. csv-table::
   :header: "Field", "Default", "Description"
   :widths: 28, 22, 45

   "``use_master_calibrations``", "``True``", "Query KOA for stage-0 keys not in YAML"
   "``derive_stage1_trace_from_sim``", "``True``", "Derive trace/fiberflat from simulated fiberflats; do not auto-load KOA/local trace masters"
   "``calibration_config``", "``None``", "Path to YAML or dict (``calibrations``, ``fiberflat_rates``, ``koa``)"
   "``connect_remote``", "``True``", "KOA remote connection"
   "``sync_caldb_first``", "``True``", "Sync KOA metadata before fetch"
   "``download_missing_koa``", "``True``", "Download missing KOA files"
   "``calibration_kwargs``", "``{}``", "Passed to ``create_simulated_calibration_models``"

Stage toggles
~~~~~~~~~~~~~

.. csv-table::
   :header: "Field", "Default", "Description"
   :widths: 28, 22, 45

   "``run_stage1``", "``True``", "Run fiberflat pipeline + Stage 1"
   "``run_stage2``", "``True``", "Run blaze + barycorr + stellar RV"
   "``make_preview_html``", "``True``", "Build ``index.html`` preview report"

Target and template
~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Field", "Default", "Description"
   :widths: 28, 22, 45

   "``radial_velocity_km_s``", "0.0", "Extra RV added to Gaia target (km s⁻¹)"
   "``use_gaia_target``", "``True``", "Resolve star via VizieR / Gaia DR3"
   "``gaia_source_id``", "``3438436918943830144``", "Default Gaia DR3 source"
   "``gaia_max_g_mag``", "8.0", "Max G mag for random star selection (if no source_id)"
   "``gaia_random_seed``", "``None``", "RNG seed for random Gaia selection"
   "``template_filename``", "``lte049.0-2.5-...h5``", "BT-Settl template HDF5"
   "``template_mag_ab``", "10.0", "Fallback normalization AB mag"
   "``template_2mass_band``", "``J``", "2MASS band for dynamic normalization"
   "``template_mag_from_2mass``", "``True``", "Normalize to target 2MASS mag (Vega→AB)"
   "``stellar_rv_mask_path``", "packaged delta mask", "Stage 2 CCF line mask"
   "``flat_scale``", "``2e-14``", "Simulated fiberflat blackbody flux scale (1000 K default; avoid saturation)"
   "``flat_temperature_k``", "1000.0", "Simulated fiberflat blackbody temperature (K)"
   "``fiberflat_coadd_exposures``", "1", "Rate FITS per fiber (1 = no coadd; increase for optional pipeline coadd)"
   "``fiberflat_duplicate_single_reduction``", "``False``", "Duplicate one Stage 0 product vs independent ramps"
   "``fiberflat_trace_prominence_factor``", "``None`` (BSPEC per-fiber defaults)", "Trace peak-finding threshold(s); float or fiber-id dict"
   "``fiberflat_rate_map_method``", "``native``", "``native`` PRF convolver (pyechelle HDF) or ``pyechelle`` raytracing"
   "``fiberflat_pyechelle_exposure_time_s``", "300.0", "pyechelle exposure time for fiberflat rate maps (s)"

``E2EResult``
-------------

Returned by :func:`~hispecdrp.simulations.e2e_pipeline.run_e2e_simulation`:

* ``ramp_model``, ``stage0_model``, ``stage1_model``, ``stage2_model``
* ``calibration_models`` — stage-0 dict used for the run
* ``fiberflat_bundle`` — ``trace_pos``, ``trace_prof``, ``fiberflat`` models
* ``gaia_target`` — :class:`~hispecdrp.simulations.e2e_gaia.E2EGaiaTarget` or ``None``
* ``calibration_sources`` — provenance string per calibration key

Programmatic example
--------------------

.. code-block:: python

   from hispecdrp.simulations.e2e_pipeline import E2EConfig, run_e2e_simulation

   config = E2EConfig(
       output_dir="test_data",
       spectrograph="BSPEC",
       derive_stage1_trace_from_sim=True,
       use_gaia_target=True,
       template_filename="lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
   )
   result = run_e2e_simulation(config)

See :doc:`running` for the CLI entry point.
