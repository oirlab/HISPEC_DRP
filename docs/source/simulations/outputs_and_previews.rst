Outputs and Previews
====================

File products
-------------

Automatic products (default simulated-cal path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Product", "Typical path", "Created by"
   :widths: 28, 38, 30

   "Rate map", "``<output_dir>/rate_map.fits``", "``SimulateRateMap``"
   "Science ramp", "``<output_dir>/simulated_ramp.fits``", "``simulate_ramp_from_rate_map``"
   "Dummy fiberflat rates", "``<output_dir>/cals/HB.*_rate-FIBERFLAT-FIB*.fits``", "``create_dummy_fiberflat_rate_files``"
   "Trace / fiberflat masters", "``<output_dir>/cals/mastercal-{tracepos,traceprof,fiberflat}.fits``", "``ProcessFiberflatsPipeline``"
   "Stage 0 products", "``<output_dir>/HB.*_{raw,rate}-*.fits``", "``Stage0Pipeline``"
   "Stage 1 products", "``<output_dir>/HB.*_specuncal*.fits``", "``Stage1Pipeline``"
   "Stage 2 products", "``<output_dir>/HB.*_spec-*.fits``", "``Stage2Pipeline``"
   "Simulated stage-0 cals", "``<output_dir>/cals/sim_*.fits``", "``create_simulated_calibration_models`` (cached)"

Filename pattern: ``<prefix>.<date>.<time>_<group>-<step>.fits`` where ``group`` is
``raw``, ``rate``, ``specuncal``, or ``spec``.

HTML preview report
-------------------

When ``make_preview_html=True`` (default),
:func:`~hispecdrp.simulations.e2e_pipeline.generate_e2e_preview_report` writes:

* ``<output_dir>/index.html`` — scrollable report by stage and primitive
* ``<output_dir>/previews/*.png`` — rendered PNG previews

Preview policy
~~~~~~~~~~~~~~

.. csv-table::
   :header: "Stage", "Product pattern", "Preview content"
   :widths: 10, 32, 48

   "0 (pre-rampfit)", "``*_raw-*.fits``", "Mean detector image across all UTR reads"
   "0 (post-rampfit)", "``*_rate-*.fits``", "2D rate map"
   "1", "``*_specuncal-specextract.fits``", "1D extracted spectra, one panel per order per fiber"
   "2", "Existing pipeline previews", "Stellar-RV CCF PNG/HTML, mask-overlay debug pages"

The report scans ``output_dir`` for saved FITS and reuses Stage 2 PNG/HTML products
already written by :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.

Gaia metadata in the report
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a Gaia target was resolved, the HTML header includes source ID, coordinates, RV, and
2MASS normalization summary.

After Stage 2, an **RV performance summary** table lists per fiber: injected RV,
``RVCCFAVG`` measured RV, bias, aggregate uncertainty σ(⟨RV⟩), RMS inlier-chunk scatter
about ⟨RV⟩, and inlier chunk counts.

Opening the report
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   open test_data/index.html   # macOS

Per-primitive previews
----------------------

Independently of the HTML report, each pipeline primitive with ``save_preview=True``
writes its own preview next to the FITS product (PNG or Plotly HTML depending on the
primitive), under the pipeline ``output_dir`` (for E2E fiberflat masters, ``<output_dir>/cals/``).
Examples:

* ``*_mastercal-tracepos_preview.png``
* ``*_spec-stellar_rv_preview.html``
* ``*_spec-stellar_rv_mask_overlay_debug_fib1.html``

Calibration provenance
----------------------

At the end of a run, :func:`~hispecdrp.simulations.e2e_support.log_e2e_calibration_summary`
logs the source of each calibration (``koa``, ``simulated``, ``fiberflat_pipeline``,
``simulated_dummy_rate:...``, etc.) in ``result.calibration_sources``.

See :doc:`calibrations` for resolution order.
