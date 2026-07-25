Running the E2E Simulation
==========================

Command-line
------------

From the repository root with the ``hispec`` conda environment:

.. code-block:: bash

   conda activate hispec
   python -m hispecdrp.simulations.run_e2e_sim

Edit literals in :mod:`hispecdrp.simulations.run_e2e_sim` (``output_dir``, reads, cores,
Gaia source, template, etc.) or construct :class:`~hispecdrp.simulations.e2e_pipeline.E2EConfig`
programmatically and call :func:`~hispecdrp.simulations.e2e_pipeline.run_e2e_simulation`.

All logic lives under :mod:`hispecdrp.simulations`; no separate ``benchmarks`` package is
required for the E2E driver.

Pinning calibrations
--------------------

Pass a YAML path via ``calibration_config``:

.. code-block:: python

   config = E2EConfig(
       output_dir="test_data",
       calibration_config="path/to/e2e_calibrations.yaml",
   )

See ``hispecdrp/simulations/e2e_calibrations.example.yaml``.

To skip KOA entirely for stage 0:

.. code-block:: python

   config = E2EConfig(output_dir="test_data", use_master_calibrations=False)

To use KOA stage-0 cals but still derive trace/fiberflat from simulation (default):

.. code-block:: python

   config = E2EConfig(output_dir="test_data", derive_stage1_trace_from_sim=True)

KOA cache maintenance
---------------------

.. code-block:: bash

   python -m hispecdrp.simulations.run_caldb --sync --download

Requires KOA authentication (:doc:`../installation`).

Related tests
-------------

RV validation with BT-Settl + delta mask (not the full E2E driver):

.. code-block:: bash

   HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 \
     pytest tests/test_stellar_rv.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s --bspec-all-orders

Expected runtime
----------------

A full BSPEC run (10 reads, 4096² detector, 3 fibers, Stage 0–2) typically takes on the
order of **10–15 minutes** on a single core, depending on network access for Gaia/VizieR
and KOA downloads.

After completion, open ``<output_dir>/index.html`` for the aggregated preview report
(see :doc:`outputs_and_previews`).

See also
--------

* :doc:`configuration`
* :doc:`overview`
* :doc:`../quickstart/create_simulations` — lower-level rate-map API without full E2E
