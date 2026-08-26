Limitations and Known Issues
============================

Simulation fidelity
-------------------

* **Tellurics** — not fully modeled in the default rate-map path; line-mask utilities can
  apply telluric filters offline (:mod:`hispecdrp.utils.line_mask_utils`).
* **pyechelle photon ray-tracing** — optional; E2E default is native ``prf_convolution``
  (faster, noiseless rate map + Poisson noise applied in ``create_ramp``).
* **Fiberflat SED** — dummy fiberflats use a 1000 K blackbody, not a facility halogen
  WHITELIGHT spectrum; trace **geometry** matches science via the same HDF model.
* **Stage 2 docs** — :doc:`../pipelines/stage2` is marked under development; behavior
  follows :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.

Calibration alignment
---------------------

* **KOA fiberflat / trace masters** — products from an older instrument era may not match
  the current pyechelle HDF model. Default ``derive_stage1_trace_from_sim=True`` avoids
  this. A simulation-matched fiber flat should be registered in KOA for production use.
* **KOA download failures** — empty zip archives have been observed in development; E2E
  falls back to simulated stage-0 cals when fetch fails.

Network dependencies
--------------------

Default E2E runs query:

* **VizieR** — Gaia DR3 and 2MASS (via :mod:`hispecdrp.primitives.utils.gaia_query` and
  :mod:`hispecdrp.simulations.e2e_gaia`)
* **KOA** — optional stage-0 calibrations when ``use_master_calibrations=True``
* **Google Drive** — HDF instrument model auto-download on first use

Runs without network access require pre-cached HDF data, offline calibration YAML paths,
and ``use_gaia_target=False``.
