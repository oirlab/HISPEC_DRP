Barnard's Star PRV Example
==========================

End-to-end simulation of observation and pipeline reduction of a Barnard's star
analog as a precision RV standard.

This page shows two examples:

1. A single up-the-ramp exposure of a star with a theoretical spectrum similar to
   Barnard's star, assuming no planets.
2. A short segment of precision radial-velocity observations of the same star,
   with an orbiting super-Earth that causes the radial velocity to vary.

For each exposure, the workflow simulates a BSPEC rate map from a Phoenix
template, builds a ``RampModel``, then runs ``Stage0Pipeline`` →
``Stage1Pipeline`` → ``Stage2Pipeline``.

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/barnards_star_prv_example.ipynb"
        data-notebook-download="barnards_star_prv_example.ipynb"
        download="barnards_star_prv_example.ipynb">Download as notebook</a>
   </p>

**Environment variables** (set these in your shell / notebook kernel before running):

.. list-table::
   :header-rows: 1
   :widths: 28, 28, 44

   * - Variable
     - Required when
     - Purpose
   * - ``KOA_CALIBRATION_CACHE``
     - Always
     - Local KOA calibration cache directory (and SQLite DB)
   * - ``KECK_OBSERVER_EMAIL``
     - Optional remote sync
     - Keck Observer account email
   * - ``KECK_OBSERVER_PASSWORD``
     - Optional remote sync
     - Keck Observer account password

If ``KOA_CALIBRATION_CACHE`` is unset, the setup cell sets it under ``OUT``.

Imports and output directory
----------------------------

.. code-block:: python

   import os
   import hispecdrp

   print(f"hispecdrp {hispecdrp.__version__} @ {hispecdrp.__file__}")

   # OUTPUT DIRECTORY - update this to your desired output directory
   OUT = "/path/to/barnards_star_prv_example"
   os.makedirs(OUT, exist_ok=True)

   os.environ.setdefault(
       "KOA_CALIBRATION_CACHE",
       os.path.join(OUT, "cals", "KOA_CALIBRATION_CACHE"),
   )
   print("output:", OUT)
   print("KOA_CALIBRATION_CACHE:", os.environ["KOA_CALIBRATION_CACHE"])

1. Sync / verify KOA caldb (BSPEC)
----------------------------------

Pull BSPEC calibration metadata into the local caldb and download any missing
files.

.. code-block:: python

   from hispecdrp.calibrations import verify_koa_calibrations

   SPECTROGRAPH = "BSPEC"

   try:
       summary = verify_koa_calibrations(
           SPECTROGRAPH,
           ignore_missing_cal_types=["lfc_wavesol"],
       )
   except RuntimeError as exc:
       msg = str(exc)
       if "Failed to query metadata" not in msg and "404" not in msg:
           raise
       print(
           f"Remote KOA sync failed ({exc.__class__.__name__}); "
           "using local cache only."
       )
       summary = verify_koa_calibrations(
           SPECTROGRAPH,
           connect_remote=False,
           sync_remote=False,
           ignore_missing_cal_types=["lfc_wavesol"],
       )

   print(
       f"{summary['spectrograph']} caldb: {summary['n_records']} record(s); "
       f"downloaded {summary['n_downloaded']}/{summary['n_missing']} "
       f"missing file(s)"
   )
   print("types:", summary["cal_types"])

2. Single exposure of star with no planet
-----------------------------------------

Stellar properties are approximately similar to Barnard's star (same Teff, logg,
and metallicity).

2.1 Simulated rate map from a 1D spectrum to ``RampModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a Phoenix template, densify it, Doppler-shift the densified 1D spectrum by
``radial_velocity_km_s`` (``inject_rv_via='spectrum'``), paint with ZEMAX, then
build a raw UTR ``RampModel``. Stage-0 KOA masters plus simulation Stage-1
products (packaged line-ZP WAVE — unscaled, same bake for every RV — traces,
fiberflat) are registered into the local caldb.

1. Fiber 1 is on the star assuming ideal coupling.
2. Fiber 2 is empty (later updated to include an etalon spectrum).
3. Fiber 3 is background sky flux.

What this simulation does **not** include:

- telluric injection/correction
- wavelength solution determination (here the wave solution is derived from a
  ZEMAX instrument model)
- atmospheric variability of any kind

.. code-block:: python

   from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum

   FIBERS = [1, 3]
   READ_TIME = 1.7  # seconds

   # Injected stellar RV (km/s). Default path densifies the 1D template,
   # Doppler-shifts it, paints with ZEMAX, and attaches the fixed packaged
   # line-ZP WAVE (no WAVE×(1+z)).
   STELLAR_RV = -110.1

   exposure_time = 60  # seconds
   n_reads = int(exposure_time / READ_TIME)

   # Create a UTR map from a controlled seed and template spectrum.
   # Normalize to a specific magnitude (AB mag) and 2MASS filter.
   ramp, cals, rate_map = create_ramp_from_spectrum(
       OUT,
       spectrograph=SPECTROGRAPH,
       fibers=FIBERS,
       template_filename="lte03200-5.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRe.h5",
       mag_ab=6.15,
       filter_band="J",  # BSPEC (yJ)
       n_reads=n_reads,
       seed=0,
       radial_velocity_km_s=STELLAR_RV,
       barycentric_velocity_km_s=0.0,
       inject_rv_via="spectrum",
       sky_filename="mk_skybg_zm_1.0_1.0_ph.dat",
       sky_fiber=3,
       apply_e2e_throughput=FIBERS,
   )

   print(f"fibers: {FIBERS}")
   print(f"ramp type: {ramp.meta.model_type}")
   print(f"ramp filename: {ramp.meta.filename}")
   print(f"ramp data shape: {ramp.data.shape}")
   print(f"Stage-0 cals: {sorted(cals)}")
   print(f"rate_map shape: {rate_map.shape}")
   print(
       f"wavesol note: "
       f"{getattr(getattr(cals.get('wavesol'), 'meta', None), 'note', None)}"
   )

2.2 Stage 0 to 2 via pipeline classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce the simulated ramp with the same pipeline classes used on real data.

.. code-block:: python

   from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline

   l0 = Stage0Pipeline.call(ramp, output_dir=OUT)

   do_fibers = [str(f) for f in FIBERS]
   # Fixed packaged line-ZP WAVE from create_ramp_from_spectrum
   # (unscaled for spectrum paint)
   stage1_primitives = {
       "spec_extract": {"do_fibers": do_fibers},
       "set_wavesol": {"wavesol": cals["wavesol"]},
   }
   l1 = Stage1Pipeline.call(l0, output_dir=OUT, primitives=stage1_primitives)

   injected_m_s = float(STELLAR_RV) * 1000.0
   l2 = Stage2Pipeline.call(
       l1,
       output_dir=OUT,
       primitives={
           "blaze_corr": {"do_fibers": do_fibers},
           "sky_sub": {"run": True},
           "stellar_rv": {
               "ccf_input_velocity": injected_m_s,
               "do_fibers": [1],
               "save_preview": True,
               "preview_write_png": False,
           },
       },
   )

   print(f"L0 type: {l0.meta.model_type}")
   print(f"L0 filename: {l0.meta.filename}")
   print(f"L1 type: {l1.meta.model_type}")
   print(f"L1 filename: {l1.meta.filename}")
   print(f"L2 type: {l2.meta.model_type}")
   print(f"L2 filename: {l2.meta.filename}")

3. Segment of PRV observations
------------------------------

Simulate multiple exposures over several nights for a super-Earth planet orbiting
the same Barnard's star analog.

3.1 Observation times and injected RVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fiber 1 is on the star assuming ideal coupling.
2. Fiber 2 is empty (later updated to include an etalon spectrum).
3. Fiber 3 is background sky flux.

What this simulation does **not** include:

- telluric injection/correction
- wavelength solution determination (here the wave solution is derived from a
  ZEMAX instrument model)
- atmospheric variability of any kind
- stellar variability of any kind

.. code-block:: python

   import numpy as np

   FIBERS = [1, 3]
   READ_TIME = 1.7  # seconds

   # Injected stellar RV (km/s).
   STELLAR_RV = -110.1

   # observation parameters
   starting_jd = 2461293.91667  # Sep 10 2026
   nights = 3
   exposures_per_night = 3
   exposure_time = 60  # seconds
   n_reads = int(exposure_time / READ_TIME)

   nightly_start_jd_arr = starting_jd + np.arange(0, nights)
   time_offsets = np.linspace(
       0,
       (exposure_time + READ_TIME) / 86400 * (exposures_per_night - 1),
       exposures_per_night,
   )
   obs_jd_arr = np.add.outer(nightly_start_jd_arr, time_offsets).flatten()

   # planet parameters
   phase0_jd = starting_jd
   semiamplitude_K = 15  # m/s
   period = 1.75  # days

   phase_arr = (obs_jd_arr - phase0_jd) / period
   rv_arr = semiamplitude_K * np.sin(2 * np.pi * phase_arr) / 1000 + STELLAR_RV

3.2 Simulate and reduce each exposure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a helper that builds one UTR, reduces Stage 0–2, and returns the L2
product.

.. code-block:: python

   from hispecdrp.simulations.simulate_up_the_ramp import create_ramp_from_spectrum
   from hispecdrp.pipelines import Stage0Pipeline, Stage1Pipeline, Stage2Pipeline
   import gc


   def run_one_injection(
       label: str, injected_rv_km_s: float, mjd_start: float, seed: int
   ) -> dict:
       """Build one UTR, reduce Stage0–2, return L2 data product."""
       run_dir = OUT
       os.makedirs(run_dir, exist_ok=True)
       injected_m_s = float(injected_rv_km_s) * 1000.0

       print(f"\n=== {label}: inject {injected_rv_km_s:.6f} km/s, seed={seed} ===")
       ramp, cals, rate_map = create_ramp_from_spectrum(
           run_dir,
           spectrograph=SPECTROGRAPH,
           fibers=FIBERS,
           template_filename=(
               "lte03200-5.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRe.h5"
           ),
           mag_ab=6.15,
           filter_band="J",  # BSPEC (yJ)
           n_reads=n_reads,
           seed=seed,
           radial_velocity_km_s=injected_rv_km_s,
           barycentric_velocity_km_s=0.0,
           inject_rv_via="spectrum",
           sky_filename="mk_skybg_zm_1.0_1.0_ph.dat",
           sky_fiber=3,
           apply_e2e_throughput=FIBERS,
           mjd_start=mjd_start,
           register_stage1_sims=False,
           register_wavesol=False,
           save_ramp=False,
       )
       wmeta = getattr(cals.get("wavesol"), "meta", None)
       print(
           f"ramp={ramp.meta.filename}  shape={ramp.data.shape}  "
           f"wavesol note={getattr(wmeta, 'note', None)}  "
           f"injected_rv_via={getattr(wmeta, 'injected_rv_via', None)}"
       )

       do_fibers = [str(f) for f in FIBERS]
       l0 = Stage0Pipeline.call(ramp, output_dir=run_dir, save_result=False)

       del ramp
       del rate_map

       l1 = Stage1Pipeline.call(
           l0,
           output_dir=run_dir,
           save_result=False,
           primitives={
               "spec_extract": {"do_fibers": do_fibers},
               "set_wavesol": {"wavesol": cals["wavesol"]},
           },
       )

       del l0

       l2 = Stage2Pipeline.call(
           l1,
           output_dir=run_dir,
           save_result=True,
           primitives={
               "blaze_corr": {"do_fibers": do_fibers},
               # Multi-exposure loop reuses a shared cal set; skip sky_sub here.
               "sky_sub": {"run": False},
               "stellar_rv": {
                   "ccf_input_velocity": injected_m_s,
                   "do_fibers": [1],
                   "save_preview": False,
                   "preview_write_png": False,
               },
           },
       )

       del l1
       gc.collect()

       return {
           "label": label,
           "mjd": mjd_start,
           "run_dir": run_dir,
           "seed": seed,
           "inject_km_s": float(injected_rv_km_s),
           "inject_m_s": injected_m_s,
           "wavesol": cals["wavesol"],
           "l2": l2,
       }

3.3 Run the PRV segment
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   l2_results = {}

   for i in range(nights * exposures_per_night):
       l2_results[str(obs_jd_arr[i])] = run_one_injection(
           label=str(obs_jd_arr[i]),
           injected_rv_km_s=rv_arr[i],
           mjd_start=obs_jd_arr[i],
           seed=0,
       )

3.4 Injected vs recovered RV vs MJD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three-panel PRV diagnostic via `plot_injected_vs_recovered_rv`:

1. Raw injected and recovered RVs.
2. Residual `(rec - inj) - mean(rec - inj)` on a fixed ±1 m/s scale.
3. RV signal: injected about the systemic RV, and recovered minus a zero-point
   (mean recovered).

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_injected_vs_recovered_rv

   summary = plot_injected_vs_recovered_rv(
       l2_results,
       systemic_rv_km_s=STELLAR_RV,
       fiber=1,
   )
