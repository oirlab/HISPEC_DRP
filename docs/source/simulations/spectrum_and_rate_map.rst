Spectrum, Rate Map, and Ramp
============================

This page covers steps 1–2 of the E2E chain: injecting a stellar spectrum onto the
detector and building the UTR ramp that Stage 0 consumes.

Science rate map
----------------

:func:`~hispecdrp.simulations.simulate_up_the_ramp.create_ramp_and_calibrations` builds
the 2D rate map (e⁻ s⁻¹ pixel⁻¹) when no ``rate_map`` argument is supplied.

SimulateRateMap
~~~~~~~~~~~~~~~

:class:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap`:

* Places flux on fibers ``1``–``3`` (configurable via ``E2EConfig.fibers``)
* Applies optional Doppler shift via ``radial_velocity_km_s`` (or Gaia-injected RV)
* Deposits PSF stamps with ``create_native_rate_map(..., method="prf_convolution")``
* Applies detector orientation fixes (horizontal flip, rotation, 64-pixel channel pad)

Instrument HDF models
~~~~~~~~~~~~~~~~~~~~~

Trace, PSF, and wavelength sampling come from pyechelle HDF files:

.. csv-table::
   :header: "Spectrograph", "Default HDF file"
   :widths: 20, 50

   "``BSPEC``", "``HISPEC_yJ_20231006A-1_all_traces.hdf``"
   "``RSPEC``", "``HISPEC_HK_20231006B-1_all_traces.hdf``"

Resolved under ``HISPEC_TEST_DATA_DIR/HISPECDRP_TEST_DATA/`` or the Astropy download
cache; missing files are fetched via ``download_gdrive_file``.

Template spectrum
~~~~~~~~~~~~~~~~~

When ``template_filename`` is set (default in ``run_e2e_sim``:
``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5``), the template is loaded from
``hispecdrp/data/templates/``. Without a template, a flat test spectrum (0.9–2.5 µm) is used.

Magnitude normalization
~~~~~~~~~~~~~~~~~~~~~~~

Template flux can be normalized before rate-map generation:

* ``template_mag_from_2mass=True`` (default when using a Gaia target): normalize to the
  target's 2MASS magnitude (``template_2mass_band``, default ``J``), converted Vega → AB
  via VizieR (:mod:`hispecdrp.simulations.e2e_gaia`)
* ``template_mag_ab``: fallback fixed AB magnitude when no 2MASS match is found (default 10.0)

Gaia target
~~~~~~~~~~~

When ``use_gaia_target=True`` (default), :mod:`hispecdrp.simulations.e2e_gaia` resolves
the star via **VizieR** (not Gaia TAP):

* Coordinates, parallax, proper motion, Gaia DR3 radial velocity
* Barycentric correction and ``target_meta`` for FITS headers
* Optional 2MASS photometry for magnitude normalization

Default source in ``run_e2e_sim``: Gaia DR3 ``3438436918943830144`` (K-type giant matching
the default BT-Settl template).

Outputs
~~~~~~~

* ``rate_map.fits`` in ``output_dir`` when the rate map is built internally
* In-memory rate array passed to calibration completion and ramp simulation

UTR ramp
--------

:func:`~hispecdrp.simulations.simulate_up_the_ramp.simulate_ramp_from_rate_map` converts
the rate map into a :class:`~hispecdrp.datamodels.ramp.RampModel` via
:func:`~hispecdrp.simulations.utr.create_ramp`:

* Adds bias, dark, gain, detector flat, read noise
* Optional Poisson noise (``poisson_noise``, default ``True``)
* Nonlinearity polynomial ``[-1e-6, 1, 0]`` in the electron domain inside ``create_ramp``
* Optional ``uint16`` quantization (``convert_to_uint16``, default ``True``)

Default ramp parameters (``E2EConfig`` / ``run_e2e_sim``):

.. csv-table::
   :header: "Parameter", "Default", "Meaning"
   :widths: 25, 15, 40

   "``n_reads``", "10", "Number of UTR samples"
   "``readtime``", "1.7 s", "Time between reads"
   "``poisson_noise``", "``True``", "Poisson noise in ``create_ramp``"
   "``convert_to_uint16``", "``True``", "Quantize ramp to DN"
   "``max_cores``", "1", "Parallelism for ramp / primitives"
   "``shape``", "(4096, 4096)", "Detector size for simulated stage-0 cals"

Product: ``simulated_ramp.fits`` (``DEFAULT_SIMULATED_RAMP_FILENAME``).

Detector calibrations for Stage 0
---------------------------------

Before Stage 0, :mod:`hispecdrp.simulations.e2e_calibrations` resolves stage-0 products.
Any key still missing after KOA / YAML resolution is filled by
:func:`~hispecdrp.simulations.simulate_up_the_ramp.create_simulated_calibration_models`
once the rate-map shape is known.

.. csv-table::
   :header: "Key", "Model", "Stage 0 use"
   :widths: 12, 22, 40

   "``dq``", "``DQModel``", "``dq_init``"
   "``saturation``", "``SaturationModel``", "``saturation_check``"
   "``gain``", "``GainModel``", "``ramp_fit``, ``gain_corr``"
   "``rn``", "``ReadNoiseModel``", "``ramp_fit``"
   "``bias``", "``BiasModel``", "``bias_sub`` (default off in sim)"
   "``dark``", "``DarkModel``", "``dark_sub``"
   "``detflat``", "``DetectorFlatModel``", "``detflat_corr``"
   "``nonlin``", "``NonlinearCorrectionModel``", "``nonlin_corr`` (default off; identity if missing)"

Simulated calibration kwargs (``calibration_kwargs`` on ``E2EConfig``): ``gain``,
``bias_dn``, ``dark_current``, ``read_noise``, ``saturation_dn``, ``nonlin_coeffs``,
``dq_bad_rows``, ``detflat_scale``, ``seed``.

Code reference
--------------

Science rate-map deposition (same HDF path as fiber flats, different SED):

.. code-block:: python

   # hispecdrp/simulations/simulate_up_the_ramp.py (create_ramp_and_calibrations)
   srm.set_fiber_spectrum(fiber_number=f, spectrum_wavelengths=wl, spectrum_flux=flux, ...)
   rate_map = srm.create_native_rate_map(spectrograph=spectrograph, method="prf_convolution")

See :doc:`calibrations` for how calibrations are resolved and :doc:`pipeline_stages` for Stage 0.
