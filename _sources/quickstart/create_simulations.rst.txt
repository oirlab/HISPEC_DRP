============================
Create Simulated HISPEC Data
============================

This example shows how to create simulated HISPEC data using
:mod:`hispecdrp.simulations.simulate_rate_map` and :mod:`hispecdrp.simulations.simulate_up_the_ramp`.
For the full end-to-end pipeline (ramp → Stage 0–2 → RV), see
:doc:`../simulations/basic_usage`.

.. note::

   Older examples referenced a ``benchmarks/`` package at the repository root; the same
   APIs now live under :mod:`hispecdrp.simulations`.

Simulated rate maps can be created using a wrapper for the ``pyechelle`` package, or using a native DRP simulator. 
``pyechelle`` uses raytracing of individual photons and naturally includes photon noise, whereas the native simulator creates noiseless rate maps.
``pyechelle`` takes longer to run for brighter targets (more photons to trace), and the native simulator is expected to be faster for all but the faintest sources.

All simulations require downloading model PSF and trace data (~2 GB). This will be done automatically the first time you run either simulator. By default, the data is stored in the same path as the testing data.

To create simulated ramp data suitable for input to the stage0 pipeline:

* Set each fiber with a desired spectrum. Note that for continuous sources, the spectral sampling must be greater than HISPEC's spectral resolution.
* Create rate maps using either the ``pyechelle`` wrapper or the native DRP simulator.
* Create ramp data from the rate maps using the ``create_ramp`` function. To create ramp data, you will also need input flat, dark, gain maps etc.

Below is an example using the native DRP simulator to create a rate map for a flat spectrum source on fiber 1, and an emission line source on fiber 2.

.. code-block:: python

    import numpy as np
    import astropy.units as u
    import hispecdrp.simulations as simulations

    srm = simulations.SimulateRateMap()

    fiber_1_wavelengths = np.arange(0.9,2.5,2.5e-6)* u.micron
    fiber_1_flux = np.ones_like(fiber_1_wavelengths.value)*1e-12 * u.erg/u.s/u.cm**2/u.micron
    srm.set_fiber_spectrum(fiber_number=1,
                       spectrum_wavelengths=fiber_1_wavelengths,
                       spectrum_flux=fiber_1_flux,
                      continuous=True)

    fiber_2_wavelengths = np.arange(0.9,2.5,2.5e-4)* u.micron
    fiber_2_flux = np.ones_like(fiber_2_wavelengths.value)*1e-9 * u.erg/u.s/u.cm**2/u.micron
    srm.set_fiber_spectrum(fiber_number=2,
                        spectrum_wavelengths=fiber_2_wavelengths,
                        spectrum_flux=fiber_2_flux,
                        continuous=False)

    native_rate_map = srm.create_native_rate_map(spectrograph='BSPEC', method='prf_convolution')

If you then want to create ramp data from the rate map:

.. code-block:: python

    max_cores = 1
    max_slope = 50  # e-/s
    readtime = 1.7  # seconds
    n_reads = 10
    ny, nx = 4096, 4096 # size of the detector
    nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity

    gain = 1.0  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)

    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)

    dark_current = 10  # e-/s
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)

    ramp_data = simulations.create_ramp(native_rate_map,  # in e-/s
                        readtime=readtime, n_reads=n_reads,
                        nonlin_coeffs=nonlin_coeffs,
                        gain=gain_map,  # in e-/s
                        flat=detflat_map,
                        dark=dark_map,  # in e-/s
                        bias=bias_map,  # in e-/s
                        kTC_noise=50,  # in e-/s
                        poisson_noise=False, read_noise=0,  # in e-/s
                        convert_to_uint16=False, clip_ramps=True,
                        max_cores=max_cores)





