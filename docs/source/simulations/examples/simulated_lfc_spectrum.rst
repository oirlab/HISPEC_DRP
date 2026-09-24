Simulated LFC Spectrum
======================

Paint a physical laser frequency comb (LFC) on BSPEC fiber 2, fit each spot with
a 2D elliptical Gaussian, and compare WAVE at the fitted centroids to the known
injected wavelengths. A later section extracts the same comb through Stage 1–2,
attaches a mock HDF wavelength solution, fits 1D LFC lines, and plots FWHM and
resolving power.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

Modes follow ``nu_n = nu_0 + n * delta_nu`` via
:func:`~hispecdrp.simulations.e2e_pipeline.build_lfc_line_list` (same ladder as
``FitLFCLinesPrimitive`` / ``LFCNU0`` and ``LFCDELNU``).

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/simulated_lfc_example.ipynb"
        data-notebook-download="simulated_lfc_example.ipynb"
        download="simulated_lfc_example.ipynb">Download as notebook</a>
   </p>

Sections 1–4 (2D paint / elliptical fit / WAVE residuals) do **not** require
KOA credentials. Sections 5–8 extract the comb and run
``FitLFCLinesPrimitive``; those steps need a local KOA cache for traces and
fiberflats.

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from hispecdrp.simulations.e2e_pipeline import resolve_lfc_fiber_object

   OUT = Path("/path/to/hispec_simulated_lfc_example")
   OUT.mkdir(parents=True, exist_ok=True)

   SPECTROGRAPH, FIBER = "BSPEC", 2
   FIBER_OBJECT = resolve_lfc_fiber_object(SPECTROGRAPH)
   EDGE = 0.05
   MODE_STRIDE = 10  # 1 = full comb; >1 keeps every Nth mode (faster)

   A_PIX, B_PIX, PA_DEG = 0.80, 0.55, 20.0
   PA_RAD = np.deg2rad(PA_DEG)
   PEAK_VALUE = 1.0e6
   POLY_DEG = 5

   print(f"OUT = {OUT}")
   print(f"fiber={FIBER}  {FIBER_OBJECT}  mode_stride={MODE_STRIDE}")

1. Paint the LFC comb
---------------------

Build the physical mode list and paint elliptical Gaussians at HDF deposit centers.

.. code-block:: python

   import numpy as np
   from astropy.io import fits

   from hispecdrp.simulations.e2e_pipeline import (
       build_lfc_line_list,
       paint_elliptical_gaussian_comb,
   )

   pix = np.arange(4096, dtype=np.float64)
   line_list, lfc_meta = build_lfc_line_list(
       SPECTROGRAPH,
       fiber=FIBER,
       fiber_object=FIBER_OBJECT,
       edge_margin_frac=EDGE,
       mode_stride=MODE_STRIDE,
   )
   n_per = [len(v) for v in line_list.values()]
   n_tot = int(sum(n_per))
   print(f"orders={len(line_list)}  modes={n_tot}")

   rate = paint_elliptical_gaussian_comb(
       line_list,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       a_pix=A_PIX,
       b_pix=B_PIX,
       pa_rad=PA_RAD,
       peak_value=PEAK_VALUE,
   )
   rate_path = OUT / "rate_map.fits"
   hdr = fits.Header()
   hdr["SPECTRO"] = SPECTROGRAPH
   hdr["FIBER"] = FIBER
   hdr["OBJFIB2"] = FIBER_OBJECT
   hdr["LFCNU0"] = (lfc_meta["nu0"], "LFC pump frequency [Hz]")
   hdr["LFCDELNU"] = (lfc_meta["delta_nu"], "LFC mode spacing [Hz]")
   fits.PrimaryHDU(rate.astype(np.float32), header=hdr).writeto(
       rate_path, overwrite=True
   )
   print(f"rate_map peak={rate.max():.3e} -> {rate_path}")

2. Fit elliptical Gaussians
---------------------------

Fit each spot; residuals are fit minus truth for centroid and ellipse shape.

.. code-block:: python

   import numpy as np
   from astropy.table import Table

   from hispecdrp.simulations.e2e_pipeline import measure_lfc_spots_elliptical

   rows = measure_lfc_spots_elliptical(
       rate,
       line_list,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       a_pix=A_PIX,
       b_pix=B_PIX,
       pa_rad=PA_RAD,
   )
   tbl = Table(rows)
   fits_path = OUT / "lfc_elliptical_fits.ecsv"
   tbl.write(fits_path, format="ascii.ecsv", overwrite=True)
   ok = np.asarray(tbl["success"], dtype=bool) & np.isfinite(tbl["row_fit"])
   print(f"measured={len(tbl)}  success={int(ok.sum())} -> {fits_path}")

3. WAVE residuals from fitted centroids
---------------------------------------

For each successful fit, compare ``WAVE(row_fit)`` to the injected wavelength.

.. code-block:: python

   import numpy as np

   from hispecdrp.primitives.utils.spectral_utils import delta_wave_to_delta_vel
   from hispecdrp.simulations.simulate_rate_map import SimulateRateMap

   wave = SimulateRateMap().get_wavelength_solution(spectrograph=SPECTROGRAPH)[FIBER]
   per_order = {}
   for od in sorted({int(o) for o in tbl["order"]}):
       m = ok & (np.asarray(tbl["order"], dtype=int) == int(od))
       if int(m.sum()) < 5:
           continue
       lam_inj = np.asarray(tbl["lam_true_um"][m], dtype=float)
       row_fit = np.asarray(tbl["row_fit"][m], dtype=float)
       w_ord = np.asarray(wave[od]["wavelength"], dtype=np.float64)
       lam_rec = np.interp(row_fit, pix, w_ord)
       lam_mid = float(np.mean(lam_inj))
       dvel = (
           np.asarray(delta_wave_to_delta_vel(lam_rec - lam_inj, lam_mid), float)
           * 100.0
       )
       med = float(np.median(dvel))
       deg = min(POLY_DEG, int(m.sum()) - 2)
       coeffs = np.polyfit(row_fit, lam_inj, deg)
       lam_poly = np.polyval(coeffs, pix)
       okw = np.isfinite(w_ord) & np.isfinite(lam_poly)
       dvel_poly = (
           np.asarray(
               delta_wave_to_delta_vel(lam_poly[okw] - w_ord[okw], lam_mid), float
           )
           * 100.0
       )
       per_order[od] = dict(
           lam_inj=lam_inj,
           lam_rec=lam_rec,
           row_fit=row_fit,
           dvel=dvel,
           mad=float(np.median(np.abs(dvel - med))),
           bias=med,
           lam_mid=lam_mid,
           n=int(m.sum()),
           poly_vs_wave_mad_cm_s=(
               float(np.median(np.abs(dvel_poly - np.median(dvel_poly))))
               if dvel_poly.size
               else float("nan")
           ),
       )

   ods = sorted(per_order)
   mads = np.array([per_order[o]["mad"] for o in ods])
   print(
       f"orders={len(ods)}  median MAD(WAVE@row_fit)={np.median(mads):.3f} cm/s"
   )

4. Consistency check: PDF and JSON report
-----------------------------------------

.. code-block:: python

   import json

   from hispecdrp.simulations.plotting import write_lfc_elliptical_wave_report

   summary = write_lfc_elliptical_wave_report(
       OUT,
       tbl=tbl,
       success_mask=ok,
       per_order=per_order,
       a_pix=A_PIX,
       b_pix=B_PIX,
       pa_deg=PA_DEG,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       fiber_object=FIBER_OBJECT,
       lfc_meta=lfc_meta,
       mode_stride=MODE_STRIDE,
       n_modes_total=n_tot,
       n_modes_per_order=n_per,
       rate_path=rate_path,
       fits_table_path=fits_path,
   )
   print(f"wrote {summary['pdf']}")
   print(
       json.dumps(
           {k: summary[k] for k in summary if k != "per_order_mad_cm_s"},
           indent=2,
       )
   )

5. Extract the comb (Stage 1–2)
-------------------------------

Paint the same mode list with the native simulator and extract it. Skip
``set_wavesol`` — KOA has no ``lfc_wavesol`` in this example; the next section
attaches a mock WAVE from ``SimulateRateMap.get_wavelength_solution``.

.. warning::
   **Setup for this section.** Set ``KOA_CALIBRATION_CACHE`` (the first cell
   can create one under ``OUT``) and optionally ``KECK_OBSERVER_EMAIL`` /
   ``KECK_OBSERVER_PASSWORD`` for a remote sync.

.. code-block:: python

   import os

   from hispecdrp import datamodels
   from hispecdrp.calibrations import verify_koa_calibrations
   from hispecdrp.pipelines import Stage1Pipeline, Stage2Pipeline
   from hispecdrp.simulations.e2e_pipeline import paint_injected_emission_comb

   os.environ.setdefault(
       "KOA_CALIBRATION_CACHE",
       str(OUT / "cals" / "KOA_CALIBRATION_CACHE"),
   )
   verify_koa_calibrations(SPECTROGRAPH, ignore_missing_cal_types=["lfc_wavesol"])

   rate_extract = paint_injected_emission_comb(
       line_list,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       photons_per_line=1.0e3,
   )
   l0 = datamodels.L0Model(
       rate=np.asarray(rate_extract, dtype=np.float32),
       meta={
           "instrument_name": "HISPEC",
           "spectrograph": SPECTROGRAPH,
           "datetime_obs": "2027-01-01T00:00:00.000",
           "exposure_type": "SPEC",
       },
   )
   l0.rate_err = np.sqrt(np.maximum(l0.rate, 0.0)).astype(np.float32)
   l0.rate_var_poisson = np.zeros_like(l0.rate, dtype=np.float32)
   l0.rate_var_rnoise = np.zeros_like(l0.rate, dtype=np.float32)
   l0.dq = np.zeros_like(l0.rate, dtype=np.uint32)

   do_fibers = [str(FIBER)]
   l1 = Stage1Pipeline(
       output_dir=OUT,
       primitives={
           "spec_extract": {"do_fibers": do_fibers},
           "set_wavesol": {"run": False},
       },
       dataset=l0,
   ).apply()
   l2 = Stage2Pipeline(
       output_dir=OUT,
       dataset=l1,
       primitives={
           "blaze_corr": {"do_fibers": do_fibers},
           "stellar_rv": {"run": False},
           "sky_sub": {"run": False},
       },
   ).apply()

6. Mock wavelength solution
---------------------------

Copy the HDF instrument-grid WAVE onto the extracted L2. Rows follow
``HISPECConfig.get_order_list``. This is the same stacking used by
:func:`~hispecdrp.simulations.simulate_up_the_ramp.create_wavelength_solution_model_from_sim`.

.. code-block:: python

   from hispecdrp.datamodels.hispec_config import HISPECConfig
   from hispecdrp.simulations.plotting import plot_extracted_lfc_order
   from hispecdrp.simulations.simulate_up_the_ramp import (
       apply_simulate_rate_map_wavelength_solution,
   )

   apply_simulate_rate_map_wavelength_solution(
       l2, SPECTROGRAPH, fibers=[FIBER], srm=None
   )
   order_list = HISPECConfig.get_order_list(SPECTROGRAPH)
   print(f"WAVE fiber {FIBER}: {getattr(l2, f'wave_fib{FIBER}').shape}")
   plot_extracted_lfc_order(
       l2,
       fiber=FIBER,
       order=order_list[0],
       out_path=OUT / "lfc_extracted_order.png",
   )

7. Fit 1D LFC lines and plot FWHM
---------------------------------

``FitLFCLinesPrimitive`` uses the mock WAVE only to assign each peak to an LFC
mode (``WAVE``, ``MODE_NUM``). Pass the same ``nu0`` / ``delta_nu`` as
``build_lfc_line_list``.

.. code-block:: python

   from hispecdrp import FitLFCLinesPrimitive
   from hispecdrp.simulations.plotting import plot_lfc_line_fwhm

   line_fits_model = FitLFCLinesPrimitive.call(
       l2,
       do_orders=order_list,
       do_fibers=do_fibers,
       lfc_nu0=lfc_meta["nu0"],
       lfc_delta_nu=lfc_meta["delta_nu"],
       save_preview=False,
       show_preview=False,
       fwhm_bounds=(1.0, 10.0),
   )
   n_fit = len(getattr(line_fits_model, f"lfc_fitparams_fib{FIBER}"))
   print(f"fitted modes={n_fit}")
   plot_lfc_line_fwhm(
       line_fits_model,
       fiber=FIBER,
       orders=order_list,
       y="fwhm_pix",
       out_path=OUT / "lfc_line_fwhm_pix.png",
   )

8. Chebyshev WAVE and Gaussian LSF
----------------------------------

Fit a 2D Chebyshev wavelength solution to the assigned modes, then a Gaussian
LSF. Plot resolving power :math:`R = \lambda / \mathrm{FWHM}_\lambda` from the
smoothed LSF. ``ProcessLFCPipeline`` runs the same three primitives and
registers the cals.

.. code-block:: python

   from hispecdrp import (
       CalculateChebyshev2DWLSPrimitive,
       CalculateGaussianLSFPrimitive,
   )
   from hispecdrp.simulations.plotting import plot_lsf_resolving_power

   wavesol_model = CalculateChebyshev2DWLSPrimitive.call(
       line_fits_model,
       deg_inter_order=6,
       deg_intra_order=9,
       fit_orders=order_list,
       opt_weights=True,
       do_fibers=do_fibers,
   )
   lsf_model = CalculateGaussianLSFPrimitive.call(
       wavesol_model,
       do_fibers=do_fibers,
       deg_inter_order=3,
       deg_intra_order=3,
       fit_orders=order_list,
   )
   plot_lsf_resolving_power(
       lsf_model,
       fiber=FIBER,
       orders=order_list,
       out_path=OUT / "lfc_lsf_resolving_power.png",
   )

   # Apply the LFC Gaussian LSF on a uniform wavelength grid
   from hispecdrp.primitives.lsf.lsf_convolve import convolve_numba_1d_chromatic

   wave_um = np.asarray(getattr(l2, f"wave_fib{FIBER}")[0], dtype=float)
   flux = np.asarray(getattr(l2, f"spec_fib{FIBER}")[0], dtype=float)
   wmin, wmax = float(np.nanmin(wave_um)), float(np.nanmax(wave_um))
   dw = float(np.median(np.abs(np.diff(np.sort(wave_um)))))
   wave_uni = np.linspace(wmin, wmax, int(np.ceil((wmax - wmin) / dw)) + 1)
   flux_uni = np.interp(
       wave_uni,
       wave_um if wave_um[0] < wave_um[-1] else wave_um[::-1],
       flux if wave_um[0] < wave_um[-1] else flux[::-1],
   )
   kernel_grid, kernels = lsf_model.make_kernels(
       wave=wave_uni,
       fiber=FIBER,
       order=order_list[0],
       kernel_size_sigma=5.0,
   )
   fluxc = convolve_numba_1d_chromatic(flux_uni, kernels)
   print(f"kernels {kernels.shape}  fluxc {fluxc.shape}")
