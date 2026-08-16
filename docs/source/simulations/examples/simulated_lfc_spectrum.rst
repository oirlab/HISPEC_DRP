Simulated LFC Spectrum
======================

Paint a physical laser frequency comb (LFC) on BSPEC fiber 2, fit each spot with
a 2D elliptical Gaussian, and compare WAVE at the fitted centroids to the known
injected wavelengths.

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

Setup
-----

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from hispecdrp.simulations.e2e_pipeline import resolve_lfc_fiber_object

   # OUTPUT DIRECTORY - update this to your desired output directory
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

4. PDF and JSON report
----------------------

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
