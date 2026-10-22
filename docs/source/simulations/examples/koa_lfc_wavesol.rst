KOA LFC Wavelength Solution
===========================

Copy of the simulated LFC example, but the WAVE product comes from the KOA
caldb ``lfc_wavesol`` instead of painting a new comb and running
:class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`.

Section 1 still paints an HDF-centered comb only so the full-frame / three-order
zoom figure can be generated. That rate map is **not** the WAVE source.
Section 2 downloads the KOA ``lfc_wavesol``. Section 3 makes the same
``LFC − HDF`` plots versus pixel and versus wavelength.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/koa_lfc_wavesol_example.ipynb"
        data-notebook-download="koa_lfc_wavesol_example.ipynb"
        download="koa_lfc_wavesol_example.ipynb">Download as notebook</a>
   </p>

.. warning::
   **Setup.** Set these in your shell / notebook kernel before running (or let
   the first cell create a local cache under ``OUT``):

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
     - Remote sync
     - Keck Observer account email
   * - ``KECK_OBSERVER_PASSWORD``
     - Remote sync
     - Keck Observer account password

Imports and output directory
----------------------------

.. code-block:: python

   import os
   from pathlib import Path

   import numpy as np

   from hispecdrp.simulations.e2e_pipeline import resolve_lfc_fiber_object

   OUT = Path("/path/to/hispec_koa_lfc_wavesol_example")
   OUT.mkdir(parents=True, exist_ok=True)
   os.environ.setdefault("KOA_CALIBRATION_CACHE", str(OUT / "cals" / "KOA_CALIBRATION_CACHE"))

   SPECTROGRAPH, FIBER = "BSPEC", 2
   FIBER_OBJECT = resolve_lfc_fiber_object(SPECTROGRAPH)
   EDGE = 0.05
   MODE_STRIDE = 10
   A_PIX, B_PIX, PA_DEG = 0.80, 0.55, 20.0
   PA_RAD = np.deg2rad(PA_DEG)
   PEAK_VALUE = 1.0e6

   print(f"OUT = {OUT}")
   print("KOA_CALIBRATION_CACHE:", os.environ["KOA_CALIBRATION_CACHE"])
   print(f"fiber={FIBER}  {FIBER_OBJECT}  mode_stride={MODE_STRIDE}")

1. HDF LFC geometry preview
---------------------------

Paint elliptical Gaussians at HDF deposit centers so the full-frame rate map
and three order zooms can be generated (same figure as the simulated-LFC
notebook). This frame is not extracted and is not passed to
:class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`.

.. code-block:: python

   from astropy.io import fits

   from hispecdrp.simulations.e2e_pipeline import (
       build_lfc_line_list,
       paint_elliptical_gaussian_comb,
   )

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

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_lfc_rate_map_order_zooms

   plot_lfc_rate_map_order_zooms(
       rate,
       line_list,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       out_path=OUT / "lfc_rate_map_order_zooms.png",
   )

2. Download the KOA / Keck WAVE
-------------------------------

Prefer the remote KOA ``lfc_wavesol``. If observer login fails, use the
Keck-origin BSPEC master ``sim_wavesol`` from the HISPEC cal distribution
(same ``origin=KECK`` product the pipeline tests download). Packaged
``sim_hdf_wavesol.line_zp`` files are skipped.

.. code-block:: python

   from hispecdrp import datamodels
   from hispecdrp.calibrations import (
       HISPECCalibrationStore,
       seed_koa_calibration_cache,
       verify_koa_calibrations,
   )
   from hispecdrp.datamodels.hispec_config import HISPECConfig
   from hispecdrp.utils import download_gdrive_file

   KECK_MASTER_WAVESOL = {
       "BSPEC": "HISPEC/Cals/HB.20260606.00000.00.mastercal-sim_wavesol.001.fits",
       "RSPEC": "HISPEC/Cals/HR.20260606.00000.00.mastercal-sim_wavesol.001.fits",
   }


   def _is_packaged_line_zp(row: dict) -> bool:
       blob = f"{row.get('note') or ''} {row.get('filename') or ''}"
       return "sim_hdf_wavesol.line_zp" in blob


   def _as_rows(raw) -> list[dict]:
       if raw is None:
           return []
       if isinstance(raw, dict):
           return [raw]
       return [row for row in raw if isinstance(row, dict)]


   def _koa_wave_rows(rows: list[dict], cal_type: str) -> list[dict]:
       want = str(cal_type).strip().lower()
       out = []
       for row in rows:
           if str(row.get("cal_type") or "").strip().lower() != want:
               continue
           spec = str(row.get("spectrograph") or "").strip().upper()
           if spec and spec != SPECTROGRAPH:
               continue
           if _is_packaged_line_zp(row):
               continue
           out.append(row)
       out.sort(key=lambda r: float(r.get("mjd_start") or 0.0), reverse=True)
       return out


   seed = seed_koa_calibration_cache()
   print(
       f"seeded={seed.get('seeded')}  source={seed.get('source')}  "
       f"fits={seed.get('n_fits_after')}"
   )

   summary = verify_koa_calibrations(
       SPECTROGRAPH,
       require_complete=False,
       download_missing=False,
   )
   print(
       f"{summary['spectrograph']} caldb: {summary['n_records']} record(s); "
       f"mode={summary.get('remote_mode')}"
   )
   if summary.get("remote_error"):
       print("remote:", summary["remote_error"])
   print("types:", summary.get("cal_types"))

   wavesol_path = None
   wavesol_rec = {"filename": None, "origin": None}
   source = None
   with HISPECCalibrationStore(connect_remote=True, sync_on_init=False) as store:
       rows = _as_rows(store.local_db.query())
       for cal_type in ("lfc_wavesol", "sim_wavesol"):
           chosen = _koa_wave_rows(rows, cal_type)
           if not chosen and getattr(store, "remote_db", None) is not None:
               remote = _as_rows(store.remote_db.query(cal_type=cal_type))
               chosen = _koa_wave_rows(remote, cal_type)
           if chosen:
               wavesol_path, wavesol_rec = store.get_calibration(chosen[0])
               source = f"caldb:{cal_type}"
               break

   if wavesol_path is None:
       gdrive_key = KECK_MASTER_WAVESOL[SPECTROGRAPH]
       wavesol_path = download_gdrive_file(gdrive_key, output_dir=str(OUT))
       wavesol_rec = {"filename": Path(wavesol_path).name, "origin": "KECK"}
       source = "gdrive:sim_wavesol"
       print(f"KOA login/catalog empty; using Keck-origin master WAVE {gdrive_key}")

   wavesol_model = datamodels.open(str(wavesol_path))
   order_list = HISPECConfig.get_order_list(SPECTROGRAPH)
   if hasattr(wavesol_model, "make_order_vec"):
       order_list = [int(o) for o in np.asarray(wavesol_model.make_order_vec())]
   lfc_wave = np.asarray(getattr(wavesol_model, f"wave_fib{FIBER}"), dtype=float)
   print(
       f"using {wavesol_rec.get('filename')}  origin={wavesol_rec.get('origin')}  "
       f"WAVE {lfc_wave.shape}  n_orders={len(order_list)}"
   )
   print(f"local file: {wavesol_path}")

3. KOA WAVE vs ``get_wavelength_solution``
------------------------------------------

Same comparison as the simulated-LFC notebook: KOA ``wave_fib*`` minus
:meth:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap.get_wavelength_solution`,
versus pixel (full detector and 800–3500) and versus ``λ_LFC`` in 1.13–1.17 µm.

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_lfc_vs_hdf_wavesol_shift
   from hispecdrp.simulations.simulate_up_the_ramp import (
       wavelength_arrays_from_simulate_rate_map,
   )

   hdf_wave = wavelength_arrays_from_simulate_rate_map(
       SPECTROGRAPH, fibers=[FIBER]
   )[FIBER]
   if lfc_wave.shape != hdf_wave.shape:
       raise ValueError(
           f"WAVE shapes differ: KOA {lfc_wave.shape} vs HDF {hdf_wave.shape}"
       )

   def _mad_frac(a, b):
       d = (a - b) / b
       d = d[np.isfinite(d)]
       if d.size == 0:
           return np.inf
       med = np.median(d)
       return float(np.median(np.abs(d - med)))

   if _mad_frac(lfc_wave[:, ::-1], hdf_wave) < _mad_frac(lfc_wave, hdf_wave):
       lfc_wave = lfc_wave[:, ::-1]
       print("flipped KOA WAVE along the spectral axis to match HDF pixel convention")
   stats = plot_lfc_vs_hdf_wavesol_shift(
       lfc_wave,
       hdf_wave,
       orders=order_list,
       out_path=OUT / "lfc_vs_hdf_wavesol.png",
   )
   print(
       f"median Δv = {stats['median_dvel_m_s']:.3f} m/s  "
       f"MAD = {stats['mad_dvel_m_s']:.3f} m/s"
   )
   print(f"{'ord':>5} {'med_mÅ':>10} {'med_m/s':>10} {'MAD_m/s':>10}")
   for row in stats["per_order"]:
       print(
           f"{row['order']:5d} {row['median_dlam_um']*1e7:10.3f} "
           f"{row['median_dvel_m_s']:10.3f} {row['mad_dvel_m_s']:10.3f}"
       )

.. code-block:: python

   plot_lfc_vs_hdf_wavesol_shift(
       lfc_wave,
       hdf_wave,
       orders=order_list,
       xlim=(800, 3500),
       out_path=OUT / "lfc_vs_hdf_wavesol_center.png",
   )

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_lfc_vs_hdf_wavesol_vs_wavelength

   plot_lfc_vs_hdf_wavesol_vs_wavelength(
       lfc_wave,
       hdf_wave,
       orders=order_list,
       lam_xlim=(1.13, 1.17),
       out_path=OUT / "lfc_vs_hdf_wavesol_vs_lambda.png",
   )
