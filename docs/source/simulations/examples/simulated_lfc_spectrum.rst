Simulated LFC Spectrum
======================

Paint a physical laser frequency comb (LFC) on BSPEC fiber 2 using the
data-model ``LFCNU0`` / ``LFCDELNU`` ladder, deposit the spots with
the HDF deposit model (the preferred way to derive the wavelength
solution), show the 2-D rate map with zooms of three orders, extract
and blaze-correct that same map, and run
:class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`
with a per-order cubic-spline WAVE model (see §2 for why the default
global Chebyshev is not used here). The product is a **derived
wavesol object** (``lfc_wavesol.fits`` in ``OUT``): an empirical
wavelength solution anchored to the exactly known comb wavelengths at
the fitted spots. The last cells quantify how well it agrees with
:meth:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap.get_wavelength_solution`
— at the spots it exposes a coherent ≈ +39 m/s shift of the
HDF-deposited line centers relative to the deposit-model solution
(the known PSF deposition shift), and the derived WAVE is the
correction to use downstream for that and any smaller residual
shifts the deposit-model solution leaves uncorrected.

.. raw:: html

   <p><span style="color:red">The HISPEC DRP is in active development. Please submit issues and suggestions on the <a href="https://github.com/oirlab/HISPEC_DRP">GitHub</a>.</span></p>

Modes follow ``nu_n = nu_0 + n * delta_nu`` from
:attr:`~hispecdrp.datamodels.hispec_config.HISPECConfig.lfc_params`
(``LFCNU0`` / ``LFCDELNU``).

.. raw:: html

   <p class="download-notebook">
     <a class="download-notebook-btn"
        href="../../_static/notebooks/simulated_lfc_example.ipynb"
        data-notebook-download="simulated_lfc_example.ipynb"
        download="simulated_lfc_example.ipynb">Download as notebook</a>
   </p>

Sections 1–3 do **not** require KOA credentials or pyechelle.
Painting and extraction both use the sim-HDF deposit/trace model
directly. :class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`
writes ``lfc_wavesol`` / ``lsf`` under ``OUT`` and registers them in a fresh
local cache (not the KOA masters).

Imports and output directory
----------------------------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   OUT = Path("/path/to/hispec_simulated_lfc_example")
   OUT.mkdir(parents=True, exist_ok=True)

   SPECTROGRAPH, FIBER = "BSPEC", 2

   print(f"OUT = {OUT}")
   print(f"spectrograph={SPECTROGRAPH}  fiber={FIBER}")

1. Build the LFC as a 1-D spectrum, then pass it through the HDF
----------------------------------------------------------------

Build the physical mode list (``build_lfc_line_list``, comb ladder
from ``HISPECConfig.lfc_params``; pass ``nu0=`` / ``delta_nu=`` there
to simulate an arbitrary comb spacing or line count). The comb teeth
are, to the spectrograph, delta functions, so the raw 1-D spectrum
*is* a line list — one wavelength and one photon rate per mode — with
no intrinsic width at all: significantly sub-pixel by construction,
so nothing about the injected profile is imprinted on the
spectrograph and the width of every painted spot is set by the HDF
PSF alone (the extracted spot width is then a direct measurement of
the line spread function). The next cell plots that raw spectrum for
orders 110, 129, and 148 as stems at the exact comb wavelengths.

Modes falling in the wavelength overlap of two orders appear in both
orders' lists; the injected list keeps one entry per unique
wavelength, and the HDF then deposits those modes into both orders,
as a real echelle does.

The spectrum is then passed through the HDF:
``SimulateRateMap.set_fiber_spectrum(..., continuous=False)`` followed
by ``create_native_rate_map`` (``prf_convolution``), which stamps the
HDF PSF at each mode wavelength. ``apply_blaze=True`` weights each
stamp by the sinc² blaze. Note what the blaze does and does not do
here: in this HDF model the blaze peak sits near pixel ≈ 1300 rather
than mid-detector, so the efficiency falls to ~5–20% at one end of
each order but is still ~60–80% at the other — the first blaze null
lies outside the detector coverage on that side, so the comb does
*not* vanish at both ends of an order. The rate-map figure therefore
shows the tapered end faint and the bright end still populated,
which is the instrument model (the pyechelle paint shows the same
asymmetric envelope). A final cell shows the full 2-D rate map
and zooms of three orders, each with a spectral scale bar in pixels
and wavelength. Later cells extract this same rate map.

.. code-block:: python

   from hispecdrp.simulations.e2e_pipeline import build_lfc_line_list

   line_list, lfc_meta = build_lfc_line_list(SPECTROGRAPH, fiber=FIBER)
   print(lfc_meta)

   # Raw 1-D LFC spectrum: a line list, not a sampled flux density.
   # One entry per comb mode; the (arbitrary) photon rate is the same
   # for every mode, and the intrinsic width is zero. Overlapping
   # orders share modes, so the injected list uses each unique
   # wavelength once.
   lfc_wl = np.unique(
       np.concatenate([np.asarray(v, dtype=float) for v in line_list.values()])
   )
   lfc_wl.sort()
   PHOTONS_PER_LINE = 1.0e7  # ph/s per mode; sets the flux scale only
   lfc_line_rate = np.full(lfc_wl.shape, PHOTONS_PER_LINE)
   n_modes_traces = sum(len(v) for v in line_list.values())
   print(
       f"line-list spectrum: {lfc_wl.size} unique modes "
       f"({n_modes_traces} painted across all order traces), "
       f"{PHOTONS_PER_LINE:.1e} ph/s/line, "
       f"{lfc_wl.min():.4f}-{lfc_wl.max():.4f} um"
   )

.. code-block:: python

   import matplotlib.pyplot as plt

   # The raw 1-D spectrum before the HDF: delta-function comb teeth.
   # Each stem is one mode at its exact comb wavelength (uniform
   # height = uniform photon rate per line); there is no line profile
   # to resolve.
   SHOW_ORDERS = (110, 129, 148)
   fig, axes = plt.subplots(len(SHOW_ORDERS), 1, figsize=(11, 6.5), sharex=False)
   for ax, od in zip(axes, SHOW_ORDERS):
       spots = np.sort(line_list[int(od)])
       ax.vlines(spots, 0.0, PHOTONS_PER_LINE, color="tab:blue", lw=0.7)
       ax.plot(
           spots, np.full(spots.shape, PHOTONS_PER_LINE), "|",
           ms=10, color="tab:orange",
           label=f"comb modes (n={spots.size})",
       )
       ax.set_xlim(spots.min() - 0.0007, spots.max() + 0.0007)
       ax.set_ylim(0, PHOTONS_PER_LINE * 1.25)
       ax.set_ylabel("photon rate\n(ph s$^{-1}$ line$^{-1}$)")
       ax.set_title(f"order {od}", loc="left", fontsize=10)
       ax.legend(fontsize=8, loc="upper right")
   axes[-1].set_xlabel("wavelength (um)")
   fig.suptitle("Raw 1-D LFC line-list spectrum before the HDF", fontsize=13)
   fig.tight_layout()
   out_png = OUT / "lfc_1d_linelists_orders.png"
   fig.savefig(out_png, dpi=130, bbox_inches="tight")
   plt.close(fig)
   # plt.show() is a no-op on non-interactive backends; display the PNG
   # explicitly like the hispecdrp plotting helpers do.
   from IPython.display import Image, display

   display(Image(filename=str(out_png)))

.. code-block:: python

   from astropy import units as u

   from hispecdrp.simulations import SimulateRateMap

   # continuous=False makes each line-list entry a delta-function
   # sample: the PRF-convolution paint deposits the HDF PSF itself at
   # each mode wavelength, so the painted spots carry the spectrograph
   # LSF and nothing else. apply_blaze=True weights each stamp by the
   # sinc^2 blaze (peak near pixel ~1300 in this HDF, so the taper is
   # asymmetric -- see the Sec. 1 text).
   srm = SimulateRateMap()
   srm.set_fiber_spectrum(
       FIBER,
       lfc_wl * u.micron,
       lfc_line_rate * (u.ph / u.s),
       continuous=False,
   )
   rate = srm.create_native_rate_map(
       spectrograph=SPECTROGRAPH, method="prf_convolution",
       apply_blaze=True, save=False,
   )
   print(
       f"HDF rate map: shape={rate.shape}, max={rate.max():.3e} ph/s/pix, "
       f"in-channel flux={rate[:, :4096].sum():.3e} ph/s"
   )

.. code-block:: python

   from hispecdrp.simulations.plotting import plot_lfc_rate_map_order_zooms

   plot_lfc_rate_map_order_zooms(
       rate,
       line_list,
       spectrograph=SPECTROGRAPH,
       fiber=FIBER,
       out_path=OUT / "lfc_rate_map_order_zooms.png",
   )

2. Extract, blaze, and run ProcessLFCPipeline
---------------------------------------------

Box-extract the painted rate map and blaze-correct with a unity fiberflat
(no sky / stellar RV). Pass sim-HDF ``tracepos`` / ``traceprof`` and a
geometric HDF WAVE in memory so Stage 1 never queries KOA. That WAVE is
only a bootstrap for mode assignment.

``fit_lfc_lines`` is given ``fwhm_bounds=(1.0, 7.0)``, a built-in
primitive parameter. Because the injection is delta-function lines,
the extracted spots are the LSF itself: HDF-PSF spots with FWHM
≈ 3.0–3.9 pixels, outside the default (1, 3) bound, which would
reject most spots as failed fits. The blaze correction uses a unity
fiberflat, so the sinc² blaze painted in §1 remains in the extracted
spectra and the fitted amplitudes taper toward the order ends, as
they would ahead of a real flat-field correction.

``LFCNU0`` / ``LFCDELNU`` are written onto the data-model metadata.
:class:`~hispecdrp.pipelines.calibration.process_lfc_pipeline.ProcessLFCPipeline`
then runs with its default line-fit and LSF parameters, but with
``wave_model='cubic_spline_1d'`` for the WAVE step instead of the default
global 2-D Chebyshev (``chebyshev2d``, degree 9 x 10). The HDF solution
in §3 is a PCHIP through 50 deposit samples per order (knots every
~83 pixels); the global Chebyshev cannot represent that piecewise-cubic
shape and would add its own smooth systematics. A per-order cubic
spline through the fitted lines instead follows the empirical spot
positions locally — it is exact at the fitted spots by construction
— so the resulting WAVE is a derived wavelength solution tied
directly to the known comb wavelengths, correcting whatever small
shifts the deposit-model solution leaves between and at the spots.
Note the spline does not
extrapolate: WAVE is NaN outside the fitted-line pixel range of each
order, and the §3 comparisons mask those pixels.

.. code-block:: python

   import os

   from hispecdrp import BlazeCorrectionPrimitive, datamodels
   from hispecdrp.pipelines import ProcessLFCPipeline, Stage1Pipeline
   from hispecdrp.simulations.e2e_pipeline import (
       create_tracepos_model_from_sim_hdf,
       create_traceprof_model_from_sim_hdf,
   )
   from hispecdrp.simulations.ideal_spectrum_rv import geometric_hdf_wavesol

   os.environ["KOA_CALIBRATION_CACHE"] = str(OUT / "cals" / "lfc_local_caldb")

   do_fibers = [str(FIBER)]
   l0 = datamodels.L0Model(
       rate=np.asarray(rate, dtype=np.float32),
       meta={
           "instrument_name": "HISPEC",
           "spectrograph": SPECTROGRAPH,
           "datetime_obs": "2027-01-01T00:00:00.000",
           "exposure_type": "SPEC",
           "lfc_nu0": lfc_meta["nu0"],
           "lfc_delta_nu": lfc_meta["delta_nu"],
           "target_name": lfc_meta["fiber_object"],
       },
   )
   l0.rate_err = np.sqrt(np.maximum(l0.rate, 0.0)).astype(np.float32)
   l0.rate_var_poisson = np.zeros_like(l0.rate, dtype=np.float32)
   l0.rate_var_rnoise = np.zeros_like(l0.rate, dtype=np.float32)
   l0.dq = np.zeros_like(l0.rate, dtype=np.uint32)

   l1 = Stage1Pipeline(
       output_dir=OUT,
       dataset=l0,
       primitives={
           "spec_extract": {
               "do_fibers": do_fibers,
               "method": "box",
               "tracepos": create_tracepos_model_from_sim_hdf(SPECTROGRAPH),
               "traceprof": create_traceprof_model_from_sim_hdf(SPECTROGRAPH),
           },
           "set_wavesol": {"wavesol": geometric_hdf_wavesol(SPECTROGRAPH)},
       },
   ).apply()

   n_orders, n_pix = getattr(l1, f"spec_uncal_fib{FIBER}").shape
   fiberflat = datamodels.FiberflatModel(
       meta={"instrument_name": "HISPEC", "spectrograph": SPECTROGRAPH},
   )
   fiberflat[FIBER].spec = np.ones((n_orders, n_pix), dtype=np.float32)
   fiberflat[FIBER].spec_err = np.full((n_orders, n_pix), 1.0e-2, dtype=np.float32)
   fiberflat[FIBER].spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
   l2 = BlazeCorrectionPrimitive.call(
       l1,
       do_fibers=do_fibers,
       fiberflat=fiberflat,
       output_dir=OUT,
       save_result=True,
   )
   l2.meta.lfc_nu0 = l0.meta.lfc_nu0
   l2.meta.lfc_delta_nu = l0.meta.lfc_delta_nu

   wavesol_model = ProcessLFCPipeline(
       output_dir=OUT,
       dataset=l2,
       primitives={
           "fit_lfc_lines": {"do_fibers": do_fibers, "fwhm_bounds": (1.0, 7.0)},
           "lfc_wavesol": {"do_fibers": do_fibers, "wave_model": "cubic_spline_1d"},
           "lsf": {"do_fibers": do_fibers},
       },
   ).apply()["wavesol"]
   order_list = [int(o) for o in np.asarray(wavesol_model.make_order_vec())]
   wavesol_path = OUT / "lfc_wavesol.fits"
   wavesol_model.save(filepath=str(wavesol_path))
   print(
       f"{type(l1).__name__} → {type(l2).__name__} → {type(wavesol_model).__name__}  "
       f"WAVE {getattr(wavesol_model, f'wave_fib{FIBER}').shape}  "
       f"fitted modes={len(getattr(wavesol_model, f'lfc_fitparams_fib{FIBER}'))}"
   )
   print(f"wrote {wavesol_path}")

One order in detail: why the residuals spike between lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cell below dissects a single order. The top panel shows the two wavelength solutions as functions of pixel: the pipeline per-order cubic spline (orange line) threaded through every fitted LFC spot (orange points, the spline knots), and the HDF solution (blue line) that ``get_wavelength_solution`` obtains by inverting a PCHIP through the 50 HDF deposit samples of the order (blue points). Note the knot-density contrast: ~100 LFC spots versus 50 HDF samples over the same pixels.

The bottom panel is the per-pixel velocity residual between the two lines, with the residual also evaluated exactly at each knot set (orange at the LFC spots, blue at the HDF samples). The orange points sit at the at-spot offset measured in §3 (≈ −39 m/s in pipeline-minus-HDF for this order): the spline passes exactly through its knots, so its level against the deposit-model truth is precisely the PSF deposition shift of the HDF-painted spots. Between orange points the grey curve bulges, because the spline and the inverted PCHIP are different cubics between their respective knots; those deterministic bulges, largest in the shaded first/last knot intervals, are the "spikes" seen in the full-detector comparison in §3 below, and are the only sense in which the derived WAVE departs from ``get_wavelength_solution`` — an interpolation-shape difference, not a shift of the solution.

.. code-block:: python

   from scipy.constants import c as C_LIGHT
   from scipy.interpolate import CubicSpline

   import matplotlib.pyplot as plt

   from hispecdrp.simulations import SimulateRateMap
   from hispecdrp.simulations.simulate_rate_map import read_pyechelle_hdf

   ORDER_SHOW = 118 if 118 in order_list else order_list[len(order_list) // 2]
   oi = order_list.index(ORDER_SHOW)
   pix = np.arange(4096, dtype=float)

   # Pipeline spline for this order: line = WAVE row, knots = fitted LFC spots.
   pipe_wave = np.asarray(getattr(wavesol_model, f"wave_fib{FIBER}"), dtype=float)[oi]
   tab = getattr(wavesol_model, f"lfc_fitparams_fib{FIBER}")
   sel = np.asarray(tab["ORDER"]) == ORDER_SHOW
   lfc_pix = np.asarray(tab["MU_PIX"], dtype=float)[sel]
   lfc_wave_knots = np.asarray(tab["WAVE"], dtype=float)[sel]
   srt = np.argsort(lfc_pix)
   lfc_pix, lfc_wave_knots = lfc_pix[srt], lfc_wave_knots[srt]

   # HDF truth for this order: line = inverted PCHIP, knots = its 50 deposit samples.
   srm = SimulateRateMap()
   truth_wave = np.asarray(
       srm.get_wavelength_solution(SPECTROGRAPH)[FIBER][ORDER_SHOW]["wavelength"],
       dtype=float,
   )
   model_name = srm.bspec_hdf_model if SPECTROGRAPH == "BSPEC" else srm.rspec_hdf_model
   model_data = read_pyechelle_hdf(srm.hdf_file_directory + model_name)
   raw = model_data[f"fiber_{FIBER}"][str(ORDER_SHOW)]
   hdf_sample_wave = np.asarray(raw["xy_wavelengths"], dtype=float)
   # get_wavelength_solution flips the spectral axis vs the HDF x coordinate.
   hdf_sample_pix = 4095.0 - np.asarray(raw["x"], dtype=float)
   srt = np.argsort(hdf_sample_pix)
   hdf_sample_pix, hdf_sample_wave = hdf_sample_pix[srt], hdf_sample_wave[srt]

   # Per-pixel velocity residual between the two lines.
   ok = np.isfinite(pipe_wave) & np.isfinite(truth_wave)
   dvel = np.where(ok, C_LIGHT * (pipe_wave - truth_wave) / truth_wave, np.nan)
   # Residuals evaluated exactly at each knot set: rebuild the pipeline
   # spline from its knots (as the primitive does) and interpolate the
   # HDF line for the LFC positions; vice versa for the HDF positions.
   spline_fn = CubicSpline(lfc_pix, lfc_wave_knots)
   dvel_at_lfc = C_LIGHT * (lfc_wave_knots - np.interp(lfc_pix, pix, truth_wave)) / lfc_wave_knots
   dvel_at_hdf = C_LIGHT * (spline_fn(hdf_sample_pix, extrapolate=False) - hdf_sample_wave) / hdf_sample_wave

   fig, axes = plt.subplots(
       2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]}
   )
   ax = axes[0]
   ax.plot(pix, truth_wave, color="tab:blue", lw=1.2,
           label="HDF: inverted PCHIP (get_wavelength_solution)")
   ax.scatter(hdf_sample_pix, hdf_sample_wave, color="tab:blue", s=22, zorder=5,
              label=f"HDF deposit samples (n={hdf_sample_pix.size})")
   ax.plot(pix, pipe_wave, color="tab:orange", lw=1.2,
           label="Pipeline: per-order cubic spline (cubic_spline_1d)")
   ax.scatter(lfc_pix, lfc_wave_knots, color="tab:orange", s=14, zorder=5,
              label=f"Fitted LFC spots / spline knots (n={lfc_pix.size})")
   ax.set_ylabel(r"$\lambda$ ($\mu$m)")
   ax.set_title(f"Order {ORDER_SHOW}: two wavelength solutions, two knot sets")
   ax.legend(loc="best", fontsize=8, ncol=2)

   ax = axes[1]
   for lo, hi in ((lfc_pix[0], lfc_pix[1]), (lfc_pix[-2], lfc_pix[-1])):
       ax.axvspan(lo, hi, color="tab:red", alpha=0.10)
   ax.plot(pix, dvel, color="0.3", lw=0.8, label="Pipeline - HDF (per pixel)")
   ax.scatter(hdf_sample_pix, dvel_at_hdf, color="tab:blue", s=22, zorder=5,
              label="residual at HDF sample positions")
   ax.scatter(lfc_pix, dvel_at_lfc, color="tab:orange", s=16, zorder=5,
              label="residual at LFC spot positions")
   ax.axhline(0.0, color="k", lw=0.8)
   ax.set_xlabel("spectral pixel")
   ax.set_ylabel(r"$\Delta v$ (m s$^{-1}$)")
   ax.set_xlim(lfc_pix.min() - 100, lfc_pix.max() + 100)
   ax.legend(loc="best", fontsize=8)
   fig.tight_layout()
   out_png = OUT / f"lfc_vs_hdf_order{ORDER_SHOW}_interpolation.png"
   fig.savefig(out_png, dpi=150)
   plt.close(fig)
   # plt.show() is a no-op on non-interactive backends; display the PNG
   # explicitly like the hispecdrp plotting helpers do.
   from IPython.display import Image, display

   display(Image(filename=str(out_png)))

   print(
       f"order {ORDER_SHOW}: {lfc_pix.size} LFC knots "
       f"(spacing {np.median(np.diff(lfc_pix)):.1f} pix), "
       f"{hdf_sample_pix.size} HDF samples "
       f"(spacing {np.median(np.diff(hdf_sample_pix)):.1f} pix)"
   )
   print(
       f"median dv at LFC spots = {np.nanmedian(dvel_at_lfc):+.4f} m/s; "
       f"max |dv| between spots = {np.nanmax(np.abs(dvel)):.3f} m/s"
   )



3. LFC WAVE vs ``get_wavelength_solution``
------------------------------------------

Compare the pipeline ``wave_fib*`` to
:meth:`~hispecdrp.simulations.simulate_rate_map.SimulateRateMap.get_wavelength_solution`
on the same fiber, order list, and spectral pixels. The first plot is the full detector versus pixel; the next cell zooms to
spectral pixels 800–3500. A later cell remaps the same ``LFC − HDF``
velocity onto ``λ_LFC`` (one trace per order), which is the usual
``λ_LFC − λ_SIM`` versus wavelength view. Pixels the pipeline flags
``UNRELIABLE_WS`` in ``spec_dq_fib*`` are masked out of the comparison
(set to NaN). For the spline wavesol that covers pixels outside the
fitted line range plus the first and last knot interval of each order,
where the spline rests on a boundary segment and its interpolation
error is largest. No velocity clipping is applied: the residuals of
interest are at the cm/s level and below.

The primary check is at the fitted spots themselves. For every fitted
LFC line the pipeline knows the exact comb wavelength and measures
the spot position ``MU_PIX``; evaluating ``get_wavelength_solution``
at that same pixel must return the comb wavelength. The spots were deposited by the HDF PSF machinery while the truth
is the deposit-inverse solution, and the check exposes the known PSF
deposition shift: per-order median offsets of +36 to +45 m/s (robust
scatter 10–16 m/s per order). The derived wavesol, anchored at the
fitted spots, absorbs exactly this shift — it is the correction the
deposit-model solution leaves uncorrected — and the remaining
differences are the between-spot interpolation shapes quantified by
the pixel-wise plots.

.. code-block:: python

   from hispecdrp.simulations import SimulateRateMap
   from hispecdrp.simulations.plotting import plot_lfc_vs_hdf_wavesol_shift

   lfc_wave = np.asarray(getattr(wavesol_model, f"wave_fib{FIBER}"), dtype=float)
   flag = int(datamodels.DQ_FLAGS["UNRELIABLE_WS"])
   dq = np.asarray(getattr(wavesol_model, f"spec_dq_fib{FIBER}"))
   n_flagged = int(((dq & flag) != 0).sum())
   lfc_wave = np.where((dq & flag) != 0, np.nan, lfc_wave)
   print(f"masked UNRELIABLE_WS pixels: {n_flagged} of {dq.size}")
   hdf_by_order = SimulateRateMap().get_wavelength_solution(SPECTROGRAPH)[FIBER]
   hdf_wave = np.vstack(
       [np.asarray(hdf_by_order[int(od)]["wavelength"], dtype=float) for od in order_list]
   )
   stats = plot_lfc_vs_hdf_wavesol_shift(
       lfc_wave,
       hdf_wave,
       orders=order_list,
       out_path=OUT / "lfc_vs_hdf_wavesol.png",
   )
   print(
       f"median Δv = {stats['median_dvel_m_s']:.3f} m/s  "
       f"MAD = {stats['mad_dvel_m_s']:.3f} m/s  "
       f"clipped |Δv|>100 m/s: {stats['n_clipped']}"
   )
   print(f"{'ord':>5} {'med_mÅ':>10} {'med_m/s':>10} {'MAD_m/s':>10}")
   for row in stats["per_order"]:
       print(
           f"{row['order']:5d} {row['median_dlam_um']*1e7:10.3f} "
           f"{row['median_dvel_m_s']:10.3f} {row['mad_dvel_m_s']:10.3f}"
       )

.. code-block:: python

   # Headline test: get_wavelength_solution at the fitted spot positions
   # versus the exact comb wavelengths the pipeline assigned.
   import matplotlib.pyplot as plt
   from scipy.constants import c as C_LIGHT

   tab = getattr(wavesol_model, f"lfc_fitparams_fib{FIBER}")
   mu = np.asarray(tab["MU_PIX"], dtype=float)
   wv = np.asarray(tab["WAVE"], dtype=float)
   od = np.asarray(tab["ORDER"], dtype=int)
   ok = np.isfinite(mu) & np.isfinite(wv)
   mu, wv, od = mu[ok], wv[ok], od[ok]

   lam_hdf_at_mu = np.empty_like(mu)
   disp_at_mu = np.empty_like(mu)  # m/s per pixel at each spot
   for o in order_list:
       sel = od == o
       row = hdf_wave[order_list.index(o)]
       lam_hdf_at_mu[sel] = np.interp(mu[sel], pix_grid := np.arange(row.size, dtype=float), row)
       disp_at_mu[sel] = C_LIGHT * np.abs(np.gradient(row))[
           np.clip(np.rint(mu[sel]).astype(int), 0, row.size - 1)
       ] / row[np.clip(np.rint(mu[sel]).astype(int), 0, row.size - 1)]

   dv_spot = C_LIGHT * (lam_hdf_at_mu - wv) / wv
   med = np.median(dv_spot)
   mad = np.median(np.abs(dv_spot - med))
   pix_off = dv_spot / disp_at_mu
   print(f"fitted spots: {mu.size}")
   print(
       f"at-spot dv (HDF - comb): median = {med:+.4f} m/s  "
       f"robust sigma = {1.4826 * mad:.4f} m/s"
   )
   print(
       f"equivalent spectral-pixel offset: median = {np.median(pix_off):+.3f} pix"
   )

   fig, ax = plt.subplots(figsize=(13, 3.6))
   for o in order_list:
       sel = od == o
       ax.scatter(mu[sel], dv_spot[sel], s=3, label=f"order {o}" if o in (110, 148) else None)
   ax.axhline(0.0, color="k", lw=0.9)
   ax.set_xlabel("fitted spot pixel (MU_PIX)")
   ax.set_ylabel(r"$\Delta v$ at spots (m s$^{-1}$)")
   ax.set_title("get_wavelength_solution vs exact comb wavelength, at the spots")
   fig.tight_layout()
   out_png = OUT / "lfc_wavesol_at_spot_test.png"
   fig.savefig(out_png, dpi=150)
   plt.close(fig)
   from IPython.display import Image, display

   display(Image(filename=str(out_png)))

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

Derived wavesol product
-----------------------

The derived wavelength solution is the ``wavesol_model`` saved as
``lfc_wavesol.fits`` in ``OUT`` in §2 (and registered in the local
calibration cache). Its WAVE is tied to the exactly known comb
wavelengths at ≈ 4,300 fitted spots per fiber, correcting the coherent ≈ +39 m/s PSF-deposition offset of the deposit-model solution (per-order medians +36 to +45 m/s; at-spot robust scatter 10–16 m/s per order) — so downstream steps should use this derived WAVE, rather than the deposit-model solution, wherever shifts at that level or smaller matter. Pixel-by-pixel, the derived WAVE differs from ``get_wavelength_solution`` by that same offset plus a between-spot interpolation component (robust scatter ≈ 11 m/s about the median, max 69 m/s, largest in the ``UNRELIABLE_WS``-flagged first/last knot intervals), which the per-order spline WAVE already embodies.
