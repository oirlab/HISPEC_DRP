# Packaged sim-HDF line-ZP wavelength solutions

These `LFCWavesolModel` FITS files start from **deposit-inverse HDF WAVE** and
apply absorption-line corrections so labeled λ matches the apparent centers used
by stellar RV / CCF:

1. Per-order poly Doppler scale from rate-map (and optional L2 spectrum) line
   centroids — removes the ~50–60 m/s ZEMAX PSF deposition shift.
2. Optional **CCF-order scale refine** on a known-RV reference spectrum + ccfopt
   mask — folds residual order-dependent CCF systematics into WAVE (data-model
   calibration, not a chromatic RV model at science time).

| File | Spectrograph |
|------|----------------|
| `HB.sim_hdf_wavesol.line_zp.fits` | BSPEC |
| `HR.sim_hdf_wavesol.line_zp.fits` | RSPEC (optional) |

Each FITS has a JSON sidecar with per-fiber ZP provenance.

## Regenerate (controlled `pipeline_example` loop)

Bake against the **same recipe as** the docs Basic Usage walkthrough:

* `create_ramp_from_spectrum(..., seed=0, RV=0, force BERV=0)`
* Stage1/2 with **explicit** sim masters from the returned `cals` dict
  (tracepos / traceprof / fiberflat / packaged WAVE) — not KOA selectors alone
* Packaged `*.ccfopt.delta.mas` mask (see `ccfopt_mask_from_sim.ipynb`)

From the repository root (needs ZEMAX HDF cache + KOA Stage-0 masters):

```bash
export KOA_CALIBRATION_CACHE=/path/to/KOA_CALIBRATION_CACHE

# 1) Regenerate the ccfopt mask (E2E notebook), then:
# 2) Multi-fiber WAVE (rate-map ZP + L2 poly + 2× CCF-order refine):
python hispecdrp/data/wavesol/bake_pipeline_example_wavesol.py \
  --fibers 1 2 3 --reuse-existing
```

After updating the packaged FITS, refresh any KOA-cached `*lfc_wavesol*` copy
(or delete it) so Stage1 does not keep a stale WAVE file.

## E2E / notebook usage

`create_ramp_from_spectrum` registers the packaged WAVE (`meta.note =
sim_hdf_wavesol.line_zp`). The notebook injects the returned Stage-1 cals into
Stage1/2 so fiberflat / WAVE match the bake. Pair with the packaged
`*.ccfopt.delta.mas` mask (`packaged_default_line_mask_path()`).

Order-dependent CCF systematics are already folded into
`HB.sim_hdf_wavesol.line_zp.fits` — do **not** re-apply CCF→WAVE refine at
science / notebook runtime.

**Reproducibility:** Keep `seed=0`, inject the `cals` from
`create_ramp_from_spectrum`, and use the packaged ccfopt mask. Fiber-2
aggregate RV then recovers ≈0 (formal uncertainty ≈0.3 m/s). Changing `seed`,
skipping cal injection (different fiberflat), or editing the mask without a
WAVE re-bake breaks the transfer — re-run the loop above.
