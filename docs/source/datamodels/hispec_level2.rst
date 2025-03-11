HISPEC Level 2
==============

HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements. Data products from BSPEC and RSPEC are still kept separate.


Extensions
----------

.. csv-table::
   :header: "#", "Name", "HDU TYPE", "Data Type", "Dimensions", "Units", "Description"

   "1-3", "SPEC_UNCAL_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum."
   "4-6", "SPEC_UNCAL_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum error (all sources)."
   "7-9", "SPEC_UNCAL_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum variance (Poisson)."
   "10-12", "SPEC_UNCAL_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum variance (read noise)."
   "13-15", "SPEC_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s (or W/m²/µm)", "Calibrated spectrum."
   "16-18", "SPEC_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s (or W/m²/µm)", "Calibrated spectrum error (all sources)."
   "19-21", "SPEC_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)² or (W/m²/µm)²", "Calibrated spectrum variance (Poisson)."
   "22-24", "SPEC_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)² or (W/m²/µm)²", "Calibrated spectrum variance (read noise)."
   "25-27", "SPEC_DQ_[B/R]SPEC_FIB#", "Image", "Uint32", "Nord x 4096", "None", "Calibrated spectrum data-quality flags."
   "28-30", "MID_TIME_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "MJD", "Flux-weighted exposure mid-point times."
   "31-33", "BARY_CORR_[B/R]SPEC_FIB#", "Table", "See below", "See below", "See below", "Barycentric correction information."
   "34-36", "WS_[B/R]SPEC_FIB#", "Image", "Float64", "Nord x 4096", "micron", "Wavelength solution in rest frame of the instrument."
   "37-39", "FIBERFLAT_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Fiberflat spectrum."
   "40-42", "FIBERFLAT_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Fiberflat spectrum error (all sources)."
   "43-45", "FIBERFLAT_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Fiberflat spectrum variance (Poisson)."
   "46-48", "FIBERFLAT_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Fiberflat spectrum variance (read noise)."
   "49-51", "SKY_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum or model."
   "52-54", "SKY_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum error (all sources)."
   "55-57", "SKY_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum variance (Poisson)."
   "58-60", "SKY_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum variance (read noise)."
   "61-63", "STAR_MOD_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm", "Stellar model spectrum Doppler shifted according to Barycenter OR RVs."
   "64-66", "E2E_TPUT_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "End-to-end throughput derived from STAR_MOD."
   "67-69", "TEL_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Telluric model spectrum."
   "70-72", "TEL_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "None", "Telluric model spectrum error (all sources)."
   "73-75", "FLUX_ABS_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm/(e-/s)", "Spectrum in absolute flux units."
   "76-78", "FLUX_ABS_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm/(e-/s)", "Absolute spectrum error (all sources)."
   "79-81", "FLUX_ABS_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm/(e-/s)", "Absolute spectrum variance (Poisson)."
   "82-84", "FLUX_ABS_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm/(e-/s)", "Absolute spectrum variance (read noise)."
   "85-87", "ET_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Simultaneous etalon spectrum."
   "88-90", "ET_ERR_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Etalon spectrum error (all sources)."
   "91-93", "ET_VAR_POISSON_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Etalon spectrum variance (Poisson)."
   "94-96", "ET_VAR_RNOISE_[B/R]SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Etalon spectrum variance (read noise)."
   "97-99", "ET_CEN_[B/R]SPEC_FIB#", "Table", "See below", "See below", "See below", "Etalon centroids."
   "100-102", "RVCCF_[B/R]SPEC_FIB#", "Table", "See below", "See below", "See below", "RVs for each chunk."
   "103-105", "RVCCFWAVG_[B/R]SPEC_FIB#", "Table", "See below", "See below", "See below", "RVs averaged together across chunks."
   "106-108", "CCF_[B/R]SPEC_FIB#", "Table", "See below", "See below", "See below", "Cross-correlation functions for each chunk."


**BARY_CORR Columns:**

- **ORDER (int32)**: Echelle order number.
- **WAVE (float64)**: Wavelength of the middle of order in microns.
- **JDUTC (float64)**: Julian date in universal coordinate time (UTC) of the non-flux-weighted midpoint.
- **JDUTCW (float64)**: Julian date in universal coordinate time (UTC) of the flux-weighted midpoint.
- **HJD (float64)**: Heliocentric Julian date of the non-flux-weighted midpoint.
- **HJDW (float64)**: Heliocentric Julian date of the flux-weighted midpoint.
- **BJD (float64)**: Barycentric Julian date of the non-flux-weighted midpoint.
- **BJDW (float64)**: Barycentric Julian date of the flux-weighted midpoint.
- **BERV (float64)**: Barycentric velicity correction for the non-flux-weighted midpoint in m/s.
- **BERVW (float64)**: Barycentric velicity correction for the flux-weighted midpoint in m/s.

**Etalon Fit Columns:**

- **AMPLITUDE, AMPLITUDE_ERR, (float64)**: The amplitude of the etalon mode and its uncertainty in e-.
- **MEAN, MEAN_ERR (float64)**: The center of the etalon mode and its uncertainty in pixels.
- **SIGMA, SIGMA_ERR (float64)**: The standard deviation of the etalon mode and its uncertainty in pixels.
- **B0, B0_ERR (float64)**: The background level and uncertainty in e-.
- **MODE_NUM (int32)**: The mode number of this etalon peak. Etalon mode numbers are arbitrary, but are nice to keep track of.
- **WAVE, WAVE_ERR (float64)**: The measured wavelength of the etalon peak and its uncertainty as calibrated by the LFC in microns.
- **FINESSE, FINESSE_ERR (float64)**: The cavity finesse and its uncertainty.
- **CAVLEN, CAVLEN_ERR (float64)**: The effective cavity length at this wavelength.

**RV Columns (Chunks):**

- **BJDW (float64)**: Barycentric Julian date of the flux-weighted midpoint.
- **BERVW (float64)**: Barycentric velicity correction for the flux-weighted midpoint in m/s.
- **RV (float64)**: The RV in m/s.
- **RV_ERR (float64)**: The RV uncertainty in m/s.
- **WAVE (float64)**: Wavelength of the middle of order in microns.
- **SPECRES (float64)**: Average spectral resolution in this order.
- **TELL_ABS (float64)**: Average telluric absorption in this order.
- **BIS (float64)**: Bisector inverse slope (BIS) in m/s.
- **RV_ABS (float64)**: Absolute RV measured in m/s.
- **RV_ABS_ERR (float64)**: Error in absolute RV in m/s.

**RV Columns (Combined):** there will likely only be one value per column, so we may consider an alternative approach for this extension:

- **BJDW (float64)**: Average barycentric Julian date of the flux-weighted midpoint.
- **BERVW (float64)**: Barycentric velicity correction for the flux-weighted midpoint in m/s.
- **RV (float64)**: RV in m/s.
- **RV_ERR_AGG (float64)**: RV uncertainty measured by adding the individual RV errors in quadrature.
- **RV_ERR_DIST (float64)**: RV uncertainty measured from the distribution of the individual RVs assuming they are normally distributed.
- **AIRMASS (float64)**: Airmass at the midpoint of this observation.
- **BIS (float64)**: Bisector inverse slope (BIS) in m/s.
- **RV_ABS (float64)**: Absolute RV measured in m/s.
- **RV_ABS_ERR (float64)**: Error in absolute RV in m/s.

**CCF Columns:**

- **WAVE (float64)**: The average wavelength in this order (microns).
- **VEL_GRID (float64)**: The relative wavelength or velocity grid (TBD).
- **CCF (float64)**: The CCF function.