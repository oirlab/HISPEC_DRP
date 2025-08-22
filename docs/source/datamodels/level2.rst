Level 2 (Calibrated Spectra)
============================

HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements. Data products from BSPEC and RSPEC are still kept separate.

**Classes:**

* :py:class:`~hispecdrp.datamodels.hispec.level2.HISPECL2Model`
* :py:class:`~hispecdrp.datamodels.parvi.level2.PARVIL2Model`

**File suffix:** spec


Extensions
----------

.. csv-table::
   :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

   "SPEC_UNCAL_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum."
   "SPEC_UNCAL_ERR_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum error (all sources)."
   "SPEC_UNCAL_VAR_POISSON_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum variance (Poisson)."
   "SPEC_UNCAL_VAR_RNOISE_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum variance (read noise)."
   "SPEC_UNCAL_DQ_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "L1 spectrum data quality."
   "SPEC_FIB#", "Image", "Float32", "Nord x 4096", "e-/s (or W/m²/µm)", "Calibrated spectrum."
   "SPEC_ERR_FIB#", "Image", "Float32", "Nord x 4096", "e-/s", "Calibrated spectrum error (all sources)."
   "SPEC_VAR_POISSON_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Calibrated spectrum variance (Poisson)."
   "SPEC_VAR_RNOISE_FIB#", "Image", "Float32", "Nord x 4096", "(e-/s)²", "Calibrated spectrum variance (read noise)."
   "SPEC_DQ_FIB#", "Image", "Uint32", "Nord x 4096", "None", "Calibrated spectrum data-quality flags."
   "MID_TIME_FIB#", "Image", "Float32", "Nord x 4096", "MJD", "Flux-weighted exposure mid-point times."
   "BARY_CORR_FIB#", "Table", "See below", "See below", "See below", "Barycentric correction information."
   "WAVE_FIB#", "Image", "Float64", "Nord x 4096", "micron", "Drift corrected wavelength solution in rest frame of the instrument."
   "TRANS_FIB#", "Image", "Float32", "Nord x 4096", "None", "Transmission correction."
   "TRANS_ERR_FIB#", "Image", "Float32", "Nord x 4096", "None", "Transmission correction error (all sources)."
   "SKY_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum or model."
   "SKY_ERR_FIB#", "Image", "Float32", "Nord x 4096", "None", "Sky spectrum error (all sources)."
   "STAR_MOD_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm", "Stellar model spectrum Doppler shifted according to Barycenter OR RVs."
   "E2E_TPUT_FIB#", "Image", "Float32", "Nord x 4096", "None", "End-to-end throughput derived from STAR_MOD."
   "TEL_FIB#", "Image", "Float32", "Nord x 4096", "None", "Telluric model spectrum."
   "TEL_ERR_FIB#", "Image", "Float32", "Nord x 4096", "None", "Telluric model spectrum error (all sources)."
   "FLUX_ABS_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm", "Spectrum in absolute flux units."
   "FLUX_ABS_ERR_FIB#", "Image", "Float32", "Nord x 4096", "W/m²/µm", "Absolute spectrum error (all sources)."
   "FLUX_ABS_VAR_POISSON_FIB#", "Image", "Float32", "Nord x 4096", "(W/m²/µm)²", "Absolute spectrum variance (Poisson)."
   "FLUX_ABS_VAR_RNOISE_FIB#", "Image", "Float32", "Nord x 4096", "(W/m²/µm)²", "Absolute spectrum variance (read noise)."
   "DRIFT_PIX_FIB#", "Image", "Float64", "Nord x 4096", "e-/s", "Drift measured by etalon in pixels for each pixel."
   "DRIFT_ERR_PIX_FIB#", "Image", "Float64", "Nord x 4096", "e-/s", "Drift error in pixels for each pixel."
   "DRIFT_WAVE_FIB3#", "Image", "Float64", "Nord x 4096", "e-/s", "Drift measured by etalon in microns for each pixel."
   "DRIFT_ERR_WAVE_FIB#", "Image", "Float64", "Nord x 4096", "e-/s", "Drift error in microns for each pixel."
   "RVCCF_FIB#", "Table", "See below", "See below", "See below", "RVs for each chunk."
   "RVCCFAGG_FIB#", "Table", "See below", "See below", "See below", "RVs aggregated across chunks."


.. csv-table:: **BARY_CORR Columns**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "ORDER", "Int32", "Echelle order number."
   "WAVE", "Float64", "Wavelength of the middle of order in microns."
   "JDUTC", "Float64", "Julian date in universal coordinate time (UTC) of the non-flux-weighted midpoint."
   "JDUTCW", "Float64", "Julian date in universal coordinate time (UTC) of the flux-weighted midpoint."
   "HJD", "Float64", "Heliocentric Julian date of the non-flux-weighted midpoint."
   "HJDW", "Float64", "Heliocentric Julian date of the flux-weighted midpoint."
   "BJD", "Float64", "Barycentric Julian date of the non-flux-weighted midpoint."
   "BJDW", "Float64", "Barycentric Julian date of the flux-weighted midpoint."
   "BERV", "Float64", "Barycentric velicity correction for the non-flux-weighted midpoint in m/s."
   "BERVW", "Float64", "Barycentric velicity correction for the flux-weighted midpoint in m/s."


.. csv-table:: **RV Columns (Chunks)**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "BJDW", "Float64", "Barycentric Julian date of the flux-weighted midpoint."
   "BERVW", "Float64", "Barycentric velicity correction for the flux-weighted midpoint in m/s."
   "RV", "Float64", "The RV in m/s."
   "RV_ERR", "Float64", "The RV uncertainty in m/s."
   "WAVE", "Float64", "Wavelength of the middle of order in microns."
   "SPECRES", "Float64", "Average spectral resolution in this order."
   "TELL_ABS", "Float64", "Average telluric absorption in this order."
   "BIS", "Float64", "Bisector inverse slope (BIS) in m/s."
   "RV_ABS", "Float64", "Absolute RV measured in m/s."
   "RV_ABS_ERR", "Float64", "Error in absolute RV in m/s."


.. csv-table:: **RV Columns (Aggregated)**
   :header: "Name", "Data Type", "Description"
   :widths: 20, 15, 65

   "BJDW", "Float64", "Average barycentric Julian date of the flux-weighted midpoint."
   "BERVW", "Float64", "Barycentric velicity correction for the flux-weighted midpoint in m/s."
   "RV", "Float64", "RV in m/s."
   "RV_ERR_AGG", "Float64", "RV uncertainty from quadrature sum of individual RV errors."
   "RV_ERR_DIST", "Float64", "RV uncertainty from distribution of individual RVs assuming normality."
   "AIRMASS", "Float64", "Airmass at the midpoint of this observation."
   "BIS", "Float64", "Bisector inverse slope (BIS) in m/s."
   "RV_ABS", "Float64", "Absolute RV measured in m/s."
   "RV_ABS_ERR", "Float64", "Error in absolute RV in m/s."