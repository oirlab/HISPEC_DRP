
from .base_model import DataModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_barycorr_extensions, create_starmodel_extensions, create_throughput_extensions, create_rv_extensions, create_sky_extensions, create_telluric_extensions, create_trans_extensions, create_rv_extensions, create_drift_extensions, create_etalonfitparam_extensions

__all__ = ['L2Model']

def _create_l2_extensions(num_fibers : int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC_UNCAL'))
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_drift_extensions(fiber=fiber))
        ext.update(create_midtime_extensions(fiber=fiber))
        ext.update(create_barycorr_extensions(fiber=fiber))
        ext.update(create_starmodel_extensions(fiber=fiber))
        ext.update(create_throughput_extensions(fiber=fiber))
        ext.update(create_rv_extensions(fiber=fiber))
        ext.update(create_sky_extensions(fiber=fiber))
        ext.update(create_telluric_extensions(fiber=fiber))
        ext.update(create_trans_extensions(fiber=fiber))
        ext.update(create_etalonfitparam_extensions(fiber=fiber))
    return ext

class L2Model(DataModel):
    """
    HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements. Data products from BSPEC and RSPEC are still kept separate.

    Extensions
    ----------

    HDU Name                        HDU Type   Data Type   Dimensions      Units            Description
    --------                        --------   ---------   ----------      -----            -----------
    SPEC_UNCAL_FIB#                 Image      Float32     Nord x 4096     e-/s             L1 spectrum.
    SPEC_UNCAL_ERR_FIB#             Image      Float32     Nord x 4096     e-/s             L1 spectrum error (all sources).
    SPEC_UNCAL_VAR_POISSON_FIB#     Image      Float32     Nord x 4096     e-/s             L1 spectrum variance (Poisson).
    SPEC_UNCAL_VAR_RNOISE_FIB#      Image      Float32     Nord x 4096     e-/s             L1 spectrum variance (read noise).
    SPEC_UNCAL_DQ_FIB#              Image      Float32     Nord x 4096     e-/s             L1 spectrum data quality.
    SPEC_FIB#                       Image      Float32     Nord x 4096     e-/s (or W/m²/µm) Calibrated spectrum.
    SPEC_ERR_FIB#                   Image      Float32     Nord x 4096     e-/s             Calibrated spectrum error (all sources).
    SPEC_VAR_POISSON_FIB#           Image      Float32     Nord x 4096     (e-/s)²          Calibrated spectrum variance (Poisson).
    SPEC_VAR_RNOISE_FIB#            Image      Float32     Nord x 4096     (e-/s)²          Calibrated spectrum variance (read noise).
    SPEC_DQ_FIB#                    Image      Uint32      Nord x 4096     None             Calibrated spectrum data-quality flags.
    MID_TIME_FIB#                   Image      Float32     Nord x 4096     MJD              Flux-weighted exposure mid-point times.
    BARY_CORR_FIB#                  Table      See below   See below       See below        Barycentric correction information.
    WAVE_FIB#                       Image      Float64     Nord x 4096     micron           Drift corrected wavelength solution in rest frame of the instrument.
    TRANS_FIB#                      Image      Float32     Nord x 4096     None             Transmission correction.
    TRANS_ERR_FIB#                  Image      Float32     Nord x 4096     None             Transmission correction error (all sources).
    SKY_FIB#                        Image      Float32     Nord x 4096     None             Sky spectrum or model.
    SKY_ERR_FIB#                    Image      Float32     Nord x 4096     None             Sky spectrum error (all sources).
    STAR_MOD_FIB#                   Image      Float32     Nord x 4096     W/m²/µm         Stellar model spectrum Doppler shifted according to Barycenter OR RVs.
    E2E_TPUT_FIB#                   Image      Float32     Nord x 4096     None             End-to-end throughput derived from STAR_MOD.
    TEL_FIB#                        Image      Float32     Nord x 4096     None             Telluric model spectrum.
    TEL_ERR_FIB#                    Image      Float32     Nord x 4096     None             Telluric model spectrum error (all sources).
    FLUX_ABS_FIB#                   Image      Float32     Nord x 4096     W/m²/µm         Spectrum in absolute flux units.
    FLUX_ABS_ERR_FIB#               Image      Float32     Nord x 4096     W/m²/µm         Absolute spectrum error (all sources).
    FLUX_ABS_VAR_POISSON_FIB#       Image      Float32     Nord x 4096     (W/m²/µm)²      Absolute spectrum variance (Poisson).
    FLUX_ABS_VAR_RNOISE_FIB#        Image      Float32     Nord x 4096     (W/m²/µm)²      Absolute spectrum variance (read noise).
    DRIFT_PIX_FIB#                  Image      Float64     Nord x 4096     e-/s             Drift measured by etalon in pixels for each pixel.
    DRIFT_ERR_PIX_FIB#              Image      Float64     Nord x 4096     e-/s             Drift error in pixels for each pixel.
    DRIFT_WAVE_FIB3#                Image      Float64     Nord x 4096     e-/s             Drift measured by etalon in microns for each pixel.
    DRIFT_ERR_WAVE_FIB#             Image      Float64     Nord x 4096     e-/s             Drift error in microns for each pixel.
    RVCCF_FIB#                      Table      See below   See below       See below        RVs for each chunk.
    RVCCFAGG_FIB#                   Table      See below   See below       See below        RVs aggregated across chunks.
    """

    # def __init__(self, *args, **kwargs):
    #     config = kwargs.get('config', None)
    #     if config is None:
    #         config = self._get_instrument_config(
    #             meta=kwargs.get('meta', None),
    #             filepath=kwargs.get('filepath', None),
    #             instrument_name=kwargs.get('instrument_name', None),
    #         )
    #     self.EXTENSIONS = self.EXTENSIONS.copy()
    #     self.EXTENSIONS.update(_create_l2_extensions(config.num_fibers))
    #     super().__init__(*args, **kwargs)

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_l2_extensions(self.config.num_fibers))
        return ext

    @property
    def data_level_suffix(self):
        return 'spec'