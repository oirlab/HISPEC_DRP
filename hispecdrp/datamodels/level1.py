from .base_model import DataModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_barycorr_extensions

__all__ = ['L1Model']

def _create_l1_extensions(num_fibers : int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC_UNCAL'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_midtime_extensions(fiber=fiber))
        ext.update(create_barycorr_extensions(fiber=fiber))
    return ext

class L1Model(FiberAttributeProxyMixin, DataModel):
    """
    HISPEC level 1 (L1) data products consist of 1D spectra immediately following spectral extraction. These spectra include a wavelength solution derived from the most recent wavelength calibration available in the calibration database (likely from the LFC). However, they do not yet incorporate the simultaneous etalon data to account for any wavelength drift or any additional corrections.

    Extensions
    ----------

    HDU Name                     HDU Type   Data Type   Dimensions      Units      Description
    --------                     --------   ---------   ----------      -----      -----------
    SPEC_UNCAL_FIB#             Image      Float32     Nord x 4096     e-/s       Extracted 1D spectra.
    SPEC_UNCAL_ERR_SPEC_FIB#    Image      Float32     Nord x 4096     e-/s       Extracted 1D spectra errors (all sources).
    SPEC_UNCAL_VAR_POISSON_FIB# Image      Float32     Nord x 4096     (e-/s)^2   Extracted 1D spectra variance (Poisson).
    SPEC_UNCAL_VAR_RNOISE_FIB#  Image      Float32     Nord x 4096     (e-/s)^2   Extracted 1D spectra variance (read noise).
    SPEC_DQ_FIB#                Image      UInt32      Nord x 4096     None       Data quality.
    MID_TIME_FIB#               Image      Float64     Nord x 4096     MJD        Exposure mid-point times.
    WAVE_FIB#                   Image      Float64     Nord x 4096     micron     Wavelength solution.
    """

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_l1_extensions(self.config.num_fibers))
        return ext

    @property
    def data_level_suffix(self):
        return 'specuncal'
    
    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_uncal_fib{fiber}'):
                return f'spec_uncal_fib{fiber}'
        return None