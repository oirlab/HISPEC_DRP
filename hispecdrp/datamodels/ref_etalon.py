from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_etalonfitparam_extensions, create_drift_extensions, create_barycorr_extensions

__all__ = ['ReferenceEtalonModel']

def _create_refetalon_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_barycorr_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_etalonfitparam_extensions(fiber=fiber))
        ext.update(create_drift_extensions(fiber=fiber))
    return ext


class ReferenceEtalonModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    Contains the reference etalon and corresponding parameters for wavelength drift corrections.

    Extensions
    ----------

    HDU Name                HDU Type   Data Type   Dimensions      Units      Description
    --------                --------   ---------   ----------      -----      -----------
    SPEC_FIB#               Image      Float32     Nord x 4096     e-/s       Corresponding etalon spectrum.
    SPEC_ERR_FIB#           Image      Float32     Nord x 4096     e-/s       Etalon spectrum error.
    SPEC_VAR_POISSON_FIB#   Image      Float32     Nord x 4096     e-/s       Etalon spectrum Poisson variance.
    SPEC_VAR_RNOISE_FIB#    Image      Float32     Nord x 4096     e-/s       Etalon spectrum read noise variance.
    SPEC_DQ_FIB#            Image      UInt32      Nord x 4096     None       Etalon spectrum data quality.
    MID_TIME_FIB#           Image      Float64     Nord x 4096     s          Mid time for each exposure.
    WAVE_FIB#               Image      Float64     Nord x 4096     nm         Etalon spectrum wavelength grid.
    ETALON_FITPARAMS_FIB#   Table      See below   See below       See below  Etalon fit parameters for each mode, order, and fiber.
    DRIFT_FIB#              Table      See below   See below       See below  Drift correction table for each fiber.
    """
    CAL_TYPE = 'ref_etalon'

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_refetalon_extensions(self.config.num_fibers))
        return ext

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
            if self.has_extension_value(f'wave_fib{fiber}'):
                return f'wave_fib{fiber}'
        return None