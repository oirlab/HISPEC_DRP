from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_specdq_extensions, create_wave_extensions

__all__ = ['SimulatedWavesolModel']


class SimulatedWavesolModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The simulated wavelength solution.

    Extensions
    ----------

    HDU Name    HDU Type   Data Type   Dimensions   Units  Description
    --------    --------   ---------   ----------   -----  -----------
    WAVE_FIB#   Image      Float64     Nord x 4096  nm     Simulated wavelength solution.
    DQ_FIB#     Image      UInt32      Nord x 4096  None   Simulated data quality.
    """

    EXTENSIONS = {
        **create_wave_extensions(),
        **create_specdq_extensions(),
    }

    CAL_TYPE = 'sim_wavesol'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'NONE')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'wave_fib{fiber}'):
                return f'wave_fib{fiber}'
        return None
