from .base_model import CalibrationModel
from .extensions import create_specdq_extensions, create_wave_extensions

__all__ = ['SimulatedWavesolModel']


def _create_simwavesol_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_specdq_extensions(fiber=fiber))
    return ext


class SimulatedWavesolModel(CalibrationModel):
    """
    The simulated wavelength solution.

    Extensions
    ----------

    HDU Name    HDU Type   Data Type   Dimensions   Units  Description
    --------    --------   ---------   ----------   -----  -----------
    WAVE_FIB#   Image      Float64     Nord x 4096  nm     Simulated wavelength solution.
    DQ_FIB#     Image      UInt32      Nord x 4096  None   Simulated data quality.
    """
    CAL_TYPE = 'sim_wavesol'

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_simwavesol_extensions(self.config.num_fibers))
        return ext
    
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