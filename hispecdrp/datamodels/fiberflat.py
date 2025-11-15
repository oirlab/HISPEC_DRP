from .base_model import CalibrationModel
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_trans_extensions

__all__ = ['FiberflatModel']


def _create_fiberflat_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_midtime_extensions(fiber=fiber))
        ext.update(create_trans_extensions(fiber=fiber))
    return ext


class FiberflatModel(CalibrationModel):
    CAL_TYPE = 'fiberflat'
    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(_create_fiberflat_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(_create_fiberflat_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)
