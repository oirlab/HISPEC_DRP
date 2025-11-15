
from .base_model import DataModel
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


class L1Model(DataModel):

    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(_create_l1_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(_create_l1_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)

    @property
    def data_level_suffix(self):
        return 'specuncal'