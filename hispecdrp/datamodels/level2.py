
from .base_model import DataModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_barycorr_extensions, create_starmodel_extensions, create_throughput_extensions, create_rv_extensions, create_sky_extensions, create_telluric_extensions, create_trans_extensions, create_rv_extensions, create_drift_extensions, create_etalonfitparam_extensions

__all__ = ['L2Model']

def create_l2_extensions(num_fibers : int):
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

    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(create_l2_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(create_l2_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)

    @property
    def data_level_suffix(self):
        return 'spec'