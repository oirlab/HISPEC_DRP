
from .base_model import DataModel
from .hispec_base import HISPECDataModel
from .parvi_base import PARVIDataModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_barycorr_extensions, create_starmodel_extensions, create_throughput_extensions, create_rv_extensions, create_sky_extensions, create_telluric_extensions, create_trans_extensions, create_rv_extensions, create_drift_extensions, create_etalonfitparam_extensions

__all__ = ['L2Model', 'HISPECL2Model', 'PARVIL2Model']

def create_l2_extensions(num_fibers : int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC_UNCAL'))
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_drift_extensions(fiber=fiber))
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

    @property
    def data_level_suffix(self):
        return 'specuncal'
    

class HISPECL2Model(HISPECDataModel, L2Model):
    EXTENSIONS = create_l2_extensions(num_fibers=HISPECDataModel.NUM_FIBERS)


class PARVIL2Model(PARVIDataModel, L2Model):
    EXTENSIONS = create_l2_extensions(num_fibers=PARVIDataModel.NUM_FIBERS)