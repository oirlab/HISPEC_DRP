
from .base_model import DataModel
from .hispec_base import HISPECDataModel
from .parvi_base import PARVIDataModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_barycorr_extensions

__all__ = ['HISPECL1Model', 'PARVIL1Model']

def _create_l1_extensions(num_fibers : int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC_UNCAL'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_barycorr_extensions(fiber=fiber))
    return ext


class L1Model(DataModel):

    @property
    def data_level_suffix(self):
        return 'specuncal'
    

class HISPECL1Model(HISPECDataModel, L1Model):
    EXTENSIONS = _create_l1_extensions(num_fibers=HISPECDataModel.NUM_FIBERS)


class PARVIL1Model(PARVIDataModel, L1Model):
    EXTENSIONS = _create_l1_extensions(num_fibers=PARVIDataModel.NUM_FIBERS)