from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from astropy.io import fits
import numpy as np

__all__ = ['TraceProfModel', 'HISPECTraceProfModel', 'PARVITraceProfModel']


def _create_traceprof_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext[f'PROF_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'PROF_ERR_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
    return ext


class TraceProfModel(CalibrationModel):
    CAL_TYPE = 'traceprof'


class HISPECTraceProfModel(HISPECCalibrationModel, TraceProfModel):
    EXTENSIONS = _create_traceprof_extensions(num_fibers=HISPECCalibrationModel.NUM_FIBERS)


class PARVITraceProfModel(PARVICalibrationModel, TraceProfModel):
    EXTENSIONS = _create_traceprof_extensions(num_fibers=PARVICalibrationModel.NUM_FIBERS)

