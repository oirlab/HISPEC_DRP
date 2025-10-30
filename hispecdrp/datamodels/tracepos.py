from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from astropy.io import fits
import numpy as np

__all__ = ['TracePosModel', 'HISPECTracePosModel', 'PARVITracePosModel']


def _create_tracepos_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext[f'COEFF_POS_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'COEFF_FWHM_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'POS_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'FWHM_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
    return ext


class TracePosModel(CalibrationModel):
    CAL_TYPE = 'tracepos'


class HISPECTracePosModel(HISPECCalibrationModel, TracePosModel):
    EXTENSIONS = _create_tracepos_extensions(num_fibers=HISPECCalibrationModel.NUM_FIBERS)


class PARVITracePosModel(PARVICalibrationModel, TracePosModel):
    EXTENSIONS = _create_tracepos_extensions(num_fibers=PARVICalibrationModel.NUM_FIBERS)

