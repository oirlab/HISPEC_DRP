from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from astropy.io import fits

__all__ = ['HISPECInstrumentParametersModel', 'PARVIInstrumentParametersModel']


class InstrumentParametersModel(CalibrationModel):
    EXTENSIONS = {
        'PARAMS': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : {
                'name': str,
                'value': float,
                'unit': str,
                'comment': str,
            }
        }
    }
    CAL_TYPE = 'instrument_params'


class HISPECInstrumentParametersModel(HISPECCalibrationModel, InstrumentParametersModel):
    pass


class PARVIInstrumentParametersModel(PARVICalibrationModel, InstrumentParametersModel):
    pass
