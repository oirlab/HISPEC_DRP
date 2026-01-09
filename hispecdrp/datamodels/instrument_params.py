from .base_model import CalibrationModel

from astropy.io import fits

__all__ = ['InstrumentParametersModel']


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