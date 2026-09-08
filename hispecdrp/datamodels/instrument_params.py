from .base_model import CalibrationModel

from astropy.io import fits

__all__ = ['InstrumentParametersModel']


class InstrumentParametersModel(CalibrationModel):
    """
    Instrument parameters like LFC set points, etc. that are not expected to change frequently but are not necessarily constant over the lifetime of the instrument.
    """


    EXTENSIONS = {
        'PARAMS': {
            'hdu_type' : fits.BinTableHDU,
            'columns' : {
                'name': str,
                'value': float,
                'unit': str,
                'comment': str,
            },
            'title': 'Instrument parameters.',
        }
    }
    CAL_TYPE = 'instrument_params'