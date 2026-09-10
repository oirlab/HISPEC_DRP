from .base_model import CalibrationModel

from .extensions import TABLE_HDU

__all__ = ['InstrumentParametersModel']


class InstrumentParametersModel(CalibrationModel):
    """
    Instrument parameters like LFC set points, etc. that are not expected to change frequently but are not necessarily constant over the lifetime of the instrument.
    """
    
    EXTENSIONS = {
        'PARAMS': TABLE_HDU({
            'name': str,
            'value': float,
            'unit': str,
            'comment': str,
        }, title='Instrument parameters.'),
    }
    CAL_TYPE = 'instrument_params'