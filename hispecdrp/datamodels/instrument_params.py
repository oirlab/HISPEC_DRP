from .base_model import CalibrationModel

from astropy.io import fits

__all__ = ['InstrumentParametersModel']


class InstrumentParametersModel(CalibrationModel):
    """
    Instrument parameters like LFC set points, etc. that are not expected to change frequently but are not necessarily constant over the lifetime of the instrument.

    Extensions
    ----------

    HDU Name   HDU Type   Data Type   Dimensions  Units     Description
    --------   --------   ---------   ----------  -----     -----------
    PARAMS     Table      See docs    See docs    See docs  Instrument parameters.
    """


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