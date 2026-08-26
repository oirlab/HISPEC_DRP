from ..base_primitive import BasePrimitive
from ... import datamodels

from ...calibrations import HISPECCalibrationStore
from hispecdrp import __version__

import os

__all__ = ['RegisterCalibrationPrimitive']


class RegisterCalibrationPrimitive(BasePrimitive):
    """
    Registers a calibration model in the local calibration database.

    Parameters
    ----------
    input : CalibrationModel
        The calibration model to be registered in the local database.
    origin : str, optional
        The origin to register the calibration under. Default is 'local'.

    Returns
    -------
    output : CalibrationModel
        The same calibration model that was registered.
    """

    PARAMETERS = {
        'origin' : 'LOCAL',
    }

    def _perform(self):
        cal_model = datamodels.open(self.dataset)

        assert isinstance(
            cal_model, datamodels.CalibrationModel
        ), f"Input model must be a valid CalibrationModel, got {type(cal_model)}"
        
        with HISPECCalibrationStore(
            instrument_name=cal_model.meta.instrument_name,
            connect_remote=False
        ) as store:
            
            local_filepath, cal_record = store.register_calibration(
                cal_model, origin=self.origin
            )

        return dict(
            output=cal_model,
            cal_record=cal_record,
            local_filepath=local_filepath
        )