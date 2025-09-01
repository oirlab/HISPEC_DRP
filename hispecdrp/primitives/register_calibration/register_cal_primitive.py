from ..base_primitive import BasePrimitive
from ... import datamodels

from koa_middleware import CalibrationStore
from hispecdrp import __version__

import os

__all__ = ['RegisterCalibrationPrimitive']


class RegisterCalibrationPrimitive(BasePrimitive):
    """
    Registers a calibration model in the local calibration database.
    """

    ARGUMENTS = {
        'generate_cal_id' : False,
    } # NOTE: Soon to add additional args here as interface with KOA is developed.

    def _perform(self):
        cal_model = datamodels.open(self.input)
        assert isinstance(
            cal_model,
            (datamodels.HISPECCalibrationModel, datamodels.PARVICalibrationModel)
        ), f"Input model must be a valid CalibrationModel, got {type(cal_model)}"
        with CalibrationStore(orm_class=cal_model.meta.get_orm_class()) as store:
            output_dir = os.path.join(store.cache_dir, 'calibrations') + os.sep
            local_filepath = cal_model.save(output_dir=output_dir)
            cal_orm = cal_model.meta.to_orm()
            store.register_local_calibration(cal_orm)
        return dict(output=cal_model, calibration_orm=cal_orm, local_filepath=local_filepath)