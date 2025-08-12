from ..base_primitive import BasePrimitive
from ... import datamodels

from koa_middleware import CalibrationStore

__all__ = ['RegisterCalibrationPrimitive']


class RegisterCalibrationPrimitive(BasePrimitive):
    """
    Registers a calibration model in the local calibration database.
    """

    ARGUMENTS = {
    } # NOTE: Soon to add additional args here as interface with KOA is developed.

    def _perform(self):
        cal_model = datamodels.open(self.input)
        assert isinstance(
            cal_model,
            (datamodels.HISPECCalibrationModel, datamodels.PARVICalibrationModel)
        ), f"Input model must be a valid CalibrationModel, got {type(cal_model)}"
        with CalibrationStore(orm_class=cal_model.CAL_ORM_CLASS) as store:
            local_filepath, calibration_orm = store.register_local_calibration(cal_model) # Probably passing args here.
        return dict(output=cal_model, calibration_orm=calibration_orm, local_filepath=local_filepath)
