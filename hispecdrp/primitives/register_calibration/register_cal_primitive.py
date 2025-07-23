from ..base_primitive import BasePrimitive
from ... import datamodels
from keckdrpframework.models.arguments import Arguments

from koa_middleware import CalibrationStore

__all__ = ['RegisterCalibrationPrimitive']


class RegisterCalibrationPrimitive(BasePrimitive):
    """
    Registers a calibration model in the local calibration database.
    """

    ARGUMENTS = {
    } # NOTE: Soon to add additional args here as interface with KOA is developed.

    def _perform(self):
        cal_model = datamodels.open(self.args.name)
        assert isinstance(
            cal_model,
            (datamodels.HISPECCalibrationModel, datamodels.PARVICalibrationModel)
        ), f"Input model must be a valid CalibrationModel, got {type(cal_model)}"
        orm_class = self.get_orm_class(cal_model)
        with CalibrationStore(orm_class=orm_class) as store:
            calibration = store.register_local_calibration(cal_model) # Probably passing args here.
        return Arguments(name=cal_model, calibration=calibration)
