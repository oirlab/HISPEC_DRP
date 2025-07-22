from ..base_primitive import BasePrimitive
from ... import datamodels
from keckdrpframework.models.arguments import Arguments

from koa_middleware import CalibrationStore

__all__ = ['RegisterCalibrationPrimitive']


class RegisterCalibrationPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        calibration (DataModel | str): Calibration to register.
    """

    ARGUMENTS = {
    } # NOTE: Soon to add additional args here as interface with KOA is developed.

    def _perform(self):
        input_model = datamodels.open(self.args.name)
        assert isinstance(
            input_model,
            (datamodels.HISPECCalibrationModel, datamodels.PARVICalibrationModel)
        ), f"Input model must be a valid CalibrationModel, got {type(input_model)}"
        with CalibrationStore(input_model) as store:
            store.register_local_calibration(input_model) # Probably passing args here.
        return Arguments(name=input_model)