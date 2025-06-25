from ..base_primitive import BasePrimitive
from ...calibrations import get_calibration_store
from ... import datamodels
from keckdrpframework.models.arguments import Arguments

from ...calibrations import get_calibration_store

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
        store = get_calibration_store(input_model)
        store.register_local_calibration(input_model) # Probably passing args here.
        return Arguments(name=input_model)