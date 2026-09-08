from .base_model import CalibrationModel
from .extensions import  DQ_L0

__all__ = ['DQModel']


class DQModel(CalibrationModel):
    """
    """

    EXTENSIONS = {**DQ_L0}
    CAL_TYPE = 'dq'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'NONE')
        self.meta.set_default('target_name', 'NONE')
        for fiber in range(1, self.num_fibers + 1):
            self.meta.set_default(f'object_fiber{fiber}', 'NONE')