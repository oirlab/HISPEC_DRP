from .base_model import CalibrationModel
from .extensions import  DQ_L0

__all__ = ['DQModel']


class DQModel(CalibrationModel):
    """
    Extensions
    ----------

    HDU Name  HDU Type   Data Type   Dimensions     Units   Description
    --------  --------   ---------   ----------     -----   -----------
    DQ        Image      UInt32      4096 x 4160    None    Data quality.
    """
    
    EXTENSIONS = DQ_L0
    CAL_TYPE = 'dq'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'NONE')
        self.meta.set_default('target_name', 'NONE')
        for fiber in range(1, self.num_fibers + 1):
            self.meta.set_default(f'object_fiber{fiber}', 'NONE')