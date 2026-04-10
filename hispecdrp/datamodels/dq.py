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