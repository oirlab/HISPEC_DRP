from .base_model import DataModel
from .hispec_base import HISPECDataModel
from .parvi_base import PARVIDataModel
from .extensions import IMAGE_UINT16_HDU, DQ_RAW, DQ_L0
import numpy as np

__all__ = ['RampModel', 'HISPECRampModel', 'PARVIRampModel']


class RampModel(DataModel):
    EXTENSIONS = {'DATA': IMAGE_UINT16_HDU, **DQ_RAW, **DQ_L0}



    @property
    def data_level_suffix(self) -> str:
        return "raw"

    def get_delta_readtimes(self) -> float:
        """
        Get the time for each read in the ramp in seconds relative to the first read.
        The start time of the first read in MJD is given by get_first_read_mjd().

        Note: This implementation does not model any specific instrument and is meant for testing only.

        """

        n_reads, ny,nx = self.data.shape
        time_single_read = (ny*nx)/self.meta.clock_rate
        readtimes = np.arange(n_reads, dtype = np.float32) * time_single_read
        return readtimes

    def get_first_read_mjd(self) -> float:
        """
        Get the MJD times of the first reads for each pixel as a 2D array.

        Note: This implementation does not model any specific instrument and is meant for testing only.
        """
        n_reads, ny, nx = self.data.shape
        first_read_mjd = self.meta.mjd_start+np.arange(ny*nx, dtype = np.float32).reshape(ny, nx)/self.meta.clock_rate
        return first_read_mjd

class HISPECRampModel(HISPECDataModel, RampModel):
    pass


class PARVIRampModel(PARVIDataModel, RampModel):
    pass