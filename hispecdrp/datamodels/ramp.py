from .base_model import DataModel
from .extensions import IMAGE_UINT16_HDU, DQ_RAW, DQ_L0
import numpy as np

__all__ = ['RampModel']


class RampModel(DataModel):
    """
    The Teledyne Hawaii HXRG detectors use up-the-ramp (UTR) non-destructive reads, so raw data is a cube where each 2D slice is a single read.

    .. csv-table::
        :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        DATA, Image, UInt16, "Nreads x 4096 x 4160", DN, Raw UTR data.
        DQ_RAW, Image, UInt8, "4096 x 4160", None, Data quality flags unique to each pixel and read.
        DQ, Image, UInt32, "Nreads x 4096 x 4160", None, Data quality flags that apply to all reads.
    """
    EXTENSIONS = {'DATA': IMAGE_UINT16_HDU, **DQ_RAW, **DQ_L0}

    @property
    def data_level_suffix(self) -> str:
        return "raw"

    def get_delta_readtimes(self, n_channels=None, vertical = None) -> np.ndarray:
        """
        Get the time for each read in the ramp in seconds relative to the first read.
        The start time of the first read in MJD is given by get_first_read_mjd().

        Notes
        -----
        This implementation does not model any specific instrument and is meant for testing only.

        Parameters
        ----------
        n_channels : int, optional
            Number of channels the detector is read out with.
        vertical : bool, optional
            If True, the channels are assumed to be vertical. If False, the readout channels are assumed to be horizontal.
            Within each channel, we assume the pixels are readout row by row assuming vertical channels; meaning channels are read in the width direction.

        Returns
        -------
        readtimes : np.ndarray
            1D array of length n_reads with the time of each read in seconds relative to the first read.
        """
        n_reads, ny, nx = self.data.shape

        if n_channels is None:
            n_channels = self.meta.n_channels
        if vertical is None:
            vertical = self.meta.channels_are_vertical

        if vertical:
            if nx % n_channels != 0:
                raise ValueError(
                    "nx is not divisible by n_channels. Detector size and number of channels are incompatible.")
            nx_chan = nx // n_channels
            ny_chan = ny
        else:
            if ny % n_channels != 0:
                raise ValueError(
                    "ny is not divisible by n_channels. Detector size and number of channels are incompatible.")
            ny_chan = ny // n_channels
            nx_chan = nx

        time_single_read = (ny_chan * nx_chan) / self.meta.clock_rate
        readtimes = np.arange(n_reads, dtype=np.float32) * time_single_read
        return readtimes

    def get_first_read_mjd(self, n_channels=None, vertical = None) -> np.ndarray:
        """
        Get the MJD times of the first reads for each pixel as a 2D array.

        Notes
        -----
        This implementation does not model any specific instrument and is meant for testing only.


        Parameters
        ----------
        n_channels : int, optional
            Number of channels the detector is read out with.
        vertical : bool, optional
            If True, the channels are assumed to be vertical. If False, the readout channels are assumed to be horizontal.
            Within each channel, we assume the pixels are readout row by row assuming vertical channels; meaning channels are read in the width direction.

        Returns
        -------
        first_read_mjd : np.ndarray
            2D array of shape (ny, nx) with the MJD time of the first read for each pixel.
        """

        if n_channels is None:
            n_channels = self.meta.n_channels
        if vertical is None:
            vertical = self.meta.channels_are_vertical

        if vertical:
            _data = self.data
        else:
            _data = self.data.transpose(0, 2, 1)

        n_reads, ny, nx = _data.shape

        if nx % n_channels != 0:
            raise ValueError(f"Detector size {nx} and number of channels {n_channels} are incompatible.")
        nx_chan = nx // n_channels
        ny_chan = ny

        first_read_mjd_channel = self.meta.mjd_start \
                        + np.arange(ny_chan*nx_chan, dtype=np.float32).reshape(ny_chan, nx_chan) \
                        / self.meta.clock_rate

        # TODO: Are adjacent channels flipped in the direction of readout?
        # TODO:   (i.e. channel 1 reads left to right, channel 2 reads right to left, etc.)
        first_read_mjd = np.tile(first_read_mjd_channel, (1, n_channels))

        if vertical:
            return first_read_mjd
        else:
            return first_read_mjd.T
