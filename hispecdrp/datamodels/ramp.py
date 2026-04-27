from .base_model import DataModel
from .extensions import IMAGE_UINT16_HDU, DQ_RAW, DQ_L0
import numpy as np
import logging
import re
from astropy.io import fits

__all__ = ['RampModel']

logger = logging.getLogger(__name__)


class RampModel(DataModel):
    """
    The Teledyne Hawaii HXRG detectors use up-the-ramp (UTR) non-destructive reads, so raw data is a cube where each 2D slice is a single read.

    Extensions
    ----------

    HDU Name   HDU Type   Data Type   Dimensions              Units   Description
    --------   --------   ---------   ----------              -----   -----------
    DATA       Image      UInt16      Nreads x 4096 x 4160    DN      Raw UTR data.
    DQ_RAW     Image      UInt8       Nreads x 4096 x 4160    None    Data quality flags unique to each pixel and read.
    DQ         Image      UInt32      4096 x 4160             None    Data quality flags that apply to all reads.
    """
    EXTENSIONS = {'DATA': IMAGE_UINT16_HDU, **DQ_RAW, **DQ_L0}
    _PARVI_DEFAULT_READTIME_SEC = 1.467

    @property
    def data_level_suffix(self) -> str:
        return "raw"

    def _is_parvi_config(self) -> bool:
        config_name = getattr(self.config, '__name__', '')
        instrument_name = getattr(self.config, 'instrument_name', '')
        return config_name == 'PARVIConfig' and instrument_name == 'PARVI'

    def _get_parvi_timei00_mjd(self, n_reads: int) -> np.ndarray:
        times = getattr(self.meta, 'parvi_timei00_mjd', None)
        if times is None:
            raise ValueError(
                "PARVI timing requires meta.parvi_timei00_mjd. "
                "Populate it with TIMEI00-derived MJD timestamps."
            )

        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("meta.parvi_timei00_mjd must be a 1D array-like of MJD timestamps.")
        if times.size == 0:
            raise ValueError("meta.parvi_timei00_mjd is empty.")
        if times.size != n_reads:
            raise ValueError(
                f"meta.parvi_timei00_mjd length ({times.size}) does not match n_reads ({n_reads})."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("meta.parvi_timei00_mjd contains non-finite values.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("meta.parvi_timei00_mjd must be strictly increasing.")

        return times

    def _get_parvi_readtime_sec(self) -> float:
        readtime_sec = getattr(self.meta, 'parvi_readtime_sec', self._PARVI_DEFAULT_READTIME_SEC)
        if readtime_sec is None:
            readtime_sec = self._PARVI_DEFAULT_READTIME_SEC
        readtime_sec = float(readtime_sec)
        if readtime_sec <= 0:
            raise ValueError("meta.parvi_readtime_sec must be > 0.")
        return readtime_sec

    @staticmethod
    def _unix_ns_to_mjd(unix_ns: np.ndarray) -> np.ndarray:
        # MJD(Unix epoch) = 40587.0 days.
        return np.asarray(unix_ns, dtype=np.float64) / 1e9 / 86400.0 + 40587.0

    def _extract_parvi_timei00_mjd_from_header(self) -> np.ndarray | None:
        if self._filepath is None:
            return None

        pattern = re.compile(r"^TIMEI(\d+)$")
        with fits.open(self._filepath, memmap=True) as hdul:
            header = hdul[0].header

            entries = []
            for key in header:
                m = pattern.match(key)
                if m is None:
                    continue
                try:
                    ns = float(header[key])
                except (TypeError, ValueError):
                    continue
                entries.append((int(m.group(1)), ns))

        if not entries:
            return None

        entries.sort(key=lambda x: x[0])
        unix_ns = np.array([v for _, v in entries], dtype=np.float64)
        return self._unix_ns_to_mjd(unix_ns)

    def _autopopulate_parvi_timing_meta(self):
        if not self._is_parvi_config():
            return

        # Support user-provided Unix-ns sequence by converting it to MJD once.
        if getattr(self.meta, 'parvi_timei00_mjd', None) is None:
            unix_ns = getattr(self.meta, 'parvi_timei00_unix_ns', None)
            if unix_ns is not None:
                self.meta.parvi_timei00_mjd = self._unix_ns_to_mjd(unix_ns).tolist()

        if getattr(self.meta, 'parvi_timef_mjd', None) is None:
            unix_ns_f = getattr(self.meta, 'parvi_timef_unix_ns', None)
            if unix_ns_f is not None:
                self.meta.parvi_timef_mjd = self._unix_ns_to_mjd(unix_ns_f).tolist()

        # If still missing and model was loaded from FITS, parse TIMEIxx header keywords.
        if getattr(self.meta, 'parvi_timei00_mjd', None) is None:
            timei00_mjd = self._extract_parvi_timei00_mjd_from_header()
            if timei00_mjd is not None:
                self.meta.parvi_timei00_mjd = timei00_mjd.tolist()

        # Keep mjd_start consistent with first TIMEI-derived timestamp.
        if getattr(self.meta, 'parvi_timei00_mjd', None):
            try:
                first = float(np.asarray(self.meta.parvi_timei00_mjd, dtype=np.float64)[0])
            except (IndexError, TypeError, ValueError):
                return
            self.meta.mjd_start = first

    def _autopopulate_parvi_primary_data_from_filepath(self):
        if not self._is_parvi_config():
            return
        if self._filepath is None:
            return
        if self.data is not None:
            return

        with fits.open(self._filepath, memmap=False) as hdul:
            primary = hdul[0].data

        if primary is None:
            return

        arr = np.asarray(primary)
        if arr.ndim == 2:
            data_cube = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            data_cube = arr
        else:
            raise ValueError(
                f"Unsupported PARVI primary image dimensionality ({arr.ndim}) for RampModel."
            )

        self.data = data_cube.astype(np.uint16, copy=False)
        if self.dq_raw is None:
            self.dq_raw = np.zeros_like(self.data, dtype=np.uint8)
        if self.dq is None:
            self.dq = np.zeros(self.data.shape[1:], dtype=np.uint32)

    def on_init(self):
        self._autopopulate_parvi_timing_meta()
        super().on_init()
        self._autopopulate_parvi_primary_data_from_filepath()
        self._autopopulate_parvi_timing_meta()

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

        if self._is_parvi_config():
            timei00_meta = getattr(self.meta, 'parvi_timei00_mjd', None)
            if timei00_meta is None:
                logger.warning(
                    "PARVI timing metadata is missing (meta.parvi_timei00_mjd). "
                    "Falling back to clock-rate timing."
                )
            else:
                timei00_mjd = self._get_parvi_timei00_mjd(n_reads=n_reads)
                span_sec = (timei00_mjd[-1] - timei00_mjd[0]) * 86400.0
                nominal_readtime_sec = self._get_parvi_readtime_sec()
                if span_sec < 0:
                    raise ValueError("PARVI read timestamp span is negative.")

                if n_reads == 1:
                    return np.zeros(1, dtype=np.float32)

                # Infer dropped-read behavior from TIMEI00 cadence for diagnostics,
                # then produce uniformly re-spaced times over the observed span.
                deltas_sec = np.diff(timei00_mjd) * 86400.0
                dropped_reads = int(np.sum(np.maximum(np.rint(deltas_sec / nominal_readtime_sec).astype(int) - 1, 0)))
                if dropped_reads > 0:
                    logger.info("Detected %d dropped PARVI reads from TIMEI00 cadence.", dropped_reads)

                return np.linspace(0.0, span_sec, n_reads, dtype=np.float32)

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

        n_reads_native = self.data.shape[0]
        parvi_start_mjd = None
        if self._is_parvi_config():
            timei00_meta = getattr(self.meta, 'parvi_timei00_mjd', None)
            if timei00_meta is None:
                logger.warning(
                    "PARVI timing metadata is missing (meta.parvi_timei00_mjd). Falling back to meta.mjd_start."
                )
            else:
                timei00_mjd = self._get_parvi_timei00_mjd(n_reads=n_reads_native)
                parvi_start_mjd = float(timei00_mjd[0])

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

        if self._is_parvi_config() and parvi_start_mjd is not None:
            mjd_start = parvi_start_mjd
            # Convert per-pixel readout offset to days for PARVI timing.
            first_read_mjd_channel = mjd_start \
                            + np.arange(ny_chan * nx_chan, dtype=np.float64).reshape(ny_chan, nx_chan) \
                            / self.meta.clock_rate / 86400.0
        else:
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
