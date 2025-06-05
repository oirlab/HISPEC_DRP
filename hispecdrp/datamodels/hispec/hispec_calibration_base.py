from .hispec_base import HISPECDataModel
import astropy.time
from datetime import datetime

__all__ = ['HISPECCalibrationModel']


class HISPECCalibrationModel(HISPECDataModel):

    HEADER_SCHEMAS = ['hispec_calibration.yaml']

    CAL_TYPE = None # String: The type of calibration data.

    def on_init(self):
        super().on_init()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE

    def on_save(self):
        """
        Override the save method to ensure that the cal_type is set.
        """
        super().on_save()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE

    @staticmethod
    def _generate_filename(
        spectrograph : str,
        date_time : str,
        cal_type : str,
        master : bool = False,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC calibration data product.

        Format: HS_{spectrograph}_{date}_[master-]{cal_type}_{suffix}.fits
            datetime : YYYYMMDDTHHMMSS.
            cal_type : The type of calibration data.
            master : Insert 'master-' before the cal_type if `master=True`.
            suffix : optional suffix

        Args:
            spectrograph (str): The spectrograph name.
            datetime (str): The date and time of the observation.
            cal_type (str): The type of calibration data.
            master (bool, optional): If True, insert 'master-' before the cal_type. Defaults to False.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        spectrographs = {'BSPEC': 'HB', 'RSPEC': 'HR'}
        spectrograph = spectrograph.upper()
        if spectrograph in spectrographs:
            s = spectrographs[spectrograph]
        else:
            raise ValueError(f"Spectrpgrah must be BSPEC or RSPEC, got {spectrograph}")
        if master:
            cal_type = f"master-{cal_type}"
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"{s}.{date_time}.{cal_type}.fits"
        if suffix is not None:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        spectrograph = self.meta.instrument.spectrograph
        date_time = self.meta.exposure.datetime_start
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )