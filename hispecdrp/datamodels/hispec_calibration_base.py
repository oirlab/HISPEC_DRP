from .hispec_base import HISPECDataModel

__all__ = ['HISPECCalibrationModel']


class HISPECCalibrationModel(HISPECDataModel):

    HEADER_SCHEMAS = ['calibration.yaml']

    CAL_TYPE = None # String: The type of calibration data.

    def on_init(self):
        super().on_init()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE


    @staticmethod
    def _generate_filename(
        spectrograph : str,
        datetime : str,
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
        if master:
            cal_type = f"master-{cal_type}"
        filename = f"HS_{spectrograph.upper()}_{datetime}_{cal_type}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    

    def generate_filename(
            self,
            spectrograph : str | None = None,
            datetime : str | None = None,
            cal_type : str | None = None,
            master : bool = False,
            suffix : str | None = None,
        ):
        """
        See _generate_filename for details.
        """
        spectrograph = spectrograph or self.meta.instrument.spectrograph
        if datetime is None:
            datetime = self.meta.date_obs \
                       + 'T' \
                       + self.meta.utc.replace(':', '')[0:6] # HHMMSS, remove colons and decimal seconds
        cal_type = cal_type or self.CAL_TYPE
        return self._generate_filename(
            spectrograph=spectrograph,
            datetime=datetime,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )