from .parvi_base import PARVIDataModel

__all__ = ['PARVICalibrationModel']


class PARVICalibrationModel(PARVIDataModel):

    HEADER_SCHEMAS = ['parvi_calibration.yaml']

    CAL_TYPE = None # String: The type of calibration data.

    def on_init(self):
        super().on_init()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE


    @staticmethod
    def _generate_filename(
        date_time : str,
        cal_type : str,
        master : bool = False,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a PARVI calibration data product.

        Format: PR_{date}_[master-]{cal_type}_{suffix}.fits
            datetime : YYYYMMDDTHHMMSS.
            cal_type : The type of calibration data.
            master : Insert 'master-' before the cal_type if `master=True`.
            suffix : optional suffix

        Args:
            datetime (str): The date and time of the observation.
            cal_type (str): The type of calibration data.
            master (bool, optional): If True, insert 'master-' before the cal_type. Defaults to False.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        if master:
            cal_type = f"master-{cal_type}"
        filename = f"PR.{date_time}.{cal_type}.fits"
        if suffix is not None:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    

    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        date_time = self.meta.exposure.datetime_start
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        date_time = self.meta.exposure.datetime_start
        return self._generate_filename(
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )