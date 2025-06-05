from ..base_model import DataModel
import astropy.time
from datetime import datetime

__all__ = ['PARVIDataModel']

class PARVIDataModel(DataModel):

    HEADER_SCHEMAS = [
        'parvi_core.yaml',
    ]

    @staticmethod
    def _generate_filename(
        date_time : str,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a PARVI data product.

        Format: PR_{date}_{suffix}.fits
            - datetime : YYYYMMDDTHHMMSS
            - suffix : optional suffix

        Args:
            date_time (str): The date and time of the observation.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"PR.{date_time}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        date_time = self.meta.exposure.datetime_start
        if suffix is None:
            if self.data_level_suffix is not None:
                suffix = self.data_level_suffix
        else:
            if self.data_level_suffix is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"
        return self._generate_filename(
            date_time=date_time,
            suffix=suffix
        )
    