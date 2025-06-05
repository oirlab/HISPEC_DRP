from ..base_model import DataModel
from datetime import datetime
import astropy.time

class HISPECDataModel(DataModel):

    HEADER_SCHEMAS = [
        'hispec_core.yaml',
    ]

    @staticmethod
    def _generate_filename(
        spectrograph : str,
        date_time : str,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC data product.
        Format: {spectrograph}.{YYYYMMDD}.{SSSSS.SS}_{suffix}.fits

        Args:
            spectrograph (str): The spectrograph name.
            date_time (str): The properly formated datetime of the observation in .
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
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"{s}.{date_time}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        date_time = self.meta.exposure.datetime_start
        spectrograph = self.meta.instrument.spectrograph
        if suffix is None:
            if self.data_level_suffix is not None:
                suffix = self.data_level_suffix
        else:
            if self.data_level_suffix is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            suffix=suffix
        )
    