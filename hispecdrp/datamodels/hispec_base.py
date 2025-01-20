from .base_model import DataModel

class HISPECDataModel(DataModel):

    HEADER_SCHEMAS = [
        'core.yaml',
        'instrument.yaml',
        'exposure.yaml',
        'target.yaml',
        'drp_steps.yaml',
        'calibration_files.yaml',
    ]

    @staticmethod
    def _generate_filename(
        spectrograph : str,
        date : str,
        level : str | int,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC data product.

        Format: HS_{spectrograph}_{date}_LVL{level}_{suffix}.fits
            date : YYYYMMDDTHHMMSS
            level : data level
            suffix : optional suffix

        Args:
            spectrograph (str): The spectrograph name.
            date (str): The date and time of the observation.
            level (str | int): The data level.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        filename = f"HS_{spectrograph.upper()}_{date}_LVL{level}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(
            self,
            spectrograph : str | None = None,
            date : str | None = None,
            level : str | int | None = None,
            suffix : str | None = None,
        ):
        """
        See _generate_filename for details.
        """
        spectrograph = spectrograph or self.instrument.spectrograph
        date = date or self.date.replace(':', '').replace('-', '')[0:15]
        level = level or self.data_level
        return self._generate_filename(
            spectrograph=spectrograph,
            date=date,
            level=level,
            suffix=suffix
        )