from .base_model import DataModel

#### HERE: NOTE: Look through remaining level properties to use for generating koa filename

class HISPECDataModel(DataModel):

    HEADER_SCHEMAS = [
        'core.yaml',
        'instrument.yaml',
        'exposure.yaml',
        'target.yaml',
        'drp_steps.yaml',
        'calibration_files.yaml',
        'keck_program.yaml',
    ]

    def on_init(self):
        super().on_init()
        if self.meta.data_level is None and self.level is not None:
            self.meta.data_level = self.level

    @staticmethod
    def _generate_filename(
        spectrograph : str,
        date_time : str,
        level : str | int,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC data product.

        Format: HS_{spectrograph}_{date}_LVL{level}_{suffix}.fits
            datetime : YYYYMMDDTHHMMSS
            level : data level
            suffix : optional suffix

        Args:
            spectrograph (str): The spectrograph name.
            date_time (str): The date and time of the observation.
            level (str | int): The data level.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        level = '' if level is None else f"_LVL{level}"
        filename = f"HS_{spectrograph.upper()}_{date_time}{level}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        spectrograph = self.meta.instrument.spectrograph
        date_time = self.meta.date_obs \
                    + 'T' \
                    + self.meta.utc.replace(':', '')[0:6] # HHMMSS, remove colons and decimal seconds
        level = self.meta.data_level or self.level
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            level=level,
            suffix=suffix
        )
    