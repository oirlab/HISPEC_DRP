from .base_model import DataModel

class PARVIDataModel(DataModel):

    @property # EXAMPLE OF PROPERTY
    def itime(self):
        return self.headers['PRIMARY']['EXPTIME']