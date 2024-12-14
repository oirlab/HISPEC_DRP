from .base_model import DataModel

class HISPECDataModel(DataModel):

    @property # EXAMPLE OF PROPERTY
    def itime(self):
        return self.headers['PRIMARY']['ITIME']