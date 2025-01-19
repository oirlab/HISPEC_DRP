from hispecdrp import datamodels
from astropy.time import Time

def add_metadata(model : datamodels.HISPECDataModel):
    model.meta.instrument.name = "HISPEC"
    model.meta.file_type = 'FITS'
    model.meta.telescope = 'Keck-II'
    model.meta.organization = 'WMKO'
    model.meta.drp_version = '0.0.1'
    model.meta.instrument.era = '0.0.1'
    time = Time(model.meta.exposure.jd_start, format='jd')
    model.meta.date = time.isot
    model.meta.date_created = time.isot

    model.meta.exposure.date_start = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.date_mid = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.date_end = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.time_start = time.datetime.strftime('%H:%M:%S.%f')[:-3]
    model.meta.exposure.time_mid = time.datetime.strftime('%H:%M:%S.%f')[:-3]
    model.meta.exposure.time_end = time.datetime.strftime('%H:%M:%S.%f')[:-3]