
from astropy.io import fits
from .hispec_base import HISPECDataModel

class HISPECL1(HISPECDataModel):
    extensions = {}
    for i in range(1, 4):
        extensions[f'SPEC_UNCAL_BSPEC_FIB{i}']
        extensions[f'SPEC_UNCAL_ERR_ALL_BSPEC_FIB{i}']
        extensions[f'SPEC_UNCAL_ERR_FLOOR_BSPEC_FIB{i}']
        extensions[f'SPEC_DQ_BSPEC_FIB{i}']
        extensions[f'MID_TIME_BSPEC_FIB{i}']
        extensions[f'WS_BSPEC_FIB{i}']
        extensions[f'SPEC_UNCAL_RSPEC_FIB{i}']
        extensions[f'SPEC_UNCAL_ERR_ALL_RSPEC_FIB{i}']
        extensions[f'SPEC_UNCAL_ERR_FLOOR_RSPEC_FIB{i}']
        extensions[f'SPEC_DQ_RSPEC_FIB{i}']
        extensions[f'MID_TIME_RSPEC_FIB{i}']
        extensions[f'WS_RSPEC_FIB{i}']