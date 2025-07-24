from .hispec_calibration_orm import *
from .parvi_calibration_orm import *
from .parvi_selectors import *
from .hispec_selectors import *

from .. import datamodels

_local_dict = dict(locals())
DATAMODEL_ORM_MAP = {}
for _name, _orm_cls in _local_dict.items():
    if _name.endswith('ORM'):
        _prefix = _name[:-len('CalibrationORM')]
        _datamodel_name = _prefix + 'DataModel'
        _datamodel_cls = datamodels.MODEL_REGISTER.get(_datamodel_name)
        if _datamodel_cls:
            DATAMODEL_ORM_MAP[_datamodel_cls] = _orm_cls

SELECTOR_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Selector')}