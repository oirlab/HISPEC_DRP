from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..datamodels import DataModel, MetaModel

__all__ = ['generate_koa_id', 'generate_koa_filehandle']

def generate_koa_id(model : 'DataModel' | 'MetaModel') -> str:
    from ..datamodels import DataModel, MetaModel
    if isinstance(model, DataModel):
        return model.meta.filename
    elif isinstance(model, MetaModel):
        return model.filename
    else:
        raise TypeError("Model must be an instance of DataModel or MetaModel")

def generate_koa_filehandle(model : 'DataModel' | 'MetaModel') -> str:
    from ..datamodels import DataModel, MetaModel
    if isinstance(model, DataModel):
        return model.meta.filename
    elif isinstance(model, MetaModel):
        return model.filename
    else:
        raise TypeError("Model must be an instance of DataModel or MetaModel")