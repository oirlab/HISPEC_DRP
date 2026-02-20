# from __future__ import annotations
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from ..datamodels import DataModel

# __all__ = ['generate_koa_id', 'generate_koa_filehandle']

# def generate_koa_id(input : 'DataModel' | dict) -> str:
#     """
#     Generate a KOA ID from a DataModel or MetaModel dictionary.

#     Parameters
#     ----------
#     input : DataModel | dict
#         The HISPEC DataModel instance or metadata dictionary.
#     """
#     from ..datamodels import DataModel
#     if isinstance(input, DataModel):
#         return input.meta.filename
#     elif isinstance(input, dict):
#         return input['filename']
#     else:
#         raise TypeError("Model must be an instance of DataModel or metadata dictionary")

# def generate_koa_filehandle(model : 'DataModel' | dict) -> str:
#     from ..datamodels import DataModel
#     if isinstance(model, DataModel):
#         return model.meta.filename
#     elif isinstance(model, dict):
#         return model['filename']
#     else:
#         raise TypeError("Model must be an instance of DataModel or metadata dictionary")