from typing import get_args, get_origin, Union
from types import UnionType
from koa_middleware import CalibrationORM
from sqlalchemy import Column, Integer, String, Float, Boolean
from pydantic.fields import FieldInfo

from ..datamodels.meta_data import MetaModel

__all__ = ['get_calibration_orm_class', 'pydantic_field_to_sqlalchemy_column', 'get_orm_field_name']


def get_calibration_orm_class(meta : MetaModel) -> type[CalibrationORM]:
    model_attrs = {}
    model_attrs['__tablename__'] = meta.instrument_name
    model_attrs['__table_args__'] = {'extend_existing': True}
    for attr, field in meta.__class__.model_fields.items():
        if field.json_schema_extra['orm_field']:
            col = pydantic_field_to_sqlalchemy_column(attr, field)
            model_attrs[col.name] = col
    class_name = f"{meta.instrument_name}ORM"
    _class = CalibrationORM.registry._class_registry.get(class_name)
    if _class is not None:
        return _class
    else:
        return type(class_name, (CalibrationORM,), model_attrs)

def pydantic_field_to_sqlalchemy_column(attr_name: str, field: FieldInfo) -> Column:
    """Convert all flattened Pydantic fields into SQLAlchemy columns."""
    TYPE_MAP = {int: Integer, str: String, float: Float, bool: Boolean}
    
    t = field.annotation
    origin = get_origin(t)
    
    # Check if it's a Union (either Union[X, Y] or X | Y syntax)
    if origin is Union or origin is UnionType:
        all_args = get_args(t)
        has_none = type(None) in all_args
        non_none_args = [a for a in all_args if a is not type(None)]
        
        if len(non_none_args) == 1 and has_none:
            t = non_none_args[0]
            nullable = True
        else:
            nullable = False
    else:
        nullable = False
    
    col_type = TYPE_MAP.get(t, String)
    default = field.default
    comment = field.title
    sql_attr_name = get_orm_field_name(attr_name, field)
    if sql_attr_name == "id":
        col = Column(sql_attr_name, col_type, primary_key=True, comment=comment)
    else:
        col = Column(sql_attr_name, col_type, default=default, comment=comment, nullable=nullable)

    return col

def get_orm_field_name(attr: str, field: FieldInfo) -> str | None:
    orm_field = field.json_schema_extra.get('orm_field', False)
    if not orm_field:
        return None
    return orm_field if isinstance(orm_field, str) else attr