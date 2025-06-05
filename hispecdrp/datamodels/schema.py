import yaml
import os
from astropy.io import fits
from .meta_dict import MetaDict
import pandas as pd
import importlib.resources


__all__ = ['load_schema', 'create_header', 'merge_schemas', 'walk_schema']


def load_schema(schema_name : str):
    """
    Load a schema from a YAML file.
    """
    if os.path.exists(schema_name):
        schema_path = schema_name
    else:
        schema_path = find_schema_path(schema_name)
    with open(schema_path, 'r') as f:
        return yaml.safe_load(f)
    

def find_schema_path(schema_name : str) -> str:
    """Search all datamodels/*/schemas/ for a schema file by base name."""
    base = importlib.resources.files("hispecdrp.datamodels")
    for subpkg in base.iterdir():
        schemas_dir = subpkg / "schemas"
        if schemas_dir.is_dir():
            candidate = schemas_dir / schema_name
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(f"No valid schema found for: {schema_name}")





# def merge_schemas(schema1 : dict, schema2 : dict):
#     """
#     Merge two schemas, with schema2 taking precedence over schema1.
#     """
#     merged = schema1.copy()
#     for key, value in schema2.items():
#         if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
#             merged[key] = merge_schemas(merged[key], value)
#         else:
#             merged[key] = value
#     return merged

def schema_to_table(schema : dict):
    df = pd.DataFrame(columns=[
        'Keyword', 'HDU', 'Comment', 'Units', 'Datatype',
        'Default', 'Datamodel Property', 'KTL Keyword'
    ])
    for current_path, value in walk_schema(schema, objects=False):
        df.loc[len(df)] = {
            'Keyword' : value['fits_keyword'],
            'HDU' : value.get('fits_hdu', 'PRIMARY'),
            'Comment' : value.get('comment', ''),
            'Units' : value.get('unit', 'None'),
            'Datatype' : value.get('type', 'string'),
            'Default' : value.get('default', 'None'),
            'Datamodel Property' : '.'.join(current_path),
            'KTL' : value.get('ktl_keyword', 'None'),
        }
    return df


def walk_schema(_schema: dict, path: list[str] | None = None, objects: bool = False):
    """
    Walk a schema tree and yield the path and value of each property which
    represents a FITS header entry, not including intermediate levels.

    Args:
        - _schema (dict): The schema to walk.
        - path (list, optional): The path to the current schema level.
        - objects (bool, optional): If True, yield only objects, not individual properties.
    """
    if path is None:
        path = []
    
    for key, value in _schema['properties'].items():
        current_path = path + [key]
        dtype = value.get('type', 'string')
        if dtype == 'object' and 'properties' in value:
            if objects:
                yield (current_path, value)
            yield from walk_schema(value, current_path, objects)
        else:
            yield (current_path, value)