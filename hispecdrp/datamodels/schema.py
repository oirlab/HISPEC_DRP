import yaml
import os
from astropy.io import fits
from .meta_dict import MetaDict


__all__ = ['load_schema', 'create_header', 'merge_schemas', 'walk_schema']


def load_schema(schema_path: str):
    """
    Load a schema from a YAML file.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    schema_dir = os.path.join(module_dir, 'schemas')
    if not os.path.dirname(schema_path):
        schema_path = os.path.join(schema_dir, schema_path)
    else:
        schema_path = os.path.join(module_dir, schema_path)
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, 'r') as f:
        return yaml.safe_load(f)


# def create_header(
#         meta : MetaDict | None = None,
#         extname : str = 'PRIMARY',
#         add_missing : bool = False
#     ) -> fits.Header:
#     """
#     Create an Astropy FITS header dynamically from a schema dict.
#     Includes section headers as COMMENT cards.
    
#     Args:
#         - meta (MetaDict, optional): Dictionary containing data values.
#             Keys should match the schema's property paths.
#             If None, the header will be created with default values.
#         - extname (str, optional): Name of the extension. Default is "PRIMARY".
#         - add_missing (bool, optional): If True, missing entries will be added with None values. Default is False.

#     Returns:
#         astropy.io.fits.Header : FITS header populated according to the schema
#     """
#     header = fits.Header()
#     def process_properties(_meta : MetaDict | None = None):
#         for prop_name, prop_info in _meta._schema['properties'].items():
#             if prop_info.get('type', 'string') == 'object' and extname == prop_info.get('fits_hdu', 'PRIMARY'):
#                 # Add section header as COMMENT
#                 header.append(('COMMENT', '=' * 40), end=True)
#                 header.append(('COMMENT', f' {prop_name.upper()}'), end=True)
#                 if 'comment' in prop_info:
#                     header.append(('COMMENT', f' {prop_info["comment"]}'), end=True)
#                 header.append(('COMMENT', '=' * 40), end=True)
#                 # Process nested properties
#                 if prop_name in _meta:
#                     process_properties(_meta[prop_name])

#             # Handle individual properties
#             elif extname == prop_info.get('fits_hdu', 'PRIMARY'):
#                 keyword = prop_info['fits_keyword']
#                 comment = prop_info.get('comment', '')
#                 unit = prop_info.get('unit')
#                 if unit is not None:
#                     if unit.lower() != 'none':
#                         comment += f" ({unit})"
                
#                 # Get value from data or use default
#                 value = None
#                 if prop_name in _meta:
#                     value = _meta[prop_name]
#                 elif _meta.get('default') is not None:
#                     value = _meta.get('default')
                
#                 # Skip if no value and not adding empty entries
#                 if value is None and not add_missing:
#                     continue

#                 # Add to header
#                 header.append((keyword, value, comment), end=True)
    
#     # Start processing from the root properties
#     if 'properties' in meta._schema:
#         process_properties(meta)
    
#     return header


def create_header(
        meta : MetaDict | None = None,
        extname : str = 'PRIMARY',
        add_missing : bool = False
    ) -> fits.Header:
    """
    Create an Astropy FITS header dynamically from a schema dict.
    Includes section headers as COMMENT cards.
    
    Args:
        - meta (MetaDict, optional): Dictionary containing data values.
            Keys should match the schema's property paths.
            If None, the header will be created with default values.
        - extname (str, optional): Name of the extension. Default is "PRIMARY".
        - add_missing (bool, optional): If True, missing entries will be added with None values. Default is False.

    Returns:
        astropy.io.fits.Header : FITS header populated according to the schema
    """
    header = fits.Header()
    if 'properties' in meta._schema:
        for path, prop_info in walk_schema(meta._schema, objects=True):

            # FITS HDU and kind of entry
            prop_extname = prop_info.get('fits_hdu', 'PRIMARY')
            if extname != prop_extname:
                continue
            dtype = prop_info.get('type', 'string')

            # Prop name
            prop_name = path[-1]

            # Section header
            if dtype == 'object':
                # Add section header as COMMENT
                header.append(('COMMENT', '=' * 40), end=True)
                header.append(('COMMENT', f' {prop_name.upper()}'), end=True)
                if 'comment' in prop_info:
                    header.append(('COMMENT', f' {prop_info["comment"]}'), end=True)
                header.append(('COMMENT', '=' * 40), end=True)

            # Handle individual properties
            else:

                keyword = prop_info['fits_keyword']
                comment = prop_info.get('comment', '')
                unit = prop_info.get('unit')
                default_val = prop_info.get('default')
                if unit is not None:
                    if unit.lower() != 'none':
                        comment += f" ({unit})"
                
                # Get value from data or use default
                meta_val = meta[path]
                if meta_val is not None:
                    value = meta_val
                elif default_val is not None:
                    value = prop_info.get('default')
                else:
                    value = None
                
                # Skip if no value and not adding empty entries
                if value is None and not add_missing:
                    continue

                # Add to header
                header.append((keyword, value, comment), end=True)

    return header


def merge_schemas(schema1 : dict, schema2 : dict):
    """
    Merge two schemas, with schema2 taking precedence over schema1.
    """
    merged = schema1.copy()
    for key, value in schema2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_schemas(merged[key], value)
        else:
            merged[key] = value
    return merged


# def walk_schema(_schema : dict, path : list[str] | None = None, objects : bool = False):
#     """
#     Walk a schema tree and yield the path and value of each property which
#     represents a FITS header entry, not including intermediate levels.

#     Args:
#         - _schema (dict): The schema to walk.
#         - path (list, optional): The path to the current schema level.
#         - objects (bool, optional): If True, yield only objects, not individual properties.
#     """
#     if path is None:
#         path = []
#     for key, value in _schema['properties'].items():
#         current_path = path + [key]
#         if isinstance(value, type(_schema)):
#             if value.get('type', 'string') == 'object' and 'properties' in value:
#                 yield from walk_schema(value, current_path)
#             else:
#                 if value.get('type', 'string') != 'object':
#                     yield (current_path, value)


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