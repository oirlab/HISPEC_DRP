import importlib.resources
import os
import yaml
import copy
import numbers

__all__ = ['MetaDict']


class MetaDict:
    """
    A dictionary that allows attribute-style access and creates nested MetaDicts
    during assignment or access and validates against a schema.
    
    Schema fields:
        - keyword: The FITS keyword to be used in the header.
        - comment: The comment to be added to the FITS header. Defaults to ''.
        - unit: The unit of the value, not used for validation. Defaults to None.
        - datatype: The type of the value, used for validation (string, bool, integer, number). Defaults to string.
        - default: The default value to be used if not provided. Defaults to None.
        - fits_hdu: The HDU to which the keyword belongs. Defaults to 'PRIMARY'.
    """

    _DEFAULTS = {
        'keyword': None,
        'comment': None,
        'unit': None,
        'datatype': 'string',
        'default': None,
        'fits_hdu': 'PRIMARY',
    }

    def __init__(self, schema : str | list[str] | dict, validate : bool = True):
        """
        Construct a new MetaDict object.

        Args:
            _validate (bool): Whether to validate the data against the schema.
        """
        self._validate = validate
        if isinstance(schema, str):
            schema = [schema]
            self.load_merged_schema(schema)
        elif isinstance(schema, list):
            self.load_merged_schema(schema)
        elif isinstance(schema, dict):
            self._schema = schema
        self.init_data()

    def init_data(self):
        """
        Set default values for properties defined in the schema.
        """
        self._data = {}
        for path, info in self.walk_schema(sections=False):
            if info.get('default') is not None:
                #self[path] = info['default']
                self.__setattr__(path, info['default'])
    
    def __getattr__(self, name : str | list[str] | tuple[str]):
        return self[name]
    
    def __getitem__(self, prop):
        if isinstance(prop, (list, tuple)):
            current = self
            for k in prop:
                current = current[k]
            return current
        
        # Handle dot notation string
        if isinstance(prop, str) and '.' in prop:
            return self[prop.split('.')]
        
        # Assert name in properties
        assert prop in self._schema, f"Property '{prop}' not found in schema"
        
        # Automatically create intermediate MetaDicts for missing keys
        if prop not in self._data:
            if 'keyword' not in self._schema[prop]:
                sub_schema = self._schema[prop]
                self._data[prop] = MetaDict(schema=sub_schema, validate=self._validate)
            else:
                return None

        return self._data[prop]
    
    def __setattr__(self, name, value):
        if isinstance(name, (list, tuple)):
            name = '.'.join(name)
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __setitem__(self, prop, value):
        if isinstance(prop, (list, tuple)):
            current = self
            for k in prop[:-1]:
                if k not in current:
                    sub_schema = current._schema[k]
                    current[k] = MetaDict(schema=sub_schema, validate=self._validate)
                current = current[k]
            current[prop[-1]] = value
            return
        
        # Handle dot notation string
        if isinstance(prop, str) and '.' in prop:
            self[prop.split('.')] = value
            return
        
        if self._validate:
            self.validate(prop, value)
        self._data[prop] = value
    
    def pprint(self, show_none : bool = False):
        """
        Pretty-print MetaDict contents with schema info from DataFrame _schema.

        Args:
            show_none: If False, skip uninitialized fields.
        """
        for prop, info in self.walk_schema(sections=False):
            prop = '.'.join(prop) if isinstance(prop, (list, tuple)) else prop
            value = self[prop] # Live value
            if value is not None or show_none:
                print(f"{prop}: {value}")
                print(f"  comment: {info.get('comment', self._DEFAULTS['comment'])}")
                print(f"  keyword: {info['keyword']}")
                print(f"  fits_hdu: {info.get('fits_hdu', self._DEFAULTS['fits_hdu'])}")
                print(f"  unit: {info.get('unit', self._DEFAULTS['unit'])}")
                print(f"  datatype: {info.get('datatype', self._DEFAULTS['datatype'])}")
    
    def __dir__(self):
        """Add keys to dir() for tab completion."""
        return list(set(super().__dir__() + list(self._data.keys())))
    
    def __contains__(self, item):
        if isinstance(item, str):
            keys = item.split(".")
        d = self
        for key in keys:
            if isinstance(d, (dict, MetaDict)) and key in d._data:
                d = d[key]
            else:
                return False
        return True
        
    def validate(self, prop, value):
        """
        Validate a property against the schema.
        
        Args:
            prop (str): The property name.
            value: The value to validate.
        
        Raises:
            ValueError: If the value does not match the schema type.
        """
        if isinstance(value, MetaDict):
            assert prop in self._schema, f"Property '{prop}' not found in schema"
            return
        schema_type = self._schema[prop].get('datatype', self._DEFAULTS['datatype'])
        type_passed = type(value)
        if 'keyword' not in self._schema[prop]:
            if value is None:
                return MetaDict(schema=self._schema[prop], validate=self._validate)
            else:
                raise ValueError(f"Cannot set '{prop}' as {type_passed}")
        if schema_type in ('string', 'None'):
            if not isinstance(value, str) and value is not None:
                raise ValueError(f"Property '{prop}' must be a string")
        elif schema_type == 'number':
            if not isinstance(value, numbers.Number) and value is not None:
                raise ValueError(f"Property '{prop}' must be a number")
        elif schema_type == 'integer':
            if not (isinstance(value, numbers.Integral) and not isinstance(value, bool)) and value is not None:
                raise ValueError(f"Property '{prop}' must be an integer")
        elif schema_type == 'boolean':
            if not isinstance(value, bool) and value is not None:
                raise ValueError(f"Property '{prop}' must be a boolean")
    
    def load_merged_schema(self, schema_name : str | list[str]):
        """
        Load a schema from a CSV file.
        """
        self._schema = {}
        if isinstance(schema_name, str):
            schema_name = [schema_name]
        for _schema_name in schema_name:
            if os.path.exists(_schema_name):
                schema_path = _schema_name
            else:
                schema_path = MetaDict.find_schema_path(_schema_name)
                with open(schema_path, 'r') as f:
                    _schema = yaml.safe_load(f)
                self._schema.update(_schema)
        return self._schema

    def walk_schema(self, schema : dict | None = None, path : list[str] | None = None, sections : bool = False):
        """
        Walk a schema tree and yield the path and value of each property which
        represents a FITS header entry, not including intermediate levels.

        Args:
            - schema (dict): The schema to walk.
            - path (list, optional): The path to the current schema level.
            - show_none (bool, optional): If True, yield only objects, not individual properties.
        """
        if path is None:
            path = []
        if schema is None:
            schema = self._schema
        
        for key, info in schema.items():
            current_path = path + [key]
            if 'keyword' not in info:
                if sections:
                    yield (current_path, info)
                yield from self.walk_schema(info, current_path, sections=sections)
            else:
                yield (current_path, info)
    
    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def find_schema_path(schema_name : str) -> str:
        """
        Search all `datamodels/*/schemas/` for a schema file by base name.
        """
        base = importlib.resources.files("hispecdrp.datamodels")
        for subpkg in base.iterdir():
            schemas_dir = subpkg / "schemas"
            if schemas_dir.is_dir():
                candidate = schemas_dir / schema_name
                if candidate.is_file():
                    return str(candidate)
        raise FileNotFoundError(f"No valid schema found for: {schema_name}")
    
    def update(self, other, override : bool = True):
        """
        Update the MetaDict with another dictionary or MetaDict.
        
        Args:
            other (dict | MetaDict): The dictionary or MetaDict to merge into this one.
        """
        for prop, _ in self.walk_schema(sections=False):
            prop = '.'.join(prop) if isinstance(prop, (list, tuple)) else prop
            if prop not in other:
                continue
            if self[prop] is None or override:
                self[prop] = other[prop]
    
    def to_dict(self):
        """
        Convert the MetaDict to a regular nested dictionary.
        
        Returns:
            dict: A nested dictionary representation of the MetaDict
        """
        result = {}
        
        for key, value in self._data.items():
            if isinstance(value, MetaDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        
        return result

    def to_flat_dict(self, sep='.', prefix=''):
        """
        Convert the MetaDict to a flattened dictionary with dot-separated keys.
        
        Args:
            sep (str): Separator to use between nested levels (default: '.')
            prefix (str): Prefix for keys (used internally for recursion)
        
        Returns:
            dict: A flattened dictionary with dot-separated keys
        """
        result = {}
        
        for key, value in self._data.items():
            full_key = f"{prefix}{sep}{key}" if prefix else key
            
            if isinstance(value, MetaDict):
                # Recursively flatten nested MetaDict
                result.update(value.to_flat_dict(sep=sep, prefix=full_key))
            else:
                # Add the leaf value
                result[full_key] = value
        
        return result

    def __delitem__(self, key):
        """
        Delete an item from the MetaDict.
        
        Args:
            key (str | list | tuple): The key to delete. Can be:
                - A string with dot notation (e.g., 'level1.level2.key')
                - A list/tuple of keys (e.g., ['level1', 'level2', 'key'])
                - A simple string key
        """
        if isinstance(key, (list, tuple)):
            # Handle nested deletion for list/tuple paths
            if len(key) == 1:
                del self._data[key[0]]
            else:
                current = self
                for k in key[:-1]:
                    current = current[k]
                del current._data[key[-1]]
            return
        
        # Handle dot notation string
        if isinstance(key, str) and '.' in key:
            self.__delitem__(key.split('.'))
            return
        
        # Handle simple key deletion
        if key in self._data:
            del self._data[key]
        else:
            raise KeyError(f"Key '{key}' not found in MetaDict")