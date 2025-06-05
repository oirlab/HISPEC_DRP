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
    
    CSV Columns:
        - fits_keyword: The FITS keyword to be used in the header.
        - comment: The comment to be added to the FITS header. Defaults to None.
        - unit: The unit of the value, not used for validation. Defaults to None.
        - type: The type of the value, used for validation. Defaults to str.
        - default: The default value to be used if not provided. Defaults to None.
        - fits_hdu: The HDU to which the keyword belongs. Defaults to 'PRIMARY'.
        - Property: The Python path to the property (e.g. meta.instrument.name).
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
        if isinstance(item, (list, tuple)):
            item = '.'.join(item)
        return item in self._data
        
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

    @staticmethod
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
    
    @staticmethod
    def flatten_dict(nested_dict, parent_key='', sep='.'):
        """
        Flatten a nested dictionary using dots to separate nested levels.
        Preserves the structure of leaf dictionaries that contain non-dict values.
        
        Args:
            nested_dict (dict): The nested dictionary to flatten
            parent_key (str): The parent key for recursion (used internally)
            sep (str): Separator to use between nested levels (default: '.')
        
        Returns:
            dict: Flattened dictionary with dot-separated keys
        """
        items = []
        
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                if all(not isinstance(v, dict) for v in value.values()):
                    items.append((new_key, value))
                else:
                    items.extend(MetaDict.flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        
        return dict(items)

    @staticmethod
    def unflatten_dict(flat_dict, sep='.'):
        """
        Convert a flattened dictionary back to nested structure.
        
        Args:
            flat_dict (dict): Dictionary with dot-separated keys
            sep (str): Separator used between nested levels (default: '.')
        
        Returns:
            dict: Nested dictionary structure
        """
        result = {}
        
        for key, value in flat_dict.items():
            keys = key.split(sep)
            current = result
            
            # Navigate/create nested structure
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the final value
            current[keys[-1]] = value
        
        return result