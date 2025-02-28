from typing import Self


__all__ = ['MetaDict']


class MetaDict:
    """
    A dictionary that allows attribute-style access and creates nested MetaDicts
    during assignment or access and validates against a schema.
    
    Schema Fields:
        - comment: The comment to be added to the FITS header.
        - fits_keyword: The FITS keyword to be used in the header.
        - unit: The unit of the value, not used for validation.
        - type: The type of the value, used for validation.
        - default: The default value to be used if not provided.
        - fits_hdu: The HDU to which the keyword belongs. Defaults to 'PRIMARY'.
    """

    def __init__(self, data: dict | Self | None = None, _schema: dict | None = None, _validate: bool = True):
        """
        Construct a new MetaDict object.

        Args:
            data (dict | MetaDict | None): The data to be stored in the MetaDict.
            _schema (dict | None): The schema to be used for validation.
            _validate (bool): Whether to validate the data against the schema.
        """
        self._schema = _schema
        self._validate = _validate
        self._data = {}
        
        if isinstance(data, dict):
            self._data.update(self.from_dict(data)._data)
        elif isinstance(data, MetaDict):
            self._data.update(data._data)


    @classmethod
    def from_dict(cls, d: dict | Self):
        if isinstance(d, dict) and not isinstance(d, MetaDict):
            return MetaDict({k: cls.from_dict(v) for k, v in d.items()})
        return d


    def get_sub_schema(self, prop: str):
        if not self._schema:
            return None
        else:
            assert prop in self._schema['properties'], f"Property '{prop}' not found in schema"
            return self._schema['properties'][prop]


    def validate(self, prop: str, value = None):
        if not self._schema:
            return 
        
        # Assert prop is valid
        assert prop in self._schema['properties']

        if value is not None:
            self.validate_header_entry(prop, value)


    def validate_header_entry(self, prop: str, value):
        if not self._schema:
            return False
        
        header_props = self._schema['properties'][prop]

        # Types
        schema_type = header_props['type'] if 'type' in header_props else 'string'
        type_passed = type(value)

        # Make sure value is the correct type
        if schema_type == 'string':
            if type_passed is not str:
                raise ValueError(f"Property '{prop}' must be a string")
        elif schema_type == 'number':
            if type_passed not in [int, float]:
                raise ValueError(f"Property '{prop}' must be a number")
        elif schema_type == 'integer':
            if type_passed is not int:
                raise ValueError(f"Property '{prop}' must be an integer")
        elif schema_type == 'boolean':
            if type_passed is not bool:
                raise ValueError(f"Property '{prop}' must be a boolean")


    def __getattr__(self, name : str | list[str] | tuple[str]):
        if isinstance(name, (list, tuple)):
            current = self
            for k in name:
                current = current[k]
            return current
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Assert name in properties
        if self._schema is not None:
            assert name in self._schema['properties'], f"Property '{name}' not found in schema"
        
        # Automatically create intermediate MetaDicts for missing keys
        if name not in self._data:
            if self._schema is not None:
                if self._schema['properties'][name]['type'] == 'object':
                    self._data[name] = MetaDict(_schema=self.get_sub_schema(name), _validate=self._validate)
                else:
                    return None
            else:
                self._data[name] = MetaDict()

        return self._data[name]


    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        self[name] = value


    def __setitem__(self, key, value):
        # Handle list of keys
        if isinstance(key, list):
            current = self
            for k in key[:-1]:
                if k not in current:
                    if self._schema is not None:
                        sub_schema = current.get_sub_schema(k)
                        current[k] = MetaDict(_schema=sub_schema, _validate=self._validate)
                    else:
                        current[k] = MetaDict()
                current = current[k]
            current[key[-1]] = value
            return
        
        # Handle dot notation string
        if isinstance(key, str) and '.' in key:
            self[key.split('.')] = value
            return
        
        if isinstance(value, dict) and not isinstance(value, MetaDict):
            value = MetaDict(value)
        if self._schema is not None and self._validate:
            self.validate(key, value)
        self._data[key] = value


    def __getitem__(self, key):
        return self.__getattr__(key)


    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
            return
        try:
            del self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def __dir__(self):
        """Add keys to dir() for tab completion."""
        return list(set(super().__dir__() + list(self._data.keys())))
    

    def __contains__(self, item):
        return item in self._data


    def pprint(self, parent=None, prefix=''):
        """
        Print the contents of the MetaDict in a human-readable format.
        """
        if parent is None:
            parent = self
        if isinstance(parent, MetaDict):
            for key, value in parent.items():
                if isinstance(value, MetaDict):
                    parent.pprint(value, prefix + key + '.')
                else:
                    header_schema = parent._schema['properties'].get(key, {})
                    print(f"{prefix}{key}: {value}")
                    print(f"  comment: {header_schema.get('comment', 'No comment available.')}")
                    print(f"  fits_keyword: {header_schema.get('fits_keyword', 'N/A')}")
                    print(f"  fits_hdu: {header_schema.get('fits_hdu', 'PRIMARY')}")
                    print(f"  unit: {header_schema.get('unit', 'None')}")
                    print(f"  type: {header_schema.get('type', 'string')}")
        else:
            print(f"{prefix}{parent}")


    def __repr__(self):
        return f"{type(self).__name__}({self._data})"


    def merge(self, other: Self):
        return MetaDict.merge(self, other)


    @staticmethod
    def merge(meta1: Self | dict, meta2: Self | dict):
        if isinstance(meta1, dict):
            meta1 = MetaDict(meta1)
        merged = MetaDict(meta1._data.copy())
        for key, value in meta2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merged.merge(merged[key], value)
            else:
                merged[key] = value
        return merged


    # Add dictionary interface methods
    def items(self):
        return self._data.items()


    def keys(self):
        return self._data.keys()


    def values(self):
        return self._data.values()

    
    def copy(self):
        return MetaDict(self._data.copy(), _schema=self._schema, _validate=self._validate)


    def get(self, key, default=None):
        return self._data.get(key, default)


    def update(self, other):
        if isinstance(other, MetaDict):
            self._data.update(other._data)
        else:
            self._data.update(other)


    def pop(self, key, default=None):
        return self._data.pop(key, default)


    def clear(self):
        self._data.clear()


    def popitem(self):
        return self._data.popitem()


# class MetaDict:
#     """
#     A dictionary that allows attribute-style access and creates nested MetaDicts
#     during assignment or access and validates against a schema.
    
#     Schema Fields:
#         - comment: The comment to be added to the FITS header.
#         - fits_keyword: The FITS keyword to be used in the header.
#         - unit: The unit of the value, not used for validation.
#         - type: The type of the value, used for validation.
#         - default: The default value to be used if not provided.
#         - fits_hdu: The HDU to which the keyword belongs. Defaults to 'PRIMARY'.
#     """
    
#     ##ALLOWED_ATTRIBUTES = ['items', 'keys', 'values', 'copy', 'get', 'update', 'pop', 'clear', 'popitem', '_schema', '_validate']
    

#     def __init__(self, data : dict | Self | None = None, _schema: dict | None = None, _validate : bool = True):
#         """
#         Construct a new MetaDict object.

#         Args:
#             data (dict | MetaDict | None): The data to be stored in the MetaDict.
#             _schema (dict | None): The schema to be used for validation.
#             _validate (bool): Whether to validate the data against the schema.
#         """
#         self._schema = _schema
#         self._validate = _validate
#         if type(data) is dict:
#             super().__init__(self.from_dict(data))
#         elif isinstance(data, MetaDict):
#             super().__init__(data)
#         else:
#             super().__init__()


#     @classmethod
#     def from_dict(cls, d : dict | Self):
#         if isinstance(d, dict) and not isinstance(d, MetaDict):
#             return MetaDict({k: cls.from_dict(v) for k, v in d.items()})
#         return d
    

#     def get_sub_schema(self, prop : str):
#         if not self._schema:
#             return None
#         else:
#             assert prop in self._schema['properties'], f"Property '{prop}' not found in schema"
#             return self._schema['properties'][prop]
    

#     def validate(self, prop : str, value = None):
#         if not self._schema:
#             return 
        
#         # Assert prop is valid
#         assert prop in self._schema['properties']

#         # Return if a dict or MetaDict is passed
#         #if isinstance(value, dict):
#         #    return
#         if value is not None:
#             self.validate_header_entry(prop, value)


#     def validate_header_entry(self, prop : str, value):
#         if not self._schema:
#             return False
        
#         header_props = self._schema['properties'][prop]

#         # Types
#         schema_type = header_props['type'] if 'type' in header_props else 'string'
#         type_passed = type(value)

#         # Make sure value is the correct type
#         if schema_type == 'string':
#             if type_passed is not str:
#                 raise ValueError(f"Property '{prop}' must be a string")
#         elif schema_type == 'number':
#             if type_passed not in [int, float]:
#                 raise ValueError(f"Property '{prop}' must be a number")
#         elif schema_type == 'integer':
#             if type_passed is not int:
#                 raise ValueError(f"Property '{prop}' must be an integer")
#         elif schema_type == 'boolean':
#             if type_passed is not bool:
#                 raise ValueError(f"Property '{prop}' must be a boolean")


#     def __getattr__(self, name):
#         # Allow access to specific attributes and block private ones
#         if name in self.ALLOWED_ATTRIBUTES:
#             return super().__getattribute__(name)
        
#         if name.startswith('_'):
#             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
#         # Assert name in properties
#         if self._schema is not None:
#             assert name in self._schema['properties'], f"Property '{name}' not found in schema"
        
#         # Automatically create intermediate MetaDicts for missing keys
#         if name not in self:
#             if self._schema is not None:
#                 if self._schema['properties'][name]['type'] == 'object':
#                     self[name] = MetaDict(_schema=self.get_sub_schema(name), _validate=self._validate)
#                 else:
#                     return None
#             else:
#                 self[name] = MetaDict()

#         return self[name]


#     def __setattr__(self, name, value):
#         if name.startswith('_'):
#             super().__setattr__(name, value)
#             return
#         self[name] = value


#     def __setitem__(self, key, value):
#         # Handle list of keys
#         if isinstance(key, list):
#             current = self
#             for k in key[:-1]:
#                 if k not in current:
#                     if self._schema is not None:
#                         sub_schema = current.get_sub_schema(k)
#                         current[k] = MetaDict(_schema=sub_schema, _validate=self._validate)
#                     else:
#                         current[k] = MetaDict()
#                 current = current[k]
#             current[key[-1]] = value
#             return
        
#         # Handle dot notation string
#         if isinstance(key, str) and '.' in key:
#             self[key.split('.')] = value
#             return
#         if isinstance(value, dict) and not isinstance(value, MetaDict):
#             value = MetaDict(value)
#         if self._schema is not None and self._validate:
#             self.validate(key, value)
#         super().__setitem__(key, value)


#     def __getitem__(self, key):
#         return self.__getattr__(key)
#         # # Handle list of keys
#         # if isinstance(key, (list, tuple)):
#         #     current = self
#         #     for k in key:
#         #         current = current[k]
#         #     return current
        
#         # # Handle dot notation string
#         # if isinstance(key, str) and '.' in key:
#         #     return self[key.split('.')]
#         # return super().__getitem__(key)


#     def __delattr__(self, name):
#         if name.startswith('_'):
#             super().__delattr__(name)
#             return
#         try:
#             del self[name]
#         except KeyError:
#             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


#     def __dir__(self):
#         """Add keys to dir() for tab completion."""
#         return list(set(super().__dir__() + list(self.keys())))
    
    
#     def __contains__(self, item):
#         try:
#             self.item
#             return True
#         except:
#             return False
    

#     def pprint(self, parent=None, prefix=''):
#         """
#         Print the contents of the MetaDict in a human-readable format.
#         """
#         if parent is None:
#             parent = self
#         if isinstance(parent, MetaDict):
#             for key, value in parent.items():
#                 if isinstance(value, MetaDict):
#                     parent.pprint(value, prefix + key + '.')
#                 else:
#                     header_schema = parent._schema['properties'].get(key, {})
#                     print(f"{prefix}{key}: {value}")
#                     print(f"  comment: {header_schema.get('comment', 'No comment available.')}")
#                     print(f"  fits_keyword: {header_schema.get('fits_keyword', 'N/A')}")
#                     print(f"  fits_hdu: {header_schema.get('fits_hdu', 'PRIMARY')}")
#                     print(f"  unit: {header_schema.get('unit', 'None')}")
#                     print(f"  type: {header_schema.get('type', 'string')}")
#         else:
#             print(f"{prefix}{parent}")


#     def __repr__(self):
#         return f"{type(self).__name__}({super().__repr__()})"
    
    
#     def merge(self, other : Self):
#         return MetaDict.merge_meta(self, other)
    

#     @staticmethod
#     def merge(meta1 : Self | dict, meta2 : Self | dict):
#         if type(meta1) is dict:
#             meta1 = MetaDict(meta1)
#         merged = meta1.copy()
#         for key, value in meta2.items():
#             if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
#                 merged[key] = merged.merge(merged[key], value)
#             else:
#                 merged[key] = value
#         return merged