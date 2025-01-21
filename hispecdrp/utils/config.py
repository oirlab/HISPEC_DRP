import yaml
import os
import importlib
import copy

__all__ = ['Config']

from typing import Self

class Config(dict):
    """
    A dictionary that allows attribute-style access at the top level while maintaining
    regular dictionary behavior for nested structures. Can be initialized from either
    a primitive config file (LOGGER + ARGUMENTS) or a pipeline config file 
    (LOGGER + ARGUMENTS + PRIMITIVES).
    """
    
    ALLOWED_ATTRIBUTES = ['items', 'keys', 'values', 'copy', 'get', 'update', 'pop', 'clear', 'popitem']
    PRIMITIVE_SECTIONS = ['logger', 'arguments']
    PIPELINE_SECTIONS = PRIMITIVE_SECTIONS + ['primitives']

    def __init__(
        self,
        data: dict | Self | None = None,
        config_file: str | None = None,
        root_dir: str | None = None
    ):
        """
        Construct a new Config object.

        Args:
            data (dict | Config | None): The data to be stored in the Config.
            config_file (str, optional): The path to the YAML configuration file to load.
            root_dir (str, optional): The root directory for resolving relative paths in config files.
        """
        super().__init__()
        self._root_dir = root_dir or (os.path.dirname(config_file) if config_file else os.getcwd())
        self._is_pipeline = False
        
        if config_file is not None:
            self.load(config_file)
        elif isinstance(data, dict):
            self.update(data)


    def update(self, other : Self | dict):
        return super().update(self.merge(self, other))
    

    def __getattr__(self, name):
        if name.startswith('_') or name in self.ALLOWED_ATTRIBUTES:
            return super().__getattribute__(name)
        
        # Create regular dict for missing keys
        if name not in self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self[name]


    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self[name] = value


    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = self.__class__(value)
        super().__setitem__(key, value)


    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
            return
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def __dir__(self):
        """Add keys to dir() for tab completion."""
        return list(set(super().__dir__() + list(self.keys())))


    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"


    @staticmethod
    def merge(_config : Self | dict, other: Self | dict):
        merged = _config.copy()
        for key, value in other.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    

    def copy(self):
        return copy.deepcopy(self)
    

    def load(self, filename : str):
        """
        Load a YAML configuration file and optionally all sub-configuraiton files within the PRIMITIVES section.
        """
        with open(filename, 'r') as f:
            _config = yaml.safe_load(f)
        self.update(_config)


    def save(self, filename : str):
        """
        Save this configuration to a YAML file.
        """
        with open(filename) as f:
            yaml.safe_dump(self, f)


    @staticmethod
    def _resolve_primitive_class(name: str):
        def try_import(module_path, class_name):
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name, None)
            except (ImportError, AttributeError):
                return None
        
        # Check if name contains a module path
        if '.' in name:
            module_path, class_name = name.rsplit('.', 1)
            
            # Try the class name as is, and with 'Primitive' and 'Pipeline' suffixes
            for suffix in ('', 'Primitive', 'Pipeline'):
                resolved_class = try_import(module_path, class_name + suffix)
                if resolved_class:
                    return resolved_class
        else:
            # If no module path, get the top-level module
            top_level_module = __name__.split('.')[0]
            
            # Try resolving the class in the top-level module with suffixes
            for suffix in ('', 'Primitive', 'Pipeline'):
                resolved_class = try_import(top_level_module, name + suffix)
                if resolved_class:
                    return resolved_class
        
        # Class not found
        return None
