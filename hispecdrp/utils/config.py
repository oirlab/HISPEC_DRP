from keckdrpframework.config.framework_config import ConfigClass as _ConfigClass
import yaml
import os
import copy

__all__ = ['PrimitiveConfig']

from typing import Self

class PrimitiveConfig(dict):
    """
    A dictionary that allows attribute-style access at the top level while maintaining
    regular dictionary behavior for nested structures. Can be initialized from either
    a primitive config file (LOGGER + ARGUMENTS) or a pipeline config file 
    (LOGGER + ARGUMENTS + PRIMITIVES).
    """
    
    ALLOWED_ATTRIBUTES = ['items', 'keys', 'values', 'copy', 'get', 'update', 'pop', 'clear', 'popitem']
    PRIMITIVE_SECTIONS = ['LOGGER', 'ARGUMENTS']
    PIPELINE_SECTIONS = PRIMITIVE_SECTIONS + ['PRIMITIVES']

    # def __new__(cls, *args, **kwargs):
    #     if isinstance(args[0], PrimitiveConfig):
    #         # Return the existing object directly
    #         return args[0]
    #     # Create a new instance for other cases
    #     return super().__new__(cls)

    # TODO: HERE. SORT THIS CLASS OUT. RUNNING test_primitive.py GIVES AN ERROR HERE.

    def __init__(
        self,
        data: dict | Self | None = None,
        config_file: str | None = None,
        root_dir: str | None = None
    ):
        """
        Construct a new PrimitiveConfig object.

        Args:
            data (dict | PrimitiveConfig | None): The data to be stored in the PrimitiveConfig.
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
            return None
        return self[name]


    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self[name] = value


    def __setitem__(self, key, value):
        # Handle list of keys
        if isinstance(key, list):
            current = self
            for k in key[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[key[-1]] = value
            return
        
        # Handle dot notation string
        if isinstance(key, str) and '.' in key:
            self[key.split('.')] = value
            return

        super().__setitem__(key, value)


    def __getitem__(self, key):
        # Handle list of keys
        if isinstance(key, (list, tuple)):
            current = self
            for k in key:
                current = current[k]
            return current
        
        # Handle dot notation string
        if isinstance(key, str) and '.' in key:
            return self[key.split('.')]
        return super().__getitem__(key)


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
    

    @property
    def is_pipeline(self) -> bool:
        """Return whether this config is a pipeline config."""
        return 'PRIMITIVES' in self
    

    def init_primitives(self, _config : dict):
        """
        Process the PRIMITIVES section, loading any nested config files.

        Args:
            _config (dict): The raw config loaded as a pure dict.
        """
        processed_primitives = {}
        
        if 'PRIMITIVES' in _config:
            breakpoint()
            for primitive_name, primitive_config in _config['PRIMITIVES'].items():
                if primitive_config is None:
                    processed_primitives[primitive_name] = {}
                    continue
                    
                if not isinstance(primitive_config, dict):
                    raise ValueError(f"Primitive configuration for '{primitive_name}' must be a dictionary or None")
                
                # Check for config_file field
                if 'config_file' in primitive_config:
                    config_path = primitive_config['config_file']
                    if not os.path.isabs(config_path):
                        config_path = os.path.join(self._root_dir, config_path)
                    
                    # Load the nested config file
                    nested_config = PrimitiveConfig(config_file=config_path, root_dir=os.path.dirname(config_path))
                    
                    # Merge with any overrides in the pipeline config
                    primitive_overrides = {k: v for k, v in primitive_config.items() if k != 'config_file'}
                    processed_primitives[primitive_name] = nested_config.merge(primitive_overrides)
                else:
                    processed_primitives[primitive_name] = primitive_config
        
        return processed_primitives
    

    def load(self, filename : str, load_primitives : bool = True):
        """
        Load a YAML configuration file and optionally all sub-configuraiton files within the PRIMITIVES section.
        """
        with open(filename, 'r') as f:
            _config = yaml.safe_load(f)
        if 'PRIMITIVES' in _config and load_primitives:
           _config = self.init_primitives(_config['PRIMITIVES'])
        self.update(_config)


    def save(self, filename : str):
        """
        Save this configuration to a YAML file.
        """
        with open(filename) as f:
            yaml.safe_dump(self, f)