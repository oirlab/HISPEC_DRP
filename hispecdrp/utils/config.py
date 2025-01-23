import yaml
import os
import copy


import yaml
import os
import copy


__all__ = ['Config']


class Config(dict):
    """
    Lightweight configuration object for pipelines and primitives from YAML files.
    """

    def __init__(self, data : dict | None = None, config_file : str | None = None):
        """
        Initialize Config with either a dictionary or a YAML file.
        
        Args:
            data (dict, optional): Initialize from dictionary data.
            config_file (str, optional): Path to YAML configuration file.
        """
        super().__init__()
        if config_file is not None:
            self.update(**self.load(config_file))
        if data is not None:
            self.update(**data)


    def copy(self):
        """
        Return a deep copy of the Config object.
        """
        return copy.deepcopy(self)


    def save(self, filename):
        """
        Save configuration to a YAML file.
        """
        with open(filename, 'w') as f:
            yaml.safe_dump(self, f)


    @staticmethod
    def load(filename : str):
        """
        Load a YAML configuration file.

        Args:
            filename (str): Path to YAML configuration file.
        """
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        data = Config.replace_none_strings(data)
        return data
        
    @staticmethod
    def replace_none_strings(d):
        """
        Recursively replace all occurrences of the string 'None' with Python's None.
        Works for nested dictionaries only.
        """
        if isinstance(d, dict):
            for key, value in d.items():
                if value == 'None':
                    d[key] = None  # Replace 'None' with None
                elif isinstance(value, dict):
                    Config.replace_none_strings(value)  # Recurse into nested dictionaries
        return d