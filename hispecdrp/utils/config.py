from collections.abc import MutableMapping
import copy
import yaml

class Config(MutableMapping):
    """
    Lightweight configuration object for pipelines and primitives from YAML files.
    """

    def __init__(self, data: dict | None = None, config_file: str | None = None):
        """
        Initialize Config with either a dictionary or a YAML file.
        
        Args:
            data (dict, optional): Initialize from dictionary data.
            config_file (str, optional): Path to YAML configuration file.
        """
        self._data = {}
        if config_file is not None:
            self.update(**self.load(config_file))
        if data is not None:
            self.update(**data)

    def copy(self):
        """
        Return a deep copy of the Config object.
        """
        new_config = Config()
        new_config._data = copy.deepcopy(self._data)
        return new_config

    def save(self, filename):
        """
        Save configuration to a YAML file.
        """
        with open(filename, 'w') as f:
            yaml.safe_dump(self._data, f)

    @staticmethod
    def load(filename: str):
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

    # Required MutableMapping abstract methods
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # Additional dictionary-like methods
    def get(self, key, default=None):
        """
        Get a value from the configuration with a default if the key doesn't exist.
        """
        return self._data.get(key, default)

    # Attribute access
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dictionary.
        Only called if the attribute isn't found through normal means.
        """
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Forward attribute assignment to the underlying dictionary,
        except for the protected '_data' attribute.
        """
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value