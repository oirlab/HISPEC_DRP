import importlib
import yaml

__all__ = ['resolve_primitive_class', 'find_config_path', 'read_config_file', 'initialize_from_config']

import os

def read_config_file(filepath : str) -> dict:
    """
    Load a config file.

    Args:
        filepath (str): The path to the YAML configuration file.
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def find_config_path(config_name : str) -> str:
    """Search hispecdrp/configs/ for a config file."""
    if os.path.exists(config_name):
        return config_name
    else:
        config_dir = importlib.resources.files("hispecdrp.configs")
        if config_dir.is_dir():
            candidate = config_dir / config_name
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"No valid config file found for: {config_name}")

def resolve_primitive_class(name: str):
    """
    Resolve a primitive class name to its actual class.
    
    Args:
        name (str): The name of the primitive class, optionally including module path
        
    Returns:
        type or None: The resolved class or None if not found
    """
    def try_import(module_path, class_name):
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name, None)
        except (ImportError, AttributeError):
            return None

    # Check if name contains a module path
    if '.' in name:
        module_path, class_name = name.rsplit('.', 1)
        resolved_class = try_import(module_path, class_name)
        if resolved_class:
            return resolved_class
    else:
        # If no module path, get the top-level module
        print(f"{__name__=}")
        top_level_module = __name__.split('.')[0]

        # Try resolving the class in the top-level module with suffixes
        resolved_class = try_import(top_level_module, name)
        if resolved_class:
            return resolved_class

    # Class not found
    return None

def initialize_from_config(config : dict | str, **kwargs):
    """
    Create an instance of a primitive from a configuration dictionary or file.

    Args:
        config (dict | str | None): A configuration dictionary, or the path to to a configuration file.
        **kwargs: Any additional keyword arguments to set or override for the primitive or pipeline.

    Returns:
        ProcessingComponent: An instance of the correct primitive.
    """
    if isinstance(config, str):
        config = read_config_file(config)
    _class = resolve_primitive_class(config['class'])
    return _class(config=config, **kwargs) 