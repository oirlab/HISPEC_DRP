import importlib
from keckdrpframework.models.arguments import Arguments


__all__ = ['resolve_primitive_class']

def prepr_args(args : Arguments) -> str:
    s = f'name = {args.name}\n'
    for k in args.iter_kw():
        if k != 'name':
            s += f'{k} = {args[k]}\n'
    s = s[:-1]
    return s


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