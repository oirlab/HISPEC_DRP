from .selectors import *

_local_dict = dict(locals())
SELECTOR_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Selector')}