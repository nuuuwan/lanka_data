import sys
from importlib import import_module

_module = import_module("api.utils_future")
globals().update(_module.__dict__)
for _name, _module_value in list(sys.modules.items()):
    if _name == "api.utils_future" or _name.startswith("api.utils_future."):
        sys.modules["utils_future" + _name[len("api.utils_future"):]] = (
            _module_value
        )
