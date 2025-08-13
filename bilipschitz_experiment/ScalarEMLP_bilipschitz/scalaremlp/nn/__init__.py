import importlib
import pkgutil
__all__ = [] # expose objax implementation as base nn


module = importlib.import_module('.'+'objax',package=__name__)
# print(__name__)
# print(module)
# print(module.__all__)
globals().update({k: getattr(module, k) for k in module.__all__})
__all__ += module.__all__
# print(module.__all__)


# class InvarianceLayer_objax:
#     pass