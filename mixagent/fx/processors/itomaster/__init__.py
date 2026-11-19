"""Lazy entry point for ItoMaster processor tools."""

from importlib import import_module

_CLASS_TO_MODULE = {
    "Distortion_itomaster": ".itomaster_fx",
    "Limiter_itomaster": ".itomaster_fx",
    "Multiband_Compressor_itomaster": ".itomaster_fx",
}

__all__ = sorted(_CLASS_TO_MODULE)


def __getattr__(name):
    if name not in _CLASS_TO_MODULE:
        raise AttributeError(name)
    module = import_module(_CLASS_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
