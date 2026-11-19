"""Lazy entry point for DiffVox processor tools."""

from importlib import import_module

_CLASS_TO_MODULE = {
    "CompressorExpander": ".diffvox_fx",
    "CompressorExpander_diffvox": ".diffvox_fx",
    "Delay": ".diffvox_fx",
    "FDN": ".diffvox_fx",
    "FDN_diffvox": ".diffvox_fx",
    "FIR": ".diffvox_fx",
    "FSDelay": ".diffvox_fx",
    "FSDelay_diffvox": ".diffvox_fx",
    "FSSurrogateDelay": ".diffvox_fx",
    "FSSurrogateDelay_diffvox": ".diffvox_fx",
    "HighPass": ".diffvox_fx",
    "HighShelf": ".diffvox_fx",
    "LowPass": ".diffvox_fx",
    "LowShelf": ".diffvox_fx",
    "Panning": ".diffvox_fx",
    "ParametricEQ_diffvox": ".diffvox_fx",
    "Peak": ".diffvox_fx",
    "Reverb": ".diffvox_fx",
    "SendFXsAndSum": ".diffvox_fx",
    "StereoWidth": ".diffvox_fx",
    "SurrogateDelay": ".diffvox_fx",
}

__all__ = sorted(_CLASS_TO_MODULE)


def __getattr__(name):
    if name not in _CLASS_TO_MODULE:
        raise AttributeError(name)
    module = import_module(_CLASS_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
