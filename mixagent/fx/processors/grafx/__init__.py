"""Lazy entry point for Grafx processor tools."""

from importlib import import_module

_CLASS_TO_MODULE = {
    "Compressor": ".dynamics",
    "NoiseGate": ".dynamics",
    "ZeroPhaseFIREqualizer": ".eq",
    "GraphicEqualizer": ".eq",
    "NewZeroPhaseFIREqualizer": ".eq",
    "ParametricEqualizer": ".eq",
    "AllPassFilter": ".filter",
    "BandPassFilter": ".filter",
    "BandRejectFilter": ".filter",
    "BiquadFilter": ".filter",
    "FIRFilter": ".filter",
    "HighPassFilter": ".filter",
    "HighShelf": ".filter",
    "LowPassFilter": ".filter",
    "LowShelf": ".filter",
    "PeakingFilter": ".filter",
    "PoleZeroFilter": ".filter",
    "StateVariableFilter": ".filter",
    "FilteredNoiseShapingReverb": ".reverb",
    "STFTMaskedNoiseReverb": ".reverb",
    "MidSideToStereo": ".stereo",
    "MonoToStereo": ".stereo",
    "SideGainImager": ".stereo",
    "StereoGain": ".stereo",
    "StereoToMidSide": ".stereo",
}

__all__ = sorted(_CLASS_TO_MODULE)


def __getattr__(name):
    if name == "core":
        module = import_module(".core", __name__)
        globals()[name] = module
        return module
    if name not in _CLASS_TO_MODULE:
        raise AttributeError(name)
    module = import_module(_CLASS_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
