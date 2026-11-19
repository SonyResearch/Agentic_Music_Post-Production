"""Name -> processor-class registry for the ten mixing-console slots.

Each public slot name used in plans resolves to one processor class. ``compressor``
is served by the DiffVox compressor/expander under the backend name
``reverse_engineering_compressor`` (see :data:`mixagent.render.TOOL_BACKEND_ALIASES`).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    family: str
    category: str
    differentiability: str
    module: str
    class_name: str
    description: str


_G = "mixagent.fx.processors.grafx"
_D = "mixagent.fx.processors.diffvox.diffvox_fx"
_I = "mixagent.fx.processors.itomaster.itomaster_fx"

_SPECS = [
    ToolSpec("eq", "grafx", "eq", "differentiable", f"{_G}.eq", "ParametricEqualizer",
             "Reverse-engineering EQ slot alias."),
    ToolSpec("gain_panning", "grafx", "stereo", "differentiable", f"{_G}.stereo", "StereoGain",
             "Reverse-engineering gain/panning slot alias."),
    ToolSpec("noisegate", "grafx", "dynamics", "differentiable", f"{_G}.dynamics", "NoiseGate",
             "Reverse-engineering noise gate slot alias."),
    ToolSpec("reverb", "grafx", "reverb", "differentiable", f"{_G}.reverb",
             "FilteredNoiseShapingReverb", "Filtered noise shaping reverb slot alias."),
    ToolSpec("stereo_imager", "grafx", "stereo", "differentiable", f"{_G}.stereo",
             "SideGainImager", "Reverse-engineering stereo imager slot alias."),
    ToolSpec("delay", "diffvox", "delay", "differentiable", _D, "FSSurrogateDelay_diffvox",
             "Reverse-engineering default delay slot alias."),
    ToolSpec("reverse_engineering_compressor", "diffvox", "dynamics", "differentiable", _D,
             "CompressorExpander_diffvox", "Reverse-engineering default compressor slot alias."),
    ToolSpec("distortion", "itomaster", "distortion", "differentiable", _I,
             "Distortion_itomaster", "Reverse-engineering default distortion slot alias."),
    ToolSpec("limiter", "itomaster", "dynamics", "differentiable", _I, "Limiter_itomaster",
             "Reverse-engineering limiter slot alias."),
    ToolSpec("multiband_compressor", "itomaster", "dynamics", "differentiable", _I,
             "Multiband_Compressor_itomaster", "Multiband compressor slot alias."),
]

AVAILABLE_TOOLS = {spec.name: spec for spec in _SPECS}


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return AVAILABLE_TOOLS[name]
    except KeyError as exc:
        options = ", ".join(sorted(AVAILABLE_TOOLS))
        raise KeyError(f"Unknown tool {name!r}. Available tools: {options}") from exc


def get_tool_class(name: str):
    spec = get_tool_spec(name)
    module = import_module(spec.module)
    return getattr(module, spec.class_name)


def create_tool(name: str, /, **kwargs: Any):
    return get_tool_class(name)(**kwargs)


def parameter_schema(name: str, /, **kwargs: Any):
    tool = create_tool(name, **kwargs)
    parameter_size = getattr(tool, "parameter_size", None)
    if not callable(parameter_size):
        return {}
    return parameter_size()
