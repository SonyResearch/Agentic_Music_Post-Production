"""Differentiable audio-effect processors used by the mixing console.

Vendored copies of processor implementations from three open-source projects, kept
byte-for-byte except for import paths:

* ``processors/grafx``     -- GRAFX (Sungho Lee et al., "GRAFX: an open-source library
  for audio processing graphs in PyTorch"): parametric EQ, stereo gain/imager, noise
  gate, filtered-noise-shaping reverb.
* ``processors/diffvox``   -- DiffVox (Chin-Yun Yu et al.): compressor/expander and
  frequency-shaped delay.
* ``processors/itomaster`` -- ITO-Master (Junghyun Koo et al.): distortion, limiter,
  multiband compressor.

Please refer to and respect the licenses of the original projects.
"""

from .registry import AVAILABLE_TOOLS, ToolSpec, create_tool, get_tool_class, get_tool_spec, parameter_schema

__all__ = ["AVAILABLE_TOOLS", "ToolSpec", "create_tool", "get_tool_class", "get_tool_spec",
           "parameter_schema"]
