"""Mixing-console renderer: tool vocabulary, parameter schemas and the hierarchical render.

A plan is ``{"stems": [...], "groups": [...], "master_chain": [...]}`` whose steps are
``{"tool": <slot>, "params": {<raw pre-activation values>, "drywet"?}}``. Rendering is
stem chains -> per-group bus chains -> sum -> master chain, with a peak safety scale.

Ten console slots (``mixagent.fx``): eight per-stem/bus slots (``STEM_TOOL_ORDER``) and
six master slots (``MASTER_TOOL_ORDER``, adding ``multiband_compressor`` and
``limiter``). Every step accepts an optional ``drywet`` logit that blends the
processed signal with its input, ``y = w*wet + (1-w)*dry`` with ``w = sigmoid(drywet)``.

The grafx reverb draws fresh random noise on every forward pass, so re-rendering a
reverb-carrying chain is not bit-identical. :class:`HierRenderer` memoizes each chain
render, so an unchanged chain stays bit-identical across the agent's iterations and the
saved final mix equals the render the critic last heard.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .fx import create_tool, get_tool_spec, parameter_schema

STEM_TOOL_ORDER = ("eq", "distortion", "compressor", "noisegate", "stereo_imager",
                   "gain_panning", "delay", "reverb")
MASTER_TOOL_ORDER = ("eq", "distortion", "multiband_compressor", "stereo_imager",
                     "gain_panning", "limiter")
TOOL_ORDER = tuple(dict.fromkeys(STEM_TOOL_ORDER + MASTER_TOOL_ORDER))
STEM_TOOL_NAMES = frozenset(STEM_TOOL_ORDER)
MASTER_TOOL_NAMES = frozenset(MASTER_TOOL_ORDER)
TOOL_NAMES = frozenset(TOOL_ORDER)

DRYWET_PARAM = "drywet"
PARAM_SR = 44100  # parameter shapes are sample-rate independent for these tools

# public slot -> processor registry name
TOOL_BACKEND_ALIASES = {"compressor": "reverse_engineering_compressor"}
# constructor kwargs selecting the console's backend variants
CONSTRUCT_KWARGS: dict[str, dict[str, Any]] = {
    "multiband_compressor": {"expander_enabled": False},
    "limiter": {"fast_limiter": True},
}
STEREO_REQUIRED_TOOLS = frozenset({"stereo_imager"})


def _backend_tool_name(name: str) -> str:
    return TOOL_BACKEND_ALIASES.get(name, name)


# --------------------------------------------------------------------------- construction

def _try_create_tool(name: str, sample_rate: int) -> Any:
    backend_name = _backend_tool_name(name)
    family = get_tool_spec(backend_name).family
    if family == "grafx":
        base = {"sample_rate": sample_rate, "flashfftconv": False}
        if backend_name == "eq":
            attempts = [{**base, "num_filters": 10},
                        {**base, "backend": "lfilter", "num_filters": 10},
                        base, {"sample_rate": sample_rate, "flashfftconv": False},
                        {"sample_rate": sample_rate}, {}]
        elif backend_name == "reverb":
            attempts = [{"sr": sample_rate, "flashfftconv": False}, {"sr": sample_rate}, {}]
        else:
            attempts = [{**base, "num_filters": 1},
                        {**base, "backend": "lfilter", "num_filters": 1},
                        base, {"sample_rate": sample_rate, "flashfftconv": False},
                        {"sample_rate": sample_rate}, {}]
    elif family == "diffvox":
        attempts = [{"sr": sample_rate}, {"sample_rate": sample_rate}, {}]
    else:  # itomaster
        attempts = [{"sample_rate": sample_rate}]

    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            return create_tool(backend_name, **kwargs)
        except TypeError:
            stripped = {k: v for k, v in kwargs.items()
                        if k in ("sample_rate", "sr", "flashfftconv")}
            try:
                return create_tool(backend_name, **stripped)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err is not None:
        raise last_err
    return create_tool(backend_name)


def construct_tool(tool_name: str, sample_rate: int) -> Any:
    extra = CONSTRUCT_KWARGS.get(tool_name)
    if not extra:
        return _try_create_tool(tool_name, sample_rate)
    backend_name = _backend_tool_name(tool_name)
    try:
        return create_tool(backend_name, sample_rate=sample_rate, **extra)
    except TypeError:
        return create_tool(backend_name, **extra)


# --------------------------------------------------------------------------- schemas

def _flatten_schema(schema, prefix: str = "") -> list[tuple[str, Any]]:
    flat: list[tuple[str, Any]] = []
    for key, value in schema.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.extend(_flatten_schema(value, path))
        else:
            flat.append((path, value))
    return flat


def _slot_count_for_shape(shape: Any) -> int:
    if shape == 1 or shape == (1,):
        return 1
    if isinstance(shape, int):
        return max(1, shape)
    if isinstance(shape, tuple):
        return int(np.prod(shape))
    return 1


def _value_slot_count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 1


def _generic_schema(tool_name: str, sample_rate: int) -> dict[str, Any]:
    try:
        return parameter_schema(_backend_tool_name(tool_name), sample_rate=sample_rate)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc = _try_create_tool(tool_name, sample_rate)
    except Exception:  # noqa: BLE001
        return {}
    if hasattr(proc, "params"):
        import torch.nn as nn

        if isinstance(proc.params, nn.ParameterDict):
            return {k: tuple(p.shape) for k, p in proc.params.items()}
    if hasattr(proc, "parameter_size") and callable(proc.parameter_size):
        try:
            return proc.parameter_size()
        except Exception:  # noqa: BLE001
            return {}
    return {}


def schema_for_tool(tool_name: str, sample_rate: int = PARAM_SR) -> dict[str, Any]:
    """Raw parameter schema ``{name: shape}`` of one console slot."""
    extra = CONSTRUCT_KWARGS.get(tool_name)
    if extra:
        try:
            return parameter_schema(_backend_tool_name(tool_name), sample_rate=sample_rate,
                                    **extra)
        except Exception:  # noqa: BLE001
            pass
    return _generic_schema(tool_name, sample_rate)


def complete_params(tool_name: str, user_params: dict | None, *,
                    sample_rate: int = PARAM_SR) -> dict[str, Any]:
    """Schema defaults (all zeros) overlaid with ``user_params``, minus ``drywet``."""
    merged: dict[str, Any] = {}
    for key, shape in _flatten_schema(schema_for_tool(tool_name, sample_rate)):
        if "." in key:
            continue
        n = _slot_count_for_shape(shape)
        merged[key] = 0.0 if n == 1 else [0.0] * n
    if user_params:
        merged.update({k: v for k, v in user_params.items() if k != DRYWET_PARAM})
    return merged


# --------------------------------------------------------------------------- validation

@dataclass
class PlanValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_SCOPE_LABEL = {"both": "stem chains and master_chain", "dry": "stem chains only",
                "master": "master_chain only"}


def _tool_scope(tool: str) -> str:
    in_dry, in_master = tool in STEM_TOOL_NAMES, tool in MASTER_TOOL_NAMES
    if in_dry and in_master:
        return "both"
    return "dry" if in_dry else "master"


def validate_step_params(tool: str, params: dict, path: str, errs: list[str]) -> None:
    """Shape-check raw params (every schema key required), tolerating ``drywet``."""
    if DRYWET_PARAM in params and _value_slot_count(params[DRYWET_PARAM]) != 1:
        errs.append(f"{path}.params.{DRYWET_PARAM} must be a single number")
    expected = {k: s for k, s in _flatten_schema(schema_for_tool(tool)) if "." not in k}
    if not expected:
        return
    processor_params = {k: v for k, v in params.items() if k != DRYWET_PARAM}
    missing = sorted(set(expected) - set(processor_params))
    extra = sorted(set(processor_params) - set(expected))
    if missing:
        errs.append(f"{path}.params missing required raw parameter(s): {missing}")
    if extra:
        errs.append(f"{path}.params has unknown parameter(s): {extra}")
    for key, shape in expected.items():
        if key not in processor_params:
            continue
        want, got = _slot_count_for_shape(shape), _value_slot_count(processor_params[key])
        if got != want:
            errs.append(f"{path}.params.{key} has {got} raw slot(s), expected {want}")


def _validate_step(step, path, allowed, where, errs) -> None:
    if not isinstance(step, dict):
        errs.append(f"{path} is not a dict")
        return
    tool = step.get("tool")
    if not isinstance(tool, str) or tool not in TOOL_NAMES:
        errs.append(f"{path}.tool={tool!r} is not an allowed tool")
        return
    if tool not in allowed:
        errs.append(f"{path}.tool={tool!r} is not allowed in {where}; it is "
                    f"{_SCOPE_LABEL[_tool_scope(tool)]}")
        return
    params = step.get("params", {})
    if not isinstance(params, dict):
        errs.append(f"{path}.params is not a dict")
        return
    validate_step_params(tool, params, path, errs)


def validate_plan(plan: dict, stem_count: int) -> PlanValidation:
    """Validate stem chains + master chain (vocabulary, scope, raw parameter shapes)."""
    errs: list[str] = []
    warns: list[str] = []
    if not isinstance(plan, dict):
        return PlanValidation(False, ["plan is not a dict"])
    stems = plan.get("stems", [])
    if not isinstance(stems, list):
        errs.append("'stems' is not a list")
        stems = []
    seen: set[int] = set()
    for i, entry in enumerate(stems):
        if not isinstance(entry, dict):
            errs.append(f"stems[{i}] is not a dict")
            continue
        idx = entry.get("stem_index")
        if not isinstance(idx, int) or idx < 0 or idx >= stem_count:
            errs.append(f"stems[{i}].stem_index invalid: {idx!r} (need 0..{stem_count - 1})")
        else:
            if idx in seen:
                warns.append(f"stem_index {idx} appears multiple times")
            seen.add(idx)
        chain = entry.get("chain", [])
        if not isinstance(chain, list):
            errs.append(f"stems[{i}].chain is not a list")
            continue
        for j, step in enumerate(chain):
            _validate_step(step, f"stems[{i}].chain[{j}]", STEM_TOOL_NAMES, "stem chains",
                           errs)
    mc = plan.get("master_chain", [])
    if mc and not isinstance(mc, list):
        errs.append("'master_chain' is not a list")
    elif isinstance(mc, list):
        for j, step in enumerate(mc):
            _validate_step(step, f"master_chain[{j}]", MASTER_TOOL_NAMES, "master_chain",
                           errs)
    return PlanValidation(ok=not errs, errors=errs, warnings=warns)


# --------------------------------------------------------------------------- DSP

def ensure_2d(audio: np.ndarray) -> np.ndarray:
    return audio[:, np.newaxis] if audio.ndim == 1 else audio


def to_stereo(x: np.ndarray) -> np.ndarray:
    x = ensure_2d(x)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x.astype(np.float32, copy=False)


def sum_signals(sigs: list[np.ndarray]) -> np.ndarray:
    fixed = [to_stereo(s) for s in sigs]
    n = max(s.shape[0] for s in fixed)
    fixed = [np.pad(s, ((0, n - s.shape[0]), (0, 0))) if s.shape[0] < n else s
             for s in fixed]
    return np.sum(np.stack(fixed, axis=0), axis=0).astype(np.float32)


def _numpy_to_torch(audio: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(audio.T)).float().unsqueeze(0)


def _torch_to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, tuple):
        parts = [_torch_to_numpy(t) for t in tensor if not isinstance(t, dict)]
        if len(parts) == 2 and parts[0].shape[1] == 1 and parts[1].shape[1] == 1:
            return np.concatenate(parts, axis=1)
        if parts:
            return parts[0]
        raise ValueError("processor returned tuple with no audio tensors")
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        return np.ascontiguousarray(arr[0].T).astype(np.float32)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr.T).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _coerce_value(value: Any, shape: Any):
    import torch

    data = [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)]
    if shape == 1 or shape == (1,):
        return torch.tensor(data[0], dtype=torch.float32)
    if isinstance(shape, int) and shape > 1:
        if len(data) < shape:
            data = data + [0.0] * (shape - len(data))
        return torch.tensor(data[:shape], dtype=torch.float32)
    if isinstance(shape, tuple):
        expected = int(np.prod(shape))
        if len(data) < expected:
            data = data + [0.0] * (expected - len(data))
        return torch.tensor(data[:expected], dtype=torch.float32).reshape(shape)
    return torch.tensor(data[0], dtype=torch.float32)


def _run_processor(proc: Any, audio_t, params: dict, sample_rate: int):
    """Raw params straight into ``forward`` (every processor activates them itself)."""
    import torch

    if not hasattr(proc, "sample_rate"):
        proc.sample_rate = sample_rate
    if not hasattr(proc, "ref_sample_rate"):
        proc.ref_sample_rate = getattr(proc, "sample_rate", sample_rate)
    if hasattr(proc, "parameter_size"):
        flat = dict(_flatten_schema(proc.parameter_size()))
        raw = {k: _coerce_value(v, flat[k]) for k, v in params.items() if k in flat}
    else:
        raw = {k: _coerce_value(v, 1) for k, v in params.items()}
    batched = {}
    for key, t in raw.items():
        if not isinstance(t, torch.Tensor):
            batched[key] = t
        elif t.dim() in (0, 1, 2):
            batched[key] = t.unsqueeze(0)
        else:
            batched[key] = t
    accepted = {p.name for p in inspect.signature(proc.forward).parameters.values()
                if p.name not in ("self", "input_signals")}
    with torch.inference_mode():
        return proc(audio_t, **{k: v for k, v in batched.items() if k in accepted})


def _blend_drywet(dry: np.ndarray, wet: np.ndarray, logit: Any) -> np.ndarray:
    weight = 1.0 / (1.0 + math.exp(-float(logit)))
    dry, wet = ensure_2d(dry), ensure_2d(wet)
    if dry.shape[1] == 1 and wet.shape[1] > 1:
        dry = np.repeat(dry, wet.shape[1], axis=1)
    elif wet.shape[1] == 1 and dry.shape[1] > 1:
        wet = np.repeat(wet, dry.shape[1], axis=1)
    if dry.shape[1] != wet.shape[1]:
        channels = min(dry.shape[1], wet.shape[1])
        dry, wet = dry[:, :channels], wet[:, :channels]
    # delay/reverb tails make the wet signal longer than the dry input
    if wet.shape[0] > dry.shape[0]:
        dry = np.pad(dry, ((0, wet.shape[0] - dry.shape[0]), (0, 0)))
    elif dry.shape[0] > wet.shape[0]:
        wet = np.pad(wet, ((0, dry.shape[0] - wet.shape[0]), (0, 0)))
    return (weight * wet + (1.0 - weight) * dry).astype(np.float32, copy=False)


def apply_step(audio: np.ndarray, step: dict, sample_rate: int) -> np.ndarray:
    """One plan step. A step that fails to construct or run passes audio through."""
    tool = step.get("tool")
    if tool not in TOOL_NAMES:
        return audio
    ch = audio.shape[1]
    if (tool in STEREO_REQUIRED_TOOLS or tool == "reverb") and ch == 1:
        audio = np.repeat(audio, 2, axis=1)
        ch = 2
    step_params = step.get("params") or {}
    drywet = step_params.get(DRYWET_PARAM)
    try:
        proc = construct_tool(tool, sample_rate)
        params = complete_params(tool, step_params, sample_rate=sample_rate)
    except Exception:  # noqa: BLE001
        return audio
    try:
        out = _torch_to_numpy(_run_processor(proc, _numpy_to_torch(audio), params,
                                             sample_rate))
    except Exception:  # noqa: BLE001
        return audio
    out = ensure_2d(out)
    if drywet is not None:
        out = _blend_drywet(audio, out, drywet)
    return out


def apply_chain(audio: np.ndarray, sample_rate: int, chain: list[dict]) -> np.ndarray:
    audio = ensure_2d(audio).astype(np.float32, copy=False)
    for step in chain:
        audio = apply_step(audio, step, sample_rate)
    return audio


# --------------------------------------------------------------------------- renderer

def load_stem(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data, sr


def chain_key(chain: list[dict]) -> str:
    return json.dumps(chain, sort_keys=True)


class HierRenderer:
    """Memoizing hierarchical renderer (stem chains -> bus chains -> master).

    Caches are keyed by chain content, so any unchanged part of the plan is reused
    verbatim; the partial renders (group submix, premaster) share those caches.
    """

    def __init__(self, stem_paths: list[Path]):
        audios, sr = [], None
        for p in stem_paths:
            data, s = load_stem(p)
            sr = s if sr is None else sr
            audios.append(data)
        max_len = max(a.shape[0] for a in audios)
        self.audios = [np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) for a in audios]
        self.sr = sr
        self._stem, self._bus, self._pm, self._master = {}, {}, {}, {}

    def _proc(self, i: int, chain: list[dict]) -> np.ndarray:
        k = (i, chain_key(chain))
        v = self._stem.get(k)
        if v is None:
            v = to_stereo(apply_chain(self.audios[i], self.sr, chain))
            self._stem[k] = v
        return v

    def render_group(self, plan: dict, gi: int, *, include_bus: bool = False) -> np.ndarray:
        g = plan["groups"][gi]
        members = [m for m in g["member_stem_indices"] if 0 <= m < len(self.audios)]
        procs = [self._proc(m, plan["stems"][m]["chain"]) for m in members]
        summed = sum_signals(procs)
        if include_bus and g["chain"]:
            bk = (gi, tuple(id(p) for p in procs), chain_key(g["chain"]))
            v = self._bus.get(bk)
            if v is None:
                v = to_stereo(apply_chain(summed, self.sr, g["chain"]))
                self._bus[bk] = v
            return v
        return summed

    def render_premaster(self, plan: dict) -> np.ndarray:
        procs = [self._proc(i, plan["stems"][i]["chain"]) for i in range(len(self.audios))]
        grouped, buses = set(), []
        for gi, g in enumerate(plan["groups"]):
            members = [m for m in g["member_stem_indices"] if 0 <= m < len(procs)]
            if not members:
                continue
            grouped.update(members)
            buses.append(self.render_group(plan, gi, include_bus=True))
        ungrouped = [procs[i] for i in range(len(procs)) if i not in grouped]
        pmkey = tuple(id(x) for x in buses + ungrouped)
        v = self._pm.get(pmkey)
        if v is None:
            v = sum_signals(buses + ungrouped)
            self._pm[pmkey] = v
        return v

    def render_full(self, plan: dict) -> np.ndarray:
        procs = [self._proc(i, plan["stems"][i]["chain"]) for i in range(len(self.audios))]
        idx_of = {s["stem_index"]: pos for pos, s in enumerate(plan["stems"])}
        grouped, buses = set(), []
        for gi, g in enumerate(plan["groups"]):
            members = [idx_of[m] for m in g["member_stem_indices"] if m in idx_of]
            if not members:
                continue
            grouped.update(members)
            bk = (gi, tuple(id(procs[m]) for m in members), chain_key(g["chain"]))
            v = self._bus.get(bk)
            if v is None:
                v = to_stereo(apply_chain(sum_signals([procs[m] for m in members]),
                                          self.sr, g["chain"]))
                self._bus[bk] = v
            buses.append(v)
        ungrouped = [procs[i] for i in range(len(procs)) if i not in grouped]
        pmkey = tuple(id(x) for x in buses + ungrouped)
        pm = self._pm.get(pmkey)
        if pm is None:
            pm = sum_signals(buses + ungrouped)
            self._pm[pmkey] = pm
        mkey = (id(pm), chain_key(plan["master_chain"]))
        mix = self._master.get(mkey)
        if mix is None:
            summed = pm
            if plan["master_chain"]:
                summed = to_stereo(apply_chain(pm, self.sr, plan["master_chain"]))
            peak = float(np.max(np.abs(summed))) if summed.size else 0.0
            if peak > 1.0:
                summed = summed / peak * 0.99
            mix = summed.astype(np.float32)
            self._master[mkey] = mix
        return mix
