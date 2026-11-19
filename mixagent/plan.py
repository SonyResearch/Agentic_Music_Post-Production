"""Plan utilities: Stage-1 tools plan -> parameterized plan, sites, patches, clamping.

The chain structure (which tools, in which order) is fixed by Stage 1 and never
changes in Stage 2: the LLM only supplies parameter values for harness-enumerated
*sites* (one site = one step of one stem / bus / master chain). Every patch is
validated against the tool's raw-parameter schema and every plan update is followed
by :func:`sanitize_full_plan`, which clamps values into a safe box and asserts that
the chain fingerprint is unchanged.

Internal plan schema::

    {"stems":  [{"stem_index": i0, "stem_name", "group", "chain": [step, ...]}],
     "groups": [{"name", "member_stem_indices": [i0, ...], "chain": [step, ...]}],
     "master_chain": [step, ...]}
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

from . import render as R

# --------------------------------------------------------------------------- presets
# Initial values for every step before the planner fills it in. Raw pre-activation
# values; {} means the schema default (all zeros) is already transparent.


def _logit(w: float) -> float:
    return math.log(w / (1.0 - w))


GENERIC_PRESETS: dict[str, dict] = {
    "eq": {},
    "gain_panning": {},
    "stereo_imager": {},
    "compressor": {"cmp_ratio": -4.0, "cmp_th": -18.0, "exp_ratio": 0.5, "exp_th": -55.0,
                   "make_up": 0.0, "at": 0.0, "rt": 0.0, "avg_coef": 0.0},
    "noisegate": {"log_threshold": -3.0, "log_ratio": -1.0, "log_knee": 0.0,
                  "z_alpha_pre": 0.0},
    "distortion": {"drive_db": -5.0},
    "delay": {"delay": -3.0, "feedback": -3.0, "gain": -3.0, "eq_freq": 0.0, "eq_Q": 0.0,
              "log_damp": 0.0, R.DRYWET_PARAM: _logit(0.20)},
    "reverb": {"log_decay": [0.0] * 24, "log_gain": [0.2] * 24,
               R.DRYWET_PARAM: _logit(0.25)},
    "multiband_compressor": {R.DRYWET_PARAM: -9.0},
    "limiter": {R.DRYWET_PARAM: -9.0},
}


def preset_step(tool: str) -> dict:
    """A ``{tool, params}`` step at its preset, expanded to the full raw schema."""
    raw = json.loads(json.dumps(GENERIC_PRESETS.get(tool, {})))
    drywet = raw.get(R.DRYWET_PARAM)
    full = R.complete_params(tool, raw)
    if drywet is not None:
        full[R.DRYWET_PARAM] = drywet
    return {"tool": tool, "params": full}


def convert_tools_plan(tools_plan: dict) -> tuple[dict, list[str]]:
    """Stage-1 tools plan (1-based indices, tool-name strings) -> internal plan.

    Tools outside a scope's vocabulary are dropped; a master ``compressor`` becomes a
    ``multiband_compressor``. Returns ``(plan, deviations)``.
    """
    deviations: list[str] = []
    stems_out = []
    for s in tools_plan.get("stems", []) or []:
        chain = [preset_step(t) for t in (s.get("chain") or []) if t in R.STEM_TOOL_NAMES]
        for t in (s.get("chain") or []):
            if t not in R.STEM_TOOL_NAMES:
                deviations.append(f"stem {s['stem_index']}: dropped non-dry tool {t!r}")
        stems_out.append({"stem_index": int(s["stem_index"]) - 1,
                          "stem_name": s.get("stem_name"), "group": s.get("group"),
                          "chain": chain})
    groups_out = []
    for g in tools_plan.get("groups", []) or []:
        chain = [preset_step(t) for t in (g.get("chain") or []) if t in R.STEM_TOOL_NAMES]
        for t in (g.get("chain") or []):
            if t not in R.STEM_TOOL_NAMES:
                deviations.append(f"group {g.get('name')!r}: dropped non-dry tool {t!r}")
        groups_out.append({"name": g.get("name"),
                           "member_stem_indices": [int(m) - 1 for m in
                                                   (g.get("member_stem_indices") or [])],
                           "chain": chain})
    master_out = []
    for t in (tools_plan.get("master_chain") or []):
        if t in R.MASTER_TOOL_NAMES:
            master_out.append(preset_step(t))
        elif t == "compressor":
            master_out.append(preset_step("multiband_compressor"))
            deviations.append("master: mapped compressor -> multiband_compressor")
        else:
            deviations.append(f"master: dropped {t!r} (not a master tool)")
    return {"stems": stems_out, "groups": groups_out, "master_chain": master_out}, deviations


# --------------------------------------------------------------------------- clamping
# Safe raw ranges per tool (list-valued params use the same range per slot).
SAFE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "eq": {"w0": (-5.0, 0.8), "q_inv": (-1.2, 0.8), "log_gain": (-0.35, 0.25)},
    "distortion": {"drive_db": (-5.0, -2.0)},
    "compressor": {"at": (-2.0, 2.0), "rt": (-2.0, 2.0), "avg_coef": (-2.0, 1.0),
                   "cmp_ratio": (-4.0, -2.0), "exp_ratio": (-0.5, 2.0),
                   "cmp_th": (-30.0, -12.0), "exp_th": (-70.0, -40.0),
                   "make_up": (-3.0, 3.0)},
    "noisegate": {"log_threshold": (-3.5, -0.5), "log_ratio": (-1.5, 0.5),
                  "log_knee": (-1.0, 0.8), "z_alpha_pre": (-2.0, 2.0)},
    "stereo_imager": {"log_gain": (-0.5, 0.35)},
    "gain_panning": {"log_gain": (-0.9, 0.25)},
    "delay": {"delay": (-4.5, -1.0), "feedback": (-4.0, -1.5), "gain": (-5.0, -2.0),
              "eq_freq": (-2.0, 0.5), "eq_Q": (-1.0, 1.0), "log_damp": (-2.0, 1.0)},
    "reverb": {"log_decay": (-2.0, 0.8), "log_gain": (0.05, 0.35)},
}
# master-only tools: wide guard rails against absurd raw values
MASTER_SAFE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "limiter": {"threshold": (-1.0, 5.5), "at": (-3.0, 3.0), "rt": (-3.0, 3.0)},
    "multiband_compressor": {
        "low_cutoff": (-2.5, 2.5), "high_cutoff": (-2.5, 2.5),
        "low_shelf_comp_thresh": (-3.0, 3.0), "mid_band_comp_thresh": (-3.0, 3.0),
        "high_shelf_comp_thresh": (-3.0, 3.0),
        "low_shelf_comp_ratio": (-4.0, 2.0), "mid_band_comp_ratio": (-4.0, 2.0),
        "high_shelf_comp_ratio": (-4.0, 2.0),
        "low_shelf_comp_at": (-3.0, 3.0), "mid_band_comp_at": (-3.0, 3.0),
        "high_shelf_comp_at": (-3.0, 3.0),
        "low_shelf_comp_rt": (-3.0, 3.0), "mid_band_comp_rt": (-3.0, 3.0),
        "high_shelf_comp_rt": (-3.0, 3.0),
    },
}
DRYWET_RANGE = (-9.0, 6.0)
DISTORTION_DRIVE_CAP = -3.5
HI_BAND_W0_THRESHOLD = -0.7   # an EQ band with w0 >= this is "high"
HI_BAND_POS_GAIN_CAP = 0.10   # cumulative positive high-band log_gain per chain


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v) -> str:
    if _is_num(v):
        r = round(float(v), 6)
        return f"{r:.1f}" if r == int(r) else f"{r:g}"
    return str(v)


def _clamp_chain(chain: list[dict], loc: str, actions: list[str]) -> None:
    """Distortion cap -> high-band EQ boost budget -> safe-range clamp (in place)."""
    for j, step in enumerate(chain):
        if step.get("tool") != "distortion":
            continue
        params = step.get("params")
        v = params.get("drive_db") if isinstance(params, dict) else None
        if _is_num(v):
            nv = min(float(v), DISTORTION_DRIVE_CAP)
            if abs(nv - v) > 1e-9:
                actions.append(f"{loc}[{j}] distortion: drive_db {_fmt(v)} -> {_fmt(nv)}")
                params["drive_db"] = nv

    total_kept = 0.0  # bands are budgeted in encounter order across the chain's EQs
    for j, step in enumerate(chain):
        if step.get("tool") != "eq":
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        w0, log_gain = params.get("w0"), params.get("log_gain")
        if not (isinstance(w0, list) and isinstance(log_gain, list)):
            continue
        for slot in range(min(len(w0), len(log_gain))):
            wv, gv = w0[slot], log_gain[slot]
            if not (_is_num(wv) and _is_num(gv)):
                continue
            if wv < HI_BAND_W0_THRESHOLD or gv <= 0.0:
                continue
            remaining = HI_BAND_POS_GAIN_CAP - total_kept
            kept = 0.0 if remaining <= 0.0 else min(float(gv), remaining)
            total_kept += kept
            if abs(kept - gv) > 1e-9:
                actions.append(f"{loc}[{j}] eq: hi-band boost budget clipped "
                               f"log_gain[{slot}] {_fmt(gv)} -> {_fmt(kept)}")
                log_gain[slot] = kept

    for j, step in enumerate(chain):
        tool = step.get("tool")
        ranges = SAFE_RANGES.get(tool)
        params = step.get("params")
        if not ranges or not isinstance(params, dict):
            continue
        for pname, (lo, hi) in ranges.items():
            if pname not in params:
                continue
            val = params[pname]
            if isinstance(val, list):
                for slot, sv in enumerate(val):
                    if _is_num(sv):
                        nv = min(max(float(sv), lo), hi)
                        if abs(nv - sv) > 1e-9:
                            actions.append(f"{loc}[{j}] {tool}: {pname}[{slot}] "
                                           f"{_fmt(sv)} -> {_fmt(nv)}")
                            val[slot] = nv
            elif _is_num(val):
                nv = min(max(float(val), lo), hi)
                if abs(nv - val) > 1e-9:
                    actions.append(f"{loc}[{j}] {tool}: {pname} {_fmt(val)} -> {_fmt(nv)}")
                    params[pname] = nv


def _clamp_step_extra(step: dict, actions: list[str], where: str) -> None:
    """Master-tool ranges + the drywet range, for every step."""
    tool = step.get("tool")
    params = step.get("params") or {}

    def clamp_val(key, v, lo, hi):
        nv = min(max(float(v), lo), hi)
        if abs(nv - float(v)) > 1e-9:
            actions.append(f"{where} {tool}.{key}: {v} -> {nv}")
            return nv
        return float(v)

    for key, (lo, hi) in (MASTER_SAFE_RANGES.get(tool) or {}).items():
        v = params.get(key)
        if _is_num(v):
            params[key] = clamp_val(key, v, lo, hi)
        elif isinstance(v, list):
            params[key] = [clamp_val(f"{key}[{i}]", x, lo, hi) if _is_num(x) else x
                           for i, x in enumerate(v)]
    dw = params.get(R.DRYWET_PARAM)
    if _is_num(dw):
        nv = min(max(float(dw), DRYWET_RANGE[0]), DRYWET_RANGE[1])
        if abs(nv - float(dw)) > 1e-9:
            actions.append(f"{where} {tool}.{R.DRYWET_PARAM}: {dw} -> {nv}")
            params[R.DRYWET_PARAM] = nv


def chain_fingerprint(plan: dict) -> str:
    return json.dumps({
        "stems": [[st["tool"] for st in s["chain"]] for s in plan["stems"]],
        "groups": [[st["tool"] for st in g["chain"]] for g in plan["groups"]],
        "master": [st["tool"] for st in plan["master_chain"]],
    })


def sanitize_full_plan(plan: dict, n_stems: int) -> tuple[dict, list[str], list[str]]:
    """Validate (stems + master) and clamp parameters; never touches chains.

    Returns ``(new_plan, validation_errors, clamp_actions)``; the input is not mutated.
    """
    flat = {"stems": [{"stem_index": s["stem_index"], "chain": s["chain"]}
                      for s in plan["stems"]],
            "master_chain": plan["master_chain"]}
    v = R.validate_plan(flat, n_stems)
    errors = list(v.errors) if not v.ok else []

    actions: list[str] = []
    out = copy.deepcopy(plan)
    for i, s in enumerate(out["stems"]):
        _clamp_chain(s["chain"], f"stems[{i}].chain", actions)
    for j, g in enumerate(out["groups"]):
        _clamp_chain(g["chain"], f"stems[{len(out['stems']) + j}].chain", actions)
    _clamp_chain(out["master_chain"], "master_chain", actions)
    out = json.loads(json.dumps(out))

    for s in out["stems"]:
        for st in s["chain"]:
            _clamp_step_extra(st, actions, f'stem {s["stem_index"]}')
    for g in out["groups"]:
        for st in g["chain"]:
            _clamp_step_extra(st, actions, f'bus "{g["name"]}"')
    for st in out["master_chain"]:
        _clamp_step_extra(st, actions, "master")

    if chain_fingerprint(out) != chain_fingerprint(plan):
        raise RuntimeError("sanitize changed the chain structure -- immutability broken")
    return out, errors, actions


# --------------------------------------------------------------------------- sites

@dataclass
class Site:
    sid: int                    # global site id (stable, 0-based)
    level: str                  # stem | bus | master
    owner: int                  # stem_index / group_index / 0
    step_idx: int
    tool: str
    stage: str                  # A | B | C
    group_index: int | None     # for stem sites: owning group index
    desc: str                   # human-readable location for prompts


def enumerate_sites(plan: dict, meta: dict[str, dict]) -> list[Site]:
    """Deterministic site list: stems (plan order) -> buses -> master.

    Stage A = stem sites of multi-stem groups; stage B = stem sites of single-stem
    groups (or ungrouped stems) + every bus site; stage C = master sites.
    """
    group_size = {gi: len(g["member_stem_indices"]) for gi, g in enumerate(plan["groups"])}
    group_of: dict[int, int] = {}
    for gi, g in enumerate(plan["groups"]):
        for m in g["member_stem_indices"]:
            group_of[m] = gi
    sites: list[Site] = []
    nid = 0
    for s in plan["stems"]:
        i = s["stem_index"]
        gi = group_of.get(i)
        multi = gi is not None and group_size[gi] >= 2
        name = s.get("stem_name") or f"stem_{i}"
        inst = (meta.get(name + ".flac") or {}).get("instrument", "")
        where = f'stem {i} "{name}"' + (f" ({inst})" if inst else "")
        for j, step in enumerate(s["chain"]):
            sites.append(Site(nid, "stem", i, j, step["tool"], "A" if multi else "B", gi,
                              where))
            nid += 1
    for gi, g in enumerate(plan["groups"]):
        where = f'bus "{g["name"]}" ({group_size[gi]} stems)'
        for j, step in enumerate(g["chain"]):
            sites.append(Site(nid, "bus", gi, j, step["tool"], "B", gi, where))
            nid += 1
    for j, step in enumerate(plan["master_chain"]):
        sites.append(Site(nid, "master", 0, j, step["tool"], "C", None, "master bus"))
        nid += 1
    return sites


def get_step(plan: dict, site: Site) -> dict:
    if site.level == "stem":
        step = plan["stems"][site.owner]["chain"][site.step_idx]
    elif site.level == "bus":
        step = plan["groups"][site.owner]["chain"][site.step_idx]
    else:
        step = plan["master_chain"][site.step_idx]
    if step["tool"] != site.tool:
        raise RuntimeError(f"site {site.sid}: tool mismatch {step['tool']} != {site.tool}")
    return step


def _coerce_numeric(value):
    """Floats only (scalars or flat lists). Raises ValueError on anything else."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a parameter value")
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("non-finite value")
        return v
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    raise ValueError(f"non-numeric value {value!r}")


def apply_site_params(plan: dict, site: Site, raw_params) -> list[str]:
    """Patch ONE site's params (partial updates merge over current values).

    Returns a list of errors; an empty list means the patch was applied.
    """
    if not isinstance(raw_params, dict):
        return [f"site {site.sid}: params must be a JSON object"]
    if set(raw_params) == {site.tool} and isinstance(raw_params[site.tool], dict):
        raw_params = raw_params[site.tool]  # {"eq": {...}} wrapper
    step = get_step(plan, site)
    schema_keys = set(R.complete_params(site.tool, {}))
    unknown = [k for k in raw_params if k not in schema_keys and k != R.DRYWET_PARAM]
    if unknown:
        return [f"site {site.sid} ({site.tool}): unknown parameter(s) {unknown}; "
                f"allowed: {sorted(schema_keys)} + optional {R.DRYWET_PARAM}"]
    try:
        cleaned = {k: _coerce_numeric(v) for k, v in raw_params.items()}
    except ValueError as e:
        return [f"site {site.sid} ({site.tool}): {e}"]
    merged = dict(step["params"])
    merged.update(cleaned)
    drywet = merged.pop(R.DRYWET_PARAM, None)
    full = R.complete_params(site.tool, merged)
    if drywet is not None:
        full[R.DRYWET_PARAM] = float(drywet)
    errs: list[str] = []
    R.validate_step_params(site.tool, full, f"site {site.sid} ({site.tool})", errs)
    if errs:
        return errs
    step["params"] = full
    return []
