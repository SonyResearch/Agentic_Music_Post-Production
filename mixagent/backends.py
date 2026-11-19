"""Detector and critic backends for Stage 2.

``llm``      the served Qwen3-Omni planner model answering the detector / critic
             prompts (the paper's untrained, inference-only rows).
``trained``  two HTTP services: the anomaly-FX detector (``servers/detector_server.py``)
             and the trained critic (``servers/critic_server.py``, one of
             f1/f2/f3 x sft/dpo). Audio travels as sha256-addressed blobs that are
             uploaded once per process and then referenced.

Critic feedback formats: F1 = free-text advice, F2 = per-tool comments, F3 = per-tool
comments plus exact parameter values. The trained critic's instruction is rebuilt
exactly as in its training data, including the chain block's deterministic value
jitter (:func:`chain_block`).

All backend objects are shared by the song worker threads and are thread-safe.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as A
from .llm import AgentError, parse_json_reply
from .plan import get_step
from .render import DRYWET_PARAM, to_stereo
from .templates import (audio_order_text, contract_for, fill, format_critique,
                        make_corrupted_validator, owner_table, site_table)

# chain-block jitter the critic was trained with (last-significant-digit units, seed)
JITTER_K = 5.0
JITTER_SEED = 20260811

# detector vocabulary / capacity
DETECTOR_TOOLS = ("eq", "compressor", "noisegate", "distortion", "reverb", "delay",
                  "stereo_imager", "gain_panning", "limiter", "multiband_compressor")
MAX_CHAIN_LEN = 12
MAX_UNITS = 32

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 900.0

CRITIC_ARMS = ("f1", "f2", "f3")
CRITIC_MAX_NEW_TOKENS = {"f1": 768, "f2": 768, "f3": 3072}


class BackendError(RuntimeError):
    """Anything that makes one detector / critic call unusable."""


@dataclass
class Critique:
    """One critic verdict for one element.

    text     repair brief for the planner; "" = nothing to fix, None = unusable.
    hints    F3: per-tool parameter values for the revise-with-hints prompt.
    patches  F3 direct mode: {site_id: params} applied without a planner turn.
    """

    text: str | None
    hints: str | None = None
    patches: dict | None = None
    raw: str | None = None


# --------------------------------------------------------------------------- chain block
# Identical to the critic's SFT-data builder: values rounded to 3 significant digits
# after a per-example U(-k, k) last-significant-digit jitter.

def round_sig(x, sig=3):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if isinstance(x, int):
        return x
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def round_tree(obj, sig=3):
    if isinstance(obj, dict):
        return {k: round_tree(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_tree(v, sig) for v in obj]
    return round_sig(obj, sig)


def _leaf_floats(obj, out):
    if isinstance(obj, dict):
        for k in sorted(obj):
            _leaf_floats(obj[k], out)
    elif isinstance(obj, list):
        for v in obj:
            _leaf_floats(v, out)
    elif not isinstance(obj, bool) and isinstance(obj, (int, float)):
        out.append(float(obj))


def _last_digit_step(x, ref, sig=3):
    mag = abs(x) if x != 0 else ref
    if mag == 0 or not math.isfinite(mag):
        mag = 1.0
    return 10.0 ** (math.floor(math.log10(mag)) - (sig - 1))


def jitter_tree(obj, rng, k, ref):
    if isinstance(obj, dict):
        return {key: jitter_tree(v, rng, k, ref) for key, v in obj.items()}
    if isinstance(obj, list):
        return [jitter_tree(v, rng, k, ref) for v in obj]
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        return obj
    if not math.isfinite(obj):
        return obj
    return float(obj) + rng.uniform(-k, k) * _last_digit_step(float(obj), ref)


def chain_block(target_el, jitter=0.0, jitter_key=None, seed=0):
    lines = []
    rng = random.Random(f"{seed}:{jitter_key}") if jitter > 0.0 else None
    for c in target_el["chain"]:
        params = dict(c["current"])
        if rng is not None:
            leaves = []
            _leaf_floats(params, leaves)
            nz = sorted(abs(v) for v in leaves if v != 0)
            ref = nz[len(nz) // 2] if nz else 1.0
            params = jitter_tree(params, rng, jitter, ref)
        params = round_tree(params)
        lines.append(f"### {c['tool']}")
        lines.append(json.dumps(params, separators=(", ", ": ")))
    return "\n".join(lines)


# --------------------------------------------------------------------------- llm

class LLMDetector:
    """The planner model lists the corrupted elements of a stage, worst first."""

    name = "llm"

    def detect(self, st, *, tag, owners, det_audio, det_labels, mix_clip, mix_gain,
               elem_clips, context_note, scenario_kind, stage_rec):
        history = stage_rec.get("revisions") or []
        hist_text = "\n".join(
            f"iteration {h['iter']}: revised {h['target']} "
            f"(tools changed: {', '.join(h['tools']) or 'none'})"
            for h in history) or "(none yet)"
        det_prompt = fill(st.prompts["detector_sweep"], CONTEXT_NOTE=context_note,
                          HISTORY=hist_text, TARGET_TABLE=owner_table(owners, st.plan),
                          AUDIO_ORDER=audio_order_text(det_labels))
        return st.agent.ask_json(st.sid, det_audio, det_prompt, st.trace, tag,
                                 make_corrupted_validator({o["id"] for o in owners}))


class LLMCritic:
    """The planner model writes the critique in format F1, F2 or F3 (hint mode)."""

    name = "llm"

    def __init__(self, arm: str = "f2"):
        if arm not in CRITIC_ARMS:
            raise ValueError(f"LLM critic has no {arm!r} level (use f1, f2 or f3)")
        self.arm = arm

    def critique(self, st, *, tag, target, owners, flagged, others, why, elem_clip,
                 mix_clip, context_note, critic_key, stage_tag, stage_rec,
                 given_tools=None, mix_gain=None):
        osites = target["sites"]
        kw = dict(
            CONTEXT_NOTE=context_note,
            TARGET_DESC=target["desc"] + (f" -- detector's note: {why}" if why else ""),
            TARGET_PLAN=site_table(osites, st.plan, with_params=True),
            OTHER_FLAGGED="\n".join(f"- {o}" for o in others) or "(none)",
            AUDIO_ORDER=audio_order_text(
                ["the flagged element on its own, processed with its current settings",
                 "the current render of this stage (everything together)",
                 "the Reference (same song, different excerpt)"]))
        audio = [elem_clip, mix_clip, st.ref_clip]
        if self.arm == "f1":
            # prose brief: the reply IS the critique, passed to the planner verbatim
            crit_prompt = fill(st.prompts[f"{critic_key}_llmf1"], **kw)
            try:
                reply = st.agent.ask(st.sid, audio, crit_prompt, st.trace, tag)
            except AgentError:
                return Critique(text=None)
            text = (reply or "").strip()
            if not text:
                return Critique(text=None, raw=reply)
            if text.upper().startswith("LEAVE AS IS"):
                return Critique(text="", raw=reply)
            return Critique(text=text, raw=reply)
        if self.arm == "f3":
            crit_prompt = fill(st.prompts[f"{critic_key}_llmf3"],
                               TOOL_CONTRACT=contract_for(st.contract_header,
                                                          st.contract_blocks,
                                                          sorted({s.tool for s in osites})),
                               **kw)
        else:
            crit_prompt = fill(st.prompts[critic_key], **kw)
        try:
            reply = st.agent.ask(st.sid, audio, crit_prompt, st.trace, tag)
        except AgentError:
            return Critique(text=None)
        text = format_critique(reply, [s.tool for s in osites])
        if self.arm != "f3" or not text:
            return Critique(text=text, raw=reply)
        try:
            parsed = parse_json_reply(reply)
        except Exception:  # noqa: BLE001
            parsed = None
        hints = param_hints(parsed) if isinstance(parsed, dict) else ""
        return Critique(text=text, hints=hints or None, raw=reply)


# --------------------------------------------------------------------------- http

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _HttpBackend:
    """JSON POST/GET with a content-addressed blob side channel.

    A server that has forgotten a blob answers 409 ``{"missing": [...]}``; those are
    re-uploaded and the request retried once.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._uploaded: set[str] = set()
        self._sha: dict[tuple, str] = {}

    def sha(self, path: Path) -> str:
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            hit = self._sha.get(key)
        if hit is not None:
            return hit
        digest = _sha256_file(path)
        with self._lock:
            self._sha[key] = digest
        return digest

    def ref(self, path: Path, blobs: dict[str, Path]) -> dict:
        digest = self.sha(path)
        blobs[digest] = Path(path)
        return {"sha256": digest}

    def _upload(self, blobs: dict[str, Path], shas: list[str]) -> None:
        payload = {"blobs": [{"sha256": s,
                              "b64": base64.b64encode(Path(blobs[s]).read_bytes())
                              .decode("ascii")} for s in shas]}
        resp = self._request("POST", "/blobs", payload)
        if resp.status_code != 200:
            raise BackendError(f"POST /blobs -> HTTP {resp.status_code}: {resp.text[:300]}")
        with self._lock:
            self._uploaded.update(shas)

    def _request(self, method: str, path: str, payload=None):
        import requests

        url = f"{self.base_url}{path}"
        last = None
        for _attempt in (1, 2):
            try:
                with requests.Session() as s:
                    return s.request(method, url, json=payload,
                                     timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            except requests.RequestException as e:  # noqa: PERF203
                last = e
        raise BackendError(f"{method} {url} failed after 2 attempts: "
                           f"{type(last).__name__}: {last}")

    def get_json(self, path: str) -> dict:
        resp = self._request("GET", path)
        if resp.status_code != 200:
            raise BackendError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def post_json(self, path: str, payload: dict, blobs: dict[str, Path]) -> dict:
        with self._lock:
            unknown = [s for s in blobs if s not in self._uploaded]
        if unknown:
            self._upload(blobs, unknown)
        resp = self._request("POST", path, payload)
        if resp.status_code == 409:
            try:
                missing = list((resp.json() or {}).get("missing") or [])
            except ValueError:
                missing = []
            known = [s for s in missing if s in blobs]
            if not known:
                raise BackendError(f"POST {path} -> 409 for blobs we did not send: "
                                   f"{missing[:3]}")
            with self._lock:
                self._uploaded.difference_update(missing)
            self._upload(blobs, known)
            resp = self._request("POST", path, payload)
        if resp.status_code != 200:
            raise BackendError(f"POST {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


# --------------------------------------------------------------------------- detector

def write_dry_clip(audio: np.ndarray, sr: int, out: Path, seconds: float, norm: str) -> Path:
    """Detector 'dry' clip: ``solo16`` = -16 LUFS like the wet solos; ``raw`` = same
    crop, no loudness normalization, peak clamp only (the detector's training input)."""
    if norm == "solo16":
        return A.save_listen_clip(audio, sr, out, seconds, A.STEM_CLIP_LUFS)
    if norm != "raw":
        raise ValueError(f"unknown dry-norm {norm!r}")
    clip = A.center_crop(to_stereo(audio), sr, seconds)
    clip = np.clip(clip, -A.SAVE_PEAK_CEILING, A.SAVE_PEAK_CEILING).astype(np.float32)
    A.write_flac(clip, sr, out)
    return out


def write_wet_clip(audio: np.ndarray, sr: int, out: Path, seconds: float,
                   gain: float) -> Path:
    """Detector 'wet' clip: this unit's contribution scaled by the MIX clip's gain, so
    the wet clips of a scenario sum to the mix clip (as in the detector's training)."""
    clip = A.center_crop(to_stereo(audio), sr, seconds) * float(gain)
    clip = np.clip(clip, -A.SAVE_PEAK_CEILING, A.SAVE_PEAK_CEILING).astype(np.float32)
    A.write_flac(clip, sr, out)
    return out


def owner_identity(st, owner: dict) -> tuple[str, str, str]:
    """(name, instrument, level) of one element in the critic's training vocabulary
    (levels ``track`` / ``bus`` / ``master``)."""
    s0 = owner["sites"][0]
    if s0.level == "stem":
        entry = st.plan["stems"][s0.owner]
        name = entry.get("stem_name") or f"stem_{s0.owner}"
        info = st.stem_meta.get(name + ".flac") or {}
        return name, ((info.get("instrument") or "").strip() or "unknown"), "track"
    if s0.level == "bus":
        name = str(st.plan["groups"][s0.owner]["name"])
        return name, name, "bus"
    return "master", "master", "master"


def owner_chain(st, owner: dict) -> dict:
    """The element's chain as [{"tool", "current"}], params alphabetical, drywet last."""
    chain = []
    for s in owner["sites"]:
        params = get_step(st.plan, s)["params"]
        current = {k: params[k] for k in sorted(params) if k != DRYWET_PARAM}
        if DRYWET_PARAM in params:
            current[DRYWET_PARAM] = params[DRYWET_PARAM]
        chain.append({"tool": s.tool, "instance": f"{owner['id']}.{s.sid}.{s.tool}",
                      "current": current})
    return {"chain": chain}


def parse_labeled_blocks(text: str, required: tuple[str, ...]) -> dict[str, str]:
    """``### name`` headed blocks -> {name: body}, bodies kept byte-exact."""
    blocks, key = {}, None
    for line in text.split("\n"):
        if line.startswith("### "):
            key = line[4:].strip()
            blocks[key] = []
        elif key is not None:
            blocks[key].append(line)
    out = {k: "\n".join(v).strip("\n") for k, v in blocks.items()}
    missing = [k for k in required if not out.get(k)]
    if missing:
        raise BackendError(f"prompt missing block(s): {missing}")
    return out


def _module_detail(modules: list[dict]) -> str:
    flagged = [m for m in modules if m.get("corrupted")]
    picked = flagged or sorted(modules, key=lambda m: -float(m.get("probability") or 0.0))[:2]
    picked = sorted(picked, key=lambda m: -float(m.get("probability") or 0.0))
    return ", ".join(f"{m.get('tool')}@{m.get('position')} "
                     f"({float(m.get('probability') or 0.0):.2f})" for m in picked)


def flagged_tools(modules: list[dict]) -> list[str]:
    """Corrupted module tool names in chain order, deduped; never empty when the unit
    has modules (falls back to the single most suspicious one)."""
    out = []
    for m in sorted(modules, key=lambda m: int(m.get("position") or 0)):
        if m.get("corrupted") and m.get("tool") and m["tool"] not in out:
            out.append(str(m["tool"]))
    if not out and modules:
        top = max(modules, key=lambda m: float(m.get("probability") or 0.0))
        if top.get("tool"):
            out.append(str(top["tool"]))
    return out


def adapt_verdict(result: dict, unit_owners: dict[str, list[dict]], threshold: float) -> dict:
    """Detector result -> ``{"corrupted": [{"id", "why", "given_tools"}]}``, worst first.

    Context units (no owner) are dropped. A merged single-stem-group unit carries two
    owners (stem, then its bus); each gets an entry with the modules of its own slice,
    renumbered to its own chain positions.
    """
    rows = []
    for u in result.get("units") or []:
        specs = unit_owners.get(str(u.get("unit")))
        if not specs:
            continue
        prob = float(u.get("probability") or 0.0)
        corrupt = bool(u["corrupted"]) if "corrupted" in u else prob > threshold
        if not corrupt:
            continue
        modules = list(u.get("modules") or [])
        for spec in specs:
            mine = [dict(m, position=int(m.get("position") or 0) - spec["lo"])
                    for m in modules if spec["lo"] <= int(m.get("position") or 0) < spec["hi"]]
            detail = _module_detail(mine)
            why = f"detector p={prob:.2f}" + (f"; modules: {detail}" if detail else "")
            rows.append((prob, spec["owner_id"], why, flagged_tools(mine)))
    rows.sort(key=lambda r: -r[0])
    return {"corrupted": [{"id": oid, "why": why, "given_tools": tools}
                          for _p, oid, why, tools in rows]}


class TrainedDetector:
    """HTTP client for the anomaly-FX detector (``POST /predict``).

    A stage is sent as a *scenario*: one unit per element (dry input clip, wet output
    clip scaled by the mix gain, tool chain) plus the mix clip. At stage B, premaster
    contributors that are not elements of the sweep travel as context units, and a
    single-stem group (stem chain feeding a bus chain) is one merged unit.
    """

    name = "trained"

    def __init__(self, base_url: str, *, threshold: float = 0.5, dry_norm: str = "solo16",
                 wet_norm: str = "mix", context_units: bool = True):
        self.http = _HttpBackend(base_url)
        self.threshold = float(threshold)
        self.dry_norm = dry_norm
        self.wet_norm = wet_norm
        self.context_units = bool(context_units)
        self._written: set[str] = set()
        self._wlock = threading.Lock()

    def health(self) -> dict:
        info = self.http.get_json("/health")
        if not (info.get("ok") or info.get("status") == "ok"):
            raise BackendError(f"detector /health not ok: {json.dumps(info)[:300]}")
        return info

    def _once(self, path: Path) -> bool:
        key = str(Path(path).resolve())
        with self._wlock:
            fresh = key not in self._written
            self._written.add(key)
        return fresh

    def _dry_clip(self, st, path: Path, audio) -> Path:
        if self._once(path):
            write_dry_clip(audio, st.renderer.sr, path, st.clip_seconds, self.dry_norm)
        return path

    def _wet_clip(self, st, path: Path, audio, mix_gain: float) -> Path:
        if self._once(path):
            write_wet_clip(audio, st.renderer.sr, path, st.clip_seconds, mix_gain)
        return path

    def _premaster_contributors(self, st) -> list[tuple[str, int]]:
        """Everything ``HierRenderer.render_premaster`` sums, in its own order."""
        n = len(st.renderer.audios)
        grouped, out = set(), []
        for gi, g in enumerate(st.plan["groups"]):
            members = [m for m in g["member_stem_indices"] if 0 <= m < n]
            if not members:
                continue
            grouped.update(members)
            out.append(("group", gi))
        out += [("stem", i) for i in range(n) if i not in grouped]
        return out

    def _merge_pairs(self, st, owners, scenario_kind) -> dict[int, tuple[dict, dict]]:
        """Stage-B single-stem groups -> (stem owner, bus owner) to fuse into one unit."""
        if scenario_kind != "full":
            return {}
        stems, buses = {}, {}
        for o in owners:
            s0 = o["sites"][0]
            if s0.level == "stem" and s0.group_index is not None:
                if len(st.plan["groups"][s0.group_index]["member_stem_indices"]) == 1:
                    stems[s0.group_index] = o
            elif s0.level == "bus":
                buses[s0.owner] = o
        return {gi: (stems[gi], buses[gi]) for gi in stems if gi in buses}

    def build_units(self, st, *, tag, owners, elem_clips, scenario_kind, mix_gain,
                    stage_rec):
        """-> (unit dicts, unit name -> owner slices [{owner_id, lo, hi}], blobs)."""
        blobs: dict[str, Path] = {}
        units, unit_owners = [], {}
        covered = set()
        stage_rec["detector_wet_norm"] = self.wet_norm
        stage_rec["detector_dry_norm"] = self.dry_norm
        merges = self._merge_pairs(st, owners, scenario_kind)
        merged_bus_ids = {bus["id"] for _stem, bus in merges.values()}

        for o in owners:
            s0 = o["sites"][0]
            name, _inst, _lvl = owner_identity(st, o)
            if s0.level == "master" or o["id"] in merged_bus_ids:
                continue
            merge = merges.get(s0.group_index) if s0.level == "stem" else None
            if merge is not None and merge[0]["id"] != o["id"]:
                merge = None
            if merge is not None:
                stem_owner, bus_owner = merge
                gname = str(st.plan["groups"][s0.group_index]["name"])
                unit_name = f"{gname}#{stem_owner['id']}+{bus_owner['id']}"
                dry_audio = st.renderer.audios[s0.owner]
                wet_audio = st.render_owner_audio(bus_owner)
                wet_src = elem_clips[bus_owner["id"]]
                covered.add(("group", s0.group_index))
                specs, chain, dropped = self._merged_chain(stem_owner, bus_owner)
                stage_rec.setdefault("detector_merged_units", []).append(unit_name)
                head_id = stem_owner["id"]
            else:
                if s0.level == "stem":
                    dry_audio = st.renderer.audios[s0.owner]
                    covered.add(("group", s0.group_index) if s0.group_index is not None
                                else ("stem", s0.owner))
                else:
                    dry_audio = st.renderer.render_group(st.plan, s0.owner,
                                                         include_bus=False)
                    covered.add(("group", s0.owner))
                unit_name = f"{name}#{o['id']}"
                wet_audio = st.render_owner_audio(o)
                wet_src = elem_clips[o["id"]]
                chain, dropped = self._chain_for(o)
                specs = [{"owner_id": o["id"], "lo": 0, "hi": len(chain)}]
                head_id = o["id"]

            dry = self._dry_clip(st, st.trace / f"{tag}_dry{head_id}.flac", dry_audio)
            if self.wet_norm == "mix":
                wet = self._wet_clip(st, st.trace / f"{tag}_det_wet{head_id}.flac",
                                     wet_audio, mix_gain)
            else:
                wet = wet_src
            if dropped:
                stage_rec.setdefault("detector_chain_truncated", []).append(unit_name)
            units.append({"name": unit_name, "owner_id": head_id,
                          "dry": self.http.ref(dry, blobs), "wet": self.http.ref(wet, blobs),
                          "chain": chain})
            unit_owners[unit_name] = specs

        if self.context_units and scenario_kind == "full":
            for kind, idx in self._premaster_contributors(st):
                if (kind, idx) in covered:
                    continue
                if kind == "group":
                    g = st.plan["groups"][idx]
                    label = f"ctx_{g['name']}#g{idx}"
                    dry_audio = st.renderer.render_group(st.plan, idx, include_bus=False)
                    wet_audio = st.renderer.render_group(st.plan, idx, include_bus=True)
                else:
                    entry = st.plan["stems"][idx]
                    label = f"ctx_{entry.get('stem_name') or idx}#s{idx}"
                    dry_audio = st.renderer.audios[idx]
                    wet_audio = st.renderer._proc(idx, entry["chain"])
                slug = re.sub(r"[^A-Za-z0-9_-]", "_", label)
                dry = self._dry_clip(st, st.trace / f"{tag}_{slug}_dry.flac", dry_audio)
                if self.wet_norm == "mix":
                    wet = self._wet_clip(st, st.trace / f"{tag}_{slug}_wet.flac",
                                         wet_audio, mix_gain)
                else:
                    wet = self._dry_clip(st, st.trace / f"{tag}_{slug}_wet.flac", wet_audio)
                units.append({"name": label, "owner_id": None,
                              "dry": self.http.ref(dry, blobs),
                              "wet": self.http.ref(wet, blobs), "chain": []})

        if len(units) > MAX_UNITS:  # context units are expendable, owners are not
            keep = [u for u in units if u["owner_id"] is not None]
            stage_rec["detector_context_units_dropped"] = len(units) - len(keep)
            if len(keep) <= MAX_UNITS:
                units = keep + [u for u in units if u["owner_id"] is None][
                    : MAX_UNITS - len(keep)]
            else:  # more owners than one call can score: detect() batches them
                units = keep
        return units, unit_owners, blobs

    @staticmethod
    def _chain_for(owner: dict) -> tuple[list[str], bool]:
        tools = [s.tool for s in owner["sites"]]
        unknown = [t for t in tools if t not in DETECTOR_TOOLS]
        if unknown:
            raise BackendError(f"chain has tools the detector never saw: {unknown}")
        return tools[:MAX_CHAIN_LEN], len(tools) > MAX_CHAIN_LEN

    @classmethod
    def _merged_chain(cls, stem_owner: dict, bus_owner: dict):
        """Stem chain then bus chain, with the owner slice each half maps back to."""
        stem_tools, _ = cls._chain_for(stem_owner)
        bus_tools, _ = cls._chain_for(bus_owner)
        chain = (stem_tools + bus_tools)[:MAX_CHAIN_LEN]
        n_stem = min(len(stem_tools), len(chain))
        specs = [{"owner_id": stem_owner["id"], "lo": 0, "hi": n_stem},
                 {"owner_id": bus_owner["id"], "lo": n_stem, "hi": len(chain)}]
        return specs, chain, len(stem_tools) + len(bus_tools) > MAX_CHAIN_LEN

    def detect(self, st, *, tag, owners, det_audio, det_labels, mix_clip, mix_gain,
               elem_clips, context_note, scenario_kind, stage_rec):
        try:
            units, unit_owners, blobs = self.build_units(
                st, tag=tag, owners=owners, elem_clips=elem_clips,
                scenario_kind=scenario_kind, mix_gain=mix_gain, stage_rec=stage_rec)
            if not units:
                return {"corrupted": []}, []
            mix_ref = self.http.ref(mix_clip, blobs)
            batches = [units[i:i + MAX_UNITS] for i in range(0, len(units), MAX_UNITS)]
            if len(batches) > 1:
                stage_rec["detector_batches"] = len(batches)
            scored = []
            for b, chunk in enumerate(batches, start=1):
                payload = {"scenario_kind": scenario_kind, "threshold": self.threshold,
                           "mix": mix_ref, "units": chunk}
                suffix = "" if len(batches) == 1 else f".b{b}"
                (st.trace / f"{tag}{suffix}.request.json").write_text(
                    json.dumps(payload, indent=1))
                part = self.http.post_json("/predict", payload, blobs)
                (st.trace / f"{tag}{suffix}.answer.txt").write_text(json.dumps(part, indent=1))
                scored += list(part.get("units") or [])
            result = {"units": scored}
        except BackendError as e:
            stage_rec.setdefault("detector_errors", []).append(str(e))
            return None, [str(e)]
        stage_rec.setdefault("detector_probs", []).append(
            {"tag": tag, "scenario_kind": scenario_kind, "threshold": self.threshold,
             "mix_gain": round(float(mix_gain), 6), "wet_norm": self.wet_norm,
             "units": [{"unit": u.get("unit"),
                        "owner_ids": [s["owner_id"] for s in
                                      unit_owners.get(str(u.get("unit"))) or []],
                        "corrupted": u.get("corrupted"), "probability": u.get("probability"),
                        "modules": u.get("modules")} for u in scored]})
        return adapt_verdict(result, unit_owners, self.threshold), []


# --------------------------------------------------------------------------- critic

def param_hints(parsed: dict) -> str:
    """F3 reply -> PARAM_HINTS block: per tool ``### tool`` + compact JSON values."""
    lines = []
    for entry in parsed.get("tools") or []:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool", "")).strip()
        payload = dict(entry.get("parameters") or {})
        if entry.get("drywet") is not None:
            payload[DRYWET_PARAM] = entry["drywet"]
        if not tool or not payload:
            continue
        lines.append(f"### {tool}")
        lines.append(json.dumps(payload, separators=(", ", ": ")))
    return "\n".join(lines)


def direct_patches(target: dict, parsed: dict) -> dict[int, dict]:
    """F3 reply -> {site_id: params}, routed to the target's own sites by tool name."""
    used, out = set(), {}
    for entry in parsed.get("tools") or []:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool", "")).strip()
        params = dict(entry.get("parameters") or {})
        if entry.get("drywet") is not None:
            params[DRYWET_PARAM] = entry["drywet"]
        if not tool or not params:
            continue
        site = next((s for s in target["sites"] if s.tool == tool and s.sid not in used),
                    None)
        if site is None:
            continue
        used.add(site.sid)
        out[site.sid] = params
    return out


class TrainedCritic:
    """HTTP client for the trained critic (``POST /critique``), given-tools prompts."""

    name = "trained"

    def __init__(self, base_url: str, *, arm: str, max_new_tokens: int | None = None,
                 f3_mode: str = "hint", jitter_k: float = JITTER_K,
                 jitter_seed: int = JITTER_SEED):
        if arm not in CRITIC_ARMS:
            raise ValueError(f"unknown critic arm {arm!r}")
        self.http = _HttpBackend(base_url)
        self.arm = arm
        self.jitter_k = float(jitter_k)
        self.jitter_seed = int(jitter_seed)
        self.f3_mode = f3_mode
        self.max_new_tokens = int(max_new_tokens or CRITIC_MAX_NEW_TOKENS[arm])

    def health(self) -> dict:
        info = self.http.get_json("/health")
        if not (info.get("ok") or info.get("status") == "ok"):
            raise BackendError(f"critic /health not ok: {json.dumps(info)[:300]}")
        if info.get("use_dry_clip"):
            raise BackendError("critic server uses the 4-clip (dry clip) protocol, which "
                               "this agent does not send")
        return info

    def build_instruction(self, st, *, target, owners, flagged, jitter_key,
                          given_tools=None) -> str:
        clip = parse_labeled_blocks(st.prompts["critic_trained_clipdesc"], ("solo", "master"))
        name, instrument, level = owner_identity(st, target)
        # the detector's order, with the target prepended when the detector did not name it
        flagged_ids = [f["id"] for f in flagged]
        if target["id"] not in flagged_ids:
            flagged_ids = [target["id"]] + flagged_ids
        parts = []
        for fid in flagged_ids:
            o = owners[fid] if 0 <= fid < len(owners) else None
            if o is None:
                continue
            fname, finst, _ = owner_identity(st, o)
            parts.append(f"{fname} ({finst})")
        flagged_list = "; ".join(parts) if parts else f"{name} ({instrument})"
        hdr = parse_labeled_blocks(st.prompts["critic_trained_chainhdr"], ("full", "stripped"))
        tools = list(given_tools or [])
        if not tools:  # never hand the model an empty diagnosis
            for s in target["sites"]:
                if s.tool not in tools:
                    tools.append(s.tool)
        block = chain_block(owner_chain(st, target), jitter=self.jitter_k,
                            jitter_key=jitter_key, seed=self.jitter_seed)
        return fill(st.prompts[f"critic_trained_{self.arm}g"],
                    TARGET_CLIP_DESC=clip["master"] if level == "master" else clip["solo"],
                    FLAGGED_LIST=flagged_list, TARGET_NAME=name,
                    TARGET_INSTRUMENT=instrument, TARGET_LEVEL=level,
                    DIAGNOSIS_TOOLS=", ".join(tools), CHAIN_HEADER=hdr["full"],
                    CHAIN_BLOCK=block)

    def critique(self, st, *, tag, target, owners, flagged, others, why, elem_clip,
                 mix_clip, context_note, critic_key, stage_tag, stage_rec,
                 given_tools=None, mix_gain=None):
        jitter_key = f"{st.sid}:{stage_tag}:{target['id']}"
        stage_rec.setdefault("critic_given_tools", []).append(
            {"id": target["id"], "tools": list(given_tools or []),
             "source": "detector_modules" if given_tools else "element_chain"})
        try:
            instruction = self.build_instruction(st, target=target, owners=owners,
                                                 flagged=flagged, jitter_key=jitter_key,
                                                 given_tools=given_tools)
        except Exception as e:  # noqa: BLE001 - one element, never the whole sweep
            stage_rec.setdefault("critic_errors", []).append(
                f"{tag}: instruction build failed: {type(e).__name__}: {e}")
            return Critique(text=None)

        blobs: dict[str, Path] = {}
        # master elements: the target clip IS the mix clip
        payload = {"instruction": instruction,
                   "target": self.http.ref(elem_clip, blobs),
                   "mix": self.http.ref(mix_clip, blobs),
                   "ref": self.http.ref(st.ref_clip, blobs),
                   "max_new_tokens": self.max_new_tokens}
        listed = [("target", elem_clip), ("mix", mix_clip), ("ref", st.ref_clip)]
        (st.trace / f"{tag}.prompt.txt").write_text(
            "\n".join(f"[audio {i + 1}] {k}: {payload[k]['sha256']} ({p})"
                      for i, (k, p) in enumerate(listed)) + "\n\n" + instruction)
        (st.trace / f"{tag}.request.json").write_text(json.dumps(
            {**payload, "instruction": f"<see {tag}.prompt.txt>"}, indent=1))

        answer, parsed, errors = None, None, []
        for round_no in (1, 2):  # one retry when the reply carries no JSON object
            try:
                data = self.http.post_json("/critique", payload, blobs)
            except BackendError as e:
                errors.append(str(e))
                break
            answer = str(data.get("answer") or "")
            (st.trace / (f"{tag}.answer.txt" if round_no == 1
                         else f"{tag}.retry.answer.txt")).write_text(json.dumps(data, indent=1))
            try:
                parsed = parse_json_reply(answer)
            except Exception as e:  # noqa: BLE001
                errors.append(f"round {round_no}: reply carries no JSON object: {e}")
                parsed = None
                continue
            break
        if parsed is None:
            stage_rec.setdefault("critic_errors", []).extend(f"{tag}: {e}" for e in errors[:2])
            return Critique(text=None)
        return self.adapt(parsed, answer, target)

    def adapt(self, parsed: dict, answer: str, target: dict) -> Critique:
        tools = [s.tool for s in target["sites"]]
        if self.arm == "f1":
            return Critique(text=str((parsed or {}).get("feedback", "")).strip(), raw=answer)
        if self.arm == "f2":
            return Critique(text=format_critique(answer, tools), raw=answer)
        text = format_critique(answer, tools)
        if not text:
            return Critique(text=text, raw=answer)
        hints = param_hints(parsed)
        patches = direct_patches(target, parsed) if self.f3_mode == "direct" else None
        return Critique(text=text, hints=hints or None, patches=patches or None, raw=answer)


# --------------------------------------------------------------------------- factory

def build_backends(cfg) -> tuple[object, object, object]:
    """-> (detector, critic, llm_detector) from a resolved preset config."""
    llm_detector = LLMDetector()
    detector = llm_detector
    if cfg.detector_backend == "trained":
        detector = TrainedDetector(cfg.detector_url, threshold=cfg.detector_threshold,
                                   dry_norm=cfg.detector_dry_norm, wet_norm="mix",
                                   context_units=True)
    if cfg.critic_backend == "trained":
        critic = TrainedCritic(cfg.critic_url, arm=cfg.critic_arm,
                               max_new_tokens=cfg.critic_max_new_tokens,
                               f3_mode=cfg.critic_f3_mode)
    else:
        critic = LLMCritic(cfg.critic_arm)
    return detector, critic, llm_detector


def check_health(cfg, detector, critic) -> dict:
    """Both trained services up, and the critic serving the requested feedback arm."""
    info = {}
    if cfg.detector_backend == "trained":
        info["detector"] = detector.health()
    if cfg.critic_backend == "trained":
        chealth = critic.health()
        served = str(chealth.get("arm") or "")
        if served != cfg.critic_arm and not served.startswith(cfg.critic_arm + "_"):
            raise BackendError(f"critic server serves arm {served!r} but the preset "
                               f"needs {cfg.critic_arm!r}")
        info["critic"] = chealth
    return info
