"""Stage 2: hierarchical parameter tuning of a fixed tool plan (detector -> critic -> revise).

Stage 1 fixed WHICH tools every stem, bus and the master get. Stage 2 fills in their
PARAMETERS, bottom-up:

  Stage A  every multi-stem group: the member stems' chains, listening to the group
           submix (single-stem groups are handled in stage B).
  Stage B  every bus chain + the stem chains of single-stem groups, listening to the
           premaster (all buses summed).
  Stage C  the master chain, listening to the full mix.

Each stage: the planner proposes initial values -> render -> ONE detector pass lists
the corrupted elements worst first -> for each flagged element, the critic writes a
repair (F1 advice / F2 per-tool comments / F3 exact values) and the planner revises
that element's parameters (F3 "direct": the critic's values are applied without a
planner turn) -> next stage. Stage C has a single element and skips the detector.
The critic hears the stage mix the detector heard for the whole sweep.

The LLM never adds, removes or reorders tools; a chain fingerprint is asserted after
every update. GT is never read by the agent; ``--log-mrs`` only logs the MRS of the
current mix to GT as a diagnostic.

Output per song::

    <out>/<sid>/<sid>.flac            final mix, -12 LUFS, peak-clamped
               /<sid>_plan_final.json  final plan with parameters
               /<sid>.json             per-song record (stages, flags, revisions)
               /stem_order.txt
               /trace/...              every prompt / answer / clip
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from . import audio as A
from . import metrics as M
from .backends import Critique
from .llm import QwenAgent
from .plan import (Site, apply_site_params, chain_fingerprint, convert_tools_plan,
                   enumerate_sites, sanitize_full_plan)
from .render import TOOL_NAMES, HierRenderer, to_stereo
from .templates import (audio_order_text, contract_for, element_audio_order, fill,
                        group_owners, make_sites_validator, site_table)

AUDIO_EXTS = (".flac", ".wav")
_LOG_LOCK = threading.Lock()


def log(sid: str, msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] [{sid}] {msg}", flush=True)


# --------------------------------------------------------------------------- presets

@dataclass(frozen=True)
class CriticPreset:
    detector_backend: str        # llm | trained
    critic_backend: str          # llm | trained
    critic_arm: str              # f1 | f2 | f3
    critic_f3_mode: str = "hint"     # hint | direct
    detector_threshold: float = 0.5
    detector_dry_norm: str = "solo16"
    critic_max_new_tokens: int | None = None


def _trained(arm: str) -> CriticPreset:
    return CriticPreset("trained", "trained", arm,
                        critic_f3_mode="direct" if arm == "f3" else "hint",
                        detector_threshold=0.15, detector_dry_norm="raw",
                        critic_max_new_tokens=3072 if arm == "f3" else 768)


PRESETS: dict[str, CriticPreset] = {
    "f1_none": CriticPreset("llm", "llm", "f1"),
    "f2_none": CriticPreset("llm", "llm", "f2"),
    "f3_none": CriticPreset("llm", "llm", "f3"),
    **{f"{arm}_{t}": _trained(arm) for arm in ("f1", "f2", "f3") for t in ("sft", "dpo")},
}


@dataclass
class RunConfig:
    """Everything one Stage-2 run needs besides the planner agents."""
    dataset_root: Path
    tool_plans: Path
    captions_dir: Path
    out: Path
    preset: str
    detector_url: str
    critic_url: str
    clip_seconds: float = 14.0
    log_mrs: bool = True
    overwrite: bool = False

    def __getattr__(self, name):  # preset fields read through the config
        return getattr(PRESETS[self.__dict__["preset"]], name)


# --------------------------------------------------------------------------- MRS probe

class MRSMeter:
    """MRS of the current full mix to the GT mix -- logging only, never a signal.

    Both signals are centre-cropped to the listening window and normalized to
    -12 LUFS first; the value is comparable to the MRS the run is scored on.
    """

    def __init__(self, sr: int, clip_seconds: float, gt: np.ndarray | None):
        self.sr, self.clip_seconds, self.gt = sr, clip_seconds, gt

    @classmethod
    def build(cls, sid: str, gt_path: Path, sr: int, clip_seconds: float) -> "MRSMeter":
        meter = cls(sr, clip_seconds, None)
        if not gt_path.is_file():
            log(sid, f"[MRS] no GT mix at {gt_path}; MRS logging disabled for this song")
            return meter
        try:
            audio, gt_sr = A.load_audio(gt_path)
            if gt_sr != sr:
                log(sid, f"[MRS] GT is {gt_sr} Hz but the render is {sr} Hz; disabled")
                return meter
            meter.gt = meter._prep(audio)
        except Exception as e:  # noqa: BLE001 - diagnostics never fail a song
            log(sid, f"[MRS] could not load GT: {type(e).__name__}: {e}")
        return meter

    def _prep(self, audio: np.ndarray) -> np.ndarray:
        clip = A.center_crop(to_stereo(audio), self.sr, self.clip_seconds)
        return M.loudness_normalize(clip, self.sr, A.TARGET_LUFS)

    def measure(self, mix_audio: np.ndarray) -> float | None:
        if self.gt is None:
            return None
        pred = self._prep(mix_audio)
        n = min(pred.shape[0], self.gt.shape[0])
        if n <= 0:
            return None
        return M.mrs(M._to_tensor(pred[:n]), M._to_tensor(self.gt[:n]), self.sr)


def measure_mrs(st: "SongState") -> float | None:
    """MRS of the current full mix (renders it even when no GT is available, so the
    render cache is filled the same way with and without GT)."""
    if st.mrs is None:
        return None
    try:
        return st.mrs.measure(st.renderer.render_full(st.plan))
    except Exception as e:  # noqa: BLE001
        log(st.sid, f"[MRS] measurement failed: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------- inputs

def discover_stems(track_dir: Path) -> list[Path]:
    return sorted((p for p in track_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS),
                  key=lambda p: p.name.lower())


def load_stem_meta(captions_dir: Path, sid: str) -> dict[str, dict]:
    """Per-stem {instrument, caption} keyed by filename."""
    path = captions_dir / sid / f"{sid}_captions.json"
    if not path.is_file():
        return {}
    try:
        entries = json.loads(path.read_text())["stem_captions"]
    except Exception:  # noqa: BLE001
        return {}
    return {e["filename"]: {"instrument": (e.get("instrument") or "").strip(),
                            "caption": (e.get("caption") or "").strip()} for e in entries}


def tools_plan_path(tool_plans: Path, sid: str) -> Path:
    """``<dir>/<sid>/<sid>_tools_plan.json`` (Stage-1 output) or ``<dir>/<sid>_tools_plan.json``."""
    nested = tool_plans / sid / f"{sid}_tools_plan.json"
    return nested if nested.is_file() else tool_plans / f"{sid}_tools_plan.json"


# --------------------------------------------------------------------------- song state

@dataclass
class SongState:
    sid: str
    plan: dict
    sites: list[Site]
    renderer: HierRenderer
    n_stems: int
    trace: Path
    ref_clip: Path
    agent: QwenAgent
    prompts: dict[str, str]
    contract_header: str
    contract_blocks: dict[str, str]
    clip_seconds: float
    meta: dict = field(default_factory=dict)
    stem_meta: dict = field(default_factory=dict)
    detector: object = None
    critic: object = None
    mrs: object = None

    def site_by_id(self, sid_i: int) -> Site:
        return self.sites[sid_i]

    def render_owner_audio(self, owner: dict) -> np.ndarray:
        """One element as it currently sounds (stem chain / group through its bus / mix)."""
        s0 = owner["sites"][0]
        if s0.level == "stem":
            return self.renderer._proc(s0.owner, self.plan["stems"][s0.owner]["chain"])
        if s0.level == "bus":
            return self.renderer.render_group(self.plan, s0.owner, include_bus=True)
        return self.renderer.render_full(self.plan)

    def resanitize(self) -> list[str]:
        self.plan, errors, actions = sanitize_full_plan(self.plan, self.n_stems)
        if errors:
            raise RuntimeError(f"plan failed validation after patch: {errors[:3]}")
        return actions

    def _reshape_tool_keyed(self, patches: dict[int, dict], stage_rec: dict) -> dict[int, dict]:
        """Recover replies that key ONE site id per element and nest {tool: params}:
        each nested tool is routed to the matching site of the same chain."""
        out: dict[int, dict] = {}
        for sid_i, params in patches.items():
            if (isinstance(params, dict) and params
                    and all(k in TOOL_NAMES and isinstance(v, dict)
                            for k, v in params.items())):
                anchor = self.site_by_id(sid_i)
                chain_sites = [s for s in self.sites
                               if s.level == anchor.level and s.owner == anchor.owner]
                chain_sites.sort(key=lambda s: s.sid != anchor.sid)
                for tool, tool_params in params.items():
                    tgt = next((s for s in chain_sites
                                if s.tool == tool and s.sid not in out), None)
                    if tgt is None:
                        stage_rec.setdefault("patch_errors", []).append(
                            f"site {sid_i}: nested tool {tool!r} has no matching site")
                        continue
                    out[tgt.sid] = tool_params
                    stage_rec["reshaped_patches"] = stage_rec.get("reshaped_patches", 0) + 1
            elif sid_i not in out:
                out[sid_i] = params
        return out

    def apply_patches(self, patches: dict[int, dict], stage_rec: dict) -> int:
        applied = 0
        patched: set[int] = set()
        for sid_i, params in self._reshape_tool_keyed(patches, stage_rec).items():
            site = self.site_by_id(sid_i)
            errs = apply_site_params(self.plan, site, params)
            if not errs:
                applied += 1
                patched.add(sid_i)
                continue
            # wrong-site salvage: params placed under a neighbouring site id of the
            # same chain are applied to the first sibling site that accepts them
            rerouted = False
            if isinstance(params, dict):
                for cand in self.sites:
                    if (cand.level == site.level and cand.owner == site.owner
                            and cand.sid != sid_i and cand.sid not in patched
                            and not apply_site_params(self.plan, cand, params)):
                        stage_rec["rerouted_patches"] = stage_rec.get("rerouted_patches", 0) + 1
                        patched.add(cand.sid)
                        applied += 1
                        rerouted = True
                        break
            if not rerouted:
                stage_rec.setdefault("patch_errors", []).extend(errs[:2])
        if applied:
            actions = self.resanitize()
            stage_rec.setdefault("sanitize_actions", 0)
            stage_rec["sanitize_actions"] += len(actions)
        return applied


# --------------------------------------------------------------------------- sweep

def _element_tools(owner: dict) -> list[str]:
    """The element's tools in chain order, deduped (the critic's diagnosis when the
    detector named none, e.g. at stage C)."""
    out = []
    for s in owner["sites"]:
        if s.tool not in out:
            out.append(s.tool)
    return out


def _revise_prompt(st: SongState, *, context_note: str, critique: str, why: str,
                   target: dict, osites: list[Site], hints: str | None) -> str:
    kw = dict(CONTEXT_NOTE=context_note, CRITIQUE=critique, DETECTOR_REASON=why,
              TARGET_DESC=target["desc"],
              SITE_TABLE=site_table(osites, st.plan, with_params=True),
              TOOL_CONTRACT=contract_for(st.contract_header, st.contract_blocks,
                                         sorted({s.tool for s in osites})),
              AUDIO_ORDER=element_audio_order())
    if hints:
        return fill(st.prompts["revise_hints"], PARAM_HINTS=hints, **kw)
    return fill(st.prompts["revise"], **kw)


def sweep_stage(st: SongState, *, stage_tag: str, loop_sites: list[Site], critic_key: str,
                context_note: str, render_clip, stage_rec: dict,
                seed_all: bool = False) -> None:
    """One detector pass, then one critic (+ planner) turn per flagged element, worst
    first. The mix clip the detector heard stays in front of the critic for the whole
    sweep; one final render is written for the trace. ``seed_all`` (stage C) skips
    the detector and flags every element."""
    owners = group_owners(loop_sites)
    stage = loop_sites[0].stage if loop_sites else "B"
    stage_rec["framework"] = "sweep"
    stage_rec["sweep_mix_update"] = "final"
    stage_rec["renders"] = 0
    stage_rec["iterations"] = 0

    def _render() -> tuple[Path, float]:
        stage_rec["renders"] += 1
        return render_clip()

    mix_clip, mix_gain = _render()
    stage_rec["mix_gain"] = round(float(mix_gain), 6)

    mrs_cur = measure_mrs(st)
    if mrs_cur is not None:
        stage_rec.setdefault("mrs_trajectory", []).append({"event": "initial", "mrs": mrs_cur})
        log(st.sid, f"[MRS] {stage_tag} initial: {mrs_cur:.4f}")

    # element solo clips at the current params (the master element IS the stage mix)
    elem_clips: dict[int, Path] = {}
    det_audio, det_labels = [], []
    for o in owners:
        if o["sites"][0].level == "master":
            elem_clips[o["id"]] = mix_clip
            continue
        p = st.trace / f"{stage_tag}_sw_elem{o['id']}.flac"
        A.save_listen_clip(st.render_owner_audio(o), st.renderer.sr, p, st.clip_seconds,
                           A.STEM_CLIP_LUFS)
        elem_clips[o["id"]] = p
        det_audio.append(p)
        det_labels.append(f"element [{o['id']}] {o['desc']}, processed with its "
                          "current settings")
    det_audio += [mix_clip, st.ref_clip]
    det_labels += ["the current render of this stage (everything together)",
                   "the Reference (same song, different excerpt)"]

    if seed_all:
        stage_rec["detector"] = "skipped_single_element"
        flagged = [{"id": o["id"], "why": ""} for o in owners]
    else:
        verdict, _derrs = st.detector.detect(
            st, tag=f"{stage_tag}_sw_detector", owners=owners, det_audio=det_audio,
            det_labels=det_labels, mix_clip=mix_clip, mix_gain=mix_gain,
            elem_clips=elem_clips, context_note=context_note,
            scenario_kind="subgroup" if stage == "A" else "full", stage_rec=stage_rec)
        if verdict is None:
            stage_rec["stop"] = "detector_failed"
            return
        flagged = verdict["corrupted"]
    stage_rec["flagged"] = [{"id": f["id"], "desc": owners[f["id"]]["desc"], "why": f["why"]}
                            for f in flagged]
    if not flagged:
        stage_rec["stop"] = "good"
        return

    sweep: list[dict] = []
    stage_rec["sweep"] = sweep
    revised = 0
    for step, f in enumerate(flagged, start=1):
        target = owners[f["id"]]
        why = f["why"]
        osites = target["sites"]
        others = [f"{owners[g['id']]['desc']}" + (f" ({g['why']})" if g["why"] else "")
                  for g in flagged if g["id"] != f["id"]]
        entry = {"step": step, "id": target["id"], "desc": target["desc"], "why": why}
        sweep.append(entry)
        tag = f"{stage_tag}_sw{step}_e{target['id']}"

        res: Critique = st.critic.critique(
            st, tag=f"{tag}_critic", target=target, owners=owners, flagged=flagged,
            others=others, why=why, elem_clip=elem_clips[target["id"]], mix_clip=mix_clip,
            context_note=context_note, critic_key=critic_key, stage_tag=stage_tag,
            stage_rec=stage_rec, mix_gain=mix_gain,
            given_tools=f.get("given_tools") or _element_tools(target))
        if res.text is None:
            entry["outcome"] = "critic_failed"
            continue
        if res.text == "":
            entry["outcome"] = "skipped_good"
            continue

        applied, patches = 0, None
        if res.patches:  # F3 direct: the critic's values, no planner turn
            applied = st.apply_patches(dict(res.patches), stage_rec)
            patches = res.patches if applied else None
            entry["direct"] = bool(applied)
            if not applied:  # rejected values -> fall back to the planner with hints
                entry["direct_rejected"] = True
        if not applied:
            rev_prompt = _revise_prompt(st, context_note=context_note, critique=res.text,
                                        why=why, target=target, osites=osites,
                                        hints=res.hints)
            patches, _perrs = st.agent.ask_json(
                st.sid, [elem_clips[target["id"]], mix_clip, st.ref_clip], rev_prompt,
                st.trace, f"{tag}_revise", make_sites_validator({s.sid: s for s in osites}))
            if patches is None:
                entry["outcome"] = "revise_failed"
                continue
            applied = st.apply_patches(patches, stage_rec)
            if not applied:
                entry["outcome"] = "patch_rejected"
                continue

        entry["outcome"] = "revised"
        entry["n_patched"] = applied
        entry["tools"] = sorted({st.site_by_id(i).tool for i in (patches or {})})
        revised += 1
        mrs_new = measure_mrs(st)
        if mrs_new is not None:
            entry["mrs_before"], entry["mrs_after"] = mrs_cur, mrs_new
            stage_rec.setdefault("mrs_trajectory", []).append(
                {"event": f"after_step_{step}_{target['desc']}", "mrs": mrs_new})
            log(st.sid, f"[MRS] {stage_tag} after fix {step} ({target['desc']}): "
                        f"{mrs_cur:.3f} -> {mrs_new:.3f}")
            mrs_cur = mrs_new
        stage_rec.setdefault("revisions", []).append(
            {"iter": step, "target": target["desc"], "n_patched": applied,
             "flagged": [owners[g["id"]]["desc"] for g in flagged], "tools": entry["tools"]})

    _render()  # one trace clip of where the sweep landed
    stage_rec["iterations"] = revised
    stage_rec["revised"] = revised
    stage_rec["stop"] = "swept"


# --------------------------------------------------------------------------- stages

def _initial(st: SongState, rec: dict, sites: list[Site], prompt: str, audio: list[Path],
             tag: str) -> None:
    patches, errs = st.agent.ask_json(st.sid, audio, prompt, st.trace, f"{tag}_initial",
                                      make_sites_validator({s.sid: s for s in sites}))
    if patches:
        rec["initial_applied"] = st.apply_patches(patches, rec)
    else:
        rec["initial_applied"] = 0
        rec.setdefault("patch_errors", []).extend(errs[:3])


def _group_table(st: SongState, gi: int) -> str:
    g = st.plan["groups"][gi]
    lines = []
    for m in g["member_stem_indices"]:
        entry = st.plan["stems"][m]
        name = entry["stem_name"]
        info = st.stem_meta.get(name + ".flac") or {}
        inst = info.get("instrument") or "?"
        cap = (info.get("caption") or "")[:240]
        chain = " -> ".join(step["tool"] for step in entry["chain"]) or "(no tools)"
        lines.append(f'stem {m} "{name}" | instrument: {inst} | chain: {chain}'
                     + (f"\n    caption: {cap}" if cap else ""))
    return "\n".join(lines)


def stage_a(st: SongState) -> None:
    st.meta["stage_a"] = {}
    for gi, g in enumerate(st.plan["groups"]):
        members = g["member_stem_indices"]
        if len(members) < 2:
            st.meta["stage_a"][g["name"]] = {"skipped": "single_stem"}
            continue
        gsites = [s for s in st.sites if s.stage == "A" and s.group_index == gi]
        rec: dict = {"n_sites": len(gsites)}
        st.meta["stage_a"][g["name"]] = rec
        if not gsites:
            rec["skipped"] = "no_sites"
            continue
        tag = f"A_g{gi}_{re.sub(r'[^A-Za-z0-9_-]', '_', g['name'])[:24]}"

        # dry stem clips, each normalized on its own (balance is judged on the submix)
        stem_clips, labels = [], []
        for m in members:
            clip = st.trace / f"{tag}_stem{m}.flac"
            A.save_listen_clip(st.renderer.audios[m], st.renderer.sr, clip, st.clip_seconds,
                               A.STEM_CLIP_LUFS)
            stem_clips.append(clip)
            labels.append(f'dry stem {m} "{st.plan["stems"][m]["stem_name"]}"')
        labels.append("the Reference mix (same song, different excerpt)")
        prompt = fill(st.prompts["group_initial"], SONG_ID=st.sid, GROUP_NAME=str(g["name"]),
                      GROUP_TABLE=_group_table(st, gi),
                      SITE_TABLE=site_table(gsites, st.plan, with_params=False),
                      TOOL_CONTRACT=contract_for(st.contract_header, st.contract_blocks,
                                                 sorted({s.tool for s in gsites})),
                      AUDIO_ORDER=audio_order_text(labels))
        _initial(st, rec, gsites, prompt, stem_clips + [st.ref_clip], tag)

        def render_clip(gi=gi, tag=tag, counter=[0]) -> tuple[Path, float]:
            counter[0] += 1
            mix = st.renderer.render_group(st.plan, gi, include_bus=False)
            return A.save_listen_clip_gain(mix, st.renderer.sr,
                                           st.trace / f"{tag}_render{counter[0]}.flac",
                                           st.clip_seconds)

        note = (f'You are working on ONE subgroup of the song: the "{g["name"]}" group '
                f"({len(members)} stems). The render contains only this group's stems "
                "through their per-stem tool chains (no group-bus processing yet). The "
                "Reference is the FULL mix of the same song (different excerpt), so "
                "compare only this group's role in it.")
        sweep_stage(st, stage_tag=tag, loop_sites=gsites, critic_key="critic_group",
                    context_note=note, render_clip=render_clip, stage_rec=rec)
        log(st.sid, f"stage A group {g['name']!r}: revised={rec.get('iterations', 0)} "
                    f"stop={rec.get('stop')}")


def stage_b(st: SongState) -> None:
    rec: dict = {}
    st.meta["stage_b"] = rec
    bsites = [s for s in st.sites if s.stage == "B"]
    rec["n_sites"] = len(bsites)
    if not bsites:
        rec["skipped"] = "no_sites"
        return
    tag = "B_bus"
    group_clips, labels = [], []
    for gi, g in enumerate(st.plan["groups"]):
        mix = st.renderer.render_group(st.plan, gi, include_bus=False)
        clip = st.trace / f"{tag}_group{gi}.flac"
        A.save_listen_clip(mix, st.renderer.sr, clip, st.clip_seconds)
        group_clips.append(clip)
        n = len(g["member_stem_indices"])
        labels.append(f'group "{g["name"]}" submix ({n} stem{"s" if n != 1 else ""})')
    # group clips are normalized individually; the premaster keeps the balance
    pm0_clip = st.trace / f"{tag}_premaster0.flac"
    A.save_listen_clip(st.renderer.render_premaster(st.plan), st.renderer.sr, pm0_clip,
                       st.clip_seconds)
    group_clips.append(pm0_clip)
    labels.append("the current premaster sum of all groups (hear the CURRENT balance "
                  "between groups here)")
    labels.append("the Reference mix (same song, different excerpt)")
    prompt = fill(st.prompts["bus_initial"], SONG_ID=st.sid,
                  SITE_TABLE=site_table(bsites, st.plan, with_params=False),
                  TOOL_CONTRACT=contract_for(st.contract_header, st.contract_blocks,
                                             sorted({s.tool for s in bsites})),
                  AUDIO_ORDER=audio_order_text(labels))
    _initial(st, rec, bsites, prompt, group_clips + [st.ref_clip], tag)

    def render_clip(counter=[0]) -> tuple[Path, float]:
        counter[0] += 1
        return A.save_listen_clip_gain(st.renderer.render_premaster(st.plan), st.renderer.sr,
                                       st.trace / f"{tag}_render{counter[0]}.flac",
                                       st.clip_seconds)

    note = ("You are working at the BUS level: every group submix now passes through "
            "its group-bus tool chain and the results are summed (no master-bus "
            "processing yet). The render is this premaster sum. The Reference is the "
            "full mix of the same song (different excerpt).")
    sweep_stage(st, stage_tag=tag, loop_sites=bsites, critic_key="critic_mix",
                context_note=note, render_clip=render_clip, stage_rec=rec)
    log(st.sid, f"stage B: revised={rec.get('iterations', 0)} stop={rec.get('stop')}")


def stage_c(st: SongState) -> None:
    rec: dict = {}
    st.meta["stage_c"] = rec
    csites = [s for s in st.sites if s.stage == "C"]
    rec["n_sites"] = len(csites)
    if not csites:
        rec["skipped"] = "no_sites"
        return
    tag = "C_master"
    pm_clip = st.trace / f"{tag}_premaster.flac"
    A.save_listen_clip(st.renderer.render_premaster(st.plan), st.renderer.sr, pm_clip,
                       st.clip_seconds)
    prompt = fill(st.prompts["master_initial"], SONG_ID=st.sid,
                  SITE_TABLE=site_table(csites, st.plan, with_params=False),
                  TOOL_CONTRACT=contract_for(st.contract_header, st.contract_blocks,
                                             sorted({s.tool for s in csites})),
                  AUDIO_ORDER=audio_order_text(
                      ["the premaster mix (all stems and buses, no master processing)",
                       "the Reference mix (same song, different excerpt)"]))
    _initial(st, rec, csites, prompt, [pm_clip, st.ref_clip], tag)

    def render_clip(counter=[0]) -> tuple[Path, float]:
        counter[0] += 1
        return A.save_listen_clip_gain(st.renderer.render_full(st.plan), st.renderer.sr,
                                       st.trace / f"{tag}_render{counter[0]}.flac",
                                       st.clip_seconds)

    note = ("You are working on the MASTER bus: the full mix now passes through the "
            "master tool chain. The render is the complete mastered mix. The Reference "
            "is the full mix of the same song (different excerpt).")
    sweep_stage(st, stage_tag=tag, loop_sites=csites, critic_key="critic_mix",
                context_note=note, render_clip=render_clip, stage_rec=rec, seed_all=True)
    log(st.sid, f"stage C: revised={rec.get('iterations', 0)} stop={rec.get('stop')}")


# --------------------------------------------------------------------------- per song

def load_initial_plan(cfg: RunConfig, sid: str) -> tuple[dict, list[Path], list[str]]:
    """Stage-1 tools plan -> sanitized internal plan, stem files, conversion notes."""
    tools_plan = json.loads(tools_plan_path(cfg.tool_plans, sid).read_text())
    stem_paths = discover_stems(cfg.dataset_root / "Input" / sid)
    plan_names = [s["stem_name"] for s in tools_plan["stems"]]
    file_names = [p.name[:-len(p.suffix)] for p in stem_paths]
    if plan_names != file_names:
        raise RuntimeError(f"{sid}: stem name mismatch between tools plan and files")
    plan, deviations = convert_tools_plan(tools_plan)
    for i, s in enumerate(plan["stems"]):
        if s["stem_index"] != i:
            raise RuntimeError(f"{sid}: plan stems not in positional order")
    plan, verrs, _actions = sanitize_full_plan(plan, len(stem_paths))
    if verrs:
        raise RuntimeError(f"{sid}: initial plan invalid: {verrs[:3]}")
    return plan, stem_paths, deviations


def process_song(sid: str, cfg: RunConfig, agent: QwenAgent, prompts: dict[str, str],
                 contract: tuple[str, dict], backends) -> dict:
    t0 = time.time()
    out_dir = cfg.out / sid
    final_flac = out_dir / f"{sid}.flac"
    if final_flac.exists() and not cfg.overwrite:
        log(sid, "already done, skipping")
        return {"song_id": sid, "status": "skipped"}
    trace = out_dir / "trace"
    trace.mkdir(parents=True, exist_ok=True)

    plan, stem_paths, deviations = load_initial_plan(cfg, sid)
    baseline_fingerprint = chain_fingerprint(plan)
    renderer = HierRenderer(stem_paths)
    stem_meta = load_stem_meta(paths.caption_file(cfg.captions_dir, cfg.dataset_root, sid).parent.parent, sid)
    sites = enumerate_sites(plan, stem_meta)
    ref_clip = A.make_file_clip(cfg.dataset_root / "Reference" / f"{sid}.flac",
                                trace / "reference_clip.flac", cfg.clip_seconds)
    detector, critic, _llm_detector = backends
    st = SongState(sid=sid, plan=plan, sites=sites, renderer=renderer,
                   n_stems=len(stem_paths), trace=trace, ref_clip=ref_clip, agent=agent,
                   prompts=prompts, contract_header=contract[0], contract_blocks=contract[1],
                   clip_seconds=cfg.clip_seconds, stem_meta=stem_meta, detector=detector,
                   critic=critic)
    st.meta["deviations"] = deviations
    st.meta["n_sites"] = len(sites)
    st.meta["stems_variant"] = "normalized"

    if cfg.log_mrs:
        st.mrs = MRSMeter.build(sid, cfg.dataset_root / "GT" / f"{sid}.flac", renderer.sr,
                                cfg.clip_seconds)
    mrs_initial = measure_mrs(st)
    if mrs_initial is not None:
        log(sid, f"[MRS] initial (tools plan, unparameterized): {mrs_initial:.4f}")
    mrs_by_stage: dict[str, float] = {}
    for name, fn in (("stage_a", stage_a), ("stage_b", stage_b), ("stage_c", stage_c)):
        fn(st)
        value = measure_mrs(st)
        if value is not None:
            mrs_by_stage[name] = value
            log(sid, f"[MRS] after {name}: {value:.4f}")
    if mrs_initial is not None:
        st.meta["mrs_initial"] = mrs_initial
        st.meta["mrs_by_stage"] = mrs_by_stage
        st.meta["mrs_final"] = mrs_by_stage.get("stage_c", mrs_initial)

    if chain_fingerprint(st.plan) != baseline_fingerprint:
        raise RuntimeError(f"{sid}: chain fingerprint changed -- immutability broken")

    final = A.normalize_lufs(to_stereo(st.renderer.render_full(st.plan)), st.renderer.sr)
    A.write_flac(final, st.renderer.sr, final_flac)
    (out_dir / f"{sid}_plan_final.json").write_text(json.dumps(st.plan, indent=1))
    (out_dir / "stem_order.txt").write_text(
        "0-based stem_index -> file (plan JSONs use this order)\n"
        + "\n".join(f"{i}: {p.name}" for i, p in enumerate(stem_paths)))
    meta = {"song_id": sid, "status": "ok", "secs": round(time.time() - t0, 1),
            "preset": cfg.preset, "chain_immutable": True, **st.meta}
    (out_dir / f"{sid}.json").write_text(json.dumps(meta, indent=1))
    log(sid, f"DONE in {meta['secs']}s")
    return meta
