"""Stage-2 prompt templates and the text blocks / reply validators built around them.

Templates live in ``mixagent/prompts/`` and use ``[[PLACEHOLDER]]`` slots. The trained
critic's templates must match its training prompts byte for byte; they end with
``<|vision_start|>`` and no newline, so the one trailing newline kept on disk is
removed at load time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .llm import parse_json_reply
from .plan import Site, get_step

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

PROMPT_FILES = {
    "contract": "hierparam_tool_contract.txt",
    "group_initial": "hierparam_group_initial.txt",
    "bus_initial": "hierparam_bus_initial.txt",
    "master_initial": "hierparam_master_initial.txt",
    # untrained (LLM) critic, one template per feedback format
    "critic_group": "hierparam_critic_group.txt",            # F2
    "critic_mix": "hierparam_critic_mix.txt",                # F2
    "critic_group_llmf1": "hierparam_critic_group_llmf1.txt",
    "critic_mix_llmf1": "hierparam_critic_mix_llmf1.txt",
    "critic_group_llmf3": "hierparam_critic_group_llmf3.txt",
    "critic_mix_llmf3": "hierparam_critic_mix_llmf3.txt",
    "detector_sweep": "hierparam_detector_sweep.txt",        # untrained (LLM) detector
    "revise": "hierparam_revise.txt",
    "revise_hints": "hierparam_revise_hints.txt",
    # trained critic (given-tools variants)
    "critic_trained_f1g": "hierparam_critic_trained_f1g.txt",
    "critic_trained_f2g": "hierparam_critic_trained_f2g.txt",
    "critic_trained_f3g": "hierparam_critic_trained_f3g.txt",
    "critic_trained_clipdesc": "hierparam_critic_trained_clipdesc.txt",
    "critic_trained_chainhdr": "hierparam_critic_trained_chainhdr.txt",
}
_NO_TRAILING_NL = frozenset({"critic_trained_f1g", "critic_trained_f2g", "critic_trained_f3g",
                             "critic_trained_clipdesc", "critic_trained_chainhdr"})


def load_prompts() -> dict[str, str]:
    out = {}
    for k, f in PROMPT_FILES.items():
        text = (PROMPT_DIR / f).read_text(encoding="utf-8")
        if k in _NO_TRAILING_NL and text.endswith("\n"):
            text = text[:-1]
        out[k] = text
    return out


_CONTRACT_SPLIT = re.compile(r"^### tool: (\w+)\s*$", re.MULTILINE)


def split_contract(contract_text: str) -> tuple[str, dict[str, str]]:
    """-> (general rules header, {tool: block})."""
    parts = _CONTRACT_SPLIT.split(contract_text)
    header = parts[0].strip()
    blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return header, blocks


def contract_for(header: str, blocks: dict[str, str], tools: list[str]) -> str:
    seen, out = set(), [header]
    for t in tools:
        if t in blocks and t not in seen:
            out.append(f"### tool: {t}\n{blocks[t]}")
            seen.add(t)
    return "\n\n".join(out)


def fill(template: str, **kw: str) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace(f"[[{k}]]", v)
    leftover = re.findall(r"\[\[[A-Z_]+\]\]", out)
    if leftover:
        raise RuntimeError(f"unfilled prompt placeholders: {leftover}")
    return out


def _params_compact(params: dict) -> str:
    def r(v):
        if isinstance(v, list):
            return [round(float(x), 3) for x in v]
        return round(float(v), 3)
    return json.dumps({k: r(v) for k, v in params.items()})


def site_table(sites: list[Site], plan: dict, *, with_params: bool) -> str:
    lines = []
    for s in sites:
        line = f"[{s.sid}] {s.desc} | step {s.step_idx + 1}: {s.tool}"
        if with_params:
            line += f" | current: {_params_compact(get_step(plan, s)['params'])}"
        lines.append(line)
    return "\n".join(lines)


def group_owners(loop_sites: list[Site]) -> list[dict]:
    """Unique revision targets (one per stem / bus / master chain) with their sites."""
    owners: list[dict] = []
    by_key: dict[tuple, dict] = {}
    for s in loop_sites:
        k = (s.level, s.owner)
        if k not in by_key:
            by_key[k] = {"id": len(owners), "desc": s.desc, "sites": []}
            owners.append(by_key[k])
        by_key[k]["sites"].append(s)
    return owners


def owner_table(owners: list[dict], plan: dict) -> str:
    lines = []
    for o in owners:
        lines.append(f"[{o['id']}] {o['desc']} | tools: "
                     + " -> ".join(s.tool for s in o["sites"]))
        for s in o["sites"]:
            lines.append(f"    {s.tool}: {_params_compact(get_step(plan, s)['params'])}")
    return "\n".join(lines)


def audio_order_text(labels: list[str]) -> str:
    return "\n".join(f"AUDIO {i + 1}: {lab}" for i, lab in enumerate(labels))


def element_audio_order() -> str:
    return audio_order_text(
        ["the flagged element on its own, processed with its current settings",
         "the current render of this stage (everything together)",
         "the Reference (same song, different excerpt)"])


# --------------------------------------------------------------------------- validators

def make_sites_validator(allowed: dict[int, Site]):
    """Planner replies: {"sites": {"<id>": {params}}} (or a bare id -> params map)."""
    def _validate(parsed) -> tuple[object | None, list[str]]:
        if not isinstance(parsed, dict):
            return None, ["top level must be a JSON object"]
        body = parsed.get("sites", parsed)
        if not isinstance(body, dict):
            return None, ['"sites" must be an object mapping site id -> params']
        out, errs = {}, []
        for k, v in body.items():
            if k == "notes":
                continue
            try:
                sid_i = int(k)
            except (TypeError, ValueError):
                errs.append(f"site key {k!r} is not an integer id")
                continue
            if sid_i not in allowed:
                errs.append(f"site id {sid_i} is not in the allowed site list")
                continue
            out[sid_i] = v
        if not out:
            return None, errs or ["no usable site entries found"]
        return out, errs
    return _validate


def make_corrupted_validator(allowed_ids: set[int]):
    """Detector replies: {"corrupted": [{"id": k, "why": "..."}, ...]}, worst first.

    An empty list is valid and means every element sounds fine.
    """
    def _validate(parsed) -> tuple[object | None, list[str]]:
        if not isinstance(parsed, dict):
            return None, ["top level must be a JSON object"]
        body = parsed.get("corrupted")
        if body is None:
            return None, ['reply needs a "corrupted" list (empty list if all fine)']
        if not isinstance(body, list):
            return None, ['"corrupted" must be a list']
        out, errs, seen = [], [], set()
        for entry in body:
            if isinstance(entry, dict):
                raw_id, why = entry.get("id"), str(entry.get("why", "")).strip()
            else:
                raw_id, why = entry, ""
            try:
                eid = int(raw_id)
            except (TypeError, ValueError):
                errs.append(f"corrupted entry {entry!r} has no integer id")
                continue
            if eid not in allowed_ids:
                errs.append(f"corrupted id {eid} is not in the element list")
                continue
            if eid not in seen:
                seen.add(eid)
                out.append({"id": eid, "why": why})
        if errs and not out and body:
            return None, errs  # nothing usable in a non-empty list -> retry
        return {"corrupted": out}, errs
    return _validate


def format_critique(reply: str, chain_tools: list[str] | None = None) -> str | None:
    """Critic reply {"tools": [{"tool", "comment"}, ...], "keep"} -> repair brief.

    Returns "" when the critic returns an empty tools list (nothing to fix), None when
    the reply is unusable, and falls back to the raw text otherwise.
    """
    try:
        parsed = parse_json_reply(reply)
    except Exception:  # noqa: BLE001
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tools"), list):
        lines = []
        for entry in parsed["tools"]:
            if not isinstance(entry, dict):
                continue
            tool = str(entry.get("tool", "")).strip()
            comment = str(entry.get("comment", "")).strip()
            if not (tool or comment):
                continue
            note = ""
            if chain_tools is not None and tool and tool not in chain_tools:
                note = (" [note: this element has no such tool -- translate the "
                        "intent onto its available tools]")
            lines.append(f"- {tool or '?'}: {comment}{note}")
        keep = str(parsed.get("keep", "")).strip()
        if not lines:
            return ""
        if keep:
            lines.append(f"- already working, preserve: {keep}")
        return "\n".join(lines)
    if isinstance(parsed, dict) and (parsed.get("wrong") or parsed.get("fix")):
        parts = [f"what is wrong: {parsed.get('wrong', '')}".strip(),
                 f"what to fix: {parsed.get('fix', '')}".strip()]
        return "\n".join(p for p in parts if not p.endswith(": "))
    text = reply.strip()[:1500]
    return text or None
