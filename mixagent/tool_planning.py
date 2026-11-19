"""Stage 1: hierarchical grouping + retrieval-augmented tool planning (no parameters).

Three planner calls per song: (1) group the stems into buses [stem audio], (2) for
each bus, the tool chain of every member stem and of the bus [bus stem audio],
(3) the master chain [text only]. Calls 2 and 3 carry in-context examples retrieved
from a bank of expert console plans (gradient-fitted to other songs' human mixes):

* text retrieval -- the song's instrument list + stem captions vs the bank songs'
  (``gemini-embedding-001`` vectors);
* audio retrieval -- the song's Reference mix vs the bank's console renders
  (RLAT vectors);
* ``dual`` (the paper's setting) = top ``text_k`` of the text ranking unioned with
  top ``audio_k`` of the audio ranking.

Every segment of the query's own source song is excluded from the bank. Embeddings
and grouping replies are cached (shipped under ``data/<dataset>/retrieval_cache``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np

from . import paths
from .llm import VLLMBackend, conversation, write_trace

STEM_TOOLS = ("eq", "distortion", "compressor", "noisegate", "stereo_imager",
              "gain_panning", "delay", "reverb")
MASTER_TOOLS = STEM_TOOLS + ("multiband_compressor", "limiter")
AUDIO_EXTS = (".flac", ".wav")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
GROUP_PROMPT = "tools_only_group_moretools.txt"


def read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def parse_json_reply(reply: str) -> dict:
    """Strict: the last ```json fenced block, or the whole reply."""
    fences = re.findall(r"```json\s*(.*?)```", reply, flags=re.DOTALL)
    text = fences[-1] if fences else reply
    return json.loads(text.strip())


def discover_stems(track_dir: Path) -> list[Path]:
    return sorted((p for p in track_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS),
                  key=lambda p: p.name.lower())


def fmt_chain(chain: list[str]) -> str:
    return " -> ".join(chain) if chain else "(none)"


def stem_key(name: str) -> str:
    """Stem name/filename -> join key (drops a known audio extension and ``_dry``)."""
    s = str(name)
    suffix = Path(s).suffix.lower()
    if suffix in AUDIO_EXTS:
        s = s[: -len(suffix)]
    return s[:-4] if s.endswith("_dry") else s


GROUP_ABBREV: dict[str, str] = {
    "bv": "backing vocals", "lv": "lead vocals", "ld": "lead vocals", "vox": "vocals",
    "ag": "acoustic guitars", "eg": "electric guitars", "egtr": "electric guitars",
    "eguit": "electric guitars", "el_gtr": "electric guitars", "el_gtrs": "electric guitars",
    "gtr": "guitars", "gtrs": "guitars", "guit": "guitars", "guitar": "guitars",
    "perc": "percussion", "kb": "keys", "fx": "effects",
}


def short_group_name(name: str) -> str:
    """Bank bus name -> plain role name (``group_01_stem_01_drum_set`` -> ``drum set``)."""
    s = re.sub(r"^group_\d+_", "", str(name))
    medley = re.match(r"^stem_\d+_(.+)$", s)
    if medley:
        return medley.group(1).replace("_", " ").strip() or str(name)
    s = re.sub(r"^\d+_", "", s)
    s = re.sub(r"_(multi|mono|stereo)$", "", s)
    s = re.sub(r"^(s|mix)_", "", s)
    m = re.match(r"^([a-z_]+?)_?(\d+(?:_\d+)*)$", s)
    base, tail = (m.group(1), m.group(2).replace("_", " ")) if m else (s, "")
    label = GROUP_ABBREV.get(base, base.replace("_", " ")).strip()
    return (f"{label} {tail}".strip() if tail else label) or str(name)


_CHANNEL_RE = re.compile(r"[ _]([LRC])$", re.IGNORECASE)


def channel_suffix(stem_name: str) -> str:
    """``L``/``R``/``C`` when the stem name ends in a channel marker, else ``''``."""
    m = _CHANNEL_RE.search(stem_key(stem_name))
    return m.group(1).upper() if m else ""


class Context:
    """Paths and settings shared by all songs of one Stage-1 run."""

    def __init__(self, *, dataset_root: Path, bank_plans_dir: Path, captions_dir: Path,
                 cache_dir: Path, song_groups: dict[str, str], arm: str = "dual",
                 text_k: int = 3, audio_k: int = 1, max_groups: int = 8,
                 caption_chars: int = 240, embed_caption_chars: int = 150,
                 embed_max_chars: int = 7800, bank_render_dir: Path | None = None,
                 audio_device: str = "cuda:0"):
        self.dataset_root = Path(dataset_root)
        self.bank_plans_dir = Path(bank_plans_dir)
        self.captions_dir = Path(captions_dir)
        self.cache_dir = Path(cache_dir)
        self.song_groups = song_groups
        self.arm, self.text_k, self.audio_k = arm, int(text_k), int(audio_k)
        self.max_groups = int(max_groups)
        self.caption_chars = int(caption_chars)
        self.embed_caption_chars = int(embed_caption_chars)
        self.embed_max_chars = int(embed_max_chars)
        self.bank_render_dir = Path(bank_render_dir) if bank_render_dir else None
        self.audio_device = audio_device
        self.bank_ids = sorted(p.name.replace("_tools_plan.json", "")
                               for p in self.bank_plans_dir.glob("*_tools_plan.json"))

    def input_dir(self, sid: str) -> Path:
        return self.dataset_root / "Input" / sid


def load_stem_meta(sid: str, ctx: Context) -> dict[str, dict]:
    """Per-stem ``{instrument, caption}`` keyed by :func:`stem_key`."""
    path = paths.caption_file(ctx.captions_dir, ctx.dataset_root, sid)
    if not path.is_file():
        return {}
    try:
        entries = json.loads(path.read_text())["stem_captions"]
    except Exception:  # noqa: BLE001
        return {}
    return {stem_key(e["filename"]): {"instrument": (e.get("instrument") or "").strip(),
                                      "caption": (e.get("caption") or "").strip()}
            for e in entries}


def stems_text(sid: str, ctx: Context) -> str | None:
    """Retrieval document: instrument list first, then one capped caption per stem."""
    meta = load_stem_meta(sid, ctx)
    if not meta:
        return None
    stem_dir = ctx.input_dir(sid)
    keys = [stem_key(p.name) for p in discover_stems(stem_dir)] if stem_dir.is_dir() else []
    keys = [k for k in keys if k in meta] or list(meta)
    seen: set[str] = set()
    instruments, lines = [], []
    for k in keys:
        m = meta[k]
        inst = m["instrument"] or k
        if inst.lower() not in seen:
            seen.add(inst.lower())
            instruments.append(inst)
        cap = m["caption"][:ctx.embed_caption_chars].strip() if ctx.embed_caption_chars > 0 else ""
        lines.append(f"- {inst}: {cap}" if cap else f"- {inst}")
    doc = "Instruments: " + ", ".join(instruments) + "\n" + "\n".join(lines)
    if len(doc) > ctx.embed_max_chars:
        doc = doc[:ctx.embed_max_chars].rsplit("\n", 1)[0]
    return doc


def exemplar_files(sid: str, ctx: Context) -> list[str]:
    """A bank clip's stem file names in file order: listed from ``<dataset_root>/Input/<sid>``
    when that clip's audio is present (the paper's runs), else from the shipped
    ``bank_stem_files.json`` (the same listing), so your own songs need no bank audio."""
    d = ctx.input_dir(sid)
    if d.is_dir():
        return sorted(p.name for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    listing = ctx.bank_plans_dir.parent / "bank_stem_files.json"
    if listing.is_file():
        files = json.loads(listing.read_text()).get(sid)
        if files is not None:
            return list(files)
    raise FileNotFoundError(f"no stems for bank clip {sid}: neither {d} nor {listing}")


def fmt_exemplar(plan: dict, sid: str, ctx: Context, source_tag: str = "",
                 score_text: str = "") -> str:
    """One retrieved bank plan as a per-stem view nested under its buses; stems are
    renumbered to the exemplar song's own file order."""
    meta = load_stem_meta(sid, ctx)
    files = exemplar_files(sid, ctx)
    by_key = {stem_key(s["stem_name"]): s for s in plan["stems"]}
    rows, remap = [], {}
    for i, fname in enumerate(files, 1):
        k = stem_key(fname)
        s = by_key.get(k)
        if s is None:
            continue
        remap[s["stem_index"]] = i
        m = meta.get(k, {})
        cap = m.get("caption", "") if ctx.caption_chars > 0 else ""
        if cap and len(cap) > ctx.caption_chars:
            cap = cap[:ctx.caption_chars].rsplit(" ", 1)[0] + "..."
        instrument = m.get("instrument") or k
        ch = channel_suffix(fname)
        if ch and not re.search(rf"\b{ch}\b|\({ch}\)$", instrument):
            instrument = f"{instrument} ({ch})"
        rows.append({"i": i, "instrument": instrument, "caption": cap,
                     "group": s.get("group"), "chain": s.get("chain", [])})
    by_i = {r["i"]: r for r in rows}
    head = f"Example [{source_tag}]" if source_tag else "Example"
    if score_text:
        head += f" ({score_text})"
    n_direct = sum(1 for r in rows if not r["group"])
    lines = [f"{head} -- {len(rows)} stems, {len(plan.get('groups', []))} buses, "
             f"{n_direct} routed straight to master:"]
    width = min(max((len(r["instrument"]) for r in rows), default=0), 28)

    def emit(r: dict) -> None:
        chain = " -> ".join(r["chain"]) if r["chain"] else "(no processing)"
        lines.append(f"    {r['i']:>2}  {r['instrument']:<{width}} {chain}")
        if r["caption"]:
            lines.append(f"        sounds like: {r['caption']}")

    for g in plan.get("groups", []):
        members = [remap[i] for i in g["member_stem_indices"] if i in remap]
        lines.append(f"  Bus '{short_group_name(g['name'])}'  bus chain: {fmt_chain(g['chain'])}")
        for i in members:
            emit(by_i[i])
    direct = [r for r in rows if not r["group"]]
    if direct:
        lines.append("  Direct to master (no bus):")
        for r in direct:
            emit(r)
    lines.append(f"  Master chain: {fmt_chain(plan.get('master_chain', []))}")
    return "\n".join(lines)


ICL_HEADER = (
    "In-context examples from previously mixed songs, structured by an expert hierarchical\n"
    "mixing console (a gradient-descent optimizer). Guidance only -- adapt to THIS song; do\n"
    "not copy blindly. Parameters are intentionally omitted: study WHICH tools were kept per\n"
    "stem, per bus, and on the master, and how selective the choices are. Each example's stem\n"
    "numbers are LOCAL to that whole song and are unrelated to the numbering you must output.\n"
    "[text] = matched on description, [audio] = matched on audio similarity, [text+audio] = both.\n\n"
)


def build_icl(exemplars: list[tuple[str, str, str]], ctx: Context) -> str:
    blocks = []
    for sid, tag, score_text in exemplars:
        path = ctx.bank_plans_dir / f"{sid}_tools_plan.json"
        if path.is_file():
            blocks.append(fmt_exemplar(json.loads(path.read_text()), sid, ctx, tag, score_text))
    return ICL_HEADER + "\n\n".join(blocks) + "\n" if blocks else ""


# --------------------------------------------------------------------------- retrieval

_INDEX_LOCK = threading.Lock()


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


def _load_cache(cache: Path, tag: str, ids: list[str] | None = None):
    """(ids, vecs) from a cache file when its tag (and ids, if given) match, else None."""
    if not cache.exists():
        return None
    z = np.load(cache, allow_pickle=True)
    cached_ids = [str(x) for x in z["ids"]]
    cached_tag = str(z["tag"]) if "tag" in z.files else ""
    if cached_tag != tag or (ids is not None and cached_ids != ids):
        return None
    return cached_ids, z["vecs"]


def _load_or_build(cache: Path, ids: list[str], build, tag: str) -> np.ndarray:
    with _INDEX_LOCK:
        hit = _load_cache(cache, tag, ids)
        if hit is not None:
            return hit[1]
        vecs = build()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.asarray(ids, dtype=object), vecs=vecs, tag=np.asarray(tag))
        return vecs


def gemini_text_embed(texts: list[str], task_type: str) -> np.ndarray:
    """``gemini-embedding-001`` vectors (needs ``google-genai`` and GEMINI_API_KEY);
    only used when a text-embedding cache entry is missing."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY" if task_type == "query" else "RETRIEVAL_DOCUMENT")
    vectors = []
    for start in range(0, len(texts), 25):
        resp = client.models.embed_content(model="gemini-embedding-001",
                                           contents=texts[start:start + 25], config=config)
        vectors += [list(e.values) for e in resp.embeddings]
    return _l2_normalize(np.asarray(vectors, dtype=np.float32))


def rlat_audio_embed(paths: list[Path], device: str) -> np.ndarray:
    """RLAT ``zy`` vectors (10 s chunk mean at 44.1 kHz), L2-normalized; only
    used when an audio-embedding cache entry is missing."""
    from evaluation.encoders import rlat_embedding  # noqa: PLC0415
    import soundfile as sf

    vecs = []
    for p in paths:
        audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
        vecs.append(rlat_embedding(audio, int(sr), device=device).cpu().numpy())
    return _l2_normalize(np.stack(vecs, axis=0))


def _text_tag(ctx: Context) -> str:
    return (f"src=stems;caps={ctx.captions_dir.resolve().name}"
            f";cap={ctx.embed_caption_chars};max={ctx.embed_max_chars}")


def excluded_ids(sid: str, ctx: Context) -> set[str]:
    """The query itself plus every other segment of the same source song."""
    g = ctx.song_groups.get(sid)
    if g is None:
        return {sid}
    return {c for c, cg in ctx.song_groups.items() if cg == g} | {sid}


def _rank(sid: str, db_ids: list[str], docs: np.ndarray, qvec: np.ndarray, ctx: Context):
    drop = excluded_ids(sid, ctx)
    keep = [j for j, c in enumerate(db_ids) if c not in drop]
    sims = docs[keep] @ qvec
    order = np.argsort(-sims)
    return [db_ids[keep[int(j)]] for j in order], sims[order]


def rank_text(sid: str, ctx: Context):
    """Bank ranked by caption-text similarity to the query, descending."""
    tag = _text_tag(ctx)
    db_ids = [c for c in ctx.bank_ids
              if (ctx.captions_dir / c / f"{c}_captions.json").is_file()]

    def build_docs():
        return gemini_text_embed([stems_text(c, ctx) for c in db_ids], "document")

    docs = _load_or_build(ctx.cache_dir / "xref_db_stems.npz", db_ids, build_docs, tag)
    qtext = stems_text(sid, ctx)
    if not qtext:
        raise RuntimeError(f"no stem captions for query {sid}")
    qvec = _load_or_build(ctx.cache_dir / f"xref_q_stems_{sid}.npz", [sid],
                          lambda: gemini_text_embed([qtext], "query"), tag)[0]
    return _rank(sid, db_ids, docs, qvec, ctx)


def rank_audio(sid: str, ctx: Context):
    """Bank ranked by Reference-mix vs console-render similarity, descending."""
    tag = "bank=grafx_subgroup"
    cache = ctx.cache_dir / "wref_db.npz"
    with _INDEX_LOCK:
        hit = _load_cache(cache, tag)
    if hit is not None and set(hit[0]) <= set(ctx.bank_ids):
        db_ids, docs = hit
    else:
        if ctx.bank_render_dir is None:
            raise RuntimeError(f"{cache} missing/stale and no --bank-render-dir to rebuild it")
        render = lambda c: ctx.bank_render_dir / c / "grafx_pred.flac"  # noqa: E731
        db_ids = [c for c in ctx.bank_ids if render(c).is_file()]
        docs = _load_or_build(cache, db_ids, lambda: rlat_audio_embed(
            [render(c) for c in db_ids], ctx.audio_device), tag)
    ref = ctx.dataset_root / "Reference" / f"{sid}.flac"
    qvec = _load_or_build(ctx.cache_dir / f"wref_q_{sid}.npz", [sid],
                          lambda: rlat_audio_embed([ref], ctx.audio_device), "ref")[0]
    return _rank(sid, db_ids, docs, qvec, ctx)


def pick_exemplars(sid: str, ctx: Context) -> tuple[list[tuple[str, str, str]], dict]:
    """-> ([(bank_id, source_tag, score_text)], retrieval_meta) for ``ctx.arm``."""
    arm, tk, ak = ctx.arm, ctx.text_k, ctx.audio_k
    if arm == "noicl":
        return [], {"arm": arm}
    if not ctx.bank_ids:  # no retrieval database: plan without examples
        return [], {"arm": "noicl", "requested_arm": arm, "reason": "empty retrieval database"}
    if arm == "random":
        candidates = [b for b in ctx.bank_ids if b not in excluded_ids(sid, ctx)]
        k = tk + ak
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8],
                                                   "big"))
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        picks = [candidates[int(i)] for i in idx]
        return [(p, "random", "") for p in picks], {"arm": arm, "k": k, "neighbors": picks}
    if arm in ("xref", "wref"):
        k, tag = (tk, "text") if arm == "xref" else (ak, "audio")
        ids, sims = (rank_text if arm == "xref" else rank_audio)(sid, ctx)
        picks = ids[:k]
        return ([(p, tag, f"{tag} {float(s):.3f}") for p, s in zip(picks, sims[:k])],
                {"arm": arm, "k": k, "neighbors": picks,
                 "scores": [float(s) for s in sims[:k]], "pool_size": len(ids)})
    if arm != "dual":
        raise ValueError(f"unknown arm {arm}")
    t_ids, t_sims = rank_text(sid, ctx)
    a_ids, a_sims = rank_audio(sid, ctx)
    t_top, a_top = t_ids[:tk], a_ids[:ak]
    t_score = {i: float(s) for i, s in zip(t_ids, t_sims)}
    a_score = {i: float(s) for i, s in zip(a_ids, a_sims)}
    out = []
    for p in t_top + [a for a in a_top if a not in t_top]:
        in_t, in_a = p in t_top, p in a_top
        tag = "text+audio" if (in_t and in_a) else ("text" if in_t else "audio")
        parts = ([f"text {t_score[p]:.3f}"] if in_t else []) + \
                ([f"audio {a_score[p]:.3f}"] if in_a else [])
        out.append((p, tag, " / ".join(parts)))
    meta = {"arm": arm, "text_k": tk, "audio_k": ak, "neighbors": [p for p, _, _ in out],
            "text_top": t_top, "audio_top": a_top,
            "overlap": sorted(set(t_top) & set(a_top)),
            "scores": {"text": {p: t_score[p] for p in t_top},
                       "audio": {p: a_score[p] for p in a_top}}}
    return out, meta


def song_groups_from_mapping(mapping_csv: Path) -> dict[str, str]:
    """sample_id -> source-song key from a dataset's mapping.csv (``old_id`` before
    ``__seg``)."""
    with open(mapping_csv, newline="") as f:
        return {r["new_id"]: "song_" + r["old_id"].split("__seg")[0] for r in csv.DictReader(f)}


# --------------------------------------------------------------------------- LLM stages

def call(backend: VLLMBackend, audio: list[Path], text: str, trace_dir: Path, tag: str) -> str:
    result = backend.generate(conversation(audio, text))
    write_trace(trace_dir, tag, audio, text, result)
    return result.answer


def stage_grouping(backend, stems: list[Path], max_groups: int, cache_file: Path,
                   trace_dir: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    listing = "\n".join(f"{i + 1}. {p.name}" for i, p in enumerate(stems))
    text = (read_prompt("hierarchical_grouping.txt").replace("{stem_count}", str(len(stems)))
            .replace("{stem_file_list}", listing).replace("{max_groups}", str(max_groups)))
    for attempt in range(2):
        try:
            data = parse_json_reply(call(backend, stems, text, trace_dir,
                                         f"stage1_grouping_try{attempt + 1}"))
            covered = sorted(i for g in data["groups"] for i in g["stem_indices"])
            assert covered == list(range(1, len(stems) + 1)), f"coverage {covered}"
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:  # fall back to one bus holding every stem
                data = {"groups": [{"name": "all", "stem_indices": list(range(1, len(stems) + 1))}],
                        "fallback": f"{type(exc).__name__}: {exc}"}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, indent=1))
    return data


def stage_group_tools(backend, group_name: str, member_stems: list[tuple[int, Path]], icl: str,
                      trace_dir: Path) -> dict:
    listing = "\n".join(f"{k + 1}. {p.name}" for k, (_, p) in enumerate(member_stems))
    text = (read_prompt(GROUP_PROMPT).replace("{group_name}", group_name)
            .replace("{stem_count}", str(len(member_stems)))
            .replace("{stem_file_list}", listing).replace("{icl_examples}", icl))
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", group_name)[:24]
    data = parse_json_reply(call(backend, [p for _, p in member_stems], text, trace_dir,
                                 f"stage2_group_{safe}"))
    out = {"stems": [], "group_chain": [t for t in data.get("group_chain", []) if t in STEM_TOOLS]}
    for entry in data.get("stems", []):
        local = int(entry["stem_index"])
        if 1 <= local <= len(member_stems):
            out["stems"].append({"stem_index": member_stems[local - 1][0],
                                 "chain": [t for t in entry.get("chain", []) if t in STEM_TOOLS]})
    return out


def stage_master(backend, session_summary: str, icl: str, trace_dir: Path) -> list[str]:
    text = (read_prompt("tools_only_master.txt").replace("{session_summary}", session_summary)
            .replace("{icl_examples}", icl))
    data = parse_json_reply(call(backend, [], text, trace_dir, "stage3_master"))
    return [t for t in data.get("master_chain", []) if t in MASTER_TOOLS]


# --------------------------------------------------------------------------- scoring

def set_f1(pred: list[str], gt: list[str]) -> float:
    ps, gs = set(pred), set(gt)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    inter = len(ps & gs)
    p, r = inter / len(ps), inter / len(gs)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score_vs_bank(pred: dict, gt: dict) -> dict:
    """Tool-set F1 of a predicted plan vs the song's own bank plan (stems matched by name)."""
    pred_by_key = {stem_key(s["stem_name"]): s["stem_index"] for s in pred["stems"]}
    gt2pred = {s["stem_index"]: pred_by_key[k] for s in gt["stems"]
               if (k := stem_key(s["stem_name"])) in pred_by_key}

    def to_pred(indices) -> set[int]:
        return {gt2pred[i] for i in indices if i in gt2pred}

    group_f1s = []
    for gg, gtools in ((to_pred(g["member_stem_indices"]), g["chain"]) for g in gt.get("groups", [])):
        best, best_overlap = None, 0
        for pg in pred.get("groups", []):
            ov = len(gg & set(pg["member_stem_indices"]))
            if ov > best_overlap:
                best, best_overlap = pg, ov
        group_f1s.append(set_f1(best["chain"] if best else [], gtools))
    pred_stem = {s["stem_index"]: s["chain"] for s in pred["stems"]}
    stem_f1s = [set_f1(pred_stem.get(gt2pred[s["stem_index"]], []), s["chain"])
                for s in gt["stems"] if s["stem_index"] in gt2pred]
    return {
        "group_tools_f1_mean": round(float(np.mean(group_f1s)), 4) if group_f1s else None,
        "stem_tools_f1_mean": round(float(np.mean(stem_f1s)), 4) if stem_f1s else None,
        "master_tools_f1": round(set_f1(pred.get("master_chain", []), gt.get("master_chain", [])), 4),
        "num_stems_matched": len(gt2pred),
    }


# --------------------------------------------------------------------------- per song

def process_song(sid: str, backend, ctx: Context, out_dir: Path) -> dict:
    """Plan one song and write ``<out>/<sid>/<sid>_tools_plan.json``."""
    stems = discover_stems(ctx.input_dir(sid))
    exemplars, retr_meta = pick_exemplars(sid, ctx)
    icl = build_icl(exemplars, ctx)
    song_dir = out_dir / sid
    trace_dir = song_dir / "trace"
    grouping = stage_grouping(backend, stems, ctx.max_groups,
                              ctx.cache_dir / f"grouping_{sid}.json", trace_dir)

    groups_out, stems_out = [], {}
    for g in grouping["groups"]:
        members = [(i, stems[i - 1]) for i in g["stem_indices"]]
        gres = stage_group_tools(backend, g["name"], members, icl, trace_dir)
        groups_out.append({"name": g["name"], "member_stem_indices": sorted(g["stem_indices"]),
                           "chain": gres["group_chain"]})
        for s in gres["stems"]:
            stems_out[s["stem_index"]] = s["chain"]

    session_lines = []
    for g in groups_out:
        session_lines.append(f"Group '{g['name']}' bus chain: {fmt_chain(g['chain'])}")
        for i in g["member_stem_indices"]:
            session_lines.append(f"  stem {stems[i - 1].name[:40]}: "
                                 f"{fmt_chain(stems_out.get(i, []))}")
    master = stage_master(backend, "\n".join(session_lines), icl, trace_dir)

    plan = {
        "sample_id": sid, "arm": ctx.arm, "retrieval": retr_meta, "num_stems": len(stems),
        "stems": [{"stem_index": i + 1, "stem_name": p.stem,
                   "group": next((g["name"] for g in groups_out
                                  if (i + 1) in g["member_stem_indices"]), None),
                   "chain": stems_out.get(i + 1, [])} for i, p in enumerate(stems)],
        "groups": groups_out,
        "master_chain": master,
    }
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / f"{sid}_tools_plan.json").write_text(json.dumps(plan, indent=1))
    gt_path = ctx.bank_plans_dir / f"{sid}_tools_plan.json"
    scores = (score_vs_bank(plan, json.loads(gt_path.read_text())) if gt_path.is_file()
              else {"scored": False})
    (song_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    return {"sample_id": sid, **scores, "exemplars": [f"{p} [{t}]" for p, t, _ in exemplars]}
