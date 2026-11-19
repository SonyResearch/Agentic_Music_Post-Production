#!/usr/bin/env python3
"""Stage 1: group the stems into buses and choose every tool chain (no parameters).

Example (paper setting: dual retrieval, 3 text + 1 audio exemplars)::

    python run_tool_planning.py --dataset-root /path/to/dataset --songs song_a,song_b \
        --out results/tool_plans_medleydb

Writes ``<out>/<sid>/<sid>_tools_plan.json`` (+ ``scores.json``: tool-set F1 against the
song's own bank plan, and ``trace/``). The retrieval bank (``data/<dataset>/bank_plans``)
is not shipped (without it, use ``--arm noicl``). For a new song, missing text embeddings are computed with ``gemini-embedding-001``
(needs ``GEMINI_API_KEY``) and missing Reference embeddings with RLAT. Delete
``grouping_<sid>.json`` in the cache (or use ``--cache-dir``) to re-run the grouping call.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mixagent import paths
from mixagent.llm import VLLMBackend
from mixagent.tool_planning import Context, process_song, song_groups_from_mapping


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="dataset with Input/<sid>/*.flac and Reference/<sid>.flac")
    ap.add_argument("--dataset", default="medleydb", choices=("medleydb", "albumdb"),
                    help="selects the shipped bank / captions / caches under data/<dataset>/")
    ap.add_argument("--songs", default=None,
                    help="comma list, split name (data/splits/<name>.txt) or file; "
                         "default: every bank song")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="dual", choices=("dual", "xref", "wref", "random", "noicl"),
                    help="exemplar retrieval: dual = text + audio (paper), xref = text only, "
                         "wref = audio only, random, noicl = no exemplars")
    ap.add_argument("--text-k", type=int, default=3)
    ap.add_argument("--audio-k", type=int, default=1)
    ap.add_argument("--max-groups", type=int, default=8)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="embedding + grouping cache (default: a copy of "
                         "data/<dataset>/retrieval_cache inside --out)")
    ap.add_argument("--bank-render-dir", type=Path, default=None,
                    help="only to rebuild the audio bank index: <dir>/<bank_id>/grafx_pred.flac")
    ap.add_argument("--audio-device", default="cuda:0",
                    help="device for RLAT when a Reference embedding is missing")
    ap.add_argument("--planner-urls", default=paths.DEFAULT_PLANNER_URL,
                    help="comma-separated vLLM endpoints; one song per endpoint at a time")
    ap.add_argument("--planner-model", default=paths.PLANNER_SERVED_NAME)
    ap.add_argument("--planner-tokenizer", default=str(paths.PLANNER_MODEL_DIR))
    return ap


def run(args) -> int:
    data = paths.dataset_data(args.dataset)
    cache_dir = args.cache_dir
    if cache_dir is None:  # never write into the shipped cache
        cache_dir = args.out / "_cache"
        if not cache_dir.exists() and (data / "retrieval_cache").is_dir():
            shutil.copytree(data / "retrieval_cache", cache_dir)
    groups_file = data / "song_groups.json"
    mapping = args.dataset_root / "mapping.csv"
    song_groups = (json.loads(groups_file.read_text()) if groups_file.is_file()
                   else song_groups_from_mapping(mapping) if mapping.is_file() else {})
    ctx = Context(dataset_root=args.dataset_root, bank_plans_dir=data / "bank_plans",
                  captions_dir=paths.captions_dir(args.dataset, args.dataset_root),
                  cache_dir=cache_dir, song_groups=song_groups, arm=args.arm,
                  text_k=args.text_k, audio_k=args.audio_k, max_groups=args.max_groups,
                  bank_render_dir=args.bank_render_dir, audio_device=args.audio_device)
    songs = paths.read_song_list(args.songs) or ctx.bank_ids
    missing = [s for s in songs if not ctx.input_dir(s).is_dir()]
    if missing:
        raise SystemExit(f"songs not found under {args.dataset_root / 'Input'}: {missing[:5]}")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(songs=songs, n_bank=len(ctx.bank_ids), cache_dir=str(cache_dir))
    (args.out / "run_config.json").write_text(json.dumps(cfg, indent=1))

    urls = [u.strip() for u in args.planner_urls.split(",") if u.strip()]
    pool: queue.Queue = queue.Queue()
    for url in urls:
        pool.put(VLLMBackend(url, args.planner_model, args.planner_tokenizer))

    def run_one(sid: str) -> dict:
        backend = pool.get()
        try:
            print(f"  {sid} -> {backend.base_url}", flush=True)
            if (args.out / sid / f"{sid}_tools_plan.json").is_file():
                return {"sample_id": sid, "resumed": True}
            return process_song(sid, backend, ctx, args.out)
        except Exception as exc:  # noqa: BLE001 - one failed song never stops the run
            print(f"  [FAIL] {sid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return {"sample_id": sid, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            pool.put(backend)

    with ThreadPoolExecutor(max_workers=min(len(urls), len(songs))) as ex:
        summary = list(ex.map(run_one, songs))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    n_err = sum(1 for s in summary if "error" in s)
    print(f"[stage 1] {len(summary) - n_err}/{len(summary)} songs planned -> {args.out}")
    return 0 if n_err == 0 else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
