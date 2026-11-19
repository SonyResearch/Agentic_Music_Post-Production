#!/usr/bin/env python3
"""Stage 2: tune the parameters of Stage-1 tool plans (detector -> critic -> revise).

Example (the paper's headline configuration)::

    python run_param_tuning.py --dataset-root /path/to/dataset --songs song_a,song_b \
        --tool-plans results/tool_plans --out results/mixagent_f3_dpo --critic f3_dpo

``--critic`` selects the detector + critic pair (see ``mixagent.param_tuning.PRESETS``):
``f1_none``/``f2_none``/``f3_none`` use the untrained planner model as detector and
critic; ``{f1,f2,f3}_{sft,dpo}`` use the trained detector and the matching trained
critic (start ``servers/detector_server.py`` and ``servers/critic_server.py --critic
<preset>`` first). Pass the Stage-1 plans with ``--tool-plans`` (the output of
``run_tool_planning.py``).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mixagent import paths
from mixagent.backends import BackendError, build_backends, check_health
from mixagent.llm import QwenAgent, VLLMBackend
from mixagent.param_tuning import PRESETS, RunConfig, log, process_song
from mixagent.render import TOOL_ORDER
from mixagent.templates import load_prompts, split_contract


def add_planner_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--planner-urls", default=paths.DEFAULT_PLANNER_URL,
                    help="comma-separated vLLM endpoints; workers round-robin over them")
    ap.add_argument("--planner-model", default=paths.PLANNER_SERVED_NAME,
                    help="served model name on the vLLM endpoint(s)")
    ap.add_argument("--planner-tokenizer", default=str(paths.PLANNER_MODEL_DIR),
                    help="local Qwen3-Omni directory, used to count prompt tokens")
    ap.add_argument("--workers", type=int, default=3, help="songs processed in parallel")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="dataset with Input/<sid>/*.flac, Reference/<sid>.flac (GT/ optional)")
    ap.add_argument("--dataset", default="medleydb", choices=("medleydb", "albumdb"),
                    help="selects the shipped assets under data/<dataset>/")
    ap.add_argument("--songs", default=None,
                    help="comma list, a split name (data/splits/<name>.txt) or a file; "
                         "default: every song that has a tool plan")
    ap.add_argument("--out", type=Path, required=True, help="output run directory")
    ap.add_argument("--critic", default="f3_dpo", choices=sorted(PRESETS))
    ap.add_argument("--tool-plans", type=Path, default=None,
                    help="Stage-1 plans (<dir>/<sid>/<sid>_tools_plan.json); default "
                         "data/<dataset>/tool_plans")
    ap.add_argument("--captions-dir", type=Path, default=None,
                    help="per-stem captions (<dir>/<sid>/<sid>_captions.json); default "
                         "data/<dataset>/audex_captions_instr")
    add_planner_args(ap)
    ap.add_argument("--detector-url", default=paths.DEFAULT_DETECTOR_URL)
    ap.add_argument("--critic-url", default=paths.DEFAULT_CRITIC_URL)
    ap.add_argument("--clip-seconds", type=float, default=14.0)
    ap.add_argument("--log-mrs", choices=("on", "off"), default="on",
                    help="log MRS of the current mix to GT/<sid>.flac (diagnostic only)")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def make_agents(args) -> list[QwenAgent]:
    urls = [u.strip() for u in args.planner_urls.split(",") if u.strip()]
    return [QwenAgent(VLLMBackend(urls[i % len(urls)], args.planner_model,
                                  args.planner_tokenizer), log=log)
            for i in range(args.workers)]


def run(args) -> int:
    tool_plans = args.tool_plans or paths.dataset_data(args.dataset) / "tool_plans"
    cfg = RunConfig(dataset_root=args.dataset_root, tool_plans=tool_plans,
                    captions_dir=args.captions_dir
                    or paths.captions_dir(args.dataset, args.dataset_root),
                    out=args.out, preset=args.critic, detector_url=args.detector_url,
                    critic_url=args.critic_url, clip_seconds=args.clip_seconds,
                    log_mrs=args.log_mrs == "on", overwrite=args.overwrite)
    prompts = load_prompts()
    contract = split_contract(prompts["contract"])
    missing = [t for t in TOOL_ORDER if t not in contract[1]]
    if missing:
        raise SystemExit(f"tool contract missing blocks for: {missing}")
    try:
        backends = build_backends(cfg)
        health = check_health(cfg, backends[0], backends[1])
    except BackendError as e:
        raise SystemExit(f"backend startup check failed: {e}") from None
    for role, info in health.items():
        print(f"[health] {role}: {json.dumps(info)[:200]}", flush=True)

    songs = paths.read_song_list(args.songs) or sorted(
        p.name[:-len("_tools_plan.json")]
        for p in list(tool_plans.glob("*_tools_plan.json"))
        + list(tool_plans.glob("*/*_tools_plan.json"))
        if p.is_file() and (p.parent == tool_plans
                            or p.name == f"{p.parent.name}_tools_plan.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_cfg.update(preset=vars(PRESETS[args.critic]), tool_plans=str(tool_plans),
                   captions_dir=str(cfg.captions_dir), songs=songs,
                   started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (args.out / "run_config.json").write_text(json.dumps(run_cfg, indent=1))

    pool = make_agents(args)
    lock = threading.Lock()

    def worker(sid: str) -> dict:
        with lock:
            agent = pool.pop()
        try:
            return process_song(sid, cfg, agent, prompts, contract, backends)
        except Exception as e:  # noqa: BLE001
            log(sid, f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            err_dir = args.out / sid
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(json.dumps(
                {"song_id": sid, "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()}, indent=1))
            return {"song_id": sid, "status": "error"}
        finally:
            with lock:
                pool.append(agent)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(worker, songs))
    counts = {s: sum(1 for r in results if r.get("status") == s)
              for s in ("ok", "skipped", "error")}
    print(f"\n[run] {args.out}: {counts} / {len(results)}")
    return 0 if counts["error"] == 0 else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
