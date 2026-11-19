#!/usr/bin/env python3
"""MixAgent end to end: Stage 1 (tool planning) then Stage 2 (parameter tuning).

    python run_mixagent.py --dataset-root /path/to/dataset --songs song_a,song_b \
        --out results/mixagent --critic f3_dpo

Writes ``<out>/tool_plans/<sid>/<sid>_tools_plan.json`` and ``<out>/mix/<sid>/<sid>.flac``.
Score the mixes with ``python evaluation/score.py --run <out>/mix --dataset-root ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_param_tuning
import run_tool_planning
from mixagent import paths
from mixagent.param_tuning import PRESETS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--dataset", default="medleydb", choices=("medleydb", "albumdb"))
    ap.add_argument("--songs", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--critic", default="f3_dpo", choices=sorted(PRESETS))
    ap.add_argument("--planner-urls", default=paths.DEFAULT_PLANNER_URL)
    ap.add_argument("--planner-model", default=paths.PLANNER_SERVED_NAME)
    ap.add_argument("--planner-tokenizer", default=str(paths.PLANNER_MODEL_DIR))
    ap.add_argument("--detector-url", default=paths.DEFAULT_DETECTOR_URL)
    ap.add_argument("--critic-url", default=paths.DEFAULT_CRITIC_URL)
    ap.add_argument("--workers", type=int, default=3, help="Stage-2 songs in parallel")
    args = ap.parse_args()

    common = ["--dataset-root", str(args.dataset_root), "--dataset", args.dataset,
              "--planner-urls", args.planner_urls, "--planner-model", args.planner_model,
              "--planner-tokenizer", args.planner_tokenizer]
    if args.songs:
        common += ["--songs", args.songs]
    plans = args.out / "tool_plans"
    rc = run_tool_planning.run(run_tool_planning.build_parser().parse_args(
        common + ["--out", str(plans)]))
    if rc != 0:
        print("[mixagent] some songs failed in Stage 1; Stage 2 runs on the rest")
    return run_param_tuning.run(run_param_tuning.build_parser().parse_args(
        common + ["--out", str(args.out / "mix"), "--tool-plans", str(plans),
                  "--critic", args.critic, "--detector-url", args.detector_url,
                  "--critic-url", args.critic_url, "--workers", str(args.workers)]))


if __name__ == "__main__":
    raise SystemExit(main())
