#!/usr/bin/env python3
"""Caption the stems of your own songs (Stage 1 needs them; the paper's clips ship theirs).

For every stem of each song, NVIDIA Nemotron-Labs-Audex-30B-A3B listens to the dry stem,
is told its instrument, and writes a short mixing-oriented caption. This is exactly how the
shipped captions in data/<dataset>/audex_captions_instr were made (same model revision,
prompt mixagent/prompts/stem_caption_with_instrument.txt, 120 words, temperature 0.7,
top-p 0.9, 16 kHz input, no reasoning trace).

    python caption_stems.py --dataset-root my_songs --songs my_song

Writes <dataset-root>/Input/audex_captions_instr/<song>/<song>_captions.json, which
run_tool_planning.py / run_param_tuning.py / run_mixagent.py pick up automatically.
Instruments come from <dataset-root>/song_info/<song>.json (written by prepare_song.py),
else from the stem file names.

Needs one ~80 GB GPU, the model (`bash setup/download_models.sh captioner`) and the
captioner extras on top of the agent environment (setup/env_captioner.txt).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

from mixagent import paths
from mixagent.paths import read_song_list

AUDIO_EXTS = {".flac", ".wav"}
PROMPT = paths.REPO_ROOT / "mixagent" / "prompts" / "stem_caption_with_instrument.txt"
DEFAULT_MODEL = paths.MODELS / "Nemotron-Labs-Audex-30B-A3B"
MODEL_ID = "nvidia/Nemotron-Labs-Audex-30B-A3B"


class Audex:
    """Thin wrapper over the inference helpers that ship inside the model repo."""

    def __init__(self, model_dir: Path, device_map: str, device: str):
        root = model_dir.resolve()
        inf_dir = root / "inference_scripts_hf"
        ckpt = root / "checkpoint_folder_full"
        if not inf_dir.is_dir() or not ckpt.is_dir():
            raise SystemExit(f"Audex model not found under {root} "
                             "(run: bash setup/download_models.sh captioner)")
        sys.path.insert(0, str(inf_dir))
        self.inf = importlib.import_module("inference_hf")
        self.au = importlib.import_module("audio_utils")
        self.model, self.tok, self.fe, self.cfg = self.inf.load_model(
            str(ckpt), device_map=device_map, device=device, torch_dtype="auto",
            tp_plan="none", refresh_code_cache=True, hf_backend="auto")

    def caption(self, audio: Path, prompt: str, args) -> str:
        out = self.inf.generate_one(
            self.model, self.tok, self.fe, self.cfg, str(audio.resolve()), prompt,
            sample_rate=16000, reasoning=False, prompt_repitition="none",
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=0)
        _thinking, answer = self.au.split_thinking(out)
        return answer.strip()


def instrument_from_name(stem: str) -> str:
    s = re.sub(r"(?i)(^|[_\-\s])(dry|stem|track|audio)(?=$|[_\-\s])", " ", stem)
    s = re.sub(r"[_\-\.]+", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--songs", required=True, help="comma list, split name or file")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--caption-words", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device-map", default="auto", help="auto | single")
    ap.add_argument("--device", default="cuda:0", help="used with --device-map single")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    template = PROMPT.read_text(encoding="utf-8")
    out_root = args.dataset_root / "Input" / paths.CAPTIONS_DIRNAME
    songs = read_song_list(args.songs)
    todo = [s for s in songs if args.overwrite or not (out_root / s / f"{s}_captions.json").is_file()]
    if not todo:
        print("all requested songs already captioned")
        return 0
    model = Audex(args.model_dir, args.device_map, args.device)

    for sid in todo:
        stem_dir = args.dataset_root / "Input" / sid
        if not stem_dir.is_dir():
            print(f"  skip {sid}: {stem_dir} not found")
            continue
        info_path = args.dataset_root / "song_info" / f"{sid}.json"
        known = {}
        if info_path.is_file():
            known = {s["file"]: s["instrument"] for s in json.loads(info_path.read_text())["stems"]}
        entries, t0 = [], time.time()
        for p in sorted((q for q in stem_dir.iterdir() if q.suffix.lower() in AUDIO_EXTS),
                        key=lambda q: q.name.lower()):
            inst = known.get(p.name) or instrument_from_name(p.stem)
            prompt = template.format(instruments=inst, caption_words=args.caption_words)
            entries.append({"filename": p.name, "instrument": inst,
                            "caption": model.caption(p, prompt, args)})
            print(f"  {sid}/{p.name} ({inst}): {entries[-1]['caption'][:90]}...", flush=True)
        dst = out_root / sid / f"{sid}_captions.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"sample_id": sid, "model_id": MODEL_ID, "stem_captions": entries},
                                  indent=1) + "\n")
        print(f"{sid}: {len(entries)} stems captioned in {time.time() - t0:.0f}s -> {dst}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
