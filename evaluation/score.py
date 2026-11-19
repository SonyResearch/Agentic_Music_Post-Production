#!/usr/bin/env python3
"""Score a run directory of rendered mixes against the ground-truth mixes.

Metrics (all vs ``<dataset-root>/GT/<sid>.flac``, prediction loudness-normalized to
-12 LUFS first, both trimmed to the common length):

  mrs       multi-resolution STFT distance (L/R + M/S)            lower is better
  mss       mid/side multi-scale spectral loss                     lower is better
  af        audio-feature distance                                 lower is better
  tunejury  TuneJury reward (reference-free)                       higher is better
  fxencpp   Fx-Encoder++ embedding cosine to GT                    higher is better
  rlat      RLAT embedding cosine to GT                           higher is better

MRS / MSS / AF need only torch + auraloss + pyloudnorm. The three neural metrics are
optional (see ``evaluation/encoders.py`` for their code / weights) and are skipped with
a message when unavailable; ``--metrics`` selects a subset.

    python evaluation/score.py --run results/mixagent_f3_dpo --dataset-root /path/to/MedleyDB_kit

Writes ``<run>/scores.csv`` (per song) and ``<run>/scores_summary.json`` (means).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mixagent import metrics as M  # noqa: E402

ALL_METRICS = ("mrs", "mss", "af", "tunejury", "fxencpp", "rlat")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return M.ensure_stereo(audio), int(sr)


def gt_path_for(gt_dir: Path, sid: str) -> Path:
    for ext in (".flac", ".wav"):
        p = gt_dir / f"{sid}{ext}"
        if p.is_file():
            return p
    return gt_dir / f"{sid}.flac"


def load_pair(pred_path: Path, gt_path: Path, normalize: bool = True):
    pred, sr = read_audio(pred_path)
    gt, gsr = read_audio(gt_path)
    if gsr != sr:
        import torch
        import torchaudio.functional as AF

        with torch.inference_mode():
            gt = M.ensure_stereo(AF.resample(torch.from_numpy(np.ascontiguousarray(gt.T)),
                                             gsr, sr).numpy().T)
    n = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:n], gt[:n]
    if normalize:
        pred = M.loudness_normalize(pred, sr, M.SCORE_LUFS)
    return pred, gt, sr


def score_song(pred_path: Path, gt_path: Path, metrics: tuple[str, ...], device: str,
               seed: int | None = 0) -> dict:
    pred, gt, sr = load_pair(pred_path, gt_path)
    pt, gtt = M._to_tensor(pred, device), M._to_tensor(gt, device)
    row: dict = {}
    if "mrs" in metrics:
        row["mrs"] = M.mrs(pt, gtt, sr)
    if "mss" in metrics:
        row["mss"] = M.mss(pt, gtt)
    if "af" in metrics:
        row["af"] = M.af(pt, gtt, sr)
    neural = [m for m in metrics if m in ("tunejury", "fxencpp", "rlat")]
    if neural:
        from evaluation import encoders as E

        for m in neural:
            if seed is not None:  # TuneJury and RLAT are stochastic; seed per call
                import torch

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            try:
                if m == "tunejury":
                    row[m] = E.tunejury(pred, sr, device)
                elif m == "fxencpp":
                    row[m] = E.cosine(E.fxencpp_embedding(pred, sr, device),
                                      E.fxencpp_embedding(gt, sr, device))
                else:
                    row[m] = E.cosine(E.rlat_embedding(pred, sr, device),
                                      E.rlat_embedding(gt, sr, device))
            except E.MetricUnavailable as e:
                print(f"[skip] {m}: {e}", flush=True)
                row[m] = None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="run dir with <sid>/<sid>.flac")
    ap.add_argument("--dataset-root", type=Path, required=True, help="dataset with GT/<sid>.flac")
    ap.add_argument("--songs", default=None, help="comma list / split name (default: all)")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed set before each stochastic metric call (-1 = unseeded)")
    args = ap.parse_args()

    from mixagent.paths import read_song_list

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        ap.error(f"unknown metrics {sorted(unknown)}; choose from {ALL_METRICS}")
    songs = read_song_list(args.songs) or sorted(
        d.name for d in args.run.iterdir() if (d / f"{d.name}.flac").is_file())
    rows = []
    for sid in songs:
        pred = args.run / sid / f"{sid}.flac"
        gt = gt_path_for(args.dataset_root / "GT", sid)
        if not (pred.is_file() and gt.is_file()):
            print(f"[skip] {sid}: missing {pred if not pred.is_file() else gt}", flush=True)
            continue
        row = {"sample_id": sid, **score_song(pred, gt, metrics, args.device,
                                              None if args.seed < 0 else args.seed)}
        rows.append(row)
        print("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in row.items()), flush=True)
    if not rows:
        print("nothing scored")
        return 1
    with open(args.run / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", *metrics])
        w.writeheader()
        w.writerows(rows)
    summary = {"n_songs": len(rows)}
    for m in metrics:
        vals = [r[m] for r in rows if isinstance(r.get(m), float)]
        summary[m] = {"mean": float(np.mean(vals)) if vals else None,
                      "median": float(np.median(vals)) if vals else None, "count": len(vals)}
    (args.run / "scores_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
