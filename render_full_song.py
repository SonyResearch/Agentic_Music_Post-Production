#!/usr/bin/env python3
"""Apply a finished MixAgent mix to the full-length stems of your song.

MixAgent tunes the mix on a 15 s excerpt. This renders the final plan
(<mix-dir>/<song>/<song>_plan_final.json) on <dataset-root>/FullStems/<song>/, the
full-length stems that prepare_song.py wrote with the same per-stem gain as the excerpt,
and normalizes the result to -12 LUFS (less if that would clip).

    python render_full_song.py --dataset-root my_songs --mix-dir results/my_song/mix \
        --song my_song --out my_song_full_mix.flac

Effects are applied with the same processors and settings as the excerpt. Time-varying
behaviour (compressor/gate envelopes, reverb tails) simply runs over the whole song.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mixagent import audio as A
from mixagent.render import HierRenderer, to_stereo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--mix-dir", type=Path, required=True, help="the 'mix' folder of a MixAgent run")
    ap.add_argument("--song", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run = args.mix_dir / args.song
    plan = json.loads((run / f"{args.song}_plan_final.json").read_text())
    order = [ln.split(": ", 1)[1] for ln in (run / "stem_order.txt").read_text().splitlines()[1:] if ": " in ln]
    full = args.dataset_root / "FullStems" / args.song
    stems = [full / name for name in order]
    missing = [p.name for p in stems if not p.is_file()]
    if missing:
        raise SystemExit(f"{full}: missing {missing[:5]} (was prepare_song.py run without --no-full?)")
    renderer = HierRenderer(stems)
    mix = to_stereo(renderer.render_full(plan))
    # -12 LUFS like the excerpt, but a whole song has louder moments than its excerpt:
    # lower the gain instead of clipping when the peak would pass the ceiling.
    lufs = A.integrated_lufs(mix, renderer.sr)
    gain = 10.0 ** ((A.TARGET_LUFS - lufs) / 20.0) if lufs is not None else 1.0
    gain = min(gain, A.SAVE_PEAK_CEILING / max(float(np.abs(mix).max()), 1e-9))
    mix = (mix * gain).astype(np.float32)
    A.write_flac(mix, renderer.sr, args.out)
    print(f"{args.song}: full-length mix ({mix.shape[0] / renderer.sr:.1f} s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
