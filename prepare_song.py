#!/usr/bin/env python3
"""Prepare your own song for MixAgent.

Takes a folder of dry stems (one audio file per track, any sample rate) and a reference
mix (any song whose mix style you want), and writes them in the layout MixAgent reads:

    <out>/Input/<name>/<stem>.flac        a 15 s excerpt of every stem, each at -16 LUFS
    <out>/Reference/<name>.flac           a 15 s excerpt of the reference, at -12 LUFS
    <out>/FullStems/<name>/<stem>.flac    the full-length stems with the SAME per-stem gain
    <out>/song_info/<name>.json           excerpt windows, per-stem gain and instrument

MixAgent was built and evaluated on 15 s excerpts, and its planner listens to every stem
at once, so the mix is made on an excerpt. render_full_song.py then applies the finished
mix to the full-length stems. The excerpt and the levels follow the paper's data: 44.1 kHz,
each stem normalized to -16 LUFS, tracks quieter than -60 dBFS RMS left out.

    python prepare_song.py --stems my_song/stems --reference ref_song.wav \
        --name my_song --out my_songs

Instruments are read from the stem file names ("lead_vocal.wav" -> "lead vocal"). Give
better names with --instruments names.json ({"Audio 03.wav": "electric bass", ...}).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
STEM_LUFS = -16.0
MIX_LUFS = -12.0
AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator, axis=0).astype(np.float32)
    return x[:, :2]


def lufs(x: np.ndarray) -> float:
    try:
        v = float(pyln.Meter(SR).integrated_loudness(x))
    except Exception:  # noqa: BLE001 - too short / silent
        return -math.inf
    return v if math.isfinite(v) else -math.inf


def rms_dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * math.log10(r) if r > 0 else -math.inf


def loudest_window(x: np.ndarray, n: int) -> int:
    """Start sample of the n-sample window with the most energy (0.5 s grid)."""
    if x.shape[0] <= n:
        return 0
    e = np.square(x.mean(axis=1), dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(e)])
    hop = SR // 2
    starts = np.arange(0, x.shape[0] - n + 1, hop)
    return int(starts[np.argmax(c[starts + n] - c[starts])])


def cut(x: np.ndarray, start: int, n: int) -> np.ndarray:
    y = x[start:start + n]
    return np.pad(y, ((0, n - y.shape[0]), (0, 0))) if y.shape[0] < n else y


def gain_to(x: np.ndarray, target: float) -> float:
    cur = lufs(x)
    return 0.0 if cur == -math.inf else target - cur


def write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SR, subtype="PCM_24")


def safe_name(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return s or "track"


def instrument_from_name(stem: str) -> str:
    s = re.sub(r"(?i)(^|[_\-\s])(dry|stem|track|audio)(?=$|[_\-\s])", " ", stem)
    s = re.sub(r"[_\-\.]+", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", type=Path, required=True, help="folder with one dry audio file per track")
    ap.add_argument("--reference", type=Path, required=True, help="reference mix (any song)")
    ap.add_argument("--name", required=True, help="song id used by MixAgent (letters, digits, _ -)")
    ap.add_argument("--out", type=Path, required=True, help="dataset root to write into")
    ap.add_argument("--start", type=float, default=None,
                    help="excerpt start in seconds (default: the loudest 15 s of the stem sum)")
    ap.add_argument("--ref-start", type=float, default=None,
                    help="reference excerpt start in seconds (default: its loudest 15 s)")
    ap.add_argument("--seconds", type=float, default=15.0, help="excerpt length (paper: 15)")
    ap.add_argument("--instruments", type=Path, default=None,
                    help='JSON {"<stem file name>": "<instrument>"} overriding names from file names')
    ap.add_argument("--silence-dbfs", type=float, default=-60.0)
    ap.add_argument("--no-full", action="store_true", help="skip FullStems/ (no full-length render)")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.name):
        raise SystemExit("--name may only contain letters, digits, '_' and '-'")
    if re.fullmatch(r"sample_\d+", args.name):
        raise SystemExit("--name must not look like sample_NNN: those ids belong to the shipped "
                         "MedleyDB/AlbumDB clips and their cached retrieval embeddings")
    files = sorted(p for p in args.stems.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"no audio files in {args.stems}")
    overrides = json.loads(args.instruments.read_text()) if args.instruments else {}

    stems = {p: load(p) for p in files}
    length = max(x.shape[0] for x in stems.values())
    n = int(round(args.seconds * SR))
    total = np.zeros((length, 2), dtype=np.float32)
    for x in stems.values():
        total[: x.shape[0]] += x if x.shape[1] == 2 else np.repeat(x, 2, axis=1)
    start = int(round(args.start * SR)) if args.start is not None else loudest_window(total, n)

    info = {"name": args.name, "sample_rate": SR, "excerpt_start_sec": start / SR,
            "excerpt_seconds": args.seconds, "stems": [], "skipped_silent": []}
    used = set()
    for p, x in stems.items():
        clip = cut(x, start, n)
        level = rms_dbfs(clip)
        if level < args.silence_dbfs:
            info["skipped_silent"].append({"source": p.name, "rms_dbfs": round(level, 1)})
            continue
        name = safe_name(p.stem)
        while name in used:
            name += "_b"
        used.add(name)
        g = gain_to(clip, STEM_LUFS)
        lin = 10 ** (g / 20)
        write(args.out / "Input" / args.name / f"{name}.flac", clip * lin)
        if not args.no_full:
            write(args.out / "FullStems" / args.name / f"{name}.flac", x * lin)
        inst = overrides.get(p.name) or overrides.get(p.stem) or instrument_from_name(p.stem)
        info["stems"].append({"file": f"{name}.flac", "source": p.name, "instrument": inst,
                              "gain_db": round(g, 4)})

    ref = load(args.reference)
    rs = int(round(args.ref_start * SR)) if args.ref_start is not None else loudest_window(ref, n)
    rclip = cut(ref, rs, n)
    write(args.out / "Reference" / f"{args.name}.flac", rclip * 10 ** (gain_to(rclip, MIX_LUFS) / 20))
    info.update(reference=args.reference.name, reference_start_sec=rs / SR,
                full_stems=not args.no_full, full_seconds=length / SR)
    (args.out / "song_info").mkdir(parents=True, exist_ok=True)
    (args.out / "song_info" / f"{args.name}.json").write_text(json.dumps(info, indent=1) + "\n")

    print(f"{args.name}: {len(info['stems'])} stems, excerpt {start / SR:.1f}-{(start + n) / SR:.1f} s "
          f"(reference {rs / SR:.1f} s), {len(info['skipped_silent'])} silent tracks skipped")
    for s in info["stems"]:
        print(f"  {s['file']:32s} <- {s['source']:32s} instrument: {s['instrument']}")
    print(f"-> {args.out}   (check the instrument names; fix them with --instruments if needed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
