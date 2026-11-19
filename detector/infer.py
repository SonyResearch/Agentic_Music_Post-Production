#!/usr/bin/env python
"""Command-line inference for the anomaly fx detector.

    python infer.py --manifest session.json
    python infer.py --manifest session.json --threshold 0.4 --json out.json

Manifest schema: detector/example/manifest.example.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from afxdet import AnomalyFxDetector  # noqa: E402
from afxdet.detector import DEFAULT_CHECKPOINT  # noqa: E402


def format_report(result: dict) -> str:
    lines = [
        f"scenario: {result['scenario_kind']}  units: {result['n_units']}  "
        f"threshold: {result['threshold']}",
        "",
        f"{'unit':<24} {'verdict':<12} {'p':>6}   modules flagged",
        "-" * 78,
    ]
    for u in result["units"]:
        flagged = [f"{m['tool']}@{m['position']}({m['probability']:.2f})"
                   for m in u["modules"] if m["corrupted"]]
        verdict = "CORRUPTED" if u["corrupted"] else "clean"
        lines.append(
            f"{u['unit']:<24} {verdict:<12} {u['probability']:>6.3f}   "
            + (", ".join(flagged) if flagged else "-")
        )
    lines += ["", f"corrupted stems: {result['corrupted_units'] or 'none'}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSON describing the stems, their chains and the mix")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda",
                    help="cuda, cuda:N or cpu (cpu works but is slow)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold; lower favours recall")
    ap.add_argument("--json", type=Path, default=None, help="also write raw JSON here")
    ap.add_argument("--latent-mode", choices=["sample", "mode"], default="sample",
                    help="VAE latent used by FXEncoder. 'sample' matches how the "
                         "detector was trained (default); 'mode' uses the posterior "
                         "mean. Both are deterministic here (see --seed).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the VAE draw; -1 to leave it unseeded")
    args = ap.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            "Place the released weights under weights/ (or set MIXAGENT_WEIGHTS)."
        )

    detector = AnomalyFxDetector(
        checkpoint=args.checkpoint, device=args.device,
        latent_mode=args.latent_mode, seed=None if args.seed < 0 else args.seed,
    )
    result = detector.predict_manifest(args.manifest, threshold=args.threshold)
    print(format_report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
