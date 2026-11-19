#!/usr/bin/env python
"""Verify the detector's files are in place and the pipeline runs end to end.

Synthesizes a short scenario, writes it as a manifest, and runs the real
inference path over it. This checks the plumbing -- checkpoints load,
both encoders run, shapes line up, outputs are finite and correctly shaped. It
does **not** check accuracy: the audio is noise, so the predictions on it are
meaningless by construction and are not asserted.

    python detector/selftest.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from afxdet import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true", help="keep the generated audio")
    args = ap.parse_args()

    failures: list[str] = []

    print("[1/5] checking files")
    required = [
        paths.DETECTOR_CKPT,
        paths.MUSICFM_CKPT,
        paths.MUSICFM_STATS,
        paths.RLAT_DIR / "model.safetensors",
        paths.SAO_DIR / "vae" / "config.json",
        paths.SAO_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        paths.RLAT_CODE / "rlat_encoder.py",
        paths.MUSICFM_CODE / "musicfm" / "model" / "musicfm_25hz.py",
    ]
    for p in required:
        ok = p.exists()
        print(f"      {'ok  ' if ok else 'MISS'} {p}")
        if not ok:
            failures.append(f"missing {p}")
    if failures:
        print("\nFAILED: files missing (run setup/download_models.sh; put weights/ in place)")
        return 1

    print("[2/5] importing")
    import soundfile as sf

    from afxdet import AnomalyFxDetector, Unit

    print("[3/5] loading model + encoders (downloads nothing; ~30 s cold)")
    det = AnomalyFxDetector(device=args.device)
    print(f"      config: d_model={det.config.d_model} decoder={det.config.decoder} "
          f"predict_modules={det.config.predict_modules}")
    reported = det.training_metrics.get("val/f1")
    if reported:
        print(f"      checkpoint's recorded validation F1: {reported:.4f}")

    print("[4/5] running the in-memory API on a 3-unit scenario")
    rng = np.random.default_rng(0)
    sr = 44100
    units, wets = [], []
    for i, chain in enumerate([["eq", "compressor"], ["eq", "reverb", "limiter"], ["gain_panning"]]):
        dry = (rng.standard_normal((sr * 5, 2)) * 0.05).astype(np.float32)
        wet = (dry * (0.8 + 0.1 * i)).astype(np.float32)
        units.append(Unit(name=f"stem{i}", dry=dry, wet=wet, sr=sr, chain=chain))
        wets.append(wet)
    mix = np.sum(wets, axis=0).astype(np.float32)

    res = det.predict(units, mix, sr, scenario_kind="subgroup")
    assert res["n_units"] == 3, res
    assert len(res["units"]) == 3, res
    for u, unit in zip(res["units"], units):
        assert 0.0 <= u["probability"] <= 1.0, u
        assert len(u["modules"]) == len(unit.chain), (u, unit.chain)
        for m in u["modules"]:
            assert 0.0 <= m["probability"] <= 1.0, m
    print(f"      per-stem probabilities: "
          f"{[u['probability'] for u in res['units']]}")
    print(f"      module predictions present: "
          f"{sum(len(u['modules']) for u in res['units'])}")

    print("[5/5] running the manifest path")
    tmp = Path(tempfile.mkdtemp(prefix="afxdet_selftest_"))
    spec = {"scenario_kind": "subgroup", "mix": "mix.wav", "units": []}
    # Float WAV, so the comparison below isolates the code path rather than
    # 16-bit quantization (which on its own shifts probabilities by ~0.007).
    sf.write(tmp / "mix.wav", mix, sr, subtype="FLOAT")
    for u in units:
        sf.write(tmp / f"{u.name}_dry.wav", u.dry, sr, subtype="FLOAT")
        sf.write(tmp / f"{u.name}_wet.wav", u.wet, sr, subtype="FLOAT")
        spec["units"].append({"name": u.name, "dry": f"{u.name}_dry.wav",
                              "wet": f"{u.name}_wet.wav", "chain": u.chain})
    (tmp / "manifest.json").write_text(json.dumps(spec, indent=2))
    res2 = det.predict_manifest(tmp / "manifest.json")
    assert res2["n_units"] == 3, res2

    # 1e-3 rather than exact: MusicFM runs in fp16, so repeated inference on the
    # same audio varies in the fourth decimal.
    diff = float(np.max(np.abs(
        np.array([u["probability"] for u in res["units"]])
        - np.array([u["probability"] for u in res2["units"]])
    )))
    same = diff < 1e-3
    print(f"      manifest path agrees with in-memory path: {same} (max diff {diff:.1e})")
    if not same:
        failures.append("manifest and in-memory paths disagree")

    if args.keep:
        print(f"      kept: {tmp}")
    else:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll checks passed -- the detector is installed and runnable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
