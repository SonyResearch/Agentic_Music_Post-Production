"""Inference API: which stems had their fx parameters corrupted, and which modules.

The model judges each unit (a stem, or a subgroup bus) *relative to its peers and
to the mixture*, so it must be given a whole scenario at once, not one stem at a
time. Three audio signals per unit define the task:

``dry``  the unit's raw input, before its fx chain
``wet``  the unit's isolated contribution to the mix, after its fx chain
``mix``  the mixture the units currently sum to

The evidence the model actually keys on is ``wet - dry`` -- what the chain did --
compared against what the other units' chains did. Feeding it a ``wet`` rendered
under different parameters than the ``mix`` will produce nonsense, so keep the
three consistent: they must all come from the same session state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import paths
from .embed import EMBED_DIM, Embedder
from .model import (
    MAX_CHAIN_LEN,
    SCENARIO_TO_ID,
    TOOL_TO_ID,
    TOOL_TYPES,
    CorruptionDetector,
    DetectorConfig,
)

DEFAULT_CHECKPOINT = paths.DETECTOR_CKPT


@dataclass
class Unit:
    """One detection unit: a stem (or subgroup bus) and its fx chain."""

    name: str
    dry: np.ndarray
    wet: np.ndarray
    sr: int
    chain: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = [t for t in self.chain if t not in TOOL_TO_ID]
        if unknown:
            raise ValueError(
                f"unit {self.name!r}: unknown fx tool(s) {unknown}. "
                f"Known types: {sorted(TOOL_TYPES)}"
            )
        if len(self.chain) > MAX_CHAIN_LEN:
            raise ValueError(
                f"unit {self.name!r}: chain of {len(self.chain)} exceeds the "
                f"{MAX_CHAIN_LEN} the model was trained with"
            )


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 (T, C)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, int(sr)


class AnomalyFxDetector:
    """Loads the trained detector and the two audio encoders."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        embedder: Embedder | None = None,
        latent_mode: str = "sample",
        seed: int | None = 0,
    ):
        self.device = device
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        # Configs saved before a field existed simply take its default, which is
        # the value that was in force when the checkpoint was trained.
        valid = set(DetectorConfig.__dataclass_fields__)
        cfg = DetectorConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        self.config = cfg
        if cfg.embed_dim != EMBED_DIM:
            raise ValueError(
                f"checkpoint expects {cfg.embed_dim}-d embeddings but this package "
                f"produces {EMBED_DIM}-d"
            )
        self.model = CorruptionDetector(cfg).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.training_metrics = ckpt.get("metrics", {})
        self.embedder = embedder if embedder is not None else Embedder(
            device=device, latent_mode=latent_mode, seed=seed
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        units: list[Unit],
        mix: np.ndarray,
        mix_sr: int,
        scenario_kind: str = "subgroup",
        threshold: float = 0.5,
    ) -> dict:
        if not units:
            raise ValueError("no units given")
        if len(units) > self.config.max_units:
            raise ValueError(
                f"{len(units)} units exceeds the {self.config.max_units} the model "
                "was trained with"
            )
        if scenario_kind not in SCENARIO_TO_ID:
            raise ValueError(f"scenario_kind must be one of {list(SCENARIO_TO_ID)}")

        n = len(units)
        dry = np.stack([self.embedder.embed(u.dry, u.sr) for u in units])
        wet = np.stack([self.embedder.embed(u.wet, u.sr) for u in units])
        mix_emb = self.embedder.embed(mix, mix_sr)

        chain_ids = torch.zeros(1, n, MAX_CHAIN_LEN, dtype=torch.long)
        for i, u in enumerate(units):
            for j, tool in enumerate(u.chain[:MAX_CHAIN_LEN]):
                chain_ids[0, i, j] = TOOL_TO_ID[tool]

        d = torch.from_numpy(dry).float().unsqueeze(0).to(self.device)
        w = torch.from_numpy(wet).float().unsqueeze(0).to(self.device)
        m = torch.from_numpy(mix_emb).float().unsqueeze(0).to(self.device)
        chain_ids = chain_ids.to(self.device)
        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)
        kind = torch.tensor([SCENARIO_TO_ID[scenario_kind]], device=self.device)

        out = self.model.generate(d, w, m, chain_ids, mask, kind, threshold=threshold)
        preds, probs = out[0][0].cpu().numpy(), out[1][0].cpu().numpy()
        mod_probs = out[3][0].cpu().numpy() if len(out) == 4 else None

        results = []
        for i, u in enumerate(units):
            entry = {
                "unit": u.name,
                "corrupted": bool(preds[i]),
                "probability": round(float(probs[i]), 4),
                "modules": [],
            }
            if mod_probs is not None:
                for j, tool in enumerate(u.chain[:MAX_CHAIN_LEN]):
                    p = float(mod_probs[i, j])
                    entry["modules"].append({
                        "position": j,
                        "tool": tool,
                        "corrupted": bool(p > threshold),
                        "probability": round(p, 4),
                    })
            results.append(entry)

        return {
            "scenario_kind": scenario_kind,
            "threshold": threshold,
            "n_units": n,
            "corrupted_units": [r["unit"] for r in results if r["corrupted"]],
            "units": results,
        }

    def predict_manifest(self, manifest: str | Path, threshold: float = 0.5) -> dict:
        """Run on a manifest JSON (schema: detector/example/manifest.example.json)."""
        path = Path(manifest)
        spec = json.loads(path.read_text())
        base = path.parent

        def resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (base / q)

        units, sr_seen = [], set()
        for u in spec["units"]:
            dry, sr_d = load_audio(resolve(u["dry"]))
            wet, sr_w = load_audio(resolve(u["wet"]))
            if sr_d != sr_w:
                raise ValueError(f"unit {u['name']!r}: dry {sr_d} Hz vs wet {sr_w} Hz")
            sr_seen.add(sr_d)
            units.append(Unit(name=u["name"], dry=dry, wet=wet, sr=sr_d,
                              chain=list(u.get("chain", []))))
        mix, mix_sr = load_audio(resolve(spec["mix"]))
        if len(sr_seen) > 1:
            raise ValueError(f"units have mixed sample rates: {sorted(sr_seen)}")

        return self.predict(
            units, mix, mix_sr,
            scenario_kind=spec.get("scenario_kind", "subgroup"),
            threshold=threshold,
        )
