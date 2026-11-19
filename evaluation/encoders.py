"""Optional neural metrics: RLAT / Fx-Encoder++ embedding cosine and TuneJury.

Each needs external code and weights (missing pieces raise ``MetricUnavailable``):

* RLAT -- ``encoder/rlat_encoder.py``,
  ``$MIXAGENT_MODELS/rlat`` (shlee-97/rlat) and the Stable Audio Open 1.0 VAE
  (``$MIXAGENT_MODELS/stable-audio-open-1.0``). Embedding = ``zy``, mean over 10 s chunks
  at 44.1 kHz (the encoder samples its VAE latent, so values vary slightly per call).
* Fx-Encoder++ -- ``pip install fxencoder_plusplus laion_clap``,
  ``$MIXAGENT_MODELS/fxencoder_plusplus/fxenc_plusplus_default.pt`` and the LAION-CLAP
  checkpoint ``$MIXAGENT_MODELS/fxencoder_plusplus/clap-630k-audioset-best.pt``.
* TuneJury -- the ``tunejury`` package (github.com/yonghyunk1m/TuneJury) on PYTHONPATH or
  ``$TUNEJURY_ROOT``; head ``tunejury.pt`` (HF ``TuneJury/tunejury``) plus its LAION-CLAP
  music checkpoint ``music_audioset_epoch_15_esc_90.14.pt`` and ``m-a-p/MERT-v1-330M``
  (fetched by the package). Reference-free, scored with an empty prompt. CLAP takes a
  random 10 s crop of longer clips, so scores vary between calls.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from mixagent import paths

ENC_SR = 44100
RLAT_CHUNK_SECONDS = 10.0


class MetricUnavailable(RuntimeError):
    pass


def _device(device: str):
    import torch

    return torch.device(device)


def _waveform(audio: np.ndarray, sample_rate: int, device):
    """[T, C] array -> [2, T] tensor at 44.1 kHz."""
    import torch

    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    a = a[:, :2]
    wav = torch.from_numpy(np.ascontiguousarray(a.T)).float()
    if int(sample_rate) != ENC_SR:
        import julius

        wav = julius.resample_frac(wav, int(sample_rate), ENC_SR)
    return wav.contiguous().to(device)


def cosine(x, y) -> float:
    import torch.nn.functional as F

    return float(F.cosine_similarity(x.reshape(1, -1).float(), y.reshape(1, -1).float(),
                                     dim=-1, eps=1e-8).detach().cpu().item())


# --------------------------------------------------------------------------- RLAT

@lru_cache(maxsize=2)
def _rlat_model(device: str):
    root = paths.REPO_ROOT / "encoder"
    ckpt = paths.MODELS / "rlat"
    sao = paths.MODELS / "stable-audio-open-1.0"
    for p in (root / "rlat_encoder.py", ckpt, sao):
        if not p.exists():
            raise MetricUnavailable(f"RLAT needs {p}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rlat_encoder import load_rlat

    model = load_rlat(checkpoint_path=str(ckpt), sao_model_path=str(sao), device=device)
    model.eval()
    return model


def rlat_embedding(audio: np.ndarray, sample_rate: int, device: str = "cuda:0"):
    import torch

    model = _rlat_model(str(_device(device)))
    wav = _waveform(audio, sample_rate, _device(device))
    n = int(round(RLAT_CHUNK_SECONDS * ENC_SR))
    embs = []
    with torch.inference_mode():
        for start in range(0, wav.shape[-1], n):
            chunk = wav[:, start:start + n]
            if chunk.shape[-1] == 0:
                continue
            if chunk.shape[-1] < n:
                chunk = torch.nn.functional.pad(chunk, (0, n - chunk.shape[-1]))
            embs.append(model(chunk.unsqueeze(0), embedding="zy").squeeze(0))
    return torch.stack(embs, dim=0).mean(dim=0)


# --------------------------------------------------------------------------- Fx-Encoder++

@lru_cache(maxsize=2)
def _fxencpp_model(device: str):
    d = paths.MODELS / "fxencoder_plusplus"
    enc, clap = d / "fxenc_plusplus_default.pt", d / "clap-630k-audioset-best.pt"
    for p in (enc, clap):
        if not p.is_file():
            raise MetricUnavailable(f"Fx-Encoder++ needs {p}")
    try:
        import laion_clap
        from fxencoder_plusplus import load_model
    except ImportError as e:
        raise MetricUnavailable(f"Fx-Encoder++ needs fxencoder_plusplus + laion_clap: {e}")
    original = laion_clap.CLAP_Module.load_ckpt

    def load_local_clap(self, ckpt=None, model_id=-1, verbose=True):
        return original(self, ckpt=str(clap), model_id=model_id, verbose=False)

    laion_clap.CLAP_Module.load_ckpt = load_local_clap
    try:
        model = load_model("default", model_path=str(enc), device=device)
    finally:
        laion_clap.CLAP_Module.load_ckpt = original
    model.eval()
    return model


def fxencpp_embedding(audio: np.ndarray, sample_rate: int, device: str = "cuda:0"):
    import torch

    model = _fxencpp_model(str(_device(device)))
    wav = _waveform(audio, sample_rate, _device(device))
    with torch.inference_mode():
        return model.fx_encoder(wav.unsqueeze(0))["embedding"].squeeze(0)


# --------------------------------------------------------------------------- TuneJury

@lru_cache(maxsize=2)
def _tunejury_scorer(device: str):
    try:
        from tunejury.score import Scorer
    except ImportError:
        root = os.environ.get("TUNEJURY_ROOT")
        if not root:
            raise MetricUnavailable("TuneJury needs the tunejury package (set TUNEJURY_ROOT "
                                    "to a clone of github.com/yonghyunk1m/TuneJury)")
        sys.path.insert(0, root)
        from tunejury.score import Scorer
    ckpt = os.environ.get("TUNEJURY_CKPT")
    downloaded = paths.MODELS / "tunejury" / "tunejury.pt"  # setup/download_models.sh eval
    if not ckpt and downloaded.is_file():
        ckpt = str(downloaded)
    if not ckpt:
        root = os.environ.get("TUNEJURY_ROOT")
        local = Path(root) / "checkpoints" / "tunejury.pt" if root else None
        if local is not None and local.is_file():
            ckpt = str(local)
        else:
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download("TuneJury/tunejury", "tunejury.pt")
    return Scorer.from_pretrained(ckpt, device=device)


def tunejury(audio: np.ndarray, sample_rate: int, device: str = "cuda:0") -> float:
    a = np.asarray(audio, dtype=np.float32)
    return float(_tunejury_scorer(device).score_waveform(a, int(sample_rate), text=""))
