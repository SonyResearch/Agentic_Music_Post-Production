"""MusicFM + RLAT embeddings for the detector.

Identical maths to the version used to build the training set -- if this drifts,
the detector is fed features it was never trained on and its predictions become
meaningless. Paths come from ``afxdet.paths``.

Both encoders are used *globally* (time-averaged):

* MusicFM 25Hz, layer 7, 24 kHz mono, fp16 -> mean over time  (1024 dims)
* FXEncoder v3 (``LT+LP+Ly`` adapter), ``zy`` embedding, 44.1 kHz stereo (1024)

concatenated into one 2048-d vector per audio input.
"""

from __future__ import annotations

import sys

import numpy as np

from . import paths

MUSICFM_SR = 24000
RLAT_SR = 44100
MUSICFM_DIM = 1024
EMBED_DIM = 2048


def _ensure_paths() -> None:
    for p in (paths.MUSICFM_CODE, paths.RLAT_CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample (T, C) or (T,) float audio."""
    if sr_in == sr_out:
        return audio
    import torch
    import torchaudio.functional as AF

    x = torch.from_numpy(np.ascontiguousarray(audio.T if audio.ndim == 2 else audio)).float()
    y = AF.resample(x, sr_in, sr_out)
    return y.numpy().T if audio.ndim == 2 else y.numpy()


class Embedder:
    """Loads MusicFM and RLAT once; embeds numpy audio arrays."""

    def __init__(
        self,
        device: str = "cuda",
        musicfm_layer: int = 7,
        rlat_embedding: str = "zy",
        latent_mode: str = "sample",
        seed: int | None = 0,
    ):
        """``latent_mode`` controls the Stable-Audio VAE latent FXEncoder runs on.

        ``sample`` draws from the posterior, which is what the detector was
        trained on -- and which makes repeated inference on the same audio
        differ, because the draw is random. ``mode`` takes the posterior mean:
        deterministic, but a feature distribution the detector has not seen.

        The default keeps training-time behaviour and pins ``seed`` so that a
        given input reproduces a given output. Set ``seed=None`` to let it drift.
        """
        _ensure_paths()
        import torch

        if latent_mode not in ("sample", "mode"):
            raise ValueError("latent_mode must be 'sample' or 'mode'")
        self.device = device
        self.musicfm_layer = musicfm_layer
        self.rlat_embedding = rlat_embedding
        self.latent_mode = latent_mode
        self.seed = seed
        self._torch = torch

        missing = [p for p in (paths.MUSICFM_CKPT, paths.MUSICFM_STATS,
                               paths.RLAT_DIR, paths.SAO_DIR) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "missing model files:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nRun setup/download_models.sh and place the released weights "
                "under weights/ (or set MIXAGENT_MODELS / MIXAGENT_WEIGHTS)."
            )

        from musicfm.model.musicfm_25hz import MusicFM25Hz

        self.musicfm = (
            MusicFM25Hz(
                is_flash=False,
                stat_path=str(paths.MUSICFM_STATS),
                model_path=str(paths.MUSICFM_CKPT),
            )
            .to(device)
            .half()
        )
        self.musicfm.eval()

        from rlat_encoder import load_rlat

        self.rlat = load_rlat(
            checkpoint_path=str(paths.RLAT_DIR),
            sao_model_path=str(paths.SAO_DIR),
            device=device,
        )
        # Reaches the latent_mode switch in encoder/rlat_encoder.py.
        enc = getattr(self.rlat, "audio_encoder", None)
        if enc is None:
            raise RuntimeError("RLAT has no audio_encoder; encoder/rlat_encoder.py changed")
        enc.latent_mode = self.latent_mode

    def embed_musicfm(self, audio: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        mono = _resample(mono, sr, MUSICFM_SR)
        x = torch.from_numpy(np.ascontiguousarray(mono)).float().to(self.device).half()
        with torch.no_grad():
            latent = self.musicfm.get_latent(x.unsqueeze(0), layer_ix=self.musicfm_layer)
        return latent.squeeze(0).float().mean(dim=0).cpu().numpy()

    def embed_rlat(self, audio: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch
        stereo = audio if audio.ndim == 2 else np.stack([audio, audio], axis=1)
        if stereo.shape[1] == 1:
            stereo = np.repeat(stereo, 2, axis=1)
        stereo = _resample(stereo[:, :2], sr, RLAT_SR)
        x = torch.from_numpy(np.ascontiguousarray(stereo.T)).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.seed is not None and self.latent_mode == "sample":
                # Re-seed per call, so a given audio input maps to a given
                # embedding regardless of how many calls preceded it.
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            emb = self.rlat(x, embedding=self.rlat_embedding)
        return emb.squeeze().float().cpu().numpy()

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Concatenated [MusicFM ; RLAT] embedding, guaranteed finite.

        The output is sanitized, not just the input: MusicFM runs in fp16, and
        audio with extreme gain can overflow it and return inf/NaN. One NaN
        would propagate through the transformer and make every prediction in the
        scenario NaN, so it is clamped here.
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        out = np.concatenate(
            [self.embed_musicfm(audio, sr), self.embed_rlat(audio, sr)]
        ).astype(np.float32)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
