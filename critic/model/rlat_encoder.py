"""Frozen RLAT wrapper used as the critic's audio encoder.

`get_local_embedding(x)` returns the pre-pooling frame sequence z_local, shape
(B, 108, 1024) for a 10-s 44.1 kHz stereo clip. The Stable Audio Open VAE inside the
encoder draws one latent sample per call (latent_mode "sample", what the adapter was
trained on), so callers that need reproducible outputs seed torch first.

The module stays fp32 even when its parent is cast to bf16: `_apply` drops dtype casts
and keeps only device moves. Embeddings come back in the input's dtype.

The encoder implementation is the shared copy in encoder/ (override the
folder with RLAT_DIR).
"""
import os
import sys

import torch
import torch.nn as nn

from critic.paths import REPO_ROOT, env_path


def _default_checkpoint():
    return str(env_path("MIXAGENT_MODELS") / "rlat")


def _default_sao():
    return str(env_path("MIXAGENT_MODELS") / "stable-audio-open-1.0")


def _import_loader():
    enc_dir = os.environ.get("RLAT_DIR", str(REPO_ROOT / "encoder"))
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from rlat_encoder import load_rlat
    return load_rlat


class RLATLocal(nn.Module):
    def __init__(self, checkpoint_path=None, sao_model_path=None):
        super().__init__()
        load_rlat = _import_loader()
        self.model = load_rlat(
            checkpoint_path=checkpoint_path or _default_checkpoint(),
            sao_model_path=sao_model_path or _default_sao(),
            device="cpu",
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _apply(self, fn, recurse=True):
        # Probe what `fn` does to an fp32 tensor: drop pure dtype casts, keep moves.
        probe = torch.empty(0, dtype=torch.float32)
        probed = fn(probe)
        if probed.dtype != torch.float32:
            if probed.device == probe.device:
                return self
            device = probed.device
            return super()._apply(lambda t: t.to(device), recurse)
        return super()._apply(fn, recurse)

    @torch.no_grad()
    def get_local_embedding(self, x):
        """x: (B, 2, T) float waveform at 44.1 kHz -> (B, T', 1024) in x's dtype."""
        out_dtype = x.dtype
        with torch.inference_mode():
            z_local = self.model.audio_encoder(x.float())["z_local"]
        # inference tensors cannot enter autograd; hand back a normal tensor
        z_local = z_local.clone()
        return z_local.detach().to(out_dtype)
