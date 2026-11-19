"""RLAT audio encoder (inference only), shared by the critic and the detector.

Loads the released RLAT weights (``config.json`` + ``model.safetensors`` from the
Hugging Face repo shlee-97/rlat, downloaded to models/rlat by setup/download_models.sh),
an adapter that runs on the latent of the Stable Audio Open 1.0 VAE (models/stable-audio-open-1.0,
only its ``vae/`` subfolder is read). ``zy`` is the wet-audio global embedding; the
critic uses the per-frame ``z_local`` sequence (arXiv:2608.28127).

``RLATAudioEncoder.latent_mode``: "sample" (default, what the critic and the detector
were trained on) draws from the VAE posterior, "mode" takes its mean.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1.0)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        return freqs.reshape(seq_len, -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    x_rot, x_rest = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = x_rot * freqs.cos() + _rotate_half(x_rot) * freqs.sin()
    return torch.cat((x_rot, x_rest), dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.proj_in = nn.Linear(dim, inner * 2)
        self.proj_out = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(self.dropout(x * F.silu(gate)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, qk_norm: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"Attention dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.dim_head = dim // heads
        self.qk_norm = qk_norm
        self.dropout = dropout
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        default_scale = self.dim_head ** -0.5
        self._qk_prescale = 10.0 / default_scale if qk_norm else 1.0

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, d = self.heads, self.dim_head
        q = self.to_q(x).view(b, t, h, d).transpose(1, 2)
        k = self.to_k(x).view(b, t, h, d).transpose(1, 2)
        v = self.to_v(x).view(b, t, h, d).transpose(1, 2)
        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self._qk_prescale
            k = F.normalize(k, dim=-1)
        q = _apply_rotary(q, freqs)
        k = _apply_rotary(k, freqs)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.to_out(out.transpose(1, 2).reshape(b, t, h * d))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout, qk_norm=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.ffn(self.norm2(x))
        return x


class CompactTransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_head = dim // heads
        self.rotary = RotaryEmbedding(dim_head // 2)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads=heads, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.rotary(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, freqs)
        return self.final_norm(x)


class XTransformerAdaptor(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        num_blocks: int = 4,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.transformer = CompactTransformerEncoder(
            dim=hidden_dim,
            depth=num_blocks,
            heads=nhead,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x.permute(0, 2, 1)


class MeanPool(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=2)


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InversePredictor(nn.Module):
    def __init__(self, audio_dim: int = 1024, transform_dim: int = 1024, hidden_dim: int = 2048):
        super().__init__()
        self.input_proj = nn.Linear(2 * audio_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
                nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, transform_dim)

    def forward(self, z_dry: torch.Tensor, z_wet: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(torch.cat([z_dry, z_wet], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


def _load_sao_encoder(model_path: str, device: torch.device):
    try:
        from diffusers import AutoencoderOobleck
    except ImportError as exc:
        raise ImportError(
            "rlat raw-audio extraction requires diffusers (see setup/env_*.txt)."
        ) from exc

    vae = AutoencoderOobleck.from_pretrained(model_path, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False)
    return vae.to(device)


class RLATAudioEncoder(nn.Module):
    def __init__(
        self,
        sao_model_path: str,
        sao_latent_dim: int = 64,
        latent_pack_factor: int = 2,
        sao_encode_batch_size: int = 8,
    ):
        super().__init__()
        self.sao_model_path = sao_model_path
        self.sao_latent_dim = sao_latent_dim
        self.latent_pack_factor = latent_pack_factor
        self.sao_encode_batch_size = sao_encode_batch_size
        self.adaptor = XTransformerAdaptor(
            in_dim=sao_latent_dim * latent_pack_factor,
            hidden_dim=1024,
            out_dim=1024,
            num_blocks=4,
            nhead=8,
            ff_mult=4,
            dropout=0.1,
        )
        self.global_pool = MeanPool()
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self._sao_container: dict[str, nn.Module] = {}

    @property
    def _sao(self) -> Optional[nn.Module]:
        return self._sao_container.get("vae")

    def _ensure_sao_loaded(self, device: torch.device) -> None:
        if self._sao is None:
            self._sao_container["vae"] = _load_sao_encoder(self.sao_model_path, device)

    def _pack_latent(self, z: torch.Tensor) -> torch.Tensor:
        p = self.latent_pack_factor
        if p <= 1:
            return z
        b, c, t = z.shape
        t_new = (t + p - 1) // p * p
        if t_new > t:
            z = F.pad(z, (0, t_new - t))
        return z.unfold(2, p, p).transpose(2, 3).reshape(b, c * p, -1)

    def forward_from_latent(self, z_sao: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.adaptor(self._pack_latent(z_sao))
        z_global = self.global_pool(h)
        out = {
            "z_global": z_global,
            "z_feat_map": h.unsqueeze(2),
            "z_local": h.permute(0, 2, 1),
            "z_proj": self.projection_head(z_global),
        }
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_sao_loaded(x.device)
        with torch.inference_mode(), torch.amp.autocast(x.device.type, enabled=False):
            x_f32 = x.float()
            chunks = []
            for i in range(0, x_f32.shape[0], self.sao_encode_batch_size):
                encoded = self._sao.encode(x_f32[i : i + self.sao_encode_batch_size])
                # latent_mode: "sample" draws from the VAE posterior
                # (random unless the caller seeds the RNG),
                # "mode" takes the posterior mean.
                dist = encoded.latent_dist
                if getattr(self, "latent_mode", "sample") == "mode":
                    chunks.append(dist.mode() if hasattr(dist, "mode") else dist.mean)
                else:
                    chunks.append(dist.sample())
            z_sao = torch.cat(chunks, dim=0)
        return self.forward_from_latent(z_sao)


class RLAT(nn.Module):
    def __init__(self, sao_model_path: str):
        super().__init__()
        self.audio_encoder = RLATAudioEncoder(sao_model_path=sao_model_path)
        self.inverse_predictor = InversePredictor()
        self.proj_LT = ProjectionMLP()

    def forward(self, audio: torch.Tensor, embedding: str = "zy") -> torch.Tensor:
        out = self.audio_encoder(audio)
        zy = out["z_global"]
        if embedding == "zy":
            return zy
        if embedding == "zproj":
            return out["z_proj"]
        if embedding == "zt":
            dry = torch.zeros_like(zy)
            return self.inverse_predictor(dry, zy)
        if embedding == "zt_proj":
            dry = torch.zeros_like(zy)
            return self.proj_LT(self.inverse_predictor(dry, zy))
        raise ValueError("embedding must be one of: zy, zproj, zt, zt_proj")


def _clean_key(key: str) -> str:
    key = key.replace("._orig_mod.", ".")
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    return key


def _sub_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = {}
    prefix_dot = prefix + "."
    for key, value in state_dict.items():
        if key.startswith(prefix_dot):
            out[_clean_key(key[len(prefix_dot) :])] = value
    return out


# the architecture built above; config.json of the released weights must agree
_EXPECTED_CONFIG = {
    "audio": {"latent_dim": 64, "hidden_dim": 1024, "embed_dim": 1024, "depth": 4, "heads": 8,
              "pack_factor": 2, "projection_dim": 512, "global_pool": "mean"},
    "inverse": {"audio_dim": 1024, "transform_dim": 1024, "hidden_dim": 2048, "num_layers": 2},
    "projection": {"input_dim": 1024, "hidden_dim": 512, "output_dim": 128},
    "blind_inverse": True,
}


def _check_config(path: str) -> None:
    import json

    with open(path) as f:
        config = json.load(f)
    for section, expected in _EXPECTED_CONFIG.items():
        got = config.get(section)
        if isinstance(expected, dict):
            got = {k: (got or {}).get(k) for k in expected}
        if got != expected:
            raise RuntimeError(f"rlat config.json [{section}] = {got}, this code expects {expected}")


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """A folder with config.json + model.safetensors (the released weights), a single
    .safetensors file, or a Lightning training checkpoint (.ckpt)."""
    if os.path.isdir(checkpoint_path):
        config = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config):
            _check_config(config)
        checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing rlat weights: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def load_rlat(
    checkpoint_path: str,
    sao_model_path: str,
    device: str = "cuda",
) -> RLAT:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing rlat weights: {checkpoint_path} (run setup/download_models.sh)")
    if not os.path.isdir(sao_model_path):
        raise FileNotFoundError(f"Missing Stable Audio Open model directory: {sao_model_path}")

    model = RLAT(sao_model_path=sao_model_path)
    state_dict = _load_state_dict(checkpoint_path)

    missing, unexpected = model.audio_encoder.load_state_dict(
        _sub_state_dict(state_dict, "audio_encoder"),
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"rlat audio encoder load mismatch: missing={missing}, unexpected={unexpected}")

    model.inverse_predictor.load_state_dict(_sub_state_dict(state_dict, "inverse_predictor"), strict=True)
    model.proj_LT.load_state_dict(_sub_state_dict(state_dict, "proj_LT"), strict=True)
    model.to(device)
    model.eval()
    return model
