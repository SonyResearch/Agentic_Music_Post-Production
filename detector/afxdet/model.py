"""Autoregressive corruption detector over a variable-length set of units.

Task shape
----------
One example is a scenario: ``N`` detection units (stems, or subgroups in the
full-song scenario) plus the mixture rendered from the current parameters.
The model emits ``N`` binary labels, one per unit -- "were this unit's fx-chain
parameters swapped to the corrupted plan?". ``N`` varies from 1 to ~23 across
songs, so both the input and the output are variable length.

Inputs per unit
---------------
* ``dry``  -- embedding of the raw input stem (or the sum of a subgroup's raw stems)
* ``wet``  -- embedding of that unit's isolated contribution under the current params
* ``wet - dry`` -- supplied explicitly; this difference *is* the "what did this
  chain do" evidence, and making it a first-class feature saves the projection
  from having to learn a subtraction.
* chain composition -- the ordered fx tool types in the unit's chain, pooled
  into one vector. Known at inference time and tells the model which
  parameters could even have been swapped.

Architecture
------------
1. Per-unit projection to ``d_model``, plus a mixture token projected by its own
   head, plus scenario-kind and unit-position embeddings.
2. A bidirectional Transformer **encoder** over ``[mix, unit_1..unit_N]``. Every
   unit attends to the mixture and to its peers, which is what lets the model
   reason relatively ("this stem is the odd one out given the others").
3. An autoregressive Transformer **decoder** that emits labels left to right.
   Step ``t`` attends causally over previously emitted labels and cross-attends
   to the encoder memory. Modelling ``p(y_t | y_<t, x)`` captures the label
   dependencies the sampler creates -- corruption is sparse and capped at five
   modules, so units are far from independent.

Training uses teacher forcing with a padding-masked BCE; inference decodes
greedily (or by sampling) one unit at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Every fx module type that appears in these consoles.
TOOL_TYPES = [
    "eq",
    "compressor",
    "noisegate",
    "distortion",
    "reverb",
    "delay",
    "stereo_imager",
    "gain_panning",
    "limiter",
    "multiband_compressor",
]
TOOL_TO_ID = {t: i + 1 for i, t in enumerate(TOOL_TYPES)}  # 0 = padding
N_TOOL_IDS = len(TOOL_TYPES) + 1

SCENARIO_KINDS = ["subgroup", "full"]
SCENARIO_TO_ID = {k: i for i, k in enumerate(SCENARIO_KINDS)}

MAX_CHAIN_LEN = 12
MAX_UNITS = 32


@dataclass
class DetectorConfig:
    embed_dim: int = 2048          # one audio embedding: MusicFM 1024 + RLAT 1024
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    tool_embed_dim: int = 64
    max_units: int = MAX_UNITS
    max_chain_len: int = MAX_CHAIN_LEN
    # --- ablation switches -------------------------------------------------
    use_dry: bool = True           # raw stem embedding
    use_wet: bool = True           # rendered-under-current-params embedding
    use_delta: bool = True         # wet - dry, the "what did this chain do" cue
    use_chain: bool = True         # pooled fx-tool-type embedding
    use_mix: bool = True           # the scenario mixture token
    # Absolute unit position. The task is permutation-equivariant -- "which of
    # these stems was corrupted" does not depend on the order they are listed in
    # -- so a position embedding carries no task signal, but it does let the
    # model key on "unit 3 of this song", which is a memorization channel and
    # this dataset has only ~129 training songs to memorize.
    use_unit_pos: bool = True
    decoder: str = "ar"            # "ar" (autoregressive) | "parallel" (independent heads)
    predict_modules: bool = False  # also emit a per-fx-module logit (joint task)
    module_loss_weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def n_audio_features(self) -> int:
        return int(self.use_dry) + int(self.use_wet) + int(self.use_delta)


class ChainEncoder(nn.Module):
    """Pools a unit's ordered fx-tool types into one vector."""

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.tool_embed = nn.Embedding(N_TOOL_IDS, cfg.tool_embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(cfg.max_chain_len, cfg.tool_embed_dim)
        self.out = nn.Linear(cfg.tool_embed_dim * 2, cfg.tool_embed_dim * 2)

    def forward(self, chain_ids: torch.Tensor) -> torch.Tensor:
        """chain_ids: (B, U, L) -> (B, U, 2 * tool_embed_dim)"""
        b, u, length = chain_ids.shape
        mask = (chain_ids != 0).float().unsqueeze(-1)
        pos = torch.arange(length, device=chain_ids.device).clamp_max(self.pos_embed.num_embeddings - 1)
        h = self.tool_embed(chain_ids) + self.pos_embed(pos)[None, None]
        h = h * mask
        summed = h.sum(dim=2)
        count = mask.sum(dim=2).clamp_min(1.0)
        pooled = torch.cat([summed / count, summed], dim=-1)
        return self.out(pooled)


class CorruptionDetector(nn.Module):
    """Encoder over (mixture, units) + autoregressive per-unit binary decoder."""

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        chain_dim = cfg.tool_embed_dim * 2 if cfg.use_chain else 0
        if cfg.n_audio_features == 0:
            raise ValueError("at least one of use_dry/use_wet/use_delta must be enabled")
        unit_in = cfg.n_audio_features * cfg.embed_dim + chain_dim

        self.unit_proj = nn.Sequential(
            nn.LayerNorm(unit_in),
            nn.Linear(unit_in, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.mix_proj = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.chain_encoder = ChainEncoder(cfg) if cfg.use_chain else None
        self.unit_pos_embed = nn.Embedding(cfg.max_units, cfg.d_model)
        self.scenario_embed = nn.Embedding(len(SCENARIO_KINDS), cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_decoder_layers)

        # Decoder step input: previous label (0/1/BOS) + this unit's encoded token.
        self.label_embed = nn.Embedding(3, cfg.d_model)  # 0, 1, 2=BOS
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # --- joint module head ------------------------------------------
        # Localizing *which* fx module was altered is a strictly finer task than
        # naming the stem, and the two share all the evidence, so the module head
        # hangs off the same per-unit representation rather than a second tower.
        # Each module is addressed by its unit's state plus its own tool type and
        # position in the chain.
        if cfg.predict_modules:
            self.module_tool_embed = nn.Embedding(N_TOOL_IDS, cfg.tool_embed_dim, padding_idx=0)
            self.module_pos_embed = nn.Embedding(cfg.max_chain_len, cfg.tool_embed_dim)
            self.module_head = nn.Sequential(
                nn.LayerNorm(cfg.d_model + 2 * cfg.tool_embed_dim),
                nn.Linear(cfg.d_model + 2 * cfg.tool_embed_dim, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )

    def module_logits(self, unit_states: torch.Tensor, chain_ids: torch.Tensor) -> torch.Tensor:
        """Per-module logits, shape (B, U, L), aligned with ``chain_ids``."""
        b, u, length = chain_ids.shape
        tool = self.module_tool_embed(chain_ids)                     # (B,U,L,E)
        pos = torch.arange(length, device=chain_ids.device).clamp_max(
            self.module_pos_embed.num_embeddings - 1
        )
        pos = self.module_pos_embed(pos)[None, None].expand(b, u, length, -1)
        h = unit_states.unsqueeze(2).expand(b, u, length, unit_states.shape[-1])
        return self.module_head(torch.cat([h, tool, pos], dim=-1)).squeeze(-1)

    # ------------------------------------------------------------------
    def encode(
        self,
        dry: torch.Tensor,
        wet: torch.Tensor,
        mix: torch.Tensor,
        chain_ids: torch.Tensor,
        unit_mask: torch.Tensor,
        scenario_kind: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory, memory_padding_mask).

        dry/wet: (B, U, E)   mix: (B, E)   chain_ids: (B, U, L)
        unit_mask: (B, U) bool, True = real unit
        scenario_kind: (B,) long
        """
        cfg = self.cfg
        parts = []
        if cfg.use_dry:
            parts.append(dry)
        if cfg.use_wet:
            parts.append(wet)
        if cfg.use_delta:
            parts.append(wet - dry)
        if self.chain_encoder is not None:
            parts.append(self.chain_encoder(chain_ids))
        unit_tok = self.unit_proj(torch.cat(parts, dim=-1))

        u = unit_tok.shape[1]
        if cfg.use_unit_pos:
            pos = torch.arange(u, device=unit_tok.device).clamp_max(cfg.max_units - 1)
            unit_tok = unit_tok + self.unit_pos_embed(pos)[None]
        unit_tok = unit_tok + self.scenario_embed(scenario_kind)[:, None]

        if cfg.use_mix:
            mix_tok = self.mix_proj(mix).unsqueeze(1)
            tokens = torch.cat([mix_tok, unit_tok], dim=1)
            mix_ok = torch.ones(unit_mask.shape[0], 1, dtype=torch.bool, device=unit_mask.device)
            key_padding = ~torch.cat([mix_ok, unit_mask], dim=1)
        else:
            tokens = unit_tok
            key_padding = ~unit_mask
        memory = self.encoder(tokens, src_key_padding_mask=key_padding)
        return memory, key_padding

    def _unit_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Encoder outputs for the units, dropping the mixture token if present."""
        return memory[:, 1:] if self.cfg.use_mix else memory

    def _decoder_inputs(self, memory: torch.Tensor, prev_labels: torch.Tensor) -> torch.Tensor:
        """Unit tokens from memory + previous-label embedding."""
        return self._unit_memory(memory) + self.label_embed(prev_labels)

    def forward(
        self,
        dry: torch.Tensor,
        wet: torch.Tensor,
        mix: torch.Tensor,
        chain_ids: torch.Tensor,
        unit_mask: torch.Tensor,
        scenario_kind: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced unit logits (B, U).

        With ``predict_modules`` the return is ``(unit_logits, module_logits)``
        where module logits are (B, U, L) aligned with ``chain_ids``.
        """
        memory, mem_pad = self.encode(dry, wet, mix, chain_ids, unit_mask, scenario_kind)

        if self.cfg.decoder == "parallel":
            # Independent per-unit heads: no label conditioning, so the model
            # cannot exploit dependencies between units (e.g. corruption
            # sparsity). This is the ablation the AR decoder is measured against.
            states = self._unit_memory(memory)
        else:
            b, u = labels.shape
            bos = torch.full((b, 1), 2, dtype=torch.long, device=labels.device)
            prev = torch.cat([bos, labels[:, :-1].clamp(0, 1).long()], dim=1)

            tgt = self._decoder_inputs(memory, prev)
            causal = nn.Transformer.generate_square_subsequent_mask(u, device=tgt.device)
            states = self.decoder(
                tgt,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=~unit_mask,
                memory_key_padding_mask=mem_pad,
            )

        unit_logits = self.head(states).squeeze(-1)
        if self.cfg.predict_modules:
            return unit_logits, self.module_logits(states, chain_ids)
        return unit_logits

    @torch.no_grad()
    def generate(
        self,
        dry: torch.Tensor,
        wet: torch.Tensor,
        mix: torch.Tensor,
        chain_ids: torch.Tensor,
        unit_mask: torch.Tensor,
        scenario_kind: torch.Tensor,
        threshold: float = 0.5,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy (or sampled) left-to-right decode. Returns (labels, probs)."""
        memory, mem_pad = self.encode(dry, wet, mix, chain_ids, unit_mask, scenario_kind)
        b, u = unit_mask.shape

        if self.cfg.decoder == "parallel":
            states = self._unit_memory(memory)
            probs = torch.sigmoid(self.head(states).squeeze(-1))
            preds = (probs > threshold).long() * unit_mask.long()
            if self.cfg.predict_modules:
                mprobs = torch.sigmoid(self.module_logits(states, chain_ids))
                return preds, probs, (mprobs > threshold).long(), mprobs
            return preds, probs
        prev = torch.full((b, u), 2, dtype=torch.long, device=memory.device)
        probs = torch.zeros(b, u, device=memory.device)
        preds = torch.zeros(b, u, dtype=torch.long, device=memory.device)

        causal = nn.Transformer.generate_square_subsequent_mask(u, device=memory.device)
        for t in range(u):
            tgt = self._decoder_inputs(memory, prev)
            out = self.decoder(
                tgt,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=~unit_mask,
                memory_key_padding_mask=mem_pad,
            )
            p = torch.sigmoid(self.head(out[:, t]).squeeze(-1))
            y = torch.bernoulli(p).long() if sample else (p > threshold).long()
            probs[:, t] = p
            preds[:, t] = y
            if t + 1 < u:
                prev[:, t + 1] = y
        preds = preds * unit_mask.long()

        if self.cfg.predict_modules:
            # Re-run the decoder on the finally-decoded labels so the module
            # logits condition on the same prefix the unit predictions imply,
            # rather than on the partial prefix of the last decode step.
            bos = torch.full((b, 1), 2, dtype=torch.long, device=memory.device)
            states = self.decoder(
                self._decoder_inputs(memory, torch.cat([bos, preds[:, :-1]], dim=1)),
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=~unit_mask,
                memory_key_padding_mask=mem_pad,
            )
            mprobs = torch.sigmoid(self.module_logits(states, chain_ids))
            return preds, probs, (mprobs > threshold).long(), mprobs
        return preds, probs


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    unit_mask: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """BCE-with-logits averaged over real units only."""
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    per = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none", pos_weight=weight
    )
    per = per * unit_mask.float()
    return per.sum() / unit_mask.float().sum().clamp_min(1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
