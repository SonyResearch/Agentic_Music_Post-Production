"""Critic LM wrapper (Lightning module) for SFT: Qwen3.5-4B + RLAT + AL adapter (+ LoRA).

The critic hears three 10-s clips: the target element, the current mix and the
reference. Each clip is encoded by the frozen RLAT encoder, resampled to 32 vectors by
the AL adapter and spliced into the prompt at the `<|vision_start|>` token, each block
preceded by the token embeddings of a plain-text label:

    [prompt ...] "target stem audio" z_t(32) "current mix audio" z_m(32)
                 "reference audio" z_r(32) [... prompt] [answer]

Loss is computed on the answer tokens only (prompt, labels and audio positions are -100).

Stage 1 (apply_lora false): only the adapter trains; the prompt is the bare
`<|vision_start|>` and the target is the criticism JSON. The adapter is written to
`<expdir>/adapter_weights.pth` (latest) and `<expdir>/adapter_weights_step<N>.pth` at
every validation.
Stage 2 (apply_lora true): adapter + LoRA train on the chat prompt -> CoT + JSON. The
adapter starts from `projection_ckpt` (a stage-1 adapter file or Lightning ckpt);
`init_weights_from` instead restores adapter + LoRA from an earlier Lightning ckpt
(used by DPO).
"""
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from critic.model.al_adapter import AL_Adapter
from critic.model.model_setup import setup_llm
from critic.model.utils import instantiate_from_config

TARGET_AUDIO_PREFIX = "target stem audio"
MIX_AUDIO_PREFIX = "current mix audio"
REF_AUDIO_PREFIX = "reference audio"
START_OF_AUDIO = "<|vision_start|>"
START_OF_THINK = "<think>"
END_OF_MSG = "<|im_end|>"
NTL_WEIGHT = 0.3
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_projection_state(path, projector_type="adapter"):
    """Adapter weights from a stage-1 file: `*_weights.pth` or a Lightning `.ckpt`."""
    if os.path.isdir(path):
        path = os.path.join(path, f"{projector_type}_weights.pth")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = {k[len("projection."):]: v for k, v in state["state_dict"].items()
                 if k.startswith("projection.")}
    return state


class LMTrainingWrapper(pl.LightningModule):
    def __init__(self, model_config=None, optimizer_config=None, scheduler_config=None):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model_type = model_config.model_type
        self.expdir = model_config.expdir
        self.max_length = model_config.max_length
        self.audio_encoder_type = model_config.audio_encoder_type
        self.projector_type = model_config.projector_type
        self.target_dtype = self.setup_precision()
        self.audio_encoder = self.setup_audio_encoder()
        self.lm, self.tokenizer = setup_llm(
            self.model_config.model_path,
            self.model_config.attn_implementation,
            self.model_config.cache_dir,
            self.target_dtype,
        )
        self.lm_hidden_size = self.lm.config.hidden_size
        if self.projector_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(self.model_config.audio_hidden_size, self.lm_hidden_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.lm_hidden_size, self.lm_hidden_size, bias=False),
            )
        elif self.projector_type == "adapter":
            self.projection = AL_Adapter(
                num_layers=6,
                num_learnable_latents=32,
                i_dim=self.model_config.audio_hidden_size,
                o_dim=self.lm_hidden_size,
                num_heads=12,
            )
        projection_ckpt = self.model_config.get("projection_ckpt", None)
        if projection_ckpt:
            self.projection.load_state_dict(load_projection_state(projection_ckpt, self.projector_type))
            print(f"[projection_ckpt] adapter initialised from {projection_ckpt}")
        self.audio_encoder.to(self.target_dtype)   # no-op for RLAT (stays fp32)
        self.projection.to(self.target_dtype)
        self.lm.to(self.target_dtype)
        self.target_token_id = self.tokenizer.encode(TARGET_AUDIO_PREFIX)
        self.mix_token_id = self.tokenizer.encode(MIX_AUDIO_PREFIX)
        self.ref_token_id = self.tokenizer.encode(REF_AUDIO_PREFIX)
        self.think_token_id = self.tokenizer.encode(START_OF_THINK)
        self.audio_token_id = self.tokenizer.encode(START_OF_AUDIO)
        self.end_of_msg = END_OF_MSG
        self.lm.eval()
        self.audio_encoder.eval()
        self.projection.train()
        for param in self.lm.parameters():
            param.requires_grad = False
        if self.model_config.apply_lora:
            self.apply_lora()
        init_weights_from = self.model_config.get("init_weights_from", None)
        if init_weights_from:
            self.load_warm_start_weights(init_weights_from)
        if self.model_config.loss_type == "ce_ntl":
            from critic.model.ntloss import NTLoss
            activated_vocab = self.lm.lm_head.weight.shape[0]
            self.ntl_fn = NTLoss(tokenizer=self.tokenizer, activated_vocab=activated_vocab)
        self.validation_step_outputs = []

    # ------------------------------------------------------------------ setup
    def loss_fn(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        if self.model_config.loss_type == "ce":
            return ce_loss
        if self.model_config.loss_type == "ce_ntl":
            ntl = self.ntl_fn(logits, labels)
            return ce_loss + (ntl * NTL_WEIGHT)
        raise ValueError(f"unknown loss_type {self.model_config.loss_type!r}")

    def apply_lora(self):
        from peft import LoraConfig, get_peft_model
        target_modules = list(self.model_config.get("lora_target_modules", DEFAULT_LORA_TARGETS))
        lora_config = LoraConfig(
            r=self.model_config.lora_rank,
            lora_alpha=(self.model_config.lora_rank * 2),
            target_modules=target_modules,
            lora_dropout=float(self.model_config.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lm = get_peft_model(self.lm, lora_config)
        self.lm.print_trainable_parameters()

    def load_warm_start_weights(self, ckpt_path):
        """Restore adapter + LoRA tensors (names containing "projection" or "lora_")
        from a Lightning ckpt or a released `.pt`; optimizer and scheduler start fresh."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        wanted = {k: v for k, v in state.items() if ("projection" in k) or ("lora_" in k)}
        own = self.state_dict()
        loadable = {k: v for k, v in wanted.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(loadable, strict=False)
        print(f"[init_weights_from] {ckpt_path}: loaded {len(loadable)} projection/LoRA keys, "
              f"skipped {len(wanted) - len(loadable)} unmatched, "
              f"ignored {len(state) - len(wanted)} other keys")

    def setup_audio_encoder(self):
        if self.audio_encoder_type != "rlat":
            raise ValueError(f"unsupported audio_encoder_type {self.audio_encoder_type!r}")
        from critic.model.rlat_encoder import RLATLocal
        audio_encoder = RLATLocal(
            checkpoint_path=self.model_config.get("rlat_checkpoint", None),
            sao_model_path=self.model_config.get("sao_model_path", None),
        )
        audio_encoder.eval()
        return audio_encoder

    def setup_precision(self):
        if self.model_config.precision in ["bf16", "bf16-mixed"]:
            return torch.bfloat16
        if self.model_config.precision in ["32", "32-mixed"]:
            return torch.float32
        raise ValueError(f"unsupported precision {self.model_config.precision!r}")

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    # ------------------------------------------------------------------ audio
    def audio_embeddings_for(self, batch):
        return self.get_audio_embedding(batch["audio_target"], batch["audio_mix"], batch["audio_ref"])

    def get_audio_embedding(self, audio_target, audio_mix, audio_ref):
        """(z_target, z_mix, z_ref), each (B, 32, hidden). Accepts waveforms (B, 2, T)
        or precomputed z_local embeddings (B, 108, 1024) from the EmbCache."""
        B = audio_target.size(0)
        concat_audios = torch.cat([audio_target, audio_mix, audio_ref], dim=0)
        if concat_audios.shape[-1] == self.model_config.audio_hidden_size:
            audio_embeds = concat_audios
        else:
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_local_embedding(concat_audios).detach()
        # self.target_dtype, not self.dtype: the first parameter is the fp32 encoder's
        audio_embeds = audio_embeds.to(self.target_dtype)
        z_target = self.projection(audio_embeds[:B])
        z_mix = self.projection(audio_embeds[B:2 * B])
        z_ref = self.projection(audio_embeds[2 * B:])
        return z_target, z_mix, z_ref

    def embed_tokens(self, token_ids):
        return self.lm.get_input_embeddings()(token_ids)

    def prefixed_audio_blocks(self, z_target, z_mix, z_ref):
        B = z_target.shape[0]
        target_token_id = torch.tensor([self.target_token_id], device=self.device)
        mix_token_id = torch.tensor([self.mix_token_id], device=self.device)
        ref_token_id = torch.tensor([self.ref_token_id], device=self.device)
        z_target = torch.cat([self.embed_tokens(target_token_id).repeat(B, 1, 1), z_target], dim=1)
        z_mix = torch.cat([self.embed_tokens(mix_token_id).repeat(B, 1, 1), z_mix], dim=1)
        z_ref = torch.cat([self.embed_tokens(ref_token_id).repeat(B, 1, 1), z_ref], dim=1)
        return z_target, z_mix, z_ref

    # ------------------------------------------------------------------ splice
    def training_multimodal_embedding(self, input_ids, output_ids, input_attention_mask,
                                      output_attention_mask, z_target, z_mix, z_ref):
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        inputs_embeds = self.embed_tokens(input_ids).to(self.device)
        output_embeds = self.embed_tokens(output_ids).to(self.device)
        blocks = self.prefixed_audio_blocks(z_target, z_mix, z_ref)
        block_masks = [torch.ones(z.shape[:2], device=self.device) for z in blocks]

        batch_embeds, batch_attention_mask, batch_labels = [], [], []
        for b, idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[b, :idx], *[z[b, :] for z in blocks],
                                inputs_embeds[b, idx + 1:], output_embeds[b, :]], dim=0)
            attention_mask = torch.cat([input_attention_mask[b, :idx], *[m[b, :] for m in block_masks],
                                        input_attention_mask[b, idx + 1:], output_attention_mask[b, :]], dim=0)
            labels = torch.full(embeds.shape[:1], fill_value=-100, device=self.device, dtype=torch.long)
            labels[-output_embeds.shape[1]:] = output_ids[b, :]
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        batch_embeds = torch.stack(batch_embeds)
        batch_attention_mask = torch.stack(batch_attention_mask)
        batch_labels = torch.stack(batch_labels)
        batch_labels[batch_labels == self.tokenizer.pad_token_id] = -100
        L = self.max_length
        return batch_embeds[:, :L], batch_attention_mask[:, :L], batch_labels[:, :L]

    def inference_multimodal_embedding(self, input_ids, input_attention_mask, z_target, z_mix, z_ref,
                                       append_think=False):
        # The chat template (enable_thinking=True) already ends with "<think>\n", so
        # append_think stays False for stage-2 prompts.
        audio_token_id = torch.tensor([self.audio_token_id], device=self.device)
        audio_insert_index = torch.where(input_ids == audio_token_id)[1].tolist()
        inputs_embeds = self.embed_tokens(input_ids).to(self.device)
        if append_think:
            think_token_id = torch.tensor([self.think_token_id], device=self.device)
            think_emb = self.embed_tokens(think_token_id).repeat(z_target.shape[0], 1, 1)
        else:
            think_emb = inputs_embeds[:, :0, :]
        blocks = self.prefixed_audio_blocks(z_target, z_mix, z_ref)
        block_masks = [torch.ones(z.shape[:2], device=self.device) for z in blocks]
        think_mask = torch.ones(think_emb.shape[:2], device=self.device)
        batch_embeds, batch_attention_mask = [], []
        for b, idx in enumerate(audio_insert_index):
            embeds = torch.cat([inputs_embeds[b, :idx], *[z[b, :] for z in blocks],
                                inputs_embeds[b, idx + 1:], think_emb[b, :]], dim=0)
            attention_mask = torch.cat([input_attention_mask[b, :idx], *[m[b, :] for m in block_masks],
                                        input_attention_mask[b, idx + 1:], think_mask[b, :]], dim=0)
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
        return torch.stack(batch_embeds), torch.stack(batch_attention_mask)

    # ------------------------------------------------------------------ steps
    def tokenize_pair(self, input_text, output_text):
        """Left-padded prompt at input_length, right-padded answer at output_length."""
        input_tokens = self.tokenizer(
            list(input_text), return_tensors="pt", max_length=self.model_config.input_length,
            padding="max_length", truncation=True, padding_side="left")
        output_tokens = self.tokenizer(
            list(output_text), return_tensors="pt", max_length=self.model_config.output_length,
            padding="max_length", truncation=True, padding_side="right")
        return (input_tokens.input_ids.to(self.device), input_tokens.attention_mask.to(self.device),
                output_tokens.input_ids.to(self.device), output_tokens.attention_mask.to(self.device))

    def forward_pass(self, batch):
        input_ids, input_mask, output_ids, output_mask = self.tokenize_pair(
            batch["input_text"], batch["output_text"])
        z_target, z_mix, z_ref = self.audio_embeddings_for(batch)
        inputs_embeds, attention_mask, labels = self.training_multimodal_embedding(
            input_ids, output_ids, input_mask, output_mask, z_target, z_mix, z_ref)
        outputs = self.lm.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                  return_dict=True)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        return self.loss_fn(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def generation_step(self, batch, do_sample=False, temperature=0.7, top_p=0.95,
                        max_new_tokens=None, skip_special_tokens=False, append_think=False):
        input_tokens = self.tokenizer(batch["input_text"], return_tensors="pt", padding=True,
                                      truncation=True).to(self.device)
        z_target, z_mix, z_ref = self.audio_embeddings_for(batch)
        inputs_embeds, attention_mask = self.inference_multimodal_embedding(
            input_tokens.input_ids, input_tokens.attention_mask, z_target, z_mix, z_ref,
            append_think=append_think)
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_length,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(END_OF_MSG),
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        outputs = self.lm.generate(**gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log("val_loss", loss, sync_dist=True)
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.log("mean_val_loss", val_loss, sync_dist=True)
        self.validation_step_outputs = []
        if not self.model_config.apply_lora and self.trainer.is_global_zero:
            weights = {k: v.detach() for k, v in self.projection.state_dict().items()}
            torch.save(weights, os.path.join(self.expdir, f"{self.projector_type}_weights.pth"))
            torch.save(weights, os.path.join(
                self.expdir, f"{self.projector_type}_weights_step{self.global_step}.pth"))

    def configure_optimizers(self):
        optimizer = instantiate_from_config(
            self.optimizer_config, params=filter(lambda p: p.requires_grad, self.parameters()))
        scheduler = instantiate_from_config(self.scheduler_config, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
