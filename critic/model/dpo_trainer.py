"""DPO on the critic.

Policy = frozen base LM + LoRA + adapter, warm-started from the SFT checkpoint
(`init_weights_from`). The reference model is that same frozen SFT checkpoint; its
per-sequence log-probs are constants, computed once before training
(`fill_reference_logprobs`, called by critic/train.py) and stored in every pair row, so
training holds one model:

    loss = -log sigmoid( beta * [ (lp_c - ref_c) - (lp_r - ref_r) ] )

Sequence log-probs reuse the SFT tokenisation / audio splice / masking / truncation, so
policy and reference log-probs cover the same token positions. Audio embeddings are
computed once per pair and shared by the chosen and rejected forwards.
"""
import json
import os

import torch
import torch.nn.functional as F

from critic.model.trainer import LMTrainingWrapper


class DPOTrainingWrapper(LMTrainingWrapper):
    def __init__(self, model_config=None, optimizer_config=None, scheduler_config=None):
        super().__init__(model_config=model_config, optimizer_config=optimizer_config,
                         scheduler_config=scheduler_config)
        self.dpo_beta = float(model_config.get("dpo_beta", 0.1))

    def _seq_logprob(self, input_text, output_text, z_target, z_mix, z_ref):
        """Summed log-prob of each answer given its prompt and audio, shape (B,)."""
        input_ids, input_mask, output_ids, output_mask = self.tokenize_pair(input_text, output_text)
        inputs_embeds, attention_mask, labels = self.training_multimodal_embedding(
            input_ids, output_ids, input_mask, output_mask, z_target, z_mix, z_ref)
        out = self.lm.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              return_dict=True)
        logits = out.logits[:, :-1, :]
        labels = labels[:, 1:]
        mask = labels != -100
        safe = labels.masked_fill(~mask, 0)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = torch.gather(logp, 2, safe.unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask).sum(dim=1)

    def pair_logprobs(self, batch):
        z_target, z_mix, z_ref = self.audio_embeddings_for(batch)
        lp_c = self._seq_logprob(batch["input_text"], batch["chosen_text"], z_target, z_mix, z_ref)
        lp_r = self._seq_logprob(batch["input_text"], batch["rejected_text"], z_target, z_mix, z_ref)
        return lp_c, lp_r

    def dpo_loss(self, batch):
        lp_c, lp_r = self.pair_logprobs(batch)
        ref_c = batch["ref_lp_chosen"].to(self.device).float()
        ref_r = batch["ref_lp_rejected"].to(self.device).float()
        margin = (lp_c - ref_c) - (lp_r - ref_r)
        loss = -F.logsigmoid(self.dpo_beta * margin).mean()
        stats = {
            "margin": margin.mean().detach(),
            "acc": (margin > 0).float().mean().detach(),
            "reward_chosen": (lp_c - ref_c).mean().detach(),
            "reward_rejected": (lp_r - ref_r).mean().detach(),
        }
        return loss, stats

    def training_step(self, batch, batch_idx):
        loss, s = self.dpo_loss(batch)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_margin", s["margin"], sync_dist=True)
        self.log("train_acc", s["acc"], sync_dist=True, prog_bar=True)
        self.log("reward_chosen", s["reward_chosen"], sync_dist=True)
        self.log("reward_rejected", s["reward_rejected"], sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, s = self.dpo_loss(batch)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", s["acc"], sync_dist=True)
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}


AUDIO_KEYS = ("audio_target", "audio_mix", "audio_ref")


@torch.no_grad()
def fill_reference_logprobs(model, data_params, batch_size=4, device="cuda"):
    """Add ref_lp_chosen / ref_lp_rejected to the train and val pair files, in place.

    ``model`` must still hold its initial weights (the SFT checkpoint = the reference).
    Files whose rows all have reference log-probs are left untouched.
    """
    from critic.data.dpo_dataset import DPOPairDataset

    todo = []
    for path in (data_params.db_path_train, data_params.db_path_val):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if any(r.get("ref_lp_chosen") is None or r.get("ref_lp_rejected") is None for r in rows):
            todo.append(path)
    if not todo:
        return
    was_training = model.training
    model.to(device).eval()
    for path in todo:
        ds = DPOPairDataset(path, data_params.model_path, data_params.get("cache_dir"),
                            require_ref_lp=False, audio_root=data_params.get("audio_root"),
                            embcache_root=data_params.get("embcache_root"))
        rows = ds.rows
        for i in range(0, len(rows), batch_size):
            items = [ds[j] for j in range(i, min(i + batch_size, len(rows)))]
            batch = {k: [it[k] for it in items] for k in ("input_text", "chosen_text", "rejected_text")}
            for k in AUDIO_KEYS:
                batch[k] = torch.stack([it[k] for it in items]).to(device)
            lp_c, lp_r = model.pair_logprobs(batch)
            for k, it in enumerate(range(i, i + len(items))):
                rows[it]["ref_lp_chosen"] = float(lp_c[k])
                rows[it]["ref_lp_rejected"] = float(lp_r[k])
            print(f"[reference log-probs] {path}: {i + len(items)}/{len(rows)}", flush=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    model.cpu().train(was_training)
    torch.cuda.empty_cache()
