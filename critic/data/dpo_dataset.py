"""DPO preference-pair dataset.

Row (see critic/examples/dpo_*.jsonl):
    {example_id, instruction, audio_target, audio_mix, audio_ref,
     chosen_text, rejected_text[, ref_lp_chosen, ref_lp_rejected]}

The reference log-probs are filled in by critic/train.py before DPO starts
(critic.model.dpo_trainer.fill_reference_logprobs). All three clips use the FIXED
centre crop (the reference log-probs are only valid for the same audio): the embedding
cache's .o1 entry when `<embcache_root>/.complete` exists, else the waveform cropped
the same way.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from critic.data.criticism_dataset import CriticismDataset, chat_prompt, read_jsonl
from critic.model.model_setup import setup_tokenizer
from critic.paths import embcache_path


class DPOPairDataset(Dataset):
    CACHE_OFFSET = 1  # 2-s centre crop, the crop the critiques were sampled on

    def __init__(self, db_path, model_path, cache_dir=None, require_ref_lp=True,
                 audio_root=None, embcache_root=None):
        self.tokenizer = setup_tokenizer(model_path, cache_dir)
        self.audio_root = audio_root
        self.embcache_root = embcache_root
        self.rows = read_jsonl(db_path)
        if require_ref_lp:
            for r in self.rows:
                if r.get("ref_lp_chosen") is None or r.get("ref_lp_rejected") is None:
                    raise ValueError(f"{db_path}: row {r['example_id']} has no reference "
                                     "log-probs (critic/train.py fills them in)")
        self.use_cache = bool(embcache_root) and os.path.exists(
            os.path.join(embcache_root, ".complete"))
        # waveform path: the SFT dataset's loader with its fixed (val) centre crop
        self._wave = CriticismDataset.__new__(CriticismDataset)
        self._wave.audio_root, self._wave.split = audio_root, "val"

    def __len__(self):
        return len(self.rows)

    def _cached(self, audio_path):
        path = embcache_path(audio_path, self.audio_root, self.embcache_root, self.CACHE_OFFSET)
        return torch.from_numpy(np.load(path))  # (108, 1024) fp16

    def _audio(self, r):
        if self.use_cache:
            return [self._cached(r[k]) for k in ("audio_target", "audio_mix", "audio_ref")]
        w = self._wave
        target, mix, ref = (w._load_audio(r[k]) for k in ("audio_target", "audio_mix", "audio_ref"))
        off_tm = w._pick_offset(min(target.shape[1], mix.shape[1]))
        off_ref = w._pick_offset(ref.shape[1])
        return [w._crop(target, off_tm), w._crop(mix, off_tm), w._crop(ref, off_ref)]

    def build_input_text(self, row):
        return chat_prompt(self.tokenizer, row["instruction"])

    def __getitem__(self, idx):
        r = self.rows[idx]
        audio_target, audio_mix, audio_ref = self._audio(r)
        return {
            "input_text": self.build_input_text(r),
            "chosen_text": r["chosen_text"],
            "rejected_text": r["rejected_text"],
            "audio_target": audio_target,
            "audio_mix": audio_mix,
            "audio_ref": audio_ref,
            "ref_lp_chosen": float(r.get("ref_lp_chosen") or 0.0),
            "ref_lp_rejected": float(r.get("ref_lp_rejected") or 0.0),
        }
