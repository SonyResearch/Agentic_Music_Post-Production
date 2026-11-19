"""SFT dataset: one JSONL row per (state, target element) critique.

Row schema:
    {example_id, kind, split, audio_target, audio_mix, audio_ref,
     instruction (user turn, ends with the line "<|vision_start|>"),
     cot, has_cot, criticism (compact JSON string)}

Audio paths are relative to `audio_root` (absolute paths also work). With the embedding
cache (`<embcache_root>/.complete` present) each clip is read as a precomputed RLAT
z_local (108, 1024) fp16 array at one of three 10-s crops (offsets 0/2/4 s): a random
crop for train (target and mix share one, the reference draws its own), the 2-s centre
crop for val. Without the cache, waveforms are loaded and cropped the same way and the
encoder runs on the fly.

Stage 1 (use_chat false): prompt = "<|vision_start|>", answer = criticism.
Stage 2 (use_chat true): prompt = chat template of the instruction (thinking on),
answer = cot + "</think>" + criticism; rows without a CoT are skipped.
"""
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from critic.model.model_setup import setup_tokenizer
from critic.paths import embcache_path, resolve_audio

SAMPLE_RATE = 44100
CLIP_SAMPLES = 441000  # 10 s @ 44.1 kHz
START_OF_AUDIO = "<|vision_start|>"
END_OF_THINK = "</think>"
END_OF_MSG = "<|im_end|>"


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chat_prompt(tokenizer, instruction):
    """The stage-2 prompt: Qwen chat template with thinking enabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )


class CriticismDataset(Dataset):
    N_CACHE_OFFSETS = 3  # .o0/.o1/.o2 = crop offsets 0 s / 2 s / 4 s
    VAL_CACHE_OFFSET = 1

    def __init__(self, db_path, model_path, cache_dir=None, split="train", use_chat=True,
                 emb_cache=True, audio_root=None, embcache_root=None):
        self.db_path = db_path
        self.split = split
        self.use_chat = use_chat
        self.audio_root = audio_root
        self.embcache_root = embcache_root
        self.emb_cache = bool(emb_cache and embcache_root)
        self._warned_cache_miss = False
        self.tokenizer = setup_tokenizer(model_path, cache_dir)
        self.metadata = []
        for row in read_jsonl(db_path):
            if self.use_chat and not row.get("has_cot", False):
                continue  # stage 2 trains on CoT + criticism
            self.metadata.append(row)
        # all-or-nothing: a partial cache would mix (108, 1024) embeddings and
        # (2, 441000) waveforms in one batch
        if self.emb_cache and not os.path.exists(os.path.join(embcache_root, ".complete")):
            self.emb_cache = False
            print(f"[CriticismDataset] {embcache_root}/.complete not found -- training on "
                  "waveforms with the on-the-fly encoder (slow path)")

    def __len__(self):
        return len(self.metadata)

    def build_texts(self, row):
        if self.use_chat:
            input_text = chat_prompt(self.tokenizer, row["instruction"])
            output_text = f"{row['cot']}\n{END_OF_THINK}\n{row['criticism']}\n{END_OF_MSG}"
        else:
            input_text = START_OF_AUDIO
            output_text = f"{row['criticism']}\n{END_OF_MSG}"
        return input_text, output_text

    # ---- waveform path
    def _load_audio(self, path):
        audio, _ = sf.read(resolve_audio(path, self.audio_root), dtype="float32", always_2d=True)
        audio = audio.T
        if audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        return audio

    def _pick_offset(self, n_samples):
        max_offset = max(0, n_samples - CLIP_SAMPLES)
        if self.split == "train":
            return random.randint(0, max_offset)
        return max_offset // 2

    def _crop(self, audio, offset):
        c, t = audio.shape
        if t < CLIP_SAMPLES:
            audio = np.concatenate([audio, np.zeros((c, CLIP_SAMPLES - t), dtype=audio.dtype)], axis=1)
            offset = 0
        return torch.from_numpy(np.ascontiguousarray(audio[:, offset:offset + CLIP_SAMPLES]))

    # ---- embedding-cache path
    def cache_path(self, audio_path, k):
        return embcache_path(audio_path, self.audio_root, self.embcache_root, k)

    def _pick_cache_offset(self):
        if self.split == "train":
            return random.randint(0, self.N_CACHE_OFFSETS - 1)
        return self.VAL_CACHE_OFFSET

    def _try_cached_item(self, row):
        k_tm = self._pick_cache_offset()   # target and mix share one offset (time-aligned)
        k_ref = self._pick_cache_offset()  # the reference offset is independent
        paths = [self.cache_path(row["audio_target"], k_tm),
                 self.cache_path(row["audio_mix"], k_tm),
                 self.cache_path(row["audio_ref"], k_ref)]
        if not all(os.path.exists(p) for p in paths):
            return None
        return [torch.from_numpy(np.load(p)) for p in paths]

    def __getitem__(self, idx):
        row = self.metadata[idx]
        if self.emb_cache:
            cached = self._try_cached_item(row)
            if cached is not None:
                input_text, output_text = self.build_texts(row)
                return {"input_text": input_text, "output_text": output_text,
                        "audio_target": cached[0], "audio_mix": cached[1], "audio_ref": cached[2]}
            if not self._warned_cache_miss:
                self._warned_cache_miss = True
                print(f"[CriticismDataset] emb cache miss for {row['example_id']} -- "
                      "falling back to waveform + on-the-fly encoder for missing rows")
        audio_target = self._load_audio(row["audio_target"])
        audio_mix = self._load_audio(row["audio_mix"])
        audio_ref = self._load_audio(row["audio_ref"])
        offset_tm = self._pick_offset(min(audio_target.shape[1], audio_mix.shape[1]))
        offset_ref = self._pick_offset(audio_ref.shape[1])
        input_text, output_text = self.build_texts(row)
        return {
            "input_text": input_text,
            "output_text": output_text,
            "audio_target": self._crop(audio_target, offset_tm),
            "audio_mix": self._crop(audio_mix, offset_tm),
            "audio_ref": self._crop(audio_ref, offset_ref),
        }
