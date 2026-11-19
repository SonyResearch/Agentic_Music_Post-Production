"""Lightning data modules for SFT (CriticismDataModule) and DPO (DPODataModule)."""
import os
import random
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from critic.data.criticism_dataset import CriticismDataset
from critic.data.dpo_dataset import DPOPairDataset


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CriticismDataModule(pl.LightningDataModule):
    def __init__(self, db_path_train: str, db_path_val: str, model_path: str,
                 batch_size: int = 8, num_workers: int = 8, use_chat: bool = True,
                 cache_dir: Optional[str] = None, audio_root: Optional[str] = None,
                 embcache_root: Optional[str] = None):
        super().__init__()
        self.db_path_train = db_path_train
        self.db_path_val = db_path_val
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_chat = use_chat
        self.cache_dir = cache_dir
        self.audio_root = audio_root
        self.embcache_root = embcache_root

    def prepare_data(self) -> None:
        for path in (self.db_path_train, self.db_path_val):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset JSONL not found: {path}")

    def _dataset(self, path, split):
        return CriticismDataset(db_path=path, model_path=self.model_path, cache_dir=self.cache_dir,
                                split=split, use_chat=self.use_chat, audio_root=self.audio_root,
                                embcache_root=self.embcache_root)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(self.db_path_train, "train")
            self.val_dataset = self._dataset(self.db_path_val, "val")
            print(f"train rows: {len(self.train_dataset)}, val rows: {len(self.val_dataset)}")
        elif stage == "test":
            self.test_dataset = self._dataset(self.db_path_val, "val")

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=shuffle,
                          persistent_workers=self.num_workers > 0,
                          worker_init_fn=_worker_init_fn if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)


class DPODataModule(pl.LightningDataModule):
    def __init__(self, db_path_train: str, db_path_val: str, model_path: str,
                 batch_size: int = 2, num_workers: int = 4, cache_dir: Optional[str] = None,
                 audio_root: Optional[str] = None, embcache_root: Optional[str] = None):
        super().__init__()
        self.db_path_train = db_path_train
        self.db_path_val = db_path_val
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.audio_root = audio_root
        self.embcache_root = embcache_root

    def prepare_data(self):
        for p in (self.db_path_train, self.db_path_val):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

    def _dataset(self, path):
        return DPOPairDataset(path, self.model_path, self.cache_dir,
                              audio_root=self.audio_root, embcache_root=self.embcache_root)

    def setup(self, stage=None):
        self.train_dataset = self._dataset(self.db_path_train)
        self.val_dataset = self._dataset(self.db_path_val)
        print(f"DPO pairs: train {len(self.train_dataset)}, val {len(self.val_dataset)}")

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=shuffle,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)
