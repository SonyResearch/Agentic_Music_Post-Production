"""Small training utilities: config instantiation, LR schedule, activation checkpointing."""
import functools
import importlib

import numpy as np
import torch


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup to the base LR, then cosine decay; `min_lr` is a fraction of it."""

    def __init__(self, optimizer, warmup_steps=100, total_steps=10000, min_lr=1e-8,
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = self.min_lr + (1 - self.min_lr) * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + torch.cos(torch.tensor(np.pi * progress)))
            scale = self.min_lr + (1 - self.min_lr) * scale
        return [scale * base_lr for base_lr in self.base_lrs]


def set_activation_checkpointing(module, layer_type):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

    wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda submodule: isinstance(submodule, layer_type),
    )
