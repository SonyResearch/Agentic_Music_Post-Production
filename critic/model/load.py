"""Load a trained critic for inference, and split its generations.

`load_model` builds LMTrainingWrapper from a config the way training does (cast to the
training dtype first), loads the trained tensors with strict=False and then proves the
load: every adapter (`projection.*`) and LoRA tensor in the file must land in a slot, no
slot may stay unfilled, and a probe adapter tensor must change. Accepts a Lightning
`.ckpt` or a released trainable-only `weights/critic/<id>.pt` ({"state_dict", "meta"}).
"""
import torch

STOP_MARKERS = ("<|im_end|>", "<|endoftext|>")
END_OF_THINK = "</think>"


def truncate_at_stop(text):
    """Everything the model emitted before the first end-of-message marker."""
    cut = len(text)
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def split_thinking(text, chat=True):
    """(cot, answer) of a raw generation.

    Chat (stage-2) outputs are "<CoT> </think> <criticism JSON> <|im_end|>": the answer
    is the span after the LAST </think>. answer == "" when the think block never closed.
    Stage-1 outputs have no think block (cot None, whole text is the answer).
    """
    body = truncate_at_stop(text)
    if not chat:
        return None, body
    idx = body.rfind(END_OF_THINK)
    if idx == -1:
        return body, ""
    return body[:idx], body[idx + len(END_OF_THINK):]


def _target_dtype(config):
    precision = config.lightning.trainer.precision
    if precision in ("16", "16-mixed"):
        return torch.float16
    if precision in ("bf16", "bf16-mixed"):
        return torch.bfloat16
    return torch.float32


def load_model(ckpt_path, config, device):
    """-> (model on `device` in eval mode, load-info dict)."""
    from critic.model.trainer import LMTrainingWrapper

    target_dtype = _target_dtype(config)
    model_params = config.model.params
    print(f"[load] building LMTrainingWrapper (dtype={target_dtype}) ...", flush=True)
    model = LMTrainingWrapper(
        model_config=model_params.model_config,
        optimizer_config=model_params.optimizer_config,
        scheduler_config=model_params.scheduler_config,
    )
    model.to(target_dtype)

    print(f"[load] reading {ckpt_path} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    meta = ckpt.get("meta", {})
    proj_keys = [k for k in state if k.startswith("projection.")]
    lora_keys = [k for k in state if "lora_" in k]

    own = model.state_dict()
    probe_key = proj_keys[0] if proj_keys else None
    probe_before = own[probe_key].detach().clone().float() if probe_key in own else None

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_set = set(unexpected)
    loaded_proj = [k for k in proj_keys if k not in unexpected_set]
    loaded_lora = [k for k in lora_keys if k not in unexpected_set]
    print(f"[load] {len(state)} tensors in file; projection {len(loaded_proj)}/{len(proj_keys)} "
          f"loaded, lora {len(loaded_lora)}/{len(lora_keys)} loaded")

    if len(proj_keys) == 0:
        raise RuntimeError("checkpoint contains no 'projection.*' keys -- wrong checkpoint?")
    if probe_before is not None:
        delta = (model.state_dict()[probe_key].detach().float() - probe_before).abs().max().item()
        print(f"[load] probe {probe_key}: max|after-before| = {delta:.6g}")
        if delta == 0:
            raise RuntimeError(f"projection weights did not change after load_state_dict: {probe_key}")
    if len(loaded_proj) != len(proj_keys):
        raise RuntimeError(f"only {len(loaded_proj)}/{len(proj_keys)} projection keys loaded")
    missing_proj = [k for k in missing if k.startswith("projection.")]
    if missing_proj:
        raise RuntimeError(f"projection keys missing from ckpt: {missing_proj[:5]}")
    if lora_keys and len(loaded_lora) != len(lora_keys):
        raise RuntimeError(f"only {len(loaded_lora)}/{len(lora_keys)} lora keys landed -- the "
                           "config's apply_lora/lora_target_modules do not match the checkpoint")
    if not lora_keys and model_params.model_config.get("apply_lora", False):
        raise RuntimeError("config says apply_lora=True but the ckpt has no lora_* tensors")
    missing_lora = [k for k in missing if "lora_" in k]
    if missing_lora:
        raise RuntimeError(f"model has lora slots the ckpt did not fill: {missing_lora[:5]}")

    model.to(device)
    model.eval()
    info = {
        "ckpt": str(ckpt_path),
        "ckpt_tensors": len(state),
        "projection_keys_in_ckpt": len(proj_keys),
        "projection_keys_loaded": len(loaded_proj),
        "lora_keys_in_ckpt": len(lora_keys),
        "lora_keys_loaded": len(loaded_lora),
        "unmatched_keys": len(proj_keys) - len(loaded_proj) + len(lora_keys) - len(loaded_lora),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "global_step": ckpt.get("global_step", meta.get("global_step")),
        "epoch": ckpt.get("epoch", meta.get("epoch")),
        "target_dtype": str(target_dtype),
        "meta": meta,
    }
    del ckpt, state
    return model, info
