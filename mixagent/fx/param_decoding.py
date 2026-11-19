from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def clone_to_cpu(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {k: clone_to_cpu(v) for k, v in value.items()}
    return value


def tensor_dict(raw_params: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in raw_params.items():
        if isinstance(value, torch.Tensor):
            out[key] = value
    return out


def unwrap_processor(module: nn.Module) -> nn.Module:
    current = module
    seen = set()
    while True:
        if id(current) in seen:
            return current
        seen.add(id(current))
        child = getattr(current, "processor", None)
        if not isinstance(child, nn.Module):
            return current
        current = child


def default_metadata(raw_params: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        key: {
            "display_name": key,
            "unit": "unitless",
            "kind": "parameter",
        }
        for key in raw_params
    }


class DecodedParamsMixin:
    def decode_params(self, raw_params: Mapping[str, Any]) -> dict[str, Any]:
        return tensor_dict(raw_params)

    def decode_param_metadata(self, raw_params: Mapping[str, Any] | None = None) -> dict[str, dict[str, str]]:
        return default_metadata(raw_params or {})


def decode_processor_group(
    processor: nn.Module,
    raw_params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    unwrapped = unwrap_processor(processor)
    decoder = getattr(unwrapped, "decode_params", None)
    metadata_fn = getattr(unwrapped, "decode_param_metadata", None)

    raw_params = tensor_dict(raw_params)
    if callable(decoder):
        decoded = decoder(raw_params)
    else:
        decoded = raw_params

    if callable(metadata_fn):
        metadata = metadata_fn(raw_params)
    else:
        metadata = default_metadata(raw_params)

    return clone_to_cpu(decoded), metadata


def decode_all_processor_groups(
    audio_processors: Mapping[str, nn.Module],
    raw_parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decoded: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for processor_type, raw_params in raw_parameters.items():
        if processor_type not in audio_processors:
            continue
        if not isinstance(raw_params, Mapping):
            continue
        decoded_group, metadata_group = decode_processor_group(
            processor=audio_processors[processor_type],
            raw_params=raw_params,
        )
        decoded[processor_type] = decoded_group
        metadata[processor_type] = metadata_group
    return decoded, metadata
