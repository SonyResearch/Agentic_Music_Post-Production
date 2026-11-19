"""Path conventions for the critic code.

Every asset is resolved from environment variables that default to folders inside
the repository:

    MIXAGENT_MODELS   <repo>/models        Qwen3.5-4B, stable-audio-open-1.0, rlat/
    MIXAGENT_WEIGHTS  <repo>/weights       critic/<id>.pt
    MIXAGENT_DATA     <repo>/data          Stage-1 retrieval database (not shipped)
    CRITIC_DATA       <repo>/critic_data   your critic training data (see critic/examples/)
    CRITIC_RUNS       <repo>/runs/critic   training logs and checkpoints

The YAML configs refer to them as ``${oc.env:CRITIC_DATA}`` etc.; ``load_config``
fills in the defaults before resolving.

Audio paths inside the JSONL files are relative to an ``audio_root``; the
precomputed RLAT embedding of clip ``<rel>`` at crop index k lives at
``<embcache_root>/<rel>.o{k}.npy``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS = {
    "MIXAGENT_MODELS": REPO_ROOT / "models",
    "MIXAGENT_WEIGHTS": REPO_ROOT / "weights",
    "MIXAGENT_DATA": REPO_ROOT / "data",
    "CRITIC_DATA": REPO_ROOT / "critic_data",
    "CRITIC_RUNS": REPO_ROOT / "runs" / "critic",
}


def setup_env() -> None:
    """Set the path environment variables that are not already set."""
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, str(value))


def env_path(key: str) -> Path:
    setup_env()
    return Path(os.environ[key])


def load_config(path):
    """OmegaConf.load + env defaults + resolve all ${oc.env:...} interpolations."""
    from omegaconf import OmegaConf

    setup_env()
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    return config


def resolve_audio(path: str, audio_root) -> str:
    """Absolute path of a JSONL audio entry (relative entries hang off audio_root)."""
    if os.path.isabs(path) or audio_root is None:
        return path
    return os.path.join(str(audio_root), path)


def relative_audio(path: str, audio_root) -> str:
    """The audio entry relative to audio_root (relative entries are returned as is)."""
    if not os.path.isabs(path):
        return path
    if audio_root is None:
        raise ValueError(f"absolute audio path {path} needs an audio_root to map it "
                         "into the embedding cache")
    return os.path.relpath(path, str(audio_root))


def embcache_path(audio_path: str, audio_root, embcache_root, k: int) -> str:
    """Cache file of one clip at crop index k (0/1/2 = offsets 0/2/4 s)."""
    return os.path.join(str(embcache_root), relative_audio(audio_path, audio_root)) + f".o{k}.npy"
