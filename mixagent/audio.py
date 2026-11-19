"""Listening-clip helpers: loudness normalization, centre crop, FLAC writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .render import to_stereo

TARGET_LUFS = -12.0       # mix clips and the saved final mix
STEM_CLIP_LUFS = -16.0    # solo element clips
SAVE_PEAK_CEILING = 0.999


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data, sr


def integrated_lufs(audio: np.ndarray, sr: int) -> float | None:
    try:
        val = float(pyln.Meter(sr).integrated_loudness(audio))
    except Exception:  # noqa: BLE001
        return None
    return val if np.isfinite(val) else None


def normalize_lufs(audio: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    """Gain to ``target`` LUFS, then clamp to the peak ceiling."""
    cur = integrated_lufs(audio, sr)
    if cur is None:
        return audio
    gained = audio * (10.0 ** ((target - cur) / 20.0))
    return np.clip(gained, -SAVE_PEAK_CEILING, SAVE_PEAK_CEILING).astype(np.float32)


def center_crop(audio: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(round(seconds * sr))
    if audio.shape[0] <= n:
        return audio
    start = (audio.shape[0] - n) // 2
    return audio[start:start + n]


def write_flac(audio: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_24")


def save_listen_clip_gain(audio: np.ndarray, sr: int, out: Path, seconds: float,
                          target_lufs: float = TARGET_LUFS) -> tuple[Path, float]:
    """Centre crop + whole-clip loudness normalization + save; returns the gain used.

    The gain matters to the trained detector: its wet clips are scaled by the MIX
    clip's gain so that they sum to the mix clip it is shown.
    """
    cropped = center_crop(to_stereo(audio), sr, seconds)
    cur = integrated_lufs(cropped, sr)
    if cur is None:
        clip, gain = cropped, 1.0
    else:
        gain = 10.0 ** ((target_lufs - cur) / 20.0)
        clip = np.clip(cropped * gain, -SAVE_PEAK_CEILING, SAVE_PEAK_CEILING).astype(np.float32)
    write_flac(clip, sr, out)
    return out, gain


def save_listen_clip(audio: np.ndarray, sr: int, out: Path, seconds: float,
                     target_lufs: float = TARGET_LUFS) -> Path:
    return save_listen_clip_gain(audio, sr, out, seconds, target_lufs)[0]


def make_file_clip(src: Path, out: Path, seconds: float,
                   target_lufs: float = TARGET_LUFS) -> Path:
    audio, sr = load_audio(src)
    return save_listen_clip(audio, sr, out, seconds, target_lufs)
