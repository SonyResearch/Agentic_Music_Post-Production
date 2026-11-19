"""Asset locations, resolved from environment variables with repo-relative defaults.

* ``MIXAGENT_MODELS``  (default ``<repo>/models``)
* ``MIXAGENT_WEIGHTS`` (default ``<repo>/weights``)
* ``MIXAGENT_DATA``    (default ``<repo>/data``)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


MODELS = _env_path("MIXAGENT_MODELS", REPO_ROOT / "models")
WEIGHTS = _env_path("MIXAGENT_WEIGHTS", REPO_ROOT / "weights")
DATA = _env_path("MIXAGENT_DATA", REPO_ROOT / "data")

PLANNER_MODEL_DIR = MODELS / "Qwen3-Omni-30B-A3B-Thinking"
PLANNER_SERVED_NAME = "Qwen3-Omni-30B-A3B-Thinking"
CAPTIONS_DIRNAME = "audex_captions_instr"

DEFAULT_PLANNER_URL = "http://127.0.0.1:8000/v1"
DEFAULT_DETECTOR_URL = "http://127.0.0.1:8200"
DEFAULT_CRITIC_URL = "http://127.0.0.1:8201"


def dataset_data(dataset: str) -> Path:
    """``<MIXAGENT_DATA>/<dataset>`` (shipped text assets for medleydb / albumdb)."""
    return DATA / dataset


def captions_dir(dataset: str, dataset_root: Path) -> Path:
    """Per-stem caption dir: the shipped copy, else ``<dataset_root>/Input/<captions>``."""
    shipped = dataset_data(dataset) / CAPTIONS_DIRNAME
    return shipped if shipped.is_dir() else Path(dataset_root) / "Input" / CAPTIONS_DIRNAME


def caption_file(captions_dir: Path, dataset_root: Path, sid: str) -> Path:
    """Caption JSON of one song: ``<dataset_root>/Input/<captions>/<sid>/`` when present
    (your own songs, written by caption_stems.py), else the one in ``captions_dir``
    (the shipped captions of the paper's clips and of the retrieval bank)."""
    own = Path(dataset_root) / "Input" / CAPTIONS_DIRNAME / sid / f"{sid}_captions.json"
    return own if own.is_file() else Path(captions_dir) / sid / f"{sid}_captions.json"


def read_song_list(spec: str | None) -> list[str] | None:
    """``--songs`` value: a comma list, a split name under data/splits, or a file."""
    if not spec:
        return None
    p = Path(spec)
    if not p.is_file():
        cand = DATA / "splits" / f"{spec}.txt"
        p = cand if cand.is_file() else p
    if p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    return [s.strip() for s in spec.split(",") if s.strip()]
