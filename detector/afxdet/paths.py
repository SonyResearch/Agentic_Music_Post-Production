"""Where the detector finds its code and weights inside the MixAgent repo.

Assets resolve from environment variables, defaulting relative to the repo root:

* ``MIXAGENT_MODELS``  (default ``<repo>/models``)  -- downloaded third-party models (incl. RLAT)
* ``MIXAGENT_WEIGHTS`` (default ``<repo>/weights``) -- MixAgent's released weights
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_ROOT = REPO_ROOT / "detector"

MODELS = Path(os.environ.get("MIXAGENT_MODELS", REPO_ROOT / "models"))
WEIGHTS = Path(os.environ.get("MIXAGENT_WEIGHTS", REPO_ROOT / "weights"))

# Code on sys.path: vendored MusicFM (imported as ``musicfm``) and the shared
# RLAT module (imported as ``rlat_encoder``).
MUSICFM_CODE = DETECTOR_ROOT / "vendor"
RLAT_CODE = REPO_ROOT / "encoder"

DETECTOR_CKPT = WEIGHTS / "detector" / "detector.pt"
RLAT_DIR = MODELS / "rlat"                          # config.json + model.safetensors
SAO_DIR = MODELS / "stable-audio-open-1.0"          # only vae/ is read
MUSICFM_CKPT = MODELS / "musicfm" / "pretrained_msd.pt"
MUSICFM_STATS = MODELS / "musicfm" / "msd_stats.json"
