"""Anomaly FX detector -- inference package.

Detects which stems of a mix have had their fx-chain parameters corrupted, and
which fx modules within those stems.
"""

from .detector import AnomalyFxDetector, Unit, load_audio
from .embed import EMBED_DIM, Embedder
from .model import MAX_CHAIN_LEN, TOOL_TYPES

__all__ = [
    "AnomalyFxDetector",
    "Unit",
    "load_audio",
    "Embedder",
    "EMBED_DIM",
    "TOOL_TYPES",
    "MAX_CHAIN_LEN",
]
__version__ = "1.0.0"
