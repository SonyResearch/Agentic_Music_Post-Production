"""Number Token Loss (NTL), vendored.

Adapted copy of the `ntloss` package (Zausinger et al., "Regress, Don't Guess",
ICML 2025) as vendored in the LLM2Fx-Tools code base, where the loss is restricted
to the first `activated_vocab` logits. Kept byte-identical to the copy used for
training. The critic's stage-2 loss is CE + 0.3 * NTLoss (see critic/model/trainer.py).
"""
from .core import NTLoss, NTLossDotProduct  # noqa: F401
