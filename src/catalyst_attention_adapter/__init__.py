"""Softmax-attention replacement powered by catalyst-brain quantum attention heads."""

from catalyst_attention_adapter.core import (
    CatalystAttentionOutput,
    CatalystQuantumAttention,
    CatalystSoftmaxAttention,
    encode_label,
    scaled_dot_softmax_attention,
)
from catalyst_attention_adapter.license import COMMERCIAL_CONTACT, LicenseError, assert_research_use

__version__ = "0.1.0"

__all__ = [
    "COMMERCIAL_CONTACT",
    "CatalystAttentionOutput",
    "CatalystQuantumAttention",
    "CatalystSoftmaxAttention",
    "LicenseError",
    "assert_research_use",
    "encode_label",
    "scaled_dot_softmax_attention",
    "__version__",
]
