from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import catalyst_brain  # noqa: F401 - importing the public SDK initializes the packaged native module.
import catalyst_hdc as hdc

from catalyst_attention_adapter.license import COMMERCIAL_CONTACT, assert_research_use


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _as_float_vector(value: Vector, *, name: str) -> list[float]:
    out = [float(item) for item in value]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def _as_float_matrix(value: Matrix, *, name: str, dim: int | None = None) -> list[list[float]]:
    out = [_as_float_vector(row, name=f"{name}[{index}]") for index, row in enumerate(value)]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    width = dim if dim is not None else len(out[0])
    for index, row in enumerate(out):
        if len(row) != width:
            raise ValueError(f"{name}[{index}] has dim {len(row)}, expected {width}")
    return out


def _dot(left: Vector, right: Vector) -> float:
    return sum(x * y for x, y in zip(left, right)) / max(1, len(left))


def _argmax_similarity(output: Vector, values: Matrix) -> tuple[int, float]:
    scores = [_dot(output, value) for value in values]
    best = max(range(len(scores)), key=lambda index: scores[index])
    return best, scores[best]


def encode_label(label: str, dim: int) -> list[float]:
    """Encode a symbolic token/label with the public Catalyst Brain SDK wheel."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    return hdc.hv_hash_string(str(label), dim)


def scaled_dot_softmax_attention(
    query: Vector,
    keys: Matrix,
    values: Matrix,
    *,
    temperature: float = 1.0,
) -> list[float]:
    """Small pure-Python reference baseline for benchmark comparisons."""
    q = _as_float_vector(query, name="query")
    k = _as_float_matrix(keys, name="keys", dim=len(q))
    v = _as_float_matrix(values, name="values")
    if len(k) != len(v):
        raise ValueError("keys and values must have the same length")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scale = math.sqrt(len(q)) * temperature
    logits = [sum(q_i * k_i for q_i, k_i in zip(q, key)) / scale for key in k]
    offset = max(logits)
    weights = [math.exp(item - offset) for item in logits]
    total = sum(weights)
    if total <= 0.0:
        return [0.0] * len(v[0])
    out = [0.0] * len(v[0])
    for weight, value in zip(weights, v):
        alpha = weight / total
        for index, item in enumerate(value):
            out[index] += alpha * item
    return out


@dataclass(frozen=True)
class CatalystAttentionOutput:
    output: list[float]
    selected_index: int
    confidence: float
    method: str
    nqubits: int


class CatalystQuantumAttention:
    """Drop-in attention primitive backed by `PyQuantumAttentionHead`.

    It replaces scaled dot-product softmax for retrieval/routing workloads:
    pass a query vector, key vectors, and value vectors; receive the attended
    value vector. Use `forward_with_metadata` when you need routing diagnostics.
    """

    def __init__(self, *, dim: int, nqubits: int | None = None, purpose: str = "research") -> None:
        assert_research_use(purpose)
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.nqubits = nqubits or 8
        self._head = hdc.PyQuantumAttentionHead(dim, self.nqubits)

    def __call__(self, query: Vector, keys: Matrix, values: Matrix) -> list[float]:
        return self.forward(query, keys, values)

    def forward(self, query: Vector, keys: Matrix, values: Matrix) -> list[float]:
        q, k, v = self._validate(query, keys, values)
        return self._head.compute(q, k, v)

    def forward_with_metadata(self, query: Vector, keys: Matrix, values: Matrix) -> CatalystAttentionOutput:
        q, k, v = self._validate(query, keys, values)
        output = self._head.compute(q, k, v)
        selected_index, confidence = _argmax_similarity(output, v)
        return CatalystAttentionOutput(
            output=output,
            selected_index=selected_index,
            confidence=round(confidence, 6),
            method="Catalyst PyQuantumAttentionHead",
            nqubits=self.nqubits,
        )

    def route_labels(
        self,
        query_label: str,
        key_labels: Sequence[str],
        value_labels: Sequence[str] | None = None,
    ) -> CatalystAttentionOutput:
        values_source = value_labels or key_labels
        keys = [encode_label(label, self.dim) for label in key_labels]
        values = [encode_label(label, self.dim) for label in values_source]
        query = encode_label(query_label, self.dim)
        return self.forward_with_metadata(query, keys, values)

    def capabilities(self) -> dict[str, object]:
        return {
            "sdk_dependency": "catalyst-brain",
            "attention_head": "PyQuantumAttentionHead",
            "softmax_replacement": True,
            "drop_in_forward": True,
            "commercial_contact": COMMERCIAL_CONTACT,
            "claim_boundary": "quantum-inspired classical SDK behavior; no physical quantum execution claim",
        }

    def _validate(self, query: Vector, keys: Matrix, values: Matrix) -> tuple[list[float], list[list[float]], list[list[float]]]:
        q = _as_float_vector(query, name="query")
        if len(q) != self.dim:
            raise ValueError(f"query has dim {len(q)}, expected {self.dim}")
        k = _as_float_matrix(keys, name="keys", dim=self.dim)
        v = _as_float_matrix(values, name="values", dim=self.dim)
        if len(k) != len(v):
            raise ValueError("keys and values must have the same length")
        return q, k, v


class CatalystSoftmaxAttention(CatalystQuantumAttention):
    """Alias with migration-friendly naming for replacing softmax attention."""


def batch_route_labels(
    adapter: CatalystQuantumAttention,
    queries: Iterable[str],
    key_labels: Sequence[str],
    value_labels: Sequence[str] | None = None,
) -> list[CatalystAttentionOutput]:
    return [adapter.route_labels(query, key_labels, value_labels) for query in queries]
