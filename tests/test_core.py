from __future__ import annotations

import pytest


def test_quantum_attention_selects_matching_value():
    from catalyst_attention_adapter import CatalystQuantumAttention, encode_label

    dim = 256
    adapter = CatalystQuantumAttention(dim=dim, nqubits=4)
    keys = [encode_label(f"key-{index}", dim) for index in range(8)]
    values = [encode_label(f"value-{index}", dim) for index in range(8)]

    result = adapter.forward_with_metadata(keys[3], keys, values)

    assert result.selected_index == 3
    assert result.method == "Catalyst PyQuantumAttentionHead"
    assert len(result.output) == dim


def test_softmax_alias_is_drop_in_callable():
    from catalyst_attention_adapter import CatalystSoftmaxAttention, encode_label, scaled_dot_softmax_attention

    dim = 128
    adapter = CatalystSoftmaxAttention(dim=dim, nqubits=4)
    keys = [encode_label(f"route-{index}", dim) for index in range(4)]
    values = [encode_label(f"value-{index}", dim) for index in range(4)]

    catalyst = adapter(keys[1], keys, values)
    baseline = scaled_dot_softmax_attention(keys[1], keys, values)

    assert len(catalyst) == len(baseline) == dim
    assert adapter.forward_with_metadata(keys[1], keys, values).selected_index == 1


def test_label_routing_and_capabilities():
    from catalyst_attention_adapter import CatalystQuantumAttention

    adapter = CatalystQuantumAttention(dim=256, nqubits=4)
    result = adapter.route_labels("repo", ["billing", "repo", "tests"])

    assert result.selected_index == 1
    assert adapter.capabilities()["sdk_dependency"] == "catalyst-brain"
    assert adapter.capabilities()["softmax_replacement"] is True


def test_validation_and_license_boundary():
    from catalyst_attention_adapter import CatalystQuantumAttention, LicenseError

    adapter = CatalystQuantumAttention(dim=4, nqubits=2)
    with pytest.raises(ValueError):
        adapter.forward([1, 2], [[1, 2]], [[1, 2]])
    with pytest.raises(LicenseError):
        CatalystQuantumAttention(dim=4, purpose="enterprise production")


def test_quick_benchmark_shape():
    from catalyst_attention_adapter.benchmarks import run_benchmark

    results = run_benchmark(mode="quick")
    catalyst_rows = [row for row in results["attention"] if row["method"] == "Catalyst quantum attention"]

    assert results["summary"]["sdk_dependency"] == "catalyst-brain"
    assert len(catalyst_rows) == 3
    assert min(row["top1_accuracy_pct"] for row in catalyst_rows) >= 90.0
