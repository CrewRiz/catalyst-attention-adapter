# Catalyst Attention Adapter Design

This adapter is a source-available softmax-attention replacement powered by the
closed-source `catalyst-brain` SDK.

## What It Replaces

The adapter targets scaled dot-product softmax attention in retrieval/routing
workloads:

```python
output = softmax(query @ keys.T) @ values
```

The drop-in replacement is:

```python
from catalyst_attention_adapter import CatalystSoftmaxAttention

attention = CatalystSoftmaxAttention(dim=1024)
output = attention(query, keys, values)
```

`forward_with_metadata` adds the selected value index and confidence for
benchmarks and routing diagnostics.

## Boundary

This project uses the public `catalyst-brain` wheel and its packaged
`PyQuantumAttentionHead`. It does not disclose Catalyst Brain internals and
does not claim physical quantum execution. The correct public position is
quantum-inspired classical SDK behavior.

## Benchmarks

The included benchmark compares:

- Catalyst quantum attention head
- pure-Python scaled dot-product softmax reference

Metrics:

- top-1 routing accuracy
- median latency
- p95 latency
- speedup versus the softmax reference

## Commercial Use

Research and evaluation are allowed. Production model serving, hosted
inference, enterprise deployments, customer pilots, and revenue workflows
require a license through `hello@strategic-innovations.ai`.
