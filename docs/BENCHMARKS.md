# Benchmark Notes

Run:

```bash
catalyst-attention-benchmark --mode quick --out .
```

Outputs:

- `results/attention_benchmark.csv`
- `results/latest.json`
- `results/README.md`
- `charts/attention_latency.svg`
- `charts/attention_accuracy.svg`
- `charts/attention_speedup.svg`

The benchmark uses deterministic Catalyst label encodings for keys and values,
then routes noisy queries to the matching value. The baseline is a pure-Python
scaled dot-product softmax reference. This is an adapter-level benchmark, not a
claim about physical quantum hardware.
