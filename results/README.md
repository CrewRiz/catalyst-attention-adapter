# Catalyst Attention Adapter Results

| Metric | Value |
| --- | ---: |
| Largest key count | 64 |
| Largest-key top-1 accuracy | 100.00% |
| Largest-key median latency | 4703.6250 us |
| Largest-key speedup vs softmax reference | 1.62x |

This benchmark compares the public Catalyst quantum attention head against a pure-Python scaled dot-product softmax reference. It does not claim physical quantum execution.

| Dimension | Keys | Noise | Method | Accuracy | Median us | P95 us | Speedup vs softmax |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 256 | 4 | 0.00% | Catalyst quantum attention | 100.00% | 102.7295 | 105.8750 | 1.40x |
| 256 | 4 | 0.00% | Scaled dot-product softmax | 100.00% | 143.3750 | 145.3750 | 1.00x |
| 512 | 16 | 5.00% | Catalyst quantum attention | 100.00% | 603.0205 | 629.7910 | 1.60x |
| 512 | 16 | 5.00% | Scaled dot-product softmax | 100.00% | 966.8545 | 987.0420 | 1.00x |
| 1024 | 64 | 10.00% | Catalyst quantum attention | 100.00% | 4703.6250 | 4807.2910 | 1.62x |
| 1024 | 64 | 10.00% | Scaled dot-product softmax | 100.00% | 7604.7290 | 7842.0830 | 1.00x |
