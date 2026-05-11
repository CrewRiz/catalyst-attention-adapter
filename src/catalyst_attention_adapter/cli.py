from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalyst_attention_adapter import CatalystQuantumAttention, encode_label
from catalyst_attention_adapter.benchmarks import run_benchmark, write_results
from catalyst_attention_adapter.charts import write_charts


def smoke_main() -> int:
    dim = 256
    adapter = CatalystQuantumAttention(dim=dim, nqubits=4)
    labels = [f"route-{index}" for index in range(8)]
    keys = [encode_label(label, dim) for label in labels]
    values = [encode_label(f"value-{label}", dim) for label in labels]
    target = 5
    out = adapter.forward_with_metadata(keys[target], keys, values)
    print(
        json.dumps(
            {
                "capabilities": adapter.capabilities(),
                "target": target,
                "selected_index": out.selected_index,
                "confidence": out.confidence,
                "method": out.method,
                "output_dim": len(out.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if out.selected_index == target else 1


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Catalyst attention adapter benchmarks.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--out", default=".", help="Output root containing results/ and charts/.")
    args = parser.parse_args(argv)
    root = Path(args.out)
    results = run_benchmark(mode=args.mode)
    write_results(results, out_dir=root / "results")
    write_charts(results, out_dir=root / "charts")
    print(json.dumps(results["summary"], indent=2, sort_keys=True))
    return 0
