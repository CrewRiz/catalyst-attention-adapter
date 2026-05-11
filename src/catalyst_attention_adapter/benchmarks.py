from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from catalyst_attention_adapter.core import (
    CatalystQuantumAttention,
    encode_label,
    scaled_dot_softmax_attention,
)


def _measure_us(fn: Callable[[], list[float]], *, repeats: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(3):
        fn()
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "median_us": round(statistics.median(ordered), 4),
        "p95_us": round(ordered[p95_index], 4),
    }


def _dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right)) / max(1, len(left))


def _predict(output: list[float], values: list[list[float]]) -> tuple[int, float]:
    scores = [_dot(output, value) for value in values]
    best = max(range(len(scores)), key=lambda index: scores[index])
    return best, scores[best]


def _flip_deterministic(vector: list[float], *, noise_pct: float, seed: int) -> list[float]:
    if noise_pct <= 0.0:
        return list(vector)
    period = 10_000
    threshold = int(noise_pct * period)
    return [
        -value if ((index * 1_103 + seed * 9_176 + 41) % period) < threshold else value
        for index, value in enumerate(vector)
    ]


def run_benchmark(*, mode: str = "quick") -> dict[str, Any]:
    configs = (
        [(256, 4, 0.00), (512, 16, 0.05), (1024, 64, 0.10)]
        if mode == "quick"
        else [(256, 4, 0.00), (512, 16, 0.05), (1024, 64, 0.10), (2048, 128, 0.15)]
    )
    repeats = 50 if mode == "quick" else 120
    trials = 12 if mode == "quick" else 32
    rows: list[dict[str, Any]] = []

    for dim, key_count, noise_pct in configs:
        keys = [encode_label(f"attention-key-{dim}-{key_count}-{index}", dim) for index in range(key_count)]
        values = [encode_label(f"attention-value-{dim}-{key_count}-{index}", dim) for index in range(key_count)]
        nqubits = max(4, min(40, (key_count - 1).bit_length()))
        adapter = CatalystQuantumAttention(dim=dim, nqubits=nqubits)
        latency_target = key_count // 2
        latency_query = _flip_deterministic(keys[latency_target], noise_pct=noise_pct, seed=latency_target)

        catalyst_latency = _measure_us(lambda: adapter(latency_query, keys, values), repeats=repeats)
        softmax_latency = _measure_us(
            lambda: scaled_dot_softmax_attention(latency_query, keys, values),
            repeats=repeats,
        )

        catalyst_correct = 0
        softmax_correct = 0
        catalyst_confidences: list[float] = []
        softmax_confidences: list[float] = []

        for trial in range(trials):
            target = (trial * 7 + key_count // 3) % key_count
            query = _flip_deterministic(keys[target], noise_pct=noise_pct, seed=trial + dim)
            catalyst = adapter.forward_with_metadata(query, keys, values)
            softmax_output = scaled_dot_softmax_attention(query, keys, values)
            softmax_pred, softmax_conf = _predict(softmax_output, values)
            catalyst_correct += int(catalyst.selected_index == target)
            softmax_correct += int(softmax_pred == target)
            catalyst_confidences.append(catalyst.confidence)
            softmax_confidences.append(softmax_conf)

        rows.append(
            {
                "dimension": dim,
                "key_count": key_count,
                "noise_pct": round(noise_pct * 100.0, 2),
                "method": "Catalyst quantum attention",
                "nqubits": nqubits,
                "trials": trials,
                "top1_accuracy_pct": round(100.0 * catalyst_correct / trials, 4),
                "mean_target_confidence": round(statistics.fmean(catalyst_confidences), 6),
                "median_us": catalyst_latency["median_us"],
                "p95_us": catalyst_latency["p95_us"],
                "latency_vs_softmax_x": round(
                    softmax_latency["median_us"] / catalyst_latency["median_us"],
                    4,
                )
                if catalyst_latency["median_us"] > 0
                else 0.0,
                "baseline": "pure Python scaled dot-product softmax",
            }
        )
        rows.append(
            {
                "dimension": dim,
                "key_count": key_count,
                "noise_pct": round(noise_pct * 100.0, 2),
                "method": "Scaled dot-product softmax",
                "nqubits": 0,
                "trials": trials,
                "top1_accuracy_pct": round(100.0 * softmax_correct / trials, 4),
                "mean_target_confidence": round(statistics.fmean(softmax_confidences), 6),
                "median_us": softmax_latency["median_us"],
                "p95_us": softmax_latency["p95_us"],
                "latency_vs_softmax_x": 1.0,
                "baseline": "reference",
            }
        )

    catalyst_rows = [row for row in rows if row["method"] == "Catalyst quantum attention"]
    largest = max(catalyst_rows, key=lambda row: row["key_count"])
    summary = {
        "mode": mode,
        "adapter": "catalyst-attention-adapter",
        "sdk_dependency": "catalyst-brain",
        "claim_boundary": "quantum-inspired classical SDK behavior; no physical quantum execution claim",
        "largest_key_count": largest["key_count"],
        "largest_accuracy_pct": largest["top1_accuracy_pct"],
        "largest_median_us": largest["median_us"],
        "largest_speedup_vs_softmax_x": largest["latency_vs_softmax_x"],
    }
    return {"summary": summary, "attention": rows}


def write_results(results: dict[str, Any], *, out_dir: str | Path = "results") -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    rows = results["attention"]
    csv_path = path / "attention_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (path / "latest.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(results, path / "README.md")


def _write_markdown(results: dict[str, Any], path: Path) -> None:
    summary = results["summary"]
    lines = [
        "# Catalyst Attention Adapter Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Largest key count | {summary['largest_key_count']} |",
        f"| Largest-key top-1 accuracy | {summary['largest_accuracy_pct']:.2f}% |",
        f"| Largest-key median latency | {summary['largest_median_us']:.4f} us |",
        f"| Largest-key speedup vs softmax reference | {summary['largest_speedup_vs_softmax_x']:.2f}x |",
        "",
        "This benchmark compares the public Catalyst quantum attention head against a pure-Python scaled dot-product softmax reference. It does not claim physical quantum execution.",
        "",
        "| Dimension | Keys | Noise | Method | Accuracy | Median us | P95 us | Speedup vs softmax |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results["attention"]:
        lines.append(
            "| "
            f"{row['dimension']} | {row['key_count']} | {row['noise_pct']:.2f}% | {row['method']} | "
            f"{row['top1_accuracy_pct']:.2f}% | {row['median_us']:.4f} | {row['p95_us']:.4f} | "
            f"{row['latency_vs_softmax_x']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
