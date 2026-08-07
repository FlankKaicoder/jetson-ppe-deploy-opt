#!/usr/bin/env python3
"""Collect FP16/INT8 GPU-only benchmark results and apply frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--fp16-return-code", type=int, required=True)
    parser.add_argument("--int8-return-code", type=int, required=True)
    parser.add_argument("--min-latency-reduction", type=float, default=0.05)
    parser.add_argument("--min-size-reduction", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def timing_summary(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    numeric: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric.setdefault(key, []).append(float(value))
    return {
        key: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
        }
        for key, values in numeric.items()
    }


def throughput(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Throughput:\s+([0-9.]+)\s+qps", text)
    if not matches:
        raise ValueError(f"throughput missing: {path}")
    return float(matches[-1])


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    engines = {
        "fp16": Path(args.fp16_engine).resolve(),
        "int8": Path(args.int8_engine).resolve(),
    }
    return_codes = {
        "fp16": args.fp16_return_code,
        "int8": args.int8_return_code,
    }
    benchmarks: dict[str, Any] = {}
    for name in ("fp16", "int8"):
        timings = timing_summary(report_dir / f"{name}_times.json")
        compute = timings.get("computeMs") or timings.get("latencyMs")
        if compute is None:
            raise KeyError(f"compute timing missing for {name}")
        benchmarks[name] = {
            "timings": timings,
            "compute": compute,
            "throughput_qps": throughput(report_dir / f"{name}_benchmark.log"),
        }
    engine_info = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in engines.items()
    }
    latency_reduction = 1 - (
        benchmarks["int8"]["compute"]["mean"]
        / benchmarks["fp16"]["compute"]["mean"]
    )
    size_reduction = 1 - (
        engine_info["int8"]["bytes"] / engine_info["fp16"]["bytes"]
    )
    speedup = (
        benchmarks["fp16"]["compute"]["mean"]
        / benchmarks["int8"]["compute"]["mean"]
    )
    gates = {
        "return_codes": all(code == 0 for code in return_codes.values()),
        "latency_reduction": latency_reduction >= args.min_latency_reduction,
        "size_reduction": size_reduction >= args.min_size_reduction,
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp08 FP16 vs INT8 GPU-only diagnostic benchmark",
        "result": "PASS" if passed else "FAIL",
        "scope": "GPU compute only; H2D/D2H disabled; not end-to-end",
        "configuration": {
            "batch": 1,
            "input": [1, 3, 640, 640],
            "warmup_milliseconds": 500,
            "iterations": 200,
            "duration_seconds": 0,
            "cuda_graph": True,
            "spin_wait": True,
            "data_transfers": False,
            "jetson_clocks": "NOT_CHECKED_NON_ROOT",
        },
        "thresholds": {
            "min_latency_reduction": args.min_latency_reduction,
            "min_size_reduction": args.min_size_reduction,
        },
        "return_codes": return_codes,
        "engines": engine_info,
        "benchmarks": benchmarks,
        "comparison": {
            "int8_speedup_vs_fp16": speedup,
            "int8_latency_reduction_vs_fp16": latency_reduction,
            "int8_size_reduction_vs_fp16": size_reduction,
        },
        "gates": gates,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"result={summary['result']}",
        "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END",
    ]
    for name in ("fp16", "int8"):
        compute = benchmarks[name]["compute"]
        lines.extend(
            [
                f"{name}_engine_bytes={engine_info[name]['bytes']}",
                f"{name}_engine_sha256={engine_info[name]['sha256']}",
                f"{name}_compute_mean_ms={compute['mean']:.12g}",
                f"{name}_compute_p50_ms={compute['p50']:.12g}",
                f"{name}_compute_p95_ms={compute['p95']:.12g}",
                f"{name}_compute_p99_ms={compute['p99']:.12g}",
                f"{name}_throughput_qps={benchmarks[name]['throughput_qps']:.12g}",
            ]
        )
    lines.extend(
        [
            f"int8_speedup_vs_fp16={speedup:.12g}",
            f"int8_latency_reduction_vs_fp16={latency_reduction:.12g}",
            f"int8_size_reduction_vs_fp16={size_reduction:.12g}",
            f"latency_gate={'PASS' if gates['latency_reduction'] else 'FAIL'}",
            f"size_gate={'PASS' if gates['size_reduction'] else 'FAIL'}",
            "",
        ]
    )
    (report_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(" ".join(lines[-3:-1]), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
