#!/usr/bin/env python3
"""Collect paired/interleaved FP16 vs explicit-QDQ GPU-only measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--qdq-engine", required=True)
    parser.add_argument("--return-codes", required=True)
    parser.add_argument("--min-latency-reduction", type=float, default=0.05)
    parser.add_argument("--min-size-reduction", type=float, default=0.10)
    parser.add_argument("--min-favorable-pairs", type=int, default=2)
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
    values = [
        float(record["computeMs"])
        for record in records
        if isinstance(record.get("computeMs"), (int, float))
        and math.isfinite(float(record["computeMs"]))
    ]
    if not values:
        raise ValueError(f"computeMs missing: {path}")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
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
    return_codes = json.loads(Path(args.return_codes).read_text(encoding="utf-8"))
    orders = {
        1: ("fp16", "qdq"),
        2: ("qdq", "fp16"),
        3: ("fp16", "qdq"),
    }
    runs: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    for round_number, order in orders.items():
        round_means: dict[str, float] = {}
        for sequence, backend in enumerate(order, start=1):
            name = f"r{round_number}_{sequence}_{backend}"
            compute = timing_summary(report_dir / f"{name}_times.json")
            runs[name] = {
                "round": round_number,
                "sequence": sequence,
                "backend": backend,
                "compute_ms": compute,
                "throughput_qps": throughput(report_dir / f"{name}.log"),
                "return_code": int(return_codes[name]),
            }
            round_means[backend] = compute["mean"]
        reduction = 1.0 - round_means["qdq"] / round_means["fp16"]
        pairs.append(
            {
                "round": round_number,
                "order": list(order),
                "fp16_mean_ms": round_means["fp16"],
                "qdq_mean_ms": round_means["qdq"],
                "qdq_latency_reduction": reduction,
                "qdq_faster": reduction > 0,
            }
        )

    engines = {
        "fp16": Path(args.fp16_engine).resolve(),
        "qdq": Path(args.qdq_engine).resolve(),
    }
    engine_info = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in engines.items()
    }
    paired_reductions = [pair["qdq_latency_reduction"] for pair in pairs]
    median_reduction = statistics.median(paired_reductions)
    favorable_pairs = sum(pair["qdq_faster"] for pair in pairs)
    size_reduction = 1.0 - engine_info["qdq"]["bytes"] / engine_info["fp16"]["bytes"]
    gates = {
        "all_processes_succeeded": all(int(code) == 0 for code in return_codes.values()),
        "paired_median_latency_reduction": median_reduction >= args.min_latency_reduction,
        "favorable_pairs": favorable_pairs >= args.min_favorable_pairs,
        "engine_size_reduction": size_reduction >= args.min_size_reduction,
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp17 explicit Q/DQ paired GPU-only diagnostic",
        "result": "PASS" if passed else "REJECT",
        "scope": "GPU compute only; no H2D/D2H; not end-to-end and not an adoption decision",
        "configuration": {
            "batch": 1,
            "input": [1, 3, 640, 640],
            "warmup_milliseconds": 500,
            "iterations": 200,
            "duration_seconds": 0,
            "cuda_graph": True,
            "spin_wait": True,
            "data_transfers": False,
            "clock_policy": "dynamic_25W_paired_interleaved",
        },
        "thresholds": {
            "min_paired_median_latency_reduction": args.min_latency_reduction,
            "min_favorable_pairs": args.min_favorable_pairs,
            "min_engine_size_reduction": args.min_size_reduction,
        },
        "engines": engine_info,
        "runs": runs,
        "pairs": pairs,
        "comparison": {
            "paired_median_qdq_latency_reduction": median_reduction,
            "paired_min_qdq_latency_reduction": min(paired_reductions),
            "paired_max_qdq_latency_reduction": max(paired_reductions),
            "favorable_pairs": favorable_pairs,
            "qdq_engine_size_reduction": size_reduction,
        },
        "gates": gates,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"result={summary['result']}",
        "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END",
    ]
    for pair in pairs:
        lines.append(
            f"round_{pair['round']}_order={','.join(pair['order'])} "
            f"fp16_mean_ms={pair['fp16_mean_ms']:.12g} "
            f"qdq_mean_ms={pair['qdq_mean_ms']:.12g} "
            f"qdq_latency_reduction={pair['qdq_latency_reduction']:.12g}"
        )
    lines.extend(
        [
            f"paired_median_qdq_latency_reduction={median_reduction:.12g}",
            f"favorable_pairs={favorable_pairs}/3",
            f"fp16_engine_bytes={engine_info['fp16']['bytes']}",
            f"qdq_engine_bytes={engine_info['qdq']['bytes']}",
            f"qdq_engine_size_reduction={size_reduction:.12g}",
        ]
    )
    for gate, value in gates.items():
        lines.append(f"gate_{gate}={'PASS' if value else 'FAIL'}")
    lines.append("")
    (report_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"result={summary['result']} median_reduction={median_reduction:.6f}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
