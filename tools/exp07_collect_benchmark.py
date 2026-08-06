#!/usr/bin/env python3
"""Collect Exp07 FP32/FP16 GPU-only diagnostic benchmark results."""

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
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--fp32-return-code", type=int, required=True)
    parser.add_argument("--fp16-return-code", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid timing file: {path}")
    numeric: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric.setdefault(key, []).append(float(value))
    fields = {}
    for key, values in numeric.items():
        fields[key] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
        }
    return {"path": str(path), "record_count": len(records), "fields": fields}


def throughput_from_log(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Throughput:\s+([0-9.]+)\s+qps", text)
    if not matches:
        raise ValueError(f"throughput missing from {path}")
    return float(matches[-1])


def engine_manifest(path: Path, precision: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "precision": precision,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    engines = {
        "fp32": Path(args.fp32_engine).resolve(),
        "fp16": Path(args.fp16_engine).resolve(),
    }
    return_codes = {
        "fp32": args.fp32_return_code,
        "fp16": args.fp16_return_code,
    }
    benchmarks = {}
    for precision in ("fp32", "fp16"):
        benchmarks[precision] = {
            "timings": timing_summary(report_dir / f"{precision}_times.json"),
            "throughput_qps": throughput_from_log(
                report_dir / f"{precision}_benchmark.log"
            ),
        }
    overall_pass = all(code == 0 for code in return_codes.values())
    summary = {
        "experiment": "Exp07 TensorRT GPU-only diagnostic benchmark",
        "result": "PASS" if overall_pass else "FAIL",
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
        "return_codes": return_codes,
        "engines": {
            name: engine_manifest(path, name) for name, path in engines.items()
        },
        "benchmarks": benchmarks,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "============================================================",
        " Exp07 TensorRT GPU-Only Diagnostic Benchmark",
        "============================================================",
        f"result={summary['result']}",
        "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END",
    ]
    for precision in ("fp32", "fp16"):
        engine = summary["engines"][precision]
        timing = benchmarks[precision]["timings"]["fields"]
        compute = timing.get("computeMs") or timing.get("latencyMs")
        if compute is None:
            raise KeyError(f"compute timing missing for {precision}")
        lines.extend(
            [
                f"{precision}_return_code={return_codes[precision]}",
                f"{precision}_engine_size_bytes={engine['size_bytes']}",
                f"{precision}_engine_sha256={engine['sha256']}",
                f"{precision}_compute_mean_ms={compute['mean']:.12g}",
                f"{precision}_compute_p50_ms={compute['p50']:.12g}",
                f"{precision}_compute_p95_ms={compute['p95']:.12g}",
                f"{precision}_compute_p99_ms={compute['p99']:.12g}",
                f"{precision}_throughput_qps={benchmarks[precision]['throughput_qps']:.12g}",
            ]
        )
    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print((report_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
