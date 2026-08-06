#!/usr/bin/env python3
"""Collect Exp07 FP32/FP16 build, consistency, and diagnostic timing results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--fp32-build-return-code", type=int, required=True)
    parser.add_argument("--fp16-build-return-code", type=int, required=True)
    parser.add_argument("--fp32-build-seconds", type=int, required=True)
    parser.add_argument("--fp16-build-seconds", type=int, required=True)
    parser.add_argument("--fp32-consistency-return-code", type=int, required=True)
    parser.add_argument("--fp16-consistency-return-code", type=int, required=True)
    parser.add_argument("--fp32-benchmark-return-code", type=int, required=True)
    parser.add_argument("--fp16-benchmark-return-code", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_timing_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list):
        return {
            "path": str(path),
            "record_count": 0,
            "numeric_fields": {},
            "parse_note": "expected a JSON list of timing records",
        }
    numeric_values: dict[str, list[float]] = {}
    for record in data:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_values.setdefault(key, []).append(float(value))
    summaries: dict[str, Any] = {}
    for key, values in numeric_values.items():
        summaries[key] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
        }
    return {
        "path": str(path),
        "record_count": len(data),
        "numeric_fields": summaries,
    }


def engine_manifest(path: Path, precision: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "precision": precision,
            "exists": False,
        }
    return {
        "path": str(path),
        "precision": precision,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    fp32_engine = Path(args.fp32_engine).resolve()
    fp16_engine = Path(args.fp16_engine).resolve()
    fp32_consistency = load_json(report_dir / "fp32" / "summary.json")
    fp16_consistency = load_json(report_dir / "fp16" / "summary.json")
    fp32_times = summarize_timing_file(report_dir / "fp32_times.json")
    fp16_times = summarize_timing_file(report_dir / "fp16_times.json")

    return_codes = {
        "fp32_build": args.fp32_build_return_code,
        "fp16_build": args.fp16_build_return_code,
        "fp32_consistency": args.fp32_consistency_return_code,
        "fp16_consistency": args.fp16_consistency_return_code,
        "fp32_benchmark": args.fp32_benchmark_return_code,
        "fp16_benchmark": args.fp16_benchmark_return_code,
    }
    consistency_pass = bool(
        isinstance(fp32_consistency, dict)
        and fp32_consistency.get("result") == "PASS"
        and isinstance(fp16_consistency, dict)
        and fp16_consistency.get("result") == "PASS"
    )
    overall_pass = bool(
        all(value == 0 for value in return_codes.values())
        and consistency_pass
        and fp32_engine.is_file()
        and fp16_engine.is_file()
    )
    artifacts = {
        "fp32_engine": engine_manifest(fp32_engine, "fp32"),
        "fp16_engine": engine_manifest(fp16_engine, "fp16"),
    }
    summary = {
        "experiment": "Exp07 TensorRT FP32 and FP16 formal validation",
        "result": "PASS" if overall_pass else "FAIL",
        "return_codes": return_codes,
        "build": {
            "fp32_seconds": args.fp32_build_seconds,
            "fp16_seconds": args.fp16_build_seconds,
            "workspace_mib": 1024,
            "builder_optimization_level": 3,
            "tf32": False,
            "fp16_flag_for_fp16_engine": True,
        },
        "artifacts": artifacts,
        "consistency": {
            "fp32": fp32_consistency,
            "fp16": fp16_consistency,
        },
        "diagnostic_benchmark": {
            "scope": "GPU compute only; H2D/D2H disabled; not end-to-end",
            "warmup_milliseconds": 500,
            "iterations": 200,
            "duration_seconds": 0,
            "cuda_graph": True,
            "spin_wait": True,
            "jetson_clocks": "NOT_CHECKED_NON_ROOT",
            "fp32": fp32_times,
            "fp16": fp16_times,
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "artifact_manifest.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "============================================================",
        " Exp07 TensorRT FP32 / FP16 Formal Summary",
        "============================================================",
        f"result={'PASS' if overall_pass else 'FAIL'}",
    ]
    for name, value in return_codes.items():
        lines.append(f"{name}_return_code={value}")
    for precision, engine in artifacts.items():
        lines.append(f"{precision}_path={engine['path']}")
        if engine.get("exists"):
            lines.append(f"{precision}_size_bytes={engine['size_bytes']}")
            lines.append(f"{precision}_sha256={engine['sha256']}")
    lines.extend(
        [
            f"fp32_build_seconds={args.fp32_build_seconds}",
            f"fp16_build_seconds={args.fp16_build_seconds}",
        ]
    )
    for precision, consistency in (
        ("fp32", fp32_consistency),
        ("fp16", fp16_consistency),
    ):
        if isinstance(consistency, dict):
            raw = consistency["raw_tensor"]
            detections = consistency["detections"]
            lines.extend(
                [
                    f"{precision}_raw_max_abs_error={raw['max_abs_error']}",
                    f"{precision}_raw_mean_abs_error={raw['mean_abs_error']}",
                    f"{precision}_raw_relative_l2_error={raw['relative_l2_error']}",
                    f"{precision}_raw_result={raw['result']}",
                    f"{precision}_detection_count={detections['candidate_count']}",
                    f"{precision}_detection_classes_equal={detections['classes_equal']}",
                    f"{precision}_detection_max_box_abs_error={detections['max_box_abs_error']}",
                    f"{precision}_detection_max_confidence_abs_error={detections['max_confidence_abs_error']}",
                    f"{precision}_detection_result={detections['result']}",
                ]
            )
        timing = fp32_times if precision == "fp32" else fp16_times
        for key, metrics in timing.get("numeric_fields", {}).items():
            if "compute" in key.lower() or "latency" in key.lower():
                lines.append(
                    f"{precision}_{key}_mean={metrics['mean']}"
                )
                lines.append(f"{precision}_{key}_p50={metrics['p50']}")
                lines.append(f"{precision}_{key}_p95={metrics['p95']}")
                lines.append(f"{precision}_{key}_p99={metrics['p99']}")
    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print((report_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
