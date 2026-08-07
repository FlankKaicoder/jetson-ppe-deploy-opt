#!/usr/bin/env python3
"""Collect Exp09 independent-process correctness and timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--processes", type=int, default=3)
    return parser.parse_args()


def read_code(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing(values: dict[str, str], prefix: str) -> dict[str, float]:
    return {
        name: float(values[f"{prefix}_{name}_ms"])
        for name in ("mean", "p50", "p95", "p99", "min", "max")
    }


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    reference = json.loads(
        (report_dir / "reference" / "reference.json").read_text(
            encoding="utf-8"
        )
    )
    process_results: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    process_gates: list[bool] = []
    for index in range(1, args.processes + 1):
        runtime_code = read_code(report_dir / f"runtime_{index}_return_code.txt")
        compare_code = read_code(report_dir / f"compare_{index}_return_code.txt")
        runtime_values = read_key_values(
            report_dir / f"runtime_{index}_summary.txt"
        )
        comparison = json.loads(
            (report_dir / f"comparison_{index}" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        output_path = report_dir / f"cxx_output_{index}_fp32.bin"
        output_hash = file_sha256(output_path)
        output_hashes.append(output_hash)
        passed = bool(
            runtime_code == 0
            and compare_code == 0
            and runtime_values.get("result") == "PASS"
            and runtime_values.get("output_finite") == "true"
            and runtime_values.get("timed_iterations") == "200"
            and comparison.get("result") == "PASS"
        )
        process_gates.append(passed)
        process_results.append(
            {
                "index": index,
                "result": "PASS" if passed else "FAIL",
                "return_codes": {
                    "runtime": runtime_code,
                    "comparison": compare_code,
                },
                "output_sha256": output_hash,
                "host_total_ms": timing(runtime_values, "host_total"),
                "cuda_total_ms": timing(runtime_values, "cuda_total"),
                "comparison": comparison,
            }
        )

    gates = {
        "process_count": args.processes == 3,
        "all_processes": all(process_gates),
        "identical_output_sha256": len(set(output_hashes)) == 1,
        "matches_python_reference_sha256": bool(
            output_hashes
            and output_hashes[0] == reference["output"]["sha256"]
        ),
    }
    passed = all(gates.values())
    host_means = [item["host_total_ms"]["mean"] for item in process_results]
    cuda_means = [item["cuda_total_ms"]["mean"] for item in process_results]
    summary = {
        "experiment": "Exp09 TensorRT C++ Runtime formal validation",
        "result": "PASS" if passed else "FAIL",
        "scope": "preprocessed FP32 NCHW input; H2D + enqueueV3 + D2H + synchronize",
        "configuration": {
            "independent_processes": args.processes,
            "warmup_per_process": 20,
            "timed_iterations_per_process": 200,
        },
        "reference": {
            "engine_sha256": reference["engine_sha256"],
            "image_sha256": reference["image_sha256"],
            "input_sha256": reference["input"]["sha256"],
            "python_output_sha256": reference["output"]["sha256"],
        },
        "processes": process_results,
        "cross_process": {
            "output_sha256": output_hashes[0] if output_hashes else None,
            "host_mean_ms_min": min(host_means),
            "host_mean_ms_max": max(host_means),
            "cuda_mean_ms_min": min(cuda_means),
            "cuda_mean_ms_max": max(cuda_means),
        },
        "gates": gates,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"independent_processes={args.processes}",
                f"output_sha256={summary['cross_process']['output_sha256']}",
                f"host_mean_ms_min={summary['cross_process']['host_mean_ms_min']}",
                f"host_mean_ms_max={summary['cross_process']['host_mean_ms_max']}",
                f"cuda_mean_ms_min={summary['cross_process']['cuda_mean_ms_min']}",
                f"cuda_mean_ms_max={summary['cross_process']['cuda_mean_ms_max']}",
                f"all_processes_pass={gates['all_processes']}",
                f"identical_output_sha256={gates['identical_output_sha256']}",
                f"matches_python_reference_sha256={gates['matches_python_reference_sha256']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result={summary['result']} processes={args.processes} "
        f"output_sha256={summary['cross_process']['output_sha256']} "
        f"host_mean_ms={min(host_means):.6f}-{max(host_means):.6f}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
