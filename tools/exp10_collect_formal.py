#!/usr/bin/env python3
"""Collect Exp10 multi-shape correctness and performance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIXTURES = ("square", "wide", "tall", "hd_wide", "small_tall")
EXPECTED_GEOMETRY = {
    "square": (640, 640, 640, 640, 0, 0, 0, 0),
    "wide": (640, 360, 640, 360, 0, 0, 140, 140),
    "tall": (360, 640, 360, 640, 140, 140, 0, 0),
    "hd_wide": (1280, 720, 640, 360, 0, 0, 140, 140),
    "small_tall": (240, 480, 320, 640, 160, 160, 0, 0),
}
GEOMETRY_KEYS = (
    "source_width", "source_height", "resized_width", "resized_height",
    "padding_left", "padding_right", "padding_top", "padding_bottom",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--minimum-kernel-reduction", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    fixture_results: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for name in FIXTURES:
        path = run_dir / "comparison" / name / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"missing comparison summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        fixture_results[name] = summary
        gates[f"{name}_correctness"] = summary.get("result") == "PASS"
        actual = tuple(summary["geometry"][key] for key in GEOMETRY_KEYS)
        gates[f"{name}_geometry"] = actual == EXPECTED_GEOMETRY[name]

    square_metrics = fixture_results["square"]["metrics"]
    gates["square_bitwise_exact"] = all(
        square_metrics[key] == 0.0
        for key in (
            "max_abs_error", "mean_abs_error", "relative_l2_error",
            "p99_abs_error", "padding_max_abs_error",
        )
    )
    hd_timing = fixture_results["hd_wide"]["timing_ms"]
    cpu_mean = float(hd_timing["cpu"]["mean"])
    kernel_mean = float(hd_timing["cuda_kernel"]["mean"])
    reduction = 1.0 - kernel_mean / cpu_mean
    gates["kernel_mean_reduction"] = reduction >= args.minimum_kernel_reduction
    passed = all(gates.values())

    summary = {
        "experiment": "Exp10 CUDA fused preprocess formal",
        "result": "PASS" if passed else "FAIL",
        "fixtures": fixture_results,
        "performance": {
            "fixture": "hd_wide",
            "warmup": 20,
            "iterations": 200,
            "cpu_mean_ms": cpu_mean,
            "cuda_kernel_mean_ms": kernel_mean,
            "cuda_total_mean_ms": float(hd_timing["cuda_total"]["mean"]),
            "kernel_mean_reduction": reduction,
            "minimum_kernel_reduction": args.minimum_kernel_reduction,
            "cuda_total_is_diagnostic_only": True,
        },
        "gates": gates,
    }
    (run_dir / "formal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"result={summary['result']}",
        "fixture_count=5",
        f"cpu_mean_ms={cpu_mean}",
        f"cuda_kernel_mean_ms={kernel_mean}",
        f"cuda_total_mean_ms={summary['performance']['cuda_total_mean_ms']}",
        f"kernel_mean_reduction={reduction}",
        f"minimum_kernel_reduction={args.minimum_kernel_reduction}",
    ]
    for name in FIXTURES:
        metrics = fixture_results[name]["metrics"]
        lines.append(
            f"{name}=result:{fixture_results[name]['result']},"
            f"max_abs:{metrics['max_abs_error']},"
            f"mean_abs:{metrics['mean_abs_error']},"
            f"relative_l2:{metrics['relative_l2_error']},"
            f"p99_abs:{metrics['p99_abs_error']},"
            f"padding_abs:{metrics['padding_max_abs_error']}"
        )
    (run_dir / "formal_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"result={summary['result']} fixtures=5 "
        f"kernel_reduction={reduction:.6f}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
