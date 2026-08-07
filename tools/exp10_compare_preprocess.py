#!/usr/bin/env python3
"""Compare CUDA fused preprocessing with the CPU OpenCV reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--runtime-summary", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--max-abs", type=float, default=2 / 255 + 1e-7)
    parser.add_argument("--mean-abs", type=float, default=0.0005)
    parser.add_argument("--relative-l2", type=float, default=0.001)
    parser.add_argument("--p99-abs", type=float, default=1 / 255 + 1e-7)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def read_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    args = parse_args()
    expected = 1 * 3 * 640 * 640
    cpu = np.fromfile(args.cpu, dtype=np.float32)
    cuda = np.fromfile(args.cuda, dtype=np.float32)
    if cpu.size != expected or cuda.size != expected:
        raise RuntimeError(f"unexpected sizes: {cpu.size}, {cuda.size}")
    values = read_values(Path(args.runtime_summary))
    difference = cpu.astype(np.float64) - cuda.astype(np.float64)
    absolute = np.abs(difference)
    denominator = max(float(np.linalg.norm(cpu.astype(np.float64))), 1e-12)
    max_abs = float(absolute.max(initial=0.0))
    mean_abs = float(absolute.mean())
    relative_l2 = float(np.linalg.norm(difference) / denominator)
    p99_abs = float(np.quantile(absolute, 0.99))
    finite = bool(np.isfinite(cpu).all() and np.isfinite(cuda).all())

    target = int(values["target_size"])
    left = int(values["padding_left"])
    right = int(values["padding_right"])
    top = int(values["padding_top"])
    bottom = int(values["padding_bottom"])
    cuda_chw = cuda.reshape(3, target, target)
    padding_mask = np.zeros((target, target), dtype=bool)
    if top:
        padding_mask[:top, :] = True
    if bottom:
        padding_mask[target - bottom :, :] = True
    if left:
        padding_mask[:, :left] = True
    if right:
        padding_mask[:, target - right :] = True
    if padding_mask.any():
        padding_error = float(
            np.abs(cuda_chw[:, padding_mask] - np.float32(114 / 255)).max()
        )
    else:
        padding_error = 0.0

    gates = {
        "shape": bool(values.get("output_shape") == "1,3,640,640"),
        "finite": finite,
        "max_abs": max_abs <= args.max_abs,
        "mean_abs": mean_abs <= args.mean_abs,
        "relative_l2": relative_l2 <= args.relative_l2,
        "p99_abs": p99_abs <= args.p99_abs,
        "padding": padding_error <= 1e-7,
        "exact_when_required": bool(
            not args.require_exact or max_abs == 0.0
        ),
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp10 CPU OpenCV vs CUDA fused preprocess",
        "result": "PASS" if passed else "FAIL",
        "geometry": {
            key: int(values[key])
            for key in (
                "source_width", "source_height", "target_size",
                "resized_width", "resized_height", "padding_left",
                "padding_right", "padding_top", "padding_bottom",
            )
        },
        "thresholds": {
            "max_abs": args.max_abs,
            "mean_abs": args.mean_abs,
            "relative_l2": args.relative_l2,
            "p99_abs": args.p99_abs,
            "padding_abs": 1e-7,
            "require_exact": args.require_exact,
        },
        "metrics": {
            "finite": finite,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_l2_error": relative_l2,
            "p99_abs_error": p99_abs,
            "padding_max_abs_error": padding_error,
        },
        "timing_ms": {
            prefix: {
                name: float(values[f"{prefix}_{name}_ms"])
                for name in ("mean", "p50", "p95", "p99", "min", "max")
            }
            for prefix in ("cpu", "cuda_kernel", "cuda_total")
        },
        "gates": gates,
    }
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"max_abs_error={max_abs}",
                f"mean_abs_error={mean_abs}",
                f"relative_l2_error={relative_l2}",
                f"p99_abs_error={p99_abs}",
                f"padding_max_abs_error={padding_error}",
                f"cpu_mean_ms={summary['timing_ms']['cpu']['mean']}",
                f"cuda_kernel_mean_ms={summary['timing_ms']['cuda_kernel']['mean']}",
                f"cuda_total_mean_ms={summary['timing_ms']['cuda_total']['mean']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result={summary['result']} max_abs={max_abs} mean_abs={mean_abs} "
        f"p99_abs={p99_abs}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
