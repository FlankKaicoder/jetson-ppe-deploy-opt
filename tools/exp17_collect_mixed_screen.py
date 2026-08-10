#!/usr/bin/env python3
"""Collect two-order GPU-only timing screen for Exp17 mixed candidates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


BACKENDS = ("fp16", "full_qdq", "p3_classification", "classification", "dfl", "detect_head")


def mean_compute(path: Path) -> float:
    records = json.loads(path.read_text(encoding="utf-8"))
    values = [float(item["computeMs"]) for item in records if math.isfinite(float(item["computeMs"]))]
    if not values:
        raise RuntimeError(f"missing computeMs: {path}")
    return statistics.fmean(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    rounds = {
        "r1": list(BACKENDS),
        "r2": list(reversed(BACKENDS)),
    }
    results = {}
    for round_name, order in rounds.items():
        results[round_name] = {
            backend: mean_compute(args.report_dir / f"{round_name}_{sequence}_{backend}_times.json")
            for sequence, backend in enumerate(order, start=1)
        }
    candidates = {}
    for backend in BACKENDS[1:]:
        reductions = [
            1.0 - results[round_name][backend] / results[round_name]["fp16"]
            for round_name in rounds
        ]
        candidates[backend] = {
            "round_latency_reduction_vs_fp16": reductions,
            "median_latency_reduction_vs_fp16": statistics.median(reductions),
            "both_rounds_faster": all(value > 0 for value in reductions),
            "screen_latency_gate": all(value >= 0.05 for value in reductions),
        }
    summary = {
        "experiment": "Exp17 mixed-precision two-order GPU-only performance screen",
        "result": "PASS",
        "scope": "screening only; GPU compute; no H2D/D2H; not final paired adoption gate",
        "configuration": {"warmup_ms": 500, "iterations": 200, "cuda_graph": True,
                          "spin_wait": True, "clock_policy": "dynamic_25W"},
        "orders": rounds,
        "mean_compute_ms": results,
        "candidates": candidates,
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["result=PASS", "scope=GPU_ONLY_SCREEN_NOT_FINAL_ADOPTION"]
    for backend in BACKENDS:
        lines.append(
            f"{backend} r1_mean_ms={results['r1'][backend]:.12g} "
            f"r2_mean_ms={results['r2'][backend]:.12g}"
        )
    for backend, item in candidates.items():
        lines.append(
            f"{backend}_median_reduction={item['median_latency_reduction_vs_fp16']:.12g} "
            f"both_faster={item['both_rounds_faster']} gate_5pct={item['screen_latency_gate']}"
        )
    lines.append("")
    (args.report_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
