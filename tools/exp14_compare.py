#!/usr/bin/env python3
"""Aggregate three-run Exp14 no-profiler comparisons."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_runs(paths: list[Path]) -> list[dict]:
    values = []
    for path in paths:
        summary_path = path / "app_output" / "summary.json"
        validation_path = path / "validation.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if summary.get("result") != "PASS" or validation.get("result") != "PASS":
            raise RuntimeError(f"run did not pass: {path}")
        values.append({"run_dir": str(path.resolve()), "summary": summary,
                       "validation": validation})
    return values


def aggregate(paths: list[Path]) -> dict:
    if len(paths) != 3:
        raise ValueError("each variant requires exactly three runs")
    runs = load_runs(paths)
    fps = [run["summary"]["pipeline_wall_fps"] for run in runs]
    e2e_mean = [run["summary"]["timings_ms"]["end_to_end"]["mean"] for run in runs]
    e2e_p95 = [run["summary"]["timings_ms"]["end_to_end"]["p95"] for run in runs]
    e2e_p99 = [run["summary"]["timings_ms"]["end_to_end"]["p99"] for run in runs]
    timing_names = sorted({
        name
        for run in runs
        for name in run["summary"]["timings_ms"]
    })
    timing_means = {
        name: statistics.fmean(
            run["summary"]["timings_ms"][name]["mean"] for run in runs
        )
        for name in timing_names
    }
    return {
        "run_dirs": [run["run_dir"] for run in runs],
        "pipeline_wall_fps_values": fps,
        "pipeline_wall_fps_mean": statistics.fmean(fps),
        "pipeline_wall_fps_cv_percent":
            statistics.pstdev(fps) / statistics.fmean(fps) * 100.0,
        "e2e_mean_ms_values": e2e_mean,
        "e2e_mean_ms_mean": statistics.fmean(e2e_mean),
        "e2e_p95_ms_values": e2e_p95,
        "e2e_p95_ms_mean": statistics.fmean(e2e_p95),
        "e2e_p99_ms_values": e2e_p99,
        "e2e_p99_ms_mean": statistics.fmean(e2e_p99),
        "stage_mean_ms_across_runs": timing_means,
        "detection_sha256": [run["validation"]["detections_sha256"] for run in runs],
    }


def delta_percent(candidate: float, baseline: float, lower_is_better: bool) -> float:
    raw = (candidate / baseline - 1.0) * 100.0
    return -raw if lower_is_better else raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-type", choices=("file", "camera"), required=True)
    parser.add_argument("--baseline", type=Path, nargs=3, required=True)
    parser.add_argument("--a", type=Path, nargs=3, required=True)
    parser.add_argument("--b", type=Path, nargs=3, required=True)
    parser.add_argument("--c", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {
        "baseline": aggregate(args.baseline),
        "A": aggregate(args.a),
        "B": aggregate(args.b),
        "C": aggregate(args.c),
    }
    baseline = groups["baseline"]
    comparisons = {}
    for name in ("A", "B", "C"):
        candidate = groups[name]
        fps_gain = delta_percent(
            candidate["pipeline_wall_fps_mean"],
            baseline["pipeline_wall_fps_mean"], False)
        mean_gain = delta_percent(
            candidate["e2e_mean_ms_mean"],
            baseline["e2e_mean_ms_mean"], True)
        p95_regression = (
            candidate["e2e_p95_ms_mean"] /
            baseline["e2e_p95_ms_mean"] - 1.0) * 100.0
        comparisons[name] = {
            "pipeline_fps_gain_percent": fps_gain,
            "e2e_mean_improvement_percent": mean_gain,
            "e2e_p95_regression_percent": p95_regression,
            "file_performance_target_pass":
                (fps_gain >= 10.0 or mean_gain >= 10.0) and p95_regression <= 5.0,
            "camera_p95_target_pass": p95_regression <= 5.0,
        }

    if args.source_type == "file":
        acceptance_pass = comparisons["C"]["file_performance_target_pass"]
    else:
        acceptance_pass = comparisons["C"]["camera_p95_target_pass"]
    finite = all(
        math.isfinite(value)
        for group in groups.values()
        for key, value in group.items()
        if isinstance(value, float)
    )
    result = {
        "result": "PASS" if acceptance_pass and finite else "FAIL",
        "source_type": args.source_type,
        "groups": groups,
        "comparisons_vs_baseline": comparisons,
        "variant_c_acceptance_pass": acceptance_pass,
        "finite": finite,
        "thresholds": {
            "file_fps_or_mean_improvement_percent": 10.0,
            "p95_max_regression_percent": 5.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
