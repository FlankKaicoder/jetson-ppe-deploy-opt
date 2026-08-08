#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from pathlib import Path


def coefficient_of_variation(values: list[float]) -> float:
    return statistics.pstdev(values) / statistics.fmean(values) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [
        json.loads((path.resolve() / "exp12_summary.json").read_text(encoding="utf-8"))
        for path in args.run_dirs
    ]
    failures = []
    if any(item.get("result") != "PASS" or item.get("mode") != "performance" for item in summaries):
        failures.append("one_or_more_performance_runs_failed")
    means = [float(item["end_to_end_ms"]["mean"]) for item in summaries]
    p95s = [float(item["end_to_end_ms"]["p95"]) for item in summaries]
    fps_values = [float(item["effective_fps"]) for item in summaries]
    mean_cv = coefficient_of_variation(means)
    fps_cv = coefficient_of_variation(fps_values)
    p95_span = (max(p95s) - min(p95s)) / min(p95s) * 100.0
    if mean_cv > 5.0:
        failures.append("mean_latency_cv_above_5_percent")
    if fps_cv > 5.0:
        failures.append("fps_cv_above_5_percent")
    if p95_span > 10.0:
        failures.append("p95_relative_span_above_10_percent")
    result = {
        "result": "PASS" if not failures else "FAIL",
        "failures": failures,
        "run_dirs": [str(path.resolve()) for path in args.run_dirs],
        "mean_latency_ms": means,
        "p95_latency_ms": p95s,
        "effective_fps": fps_values,
        "mean_latency_cv_percent": mean_cv,
        "fps_cv_percent": fps_cv,
        "p95_relative_span_percent": p95_span,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
