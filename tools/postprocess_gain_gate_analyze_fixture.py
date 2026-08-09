#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "gpu_postprocess_cuda_ms",
    "count_sync_host_ms",
    "payload_sync_host_ms",
    "cpu_decode_filter_ms",
    "cpu_candidate_scan_ms",
    "cpu_nms_ms",
    "total_ms",
    "d2h_bytes",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "cv_percent": statistics.pstdev(values) / abs(mean) * 100.0 if mean else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    samples_path = args.run_dir / "samples.csv"
    with samples_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {path: [row for row in rows if row["path"] == path]
               for path in ("P0", "P1", "P2")}
    errors: list[str] = []
    counts = {path: len(items) for path, items in grouped.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values()), 0):
        errors.append(f"unbalanced sample counts: {counts}")
    for path, items in grouped.items():
        for row in items:
            if int(row["candidate_count"]) < 0 or int(row["detection_count"]) < 0:
                errors.append(f"invalid count in {path}")
    expected_bytes = {"P0": 235200.0, "P1": 235200.0}
    aggregates = {
        path: {metric: stats([float(row[metric]) for row in items])
               for metric in METRICS}
        for path, items in grouped.items()
    }
    for path, expected in expected_bytes.items():
        values = {float(row["d2h_bytes"]) for row in grouped[path]}
        if values != {expected}:
            errors.append(f"{path} D2H bytes mismatch: {sorted(values)}")
    mean_total = {path: aggregates[path]["total_ms"]["mean"]
                  for path in grouped}
    deltas = {
        "P0_to_P1_total_ms": mean_total["P1"] - mean_total["P0"],
        "P0_to_P1_percent": (mean_total["P1"] / mean_total["P0"] - 1.0) * 100.0,
        "P1_to_P2_total_ms": mean_total["P2"] - mean_total["P1"],
        "P1_to_P2_percent": (mean_total["P2"] / mean_total["P1"] - 1.0) * 100.0,
        "P0_to_P2_total_ms": mean_total["P2"] - mean_total["P0"],
        "P0_to_P2_percent": (mean_total["P2"] / mean_total["P0"] - 1.0) * 100.0,
    }
    result = {
        "result": "PASS" if not errors else "FAIL",
        "sample_counts": counts,
        "aggregates": aggregates,
        "mean_deltas": deltas,
        "errors": errors,
    }
    (args.run_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.run_dir / "analysis.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"result={result['result']}\n")
        for path in ("P0", "P1", "P2"):
            total = aggregates[path]["total_ms"]
            stream.write(
                f"{path}_total_mean_ms={total['mean']:.9f} "
                f"median_ms={total['median']:.9f} p95_ms={total['p95']:.9f} "
                f"cv_percent={total['cv_percent']:.6f}\n"
            )
        for key, value in deltas.items():
            stream.write(f"{key}={value:.9f}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
