#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def value(data: dict, *keys: str) -> float:
    current = data
    for key in keys:
        current = current[key]
    return float(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with (args.root / "run_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        summary = json.loads((run_dir / "app_output" / "summary.json").read_text())
        monitor = json.loads((run_dir / "monitor_summary.json").read_text())
        records.append({
            "round": int(row["round"]), "order_index": int(row["order_index"]),
            "variant": row["variant"], "run_dir": str(run_dir),
            "pipeline_wall_fps": value(summary, "pipeline_wall_fps"),
            "effective_fps": value(summary, "effective_fps"),
            "e2e_mean_ms": value(summary, "timings_ms", "end_to_end", "mean"),
            "e2e_p50_ms": value(summary, "timings_ms", "end_to_end", "p50"),
            "e2e_p95_ms": value(summary, "timings_ms", "end_to_end", "p95"),
            "e2e_p99_ms": value(summary, "timings_ms", "end_to_end", "p99"),
            "inference_cuda_mean_ms": value(summary, "timings_ms", "inference_cuda", "mean"),
            "count_sync_mean_ms": value(summary, "timings_ms", "count_sync_host", "mean"),
            "candidate_sync_mean_ms": value(summary, "timings_ms", "candidate_sync_host", "mean"),
            "cpu_nms_mean_ms": value(summary, "timings_ms", "cpu_nms", "mean"),
            "d2h_bytes_mean": value(summary, "transfer", "d2h_bytes_per_frame", "mean"),
            "max_temperature_c": monitor["max_observed_temperature_c"],
        })
    metric_names = [name for name in records[0] if name not in {
        "round", "order_index", "variant", "run_dir", "max_temperature_c"}]
    aggregate = {}
    for variant in ("F0", "P"):
        selected = [item for item in records if item["variant"] == variant]
        aggregate[variant] = {
            name: statistics.mean(item[name] for item in selected) for name in metric_names
        }
        temperatures = [item["max_temperature_c"] for item in selected
                        if item["max_temperature_c"] is not None]
        aggregate[variant]["max_temperature_c"] = max(temperatures, default=None)
    pairs = []
    p95_pass = True
    fps_favorable = 0
    mean_favorable = 0
    for round_index in (1, 2, 3):
        f0 = next(item for item in records if item["round"] == round_index and item["variant"] == "F0")
        plugin = next(item for item in records if item["round"] == round_index and item["variant"] == "P")
        fps_delta = 100.0 * (plugin["pipeline_wall_fps"] / f0["pipeline_wall_fps"] - 1.0)
        mean_delta = 100.0 * (plugin["e2e_mean_ms"] / f0["e2e_mean_ms"] - 1.0)
        p95_delta = 100.0 * (plugin["e2e_p95_ms"] / f0["e2e_p95_ms"] - 1.0)
        fps_favorable += fps_delta > 0.0
        mean_favorable += mean_delta < 0.0
        p95_pass = p95_pass and p95_delta <= 5.0
        pairs.append({"round": round_index, "fps_delta_percent": fps_delta,
                      "e2e_mean_delta_percent": mean_delta,
                      "e2e_p95_delta_percent": p95_delta,
                      "p95_pass": p95_delta <= 5.0})
    aggregate_fps_delta = 100.0 * (
        aggregate["P"]["pipeline_wall_fps"] / aggregate["F0"]["pipeline_wall_fps"] - 1.0)
    aggregate_mean_delta = 100.0 * (
        aggregate["P"]["e2e_mean_ms"] / aggregate["F0"]["e2e_mean_ms"] - 1.0)
    fps_gate = aggregate_fps_delta >= 3.0 and fps_favorable >= 2
    mean_gate = aggregate_mean_delta <= -3.0 and mean_favorable >= 2
    passed = p95_pass and (fps_gate or mean_gate)
    result = {
        "result": "PASS" if passed else "REJECTED",
        "configuration": {"rounds": 3, "order": ["F0->P", "P->F0", "F0->P"],
                          "frames": 150, "warmup": 2, "clock_policy": "dynamic",
                          "p95_max_regression_percent": 5.0,
                          "aggregate_fps_min_improvement_percent": 3.0,
                          "aggregate_mean_min_improvement_percent": 3.0,
                          "minimum_favorable_pairs": 2},
        "aggregate": aggregate, "pairs": pairs,
        "aggregate_fps_delta_percent": aggregate_fps_delta,
        "aggregate_e2e_mean_delta_percent": aggregate_mean_delta,
        "fps_favorable_pairs": fps_favorable, "mean_favorable_pairs": mean_favorable,
        "p95_all_pairs_pass": p95_pass, "fps_gate_pass": fps_gate,
        "mean_gate_pass": mean_gate, "records": records,
    }
    (args.root / "performance_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
