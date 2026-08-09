#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def nested(data: dict, *keys: str) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("formal_dir", type=Path)
    args = parser.parse_args()
    with (args.formal_dir / "run_registry.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    records = []
    for row in rows:
        summary = json.loads(
            (Path(row["run_dir"]) / "app_output" / "summary.json").read_text(
                encoding="utf-8"))
        records.append({
            "round": int(row["round"]),
            "order_index": int(row["order_index"]),
            "variant": row["variant"],
            "run_dir": row["run_dir"],
            "pipeline_wall_fps": nested(summary, "pipeline_wall_fps"),
            "effective_fps": nested(summary, "effective_fps"),
            "e2e_mean_ms": nested(summary, "timings_ms", "end_to_end", "mean"),
            "e2e_p50_ms": nested(summary, "timings_ms", "end_to_end", "p50"),
            "e2e_p95_ms": nested(summary, "timings_ms", "end_to_end", "p95"),
            "e2e_p99_ms": nested(summary, "timings_ms", "end_to_end", "p99"),
            "inference_cuda_mean_ms": nested(
                summary, "timings_ms", "inference_cuda", "mean"),
            "count_sync_mean_ms": nested(
                summary, "timings_ms", "count_sync_host", "mean"),
            "candidate_sync_mean_ms": nested(
                summary, "timings_ms", "candidate_sync_host", "mean"),
            "cpu_nms_mean_ms": nested(summary, "timings_ms", "cpu_nms", "mean"),
            "d2h_bytes_mean": nested(
                summary, "transfer", "d2h_bytes_per_frame", "mean"),
        })

    metrics = [key for key in records[0]
               if key not in {"round", "order_index", "variant", "run_dir"}]
    aggregate = {}
    for variant in ("control", "plugin"):
        selected = [record for record in records if record["variant"] == variant]
        aggregate[variant] = {
            metric: statistics.mean(record[metric] for record in selected)
            for metric in metrics
        }

    pairs = []
    gate_errors = []
    for round_index in (1, 2, 3):
        control = next(record for record in records
                       if record["round"] == round_index
                       and record["variant"] == "control")
        plugin = next(record for record in records
                      if record["round"] == round_index
                      and record["variant"] == "plugin")
        p95_delta = 100.0 * (
            plugin["e2e_p95_ms"] / control["e2e_p95_ms"] - 1.0)
        wall_fps_delta = 100.0 * (
            plugin["pipeline_wall_fps"] / control["pipeline_wall_fps"] - 1.0)
        semantic = json.loads(
            (args.formal_dir / f"round_{round_index}_semantic" / "summary.json")
            .read_text(encoding="utf-8"))
        if semantic["result"] != "PASS":
            gate_errors.append(f"round {round_index} semantic comparison failed")
        if p95_delta > 5.0:
            gate_errors.append(f"round {round_index} P95 regression exceeds 5%")
        pairs.append({
            "round": round_index,
            "plugin_vs_control_pipeline_wall_fps_percent": wall_fps_delta,
            "plugin_vs_control_e2e_p95_percent": p95_delta,
            "semantic_result": semantic["result"],
            "semantic_metrics": semantic["metrics"],
        })

    result = {
        "experiment": "Exp16 formal paired/interleaved comparison",
        "result": "PASS" if not gate_errors else "FAIL",
        "configuration": {
            "rounds": 3,
            "processes_per_variant": 3,
            "clock_policy": "dynamic",
            "frames_per_process": 150,
            "p95_max_regression_percent": 5.0,
            "box_max_abs_source_pixels": 2.0,
            "confidence_max_abs": 0.005,
        },
        "aggregate": aggregate,
        "paired": pairs,
        "runs": records,
        "errors": gate_errors,
    }
    (args.formal_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"result={result['result']}"]
    for variant in ("control", "plugin"):
        item = aggregate[variant]
        lines.append(
            f"{variant}: wall_fps={item['pipeline_wall_fps']:.6f} "
            f"e2e_mean_ms={item['e2e_mean_ms']:.6f} "
            f"e2e_p95_ms={item['e2e_p95_ms']:.6f} "
            f"d2h_bytes={item['d2h_bytes_mean']:.2f}")
    for pair in pairs:
        lines.append(
            f"round={pair['round']} fps_delta_percent="
            f"{pair['plugin_vs_control_pipeline_wall_fps_percent']:.6f} "
            f"p95_delta_percent={pair['plugin_vs_control_e2e_p95_percent']:.6f} "
            f"semantic={pair['semantic_result']}")
    lines.extend(f"error={error}" for error in gate_errors)
    (args.formal_dir / "analysis.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not gate_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
