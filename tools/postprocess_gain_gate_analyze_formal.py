#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


EXPECTED_DIGEST = "9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a"


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
    parser.add_argument("formal_dir", type=Path)
    args = parser.parse_args()
    registry_path = args.formal_dir / "run_registry.csv"
    with registry_path.open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    rows = []
    errors: list[str] = []
    for item in registry:
        run_dir = Path(item["run_dir"])
        try:
            summary = json.loads(
                (run_dir / "app_output" / "summary.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (run_dir / "validation.json").read_text(encoding="utf-8")
            )
            monitor = json.loads(
                (run_dir / "monitor" / "monitor_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as error:
            errors.append(f"{run_dir}: {error}")
            continue
        path = item["path"]
        if summary.get("processed_frames") != 150:
            errors.append(f"{path} round {item['round']}: frame count mismatch")
        if summary.get("total_detections") != 151:
            errors.append(f"{path} round {item['round']}: detection count mismatch")
        if validation.get("detections_sha256") != EXPECTED_DIGEST:
            errors.append(f"{path} round {item['round']}: digest mismatch")
        if monitor.get("result") != "PASS":
            errors.append(f"{path} round {item['round']}: monitor failed")
        d2h = summary["transfer"]["d2h_bytes_per_frame"]["mean"]
        if path in ("P0", "P1") and d2h != 235200.0:
            errors.append(f"{path} round {item['round']}: D2H mismatch {d2h}")
        timings = summary["timings_ms"]
        rows.append({
            "round": int(item["round"]),
            "order_index": int(item["order_index"]),
            "path": path,
            "run_dir": str(run_dir),
            "pipeline_wall_fps": summary["pipeline_wall_fps"],
            "e2e_mean_ms": timings["end_to_end"]["mean"],
            "e2e_p95_ms": timings["end_to_end"]["p95"],
            "e2e_p99_ms": timings["end_to_end"]["p99"],
            "gpu_postprocess_cuda_ms": timings["gpu_postprocess_cuda"]["mean"],
            "count_sync_host_ms": timings["count_sync_host"]["mean"],
            "candidate_sync_host_ms": timings["candidate_sync_host"]["mean"],
            "cpu_decode_filter_ms": timings["cpu_decode_filter"]["mean"],
            "cpu_candidate_scan_ms": timings["cpu_candidate_scan"]["mean"],
            "cpu_nms_ms": timings["cpu_nms"]["mean"],
            "d2h_bytes": d2h,
            "max_temperature_c": monitor.get("max_observed_temperature_c"),
        })
    with (args.formal_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped = {path: [row for row in rows if row["path"] == path]
               for path in ("P0", "P1", "P2")}
    counts = {path: len(items) for path, items in grouped.items()}
    if counts != {"P0": 3, "P1": 3, "P2": 3}:
        errors.append(f"formal run counts mismatch: {counts}")
    metrics = (
        "pipeline_wall_fps", "e2e_mean_ms", "e2e_p95_ms", "e2e_p99_ms",
        "gpu_postprocess_cuda_ms", "count_sync_host_ms",
        "candidate_sync_host_ms", "cpu_decode_filter_ms",
        "cpu_candidate_scan_ms", "cpu_nms_ms", "d2h_bytes",
    )
    aggregates = {
        path: {metric: stats([float(row[metric]) for row in items])
               for metric in metrics}
        for path, items in grouped.items() if items
    }
    paired = {}
    for left, right in (("P0", "P1"), ("P1", "P2"), ("P0", "P2")):
        for metric in ("pipeline_wall_fps", "e2e_mean_ms", "e2e_p95_ms"):
            deltas = []
            for round_index in (1, 2, 3):
                left_row = next(row for row in grouped[left] if row["round"] == round_index)
                right_row = next(row for row in grouped[right] if row["round"] == round_index)
                deltas.append(
                    (right_row[metric] / left_row[metric] - 1.0) * 100.0
                )
            paired[f"{left}_to_{right}_{metric}_percent"] = stats(deltas)
    result = {
        "result": "PASS" if not errors else "FAIL",
        "run_counts": counts,
        "aggregates": aggregates,
        "paired_deltas": paired,
        "errors": errors,
    }
    (args.formal_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.formal_dir / "analysis.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"result={result['result']}\n")
        for path in ("P0", "P1", "P2"):
            if path not in aggregates:
                continue
            stream.write(
                f"{path}_wall_fps_mean={aggregates[path]['pipeline_wall_fps']['mean']:.6f} "
                f"e2e_mean_ms={aggregates[path]['e2e_mean_ms']['mean']:.6f} "
                f"e2e_p95_ms={aggregates[path]['e2e_p95_ms']['mean']:.6f} "
                f"d2h_bytes={aggregates[path]['d2h_bytes']['mean']:.3f}\n"
            )
        for key, value in paired.items():
            stream.write(
                f"{key}_mean={value['mean']:.6f} "
                f"median={value['median']:.6f} min={value['min']:.6f} "
                f"max={value['max']:.6f}\n"
            )
        for error in errors:
            stream.write(f"error={error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
