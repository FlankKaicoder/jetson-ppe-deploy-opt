#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

from exp12_analyze import linear_slope, parse_tegrastats, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("smoke", "formal", "diagnostic", "stability"),
        required=True)
    parser.add_argument("--variant", choices=("v0", "vfinal"), required=True)
    parser.add_argument("--source-type", choices=("file", "camera"), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    app = json.loads((run_dir / "app_output/summary.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    monitor = json.loads((run_dir / "monitor_summary.json").read_text(encoding="utf-8"))
    with (run_dir / "app_output/frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    with (run_dir / "process_samples.csv").open(newline="", encoding="utf-8") as stream:
        process_rows = list(csv.DictReader(stream))
    capture = [float(row["capture_ms"]) for row in frames]
    total = [float(row["end_to_end_ms"]) for row in frames]
    post_capture = [whole - wait for whole, wait in zip(total, capture)]
    resources = parse_tegrastats(run_dir / "tegrastats.log")
    power_mw = resources["power_mw"].get("VDD_IN", {}).get("mean", math.nan)
    wall_fps = float(app["pipeline_wall_fps"])
    energy_j = power_mw / 1000.0 / wall_fps
    steady_start_seconds = 60.0 if args.phase == "stability" else 2.0
    steady_process_rows = [
        row for row in process_rows
        if float(row["elapsed_seconds"]) >= steady_start_seconds
    ]
    if len(steady_process_rows) < 2:
        steady_process_rows = process_rows
    rss_mib = [float(row["rss_kb"]) / 1024.0 for row in steady_process_rows]
    elapsed_min = [
        float(row["elapsed_seconds"]) / 60.0 for row in steady_process_rows]
    first_window = max(1, len(total) // 10)
    p95_drift = (stats(total[-first_window:])["p95"] /
                 stats(total[:first_window])["p95"] - 1.0) * 100.0
    thermal = [value["max"] for name, value in resources["temperatures_c"].items()
               if name in {"cpu", "gpu", "tj"}]
    thermal_max = max(thermal) if thermal else math.nan
    rss_growth = rss_mib[-1] - rss_mib[0] if rss_mib else math.nan
    rss_slope = linear_slope(elapsed_min, rss_mib) if rss_mib else math.nan
    swap_growth = (resources["swap_used_mb"]["max"] -
                   resources["swap_used_mb"]["min"])
    failures = []
    if app.get("result") != "PASS" or validation.get("result") != "PASS" or monitor.get("result") != "PASS":
        failures.append("application_validation_or_monitor_failed")
    if not math.isfinite(energy_j):
        failures.append("energy_per_frame_unavailable")
    if args.phase == "stability":
        if len(frames) != 54000:
            failures.append("stability_frame_count_mismatch")
        if stats(total)["p95"] > 40.0:
            failures.append("p95_above_40ms")
        if stats(total)["p99"] > 50.0:
            failures.append("p99_above_50ms")
        if float(app["effective_fps"]) < 29.0:
            failures.append("effective_fps_below_29")
        if not math.isfinite(thermal_max) or thermal_max >= 70.0:
            failures.append("thermal_limit")
        if p95_drift > 10.0:
            failures.append("p95_drift_above_10_percent")
        if rss_growth > 64.0 or rss_slope > 1.0:
            failures.append("rss_growth_limit")
        if swap_growth > 0.0:
            failures.append("swap_growth")
    result = {
        "result": "PASS" if not failures else "FAIL",
        "phase": args.phase,
        "variant": args.variant,
        "source_type": args.source_type,
        "failures": failures,
        "processed_frames": len(frames),
        "pipeline_wall_fps": wall_fps,
        "effective_fps": float(app["effective_fps"]),
        "capture_wait_ms": stats(capture),
        "post_capture_processing_ms": stats(post_capture),
        "frame_total_ms": stats(total),
        "d2h_bytes_per_frame": validation["mean_d2h_bytes"],
        "detections_sha256": validation["detections_sha256"],
        "resources": resources,
        "energy_per_frame_j": energy_j,
        "rss_mib": stats(rss_mib),
        "steady_state_start_seconds": steady_start_seconds,
        "steady_process_samples": len(steady_process_rows),
        "rss_growth_mib": rss_growth,
        "rss_slope_mib_per_min": rss_slope,
        "swap_growth_mb": swap_growth,
        "thermal_max_cpu_gpu_tj_c": thermal_max,
        "first_last_p95_change_percent": p95_drift,
        "monitor_wall_seconds": float(monitor["wall_seconds"]),
    }
    (run_dir / "exp19_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "exp19_summary.txt").write_text(
        "\n".join((f"result={result['result']}", f"phase={args.phase}",
                    f"variant={args.variant}", f"source_type={args.source_type}",
                    f"frames={len(frames)}", f"wall_fps={wall_fps:.9f}",
                    f"frame_p95_ms={result['frame_total_ms']['p95']:.9f}",
                    f"energy_per_frame_j={energy_j:.9f}",
                    f"failures={';'.join(failures)}")) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
