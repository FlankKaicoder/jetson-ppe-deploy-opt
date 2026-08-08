#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def stats(values: list[float]) -> dict:
    if not values:
        raise RuntimeError("cannot summarize empty values")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.0
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
    ) / denominator


def parse_tegrastats(path: Path) -> dict:
    ram_used = []
    swap_used = []
    gpu_util = []
    cpu_util = []
    cpu_freq = []
    temperatures: dict[str, list[float]] = {}
    powers: dict[str, list[float]] = {name: [] for name in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")}
    valid_lines = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        swap = re.search(r"SWAP (\d+)/(\d+)MB", line)
        gpu = re.search(r"GR3D_FREQ (\d+)%", line)
        if not (ram and swap and gpu):
            continue
        valid_lines += 1
        ram_used.append(float(ram.group(1)))
        swap_used.append(float(swap.group(1)))
        gpu_util.append(float(gpu.group(1)))
        cpu_match = re.search(r"CPU \[([^]]+)\]", line)
        if cpu_match:
            percentages = [float(value) for value in re.findall(r"(\d+)%@", cpu_match.group(1))]
            frequencies = [float(value) for value in re.findall(r"%@([0-9]+)", cpu_match.group(1))]
            if percentages:
                cpu_util.append(statistics.fmean(percentages))
            if frequencies:
                cpu_freq.append(statistics.fmean(frequencies))
        for name, value in re.findall(r"([A-Za-z0-9_]+)@" + NUMBER + r"C", line):
            temperatures.setdefault(name, []).append(float(value))
        for name in powers:
            match = re.search(rf"{name} (\d+)mW/(\d+)mW", line)
            if match:
                powers[name].append(float(match.group(1)))
    if valid_lines == 0:
        raise RuntimeError("no valid tegrastats lines")
    return {
        "valid_samples": valid_lines,
        "ram_used_mb": stats(ram_used),
        "swap_used_mb": stats(swap_used),
        "cpu_util_percent": stats(cpu_util),
        "cpu_frequency_mhz": stats(cpu_freq),
        "gpu_util_percent": stats(gpu_util),
        "temperatures_c": {name: stats(values) for name, values in temperatures.items()},
        "power_mw": {name: stats(values) for name, values in powers.items() if values},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "baseline", "performance", "stability"), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    required = [
        "summary.json", "frames.csv", "detections.csv", "tegrastats.log",
        "process_samples.csv", "monitor_summary.json", "clock_status.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    app = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    monitor = json.loads((run_dir / "monitor_summary.json").read_text(encoding="utf-8"))
    clock = json.loads((run_dir / "clock_status.json").read_text(encoding="utf-8"))
    with (run_dir / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    with (run_dir / "process_samples.csv").open(newline="", encoding="utf-8") as stream:
        process_rows = list(csv.DictReader(stream))
    e2e = [float(row["end_to_end_ms"]) for row in frames]
    e2e_stats = stats(e2e)
    window = max(1, len(e2e) // 10)
    first_p95 = stats(e2e[:window])["p95"]
    last_p95 = stats(e2e[-window:])["p95"]
    p95_degradation_percent = (last_p95 - first_p95) / first_p95 * 100.0
    steady_start_seconds = 60.0 if args.mode == "stability" else (5.0 if args.mode in {"baseline", "performance"} else 2.0)
    steady_process_rows = [
        row for row in process_rows
        if float(row["elapsed_seconds"]) >= steady_start_seconds
    ]
    if len(steady_process_rows) < 2:
        steady_process_rows = process_rows
    rss_mib = [float(row["rss_kb"]) / 1024.0 for row in steady_process_rows]
    elapsed_minutes = [float(row["elapsed_seconds"]) / 60.0 for row in steady_process_rows]
    process_cpu = [float(row["process_cpu_percent"]) for row in steady_process_rows]
    rss_growth = rss_mib[-1] - rss_mib[0] if rss_mib else math.nan
    rss_slope = linear_slope(elapsed_minutes, rss_mib) if rss_mib else math.nan
    resources = parse_tegrastats(run_dir / "tegrastats.log")
    swap_growth_mb = (
        resources["swap_used_mb"]["max"] - resources["swap_used_mb"]["min"]
    )
    thermal_names = [name for name in ("cpu", "gpu", "tj") if name in resources["temperatures_c"]]
    thermal_max = max(resources["temperatures_c"][name]["max"] for name in thermal_names)
    expected_frames = {"smoke": 300, "baseline": 1800, "performance": 1800, "stability": 54000}[args.mode]
    minimum_samples = {"smoke": 5, "baseline": 30, "performance": 50, "stability": 1500}[args.mode]
    failures = []
    if app.get("result") != "PASS" or monitor.get("result") != "PASS":
        failures.append("application_or_monitor_failed")
    if int(app.get("processed_frames", -1)) != expected_frames or len(frames) != expected_frames:
        failures.append("processed_frame_count_mismatch")
    if resources["valid_samples"] < minimum_samples or len(process_rows) < minimum_samples:
        failures.append("insufficient_resource_samples")
    if args.mode in {"performance", "stability"} and not clock.get("jetson_clocks_inferred_locked"):
        failures.append("jetson_clocks_not_locked")
    if args.mode in {"performance", "stability"}:
        if e2e_stats["p95"] > 40.0:
            failures.append("end_to_end_p95_above_40ms")
        if e2e_stats["p99"] > 50.0:
            failures.append("end_to_end_p99_above_50ms")
        if float(app.get("effective_fps", 0.0)) < 29.0:
            failures.append("effective_fps_below_29")
        if thermal_max >= 70.0:
            failures.append("thermal_passive_trip_reached")
    if args.mode == "stability":
        if p95_degradation_percent > 10.0:
            failures.append("tail_latency_degradation_above_10_percent")
        if rss_growth > 64.0:
            failures.append("rss_growth_above_64_mib")
        if rss_slope > 1.0:
            failures.append("rss_slope_above_1_mib_per_min")
        if swap_growth_mb > 0.0:
            failures.append("swap_increased_during_run")
    result = {
        "result": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "failures": failures,
        "processed_frames": len(frames),
        "total_detections": int(app["total_detections"]),
        "effective_fps": float(app["effective_fps"]),
        "end_to_end_ms": e2e_stats,
        "first_10_percent_p95_ms": first_p95,
        "last_10_percent_p95_ms": last_p95,
        "p95_degradation_percent": p95_degradation_percent,
        "monitor_wall_seconds": float(monitor["wall_seconds"]),
        "process_samples": len(process_rows),
        "steady_process_samples": len(steady_process_rows),
        "steady_state_start_seconds": steady_start_seconds,
        "process_cpu_percent": stats(process_cpu),
        "rss_mib": stats(rss_mib),
        "rss_growth_mib": rss_growth,
        "rss_slope_mib_per_min": rss_slope,
        "swap_growth_mb": swap_growth_mb,
        "thermal_max_cpu_gpu_tj_c": thermal_max,
        "resources": resources,
        "clock_status": clock,
    }
    (run_dir / "exp12_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (run_dir / "exp12_summary.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"result={result['result']}\n")
        stream.write(f"mode={args.mode}\n")
        stream.write(f"processed_frames={len(frames)}\n")
        stream.write(f"effective_fps={result['effective_fps']:.9f}\n")
        stream.write(f"end_to_end_p95_ms={e2e_stats['p95']:.9f}\n")
        stream.write(f"end_to_end_p99_ms={e2e_stats['p99']:.9f}\n")
        stream.write(f"thermal_max_cpu_gpu_tj_c={thermal_max:.3f}\n")
        stream.write(f"rss_growth_mib={rss_growth:.6f}\n")
        stream.write(f"rss_slope_mib_per_min={rss_slope:.6f}\n")
        stream.write(f"swap_growth_mb={swap_growth_mb:.6f}\n")
        stream.write(f"failures={';'.join(failures)}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
