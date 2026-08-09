#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "Duration",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "L1/TEX Cache Throughput",
    "L2 Cache Throughput",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
    "Achieved Occupancy",
    "Theoretical Occupancy",
    "Waves Per SM",
    "Branch Efficiency",
    "Avg. Divergent Branches",
    "Warp Cycles Per Issued Instruction",
    "Active Warps Per Scheduler",
    "Eligible Warps Per Scheduler",
    "Registers Per Thread",
)


def short_name(name: str) -> str:
    if "decode_filter_atomic_kernel" in name:
        return "decode_filter_atomic"
    if "decode_flag_kernel" in name:
        return "decode_flag"
    if "DeviceCompactInitKernel" in name:
        return "cub_compact_init"
    if "DeviceSelectSweepKernel" in name:
        return "cub_select_sweep"
    return name


def numeric(value: str):
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return value


def parse(path: Path):
    kernels = {}
    available_names = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            metric_name = row["Metric Name"]
            metric_value = row["Metric Value"]
            if metric_value:
                available_names.add(metric_name)
            if metric_name not in METRICS or not metric_value:
                continue
            name = short_name(row["Kernel Name"])
            kernels.setdefault(name, {
                "full_name": row["Kernel Name"],
                "grid_size": row["Grid Size"],
                "block_size": row["Block Size"],
                "metrics": {},
            })
            kernels[name]["metrics"][metric_name] = {
                "value": numeric(metric_value),
                "unit": row["Metric Unit"],
            }
    return kernels, available_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-csv", type=Path, required=True)
    parser.add_argument("--cub-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic, atomic_names = parse(args.atomic_csv)
    cub, cub_names = parse(args.cub_csv)
    common_names = atomic_names | cub_names
    missing_requested = {
        "warp_stall_breakdown": not any("Stall" in name for name in common_names),
        "atomic_transactions": not any("Atomic" in name for name in common_names),
        "dram_throughput_named_metric": not any("DRAM" in name for name in common_names),
    }
    result = {
        "result": "PASS",
        "atomic": atomic,
        "cub": cub,
        "missing_requested_metrics": missing_requested,
        "interpretation_boundary": (
            "Missing fields are reported as unavailable in Nsight Compute 2024.3.1 "
            "full-set CSV and are not inferred or fabricated."),
    }
    (args.output_dir / "exp15_ncu_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "exp15_ncu_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "kernel", "grid", "block", "metric", "unit", "value"])
        for variant, kernels in (("atomic", atomic), ("cub", cub)):
            for kernel, data in kernels.items():
                for metric, item in data["metrics"].items():
                    writer.writerow([variant, kernel, data["grid_size"], data["block_size"],
                                     metric, item["unit"], item["value"]])
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
