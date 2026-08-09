#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = ("baseline", "A", "B")


def read_registry(path: Path):
    runs = {name: [] for name in VARIANTS}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            variant, repeat, run_dir = line.strip().split(",", 2)
            runs[variant].append((int(repeat), Path(run_dir)))
    for variant in VARIANTS:
        if len(runs[variant]) != 3:
            raise ValueError(f"{path}: expected three {variant} runs")
    return runs


def aggregate(registry: Path):
    result = {}
    for variant, runs in read_registry(registry).items():
        summaries = []
        validations = []
        for _, run_dir in sorted(runs):
            summaries.append(json.loads(
                (run_dir / "app_output" / "summary.json").read_text(encoding="utf-8")))
            validations.append(json.loads(
                (run_dir / "validation.json").read_text(encoding="utf-8")))
        if any(item["result"] != "PASS" for item in validations):
            raise ValueError(f"{variant}: validation failure")
        metric = lambda fn: statistics.mean(fn(item) for item in summaries)
        result[variant] = {
            "runs": [str(path) for _, path in sorted(runs)],
            "pipeline_wall_fps": metric(lambda x: x["pipeline_wall_fps"]),
            "effective_fps": metric(lambda x: x["effective_fps"]),
            "e2e_mean_ms": metric(lambda x: x["timings_ms"]["end_to_end"]["mean"]),
            "e2e_p95_ms": metric(lambda x: x["timings_ms"]["end_to_end"]["p95"]),
            "e2e_p99_ms": metric(lambda x: x["timings_ms"]["end_to_end"]["p99"]),
            "postprocess_mean_ms": metric(lambda x: x["timings_ms"]["postprocess"]["mean"]),
            "gpu_postprocess_cuda_mean_ms": metric(
                lambda x: x["timings_ms"]["gpu_postprocess_cuda"]["mean"]),
            "count_sync_host_mean_ms": metric(
                lambda x: x["timings_ms"]["count_sync_host"]["mean"]),
            "candidate_sync_host_mean_ms": metric(
                lambda x: x["timings_ms"]["candidate_sync_host"]["mean"]),
            "candidate_count_mean": metric(
                lambda x: x["transfer"]["candidate_count"]["mean"]),
            "d2h_bytes_mean": metric(
                lambda x: x["transfer"]["d2h_bytes_per_frame"]["mean"]),
        }
    return result


def percent_change(value, baseline):
    return (value / baseline - 1.0) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-registry", type=Path, required=True)
    parser.add_argument("--camera-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "file": aggregate(args.file_registry),
        "camera": aggregate(args.camera_registry),
    }
    decisions = {}
    for variant in ("A", "B"):
        base_file = sources["file"]["baseline"]
        item_file = sources["file"][variant]
        base_camera = sources["camera"]["baseline"]
        item_camera = sources["camera"][variant]
        fps_gain = percent_change(
            item_file["pipeline_wall_fps"], base_file["pipeline_wall_fps"])
        mean_change = percent_change(
            item_file["e2e_mean_ms"], base_file["e2e_mean_ms"])
        file_p95_change = percent_change(
            item_file["e2e_p95_ms"], base_file["e2e_p95_ms"])
        camera_p95_change = percent_change(
            item_camera["e2e_p95_ms"], base_camera["e2e_p95_ms"])
        bytes_reduction = 1.0 - item_file["d2h_bytes_mean"] / 235200.0
        decisions[variant] = {
            "fps_gain_percent": fps_gain,
            "e2e_mean_change_percent": mean_change,
            "file_p95_change_percent": file_p95_change,
            "camera_p95_change_percent": camera_p95_change,
            "d2h_reduction_percent": bytes_reduction * 100.0,
            "runtime_gate_pass": (
                (fps_gain >= 3.0 or mean_change <= -3.0)
                and file_p95_change <= 5.0
                and camera_p95_change <= 5.0
                and bytes_reduction >= 0.80),
        }
    result = {
        "result": "PASS",
        "file_registry": str(args.file_registry),
        "camera_registry": str(args.camera_registry),
        "sources": sources,
        "decisions": decisions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "variant", "pipeline_wall_fps", "e2e_mean_ms",
                         "e2e_p95_ms", "e2e_p99_ms", "postprocess_mean_ms",
                         "gpu_postprocess_cuda_mean_ms", "d2h_bytes_mean"])
        for source, variants in sources.items():
            for variant, metrics in variants.items():
                writer.writerow([source, variant] + [metrics[key] for key in (
                    "pipeline_wall_fps", "e2e_mean_ms", "e2e_p95_ms", "e2e_p99_ms",
                    "postprocess_mean_ms", "gpu_postprocess_cuda_mean_ms",
                    "d2h_bytes_mean")])
    lines = ["# Exp15 three-run comparison", ""]
    for source, variants in sources.items():
        lines.extend([f"## {source}", "",
                      "| Variant | Wall FPS | E2E mean ms | E2E P95 ms | D2H B/frame |",
                      "|---|---:|---:|---:|---:|"])
        for variant, metrics in variants.items():
            lines.append(
                f"| {variant} | {metrics['pipeline_wall_fps']:.3f} | "
                f"{metrics['e2e_mean_ms']:.3f} | {metrics['e2e_p95_ms']:.3f} | "
                f"{metrics['d2h_bytes_mean']:.3f} |")
        lines.append("")
    for variant, decision in decisions.items():
        lines.append(
            f"- {variant}: runtime_gate_pass={decision['runtime_gate_pass']}; "
            f"FPS {decision['fps_gain_percent']:+.3f}%, file mean "
            f"{decision['e2e_mean_change_percent']:+.3f}%, file P95 "
            f"{decision['file_p95_change_percent']:+.3f}%, camera P95 "
            f"{decision['camera_p95_change_percent']:+.3f}%, D2H reduction "
            f"{decision['d2h_reduction_percent']:.3f}%.")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decisions, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
