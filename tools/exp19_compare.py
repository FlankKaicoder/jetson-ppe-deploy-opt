#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def improvement(final: float, baseline: float, higher_is_better: bool) -> float:
    return ((final / baseline - 1.0) if higher_is_better else
            (1.0 - final / baseline)) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sources", nargs="+", choices=("file", "camera"),
        default=("file", "camera"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.registry.open(newline="", encoding="utf-8")))
    pairs = {}
    for row in rows:
        summary = json.loads((Path(row["run_dir"]) / "exp19_summary.json").read_text(encoding="utf-8"))
        if summary["result"] != "PASS":
            raise ValueError(f"failed run: {row['run_dir']}")
        pairs.setdefault((row["source"], int(row["pair"])), {})[row["variant"]] = summary
    output_rows = []
    for (source, pair), variants in sorted(pairs.items()):
        if set(variants) != {"v0", "vfinal"}:
            raise ValueError(f"incomplete pair: {source} {pair}")
        base, final = variants["v0"], variants["vfinal"]
        output_rows.append({
            "source": source,
            "pair": pair,
            "v0_wall_fps": base["pipeline_wall_fps"],
            "vfinal_wall_fps": final["pipeline_wall_fps"],
            "v0_capture_mean_ms": base["capture_wait_ms"]["mean"],
            "vfinal_capture_mean_ms": final["capture_wait_ms"]["mean"],
            "v0_post_capture_mean_ms": base["post_capture_processing_ms"]["mean"],
            "vfinal_post_capture_mean_ms": final["post_capture_processing_ms"]["mean"],
            "v0_frame_p95_ms": base["frame_total_ms"]["p95"],
            "vfinal_frame_p95_ms": final["frame_total_ms"]["p95"],
            "v0_frame_p99_ms": base["frame_total_ms"]["p99"],
            "vfinal_frame_p99_ms": final["frame_total_ms"]["p99"],
            "v0_vdd_in_mean_mw": base["resources"]["power_mw"]["VDD_IN"]["mean"],
            "vfinal_vdd_in_mean_mw": final["resources"]["power_mw"]["VDD_IN"]["mean"],
            "v0_energy_per_frame_j": base["energy_per_frame_j"],
            "vfinal_energy_per_frame_j": final["energy_per_frame_j"],
            "v0_d2h_bytes": base["d2h_bytes_per_frame"],
            "vfinal_d2h_bytes": final["d2h_bytes_per_frame"],
            "wall_fps_improvement_percent": improvement(final["pipeline_wall_fps"], base["pipeline_wall_fps"], True),
            "post_capture_mean_improvement_percent": improvement(final["post_capture_processing_ms"]["mean"], base["post_capture_processing_ms"]["mean"], False),
            "frame_p95_improvement_percent": improvement(final["frame_total_ms"]["p95"], base["frame_total_ms"]["p95"], False),
            "frame_p99_improvement_percent": improvement(final["frame_total_ms"]["p99"], base["frame_total_ms"]["p99"], False),
            "energy_improvement_percent": improvement(final["energy_per_frame_j"], base["energy_per_frame_j"], False),
        })
    improvement_keys = (
        "wall_fps_improvement_percent",
        "post_capture_mean_improvement_percent",
        "frame_p95_improvement_percent",
        "frame_p99_improvement_percent",
        "energy_improvement_percent",
    )
    aggregate = {}
    for source in args.sources:
        selected = [row for row in output_rows if row["source"] == source]
        if len(selected) != 3:
            raise ValueError(f"expected three {source} pairs")
        aggregate[source] = {
            key: statistics.median(row[key] for row in selected)
            for key in improvement_keys
        }
        aggregate[source]["favorable_wall_pairs"] = sum(
            row["wall_fps_improvement_percent"] > 0 for row in selected)
        aggregate[source]["favorable_post_capture_pairs"] = sum(
            row["post_capture_mean_improvement_percent"] > 0 for row in selected)
    camera_p95_regression_ok = "camera" not in args.sources or all(
        row["frame_p95_improvement_percent"] >= -5.0 and
        row["frame_p99_improvement_percent"] >= -5.0
        for row in output_rows if row["source"] == "camera")
    result = {
        "result": "PASS" if camera_p95_regression_ok else "FAIL",
        "camera_tail_regression_gate_pass": camera_p95_regression_ok,
        "pairs": output_rows,
        "median_improvements": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if camera_p95_regression_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
