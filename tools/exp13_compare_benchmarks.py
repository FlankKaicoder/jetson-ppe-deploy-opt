#!/usr/bin/env python3
"""Aggregate unprofiled Exp13 runs and keep wall/app timing scopes separate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cv_percent(values: list[float]) -> float:
    if len(values) < 2 or statistics.mean(values) == 0:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--source-type", choices=("file", "camera"), required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dirs) < 3:
        raise ValueError("at least three runs are required")

    rows = []
    for raw_dir in args.run_dirs:
        run_dir = raw_dir.resolve()
        app_summary = json.loads(
            (run_dir / "app_output" / "summary.json").read_text(encoding="utf-8")
        )
        start_ns = int((run_dir / "wall_start_ns.txt").read_text().strip())
        end_ns = int((run_dir / "wall_end_ns.txt").read_text().strip())
        if app_summary["result"] != "PASS":
            raise RuntimeError(f"non-PASS app summary: {run_dir}")
        if app_summary["source_type"] != args.source_type:
            raise RuntimeError(f"source mismatch: {run_dir}")
        if int(app_summary["processed_frames"]) != args.expected_frames:
            raise RuntimeError(f"frame count mismatch: {run_dir}")
        wall_seconds = (end_ns - start_ns) / 1_000_000_000.0
        rows.append(
            {
                "run_dir": str(run_dir),
                "processed_frames": int(app_summary["processed_frames"]),
                "total_detections": int(app_summary["total_detections"]),
                "wall_seconds": wall_seconds,
                "wall_fps": args.expected_frames / wall_seconds,
                "app_effective_fps": float(app_summary["effective_fps"]),
                "e2e_mean_ms": float(app_summary["timings_ms"]["end_to_end"]["mean"]),
                "e2e_p95_ms": float(app_summary["timings_ms"]["end_to_end"]["p95"]),
                "e2e_p99_ms": float(app_summary["timings_ms"]["end_to_end"]["p99"]),
                "detections_sha256": sha256(run_dir / "app_output" / "detections.csv"),
            }
        )
    if args.source_type == "file" and len({row["detections_sha256"] for row in rows}) != 1:
        raise RuntimeError("file benchmark detections are not deterministic")

    numeric_keys = (
        "wall_fps",
        "app_effective_fps",
        "e2e_mean_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
    )
    aggregate = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        aggregate[key] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "cv_percent": cv_percent(values),
        }
    result = {
        "result": "PASS",
        "source_type": args.source_type,
        "expected_frames": args.expected_frames,
        "runs": rows,
        "aggregate": aggregate,
        "notes": [
            "wall_fps includes application startup, frame processing, output work, and shutdown.",
            "app_effective_fps uses the existing Exp11 end-to-end timing scope and excludes output annotation/CSV work.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
