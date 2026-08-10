#!/usr/bin/env python3
"""Analyze the Exp18 CUDA Graph decision boundary from an Nsight SQLite export.

The report deliberately keeps host API duration, GPU activity, and estimated
critical-path savings in separate fields.  It does not treat an NVTX range
around enqueueV3 as GPU execution time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_RANGES = (
    "frame_total",
    "capture",
    "h2d",
    "preprocess_kernel",
    "preprocess_sync",
    "tensorrt_enqueue",
    "gpu_decode_filter_compaction",
    "candidate_count_d2h",
    "candidate_count_sync",
    "candidate_payload_sync",
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"instances": 0}
    return {
        "instances": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def merged_duration(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def clipped(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> list[tuple[int, int]]:
    return [
        (max(start, start_ns), min(end, end_ns))
        for start, end in intervals
        if end > start_ns and start < end_ns
    ]


def inside(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end in intervals
        if start >= start_ns and end <= end_ns
    ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--app-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    database = args.sqlite.resolve()
    app_summary_path = args.app_summary.resolve()
    if not database.is_file() or not app_summary_path.is_file():
        raise FileNotFoundError("Nsight SQLite or app summary is missing")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app_summary = json.loads(app_summary_path.read_text(encoding="utf-8"))
    if app_summary.get("processed_frames") != args.expected_frames:
        raise RuntimeError("app summary frame count does not match the profile")

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    nvtx_rows = list(
        connection.execute(
            """
            SELECT n.start, n.end, COALESCE(n.text, s.value)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON n.textId = s.id
            WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IS NOT NULL
            """
        )
    )
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, name in nvtx_rows:
        ranges[name].append((start, end))
    frames = sorted(ranges["frame_total"])
    if len(frames) != args.expected_frames:
        raise RuntimeError(
            f"frame_total count {len(frames)} != {args.expected_frames}"
        )
    for name in REQUIRED_RANGES:
        expected = args.expected_frames
        if name in {
            "h2d",
            "preprocess_kernel",
            "preprocess_sync",
            "tensorrt_enqueue",
            "gpu_decode_filter_compaction",
            "candidate_count_d2h",
            "candidate_count_sync",
            "candidate_payload_sync",
        }:
            expected += args.warmup
        if len(ranges[name]) != expected:
            raise RuntimeError(f"NVTX {name} count {len(ranges[name])} != {expected}")

    runtime_rows = list(
        connection.execute(
            """
            SELECT r.start, r.end, COALESCE(s.value, 'unknown')
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            LEFT JOIN StringIds AS s ON r.nameId = s.id
            WHERE r.end > ? AND r.start < ?
            """,
            (frames[0][0], frames[-1][1]),
        )
    )
    kernel_rows = list(
        connection.execute(
            """
            SELECT k.start, k.end,
                   COALESCE(short_names.value, demangled.value, 'unknown')
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            LEFT JOIN StringIds AS short_names ON k.shortName = short_names.id
            LEFT JOIN StringIds AS demangled ON k.demangledName = demangled.id
            WHERE k.end > ? AND k.start < ?
            """,
            (frames[0][0], frames[-1][1]),
        )
    )
    memcpy_rows = list(
        connection.execute(
            """
            SELECT start, end, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE end > ? AND start < ?
            """,
            (frames[0][0], frames[-1][1]),
        )
    )

    formal_ranges = {
        name: sorted(inside(values, frames[0][0], frames[-1][1]))
        for name, values in ranges.items()
    }
    launch_runtime = [
        (start, end, name)
        for start, end, name in runtime_rows
        if name.startswith("cudaLaunch")
    ]
    gpu_activity = [
        (start, end, f"kernel:{name}") for start, end, name in kernel_rows
    ] + [
        (start, end, f"memcpy:{byte_count}")
        for start, end, byte_count in memcpy_rows
    ]

    frame_rows: list[dict[str, object]] = []
    for index, (frame_start, frame_end) in enumerate(frames):
        preprocess = inside(
            formal_ranges["preprocess_kernel"], frame_start, frame_end
        )
        postprocess = inside(
            formal_ranges["gpu_decode_filter_compaction"], frame_start, frame_end
        )
        enqueue = inside(
            formal_ranges["tensorrt_enqueue"], frame_start, frame_end
        )
        count_sync = inside(
            formal_ranges["candidate_count_sync"], frame_start, frame_end
        )
        payload_sync = inside(
            formal_ranges["candidate_payload_sync"], frame_start, frame_end
        )
        if len(preprocess) != 1 or len(postprocess) != 1 or len(enqueue) != 1:
            raise RuntimeError(f"ambiguous graph boundary in frame {index}")

        # The permitted Graph starts at the preprocess launch and ends after
        # GPU decode/CUB submission.  H2D and both D2H waits remain outside.
        host_boundary_start = preprocess[0][0]
        host_boundary_end = postprocess[0][1]
        launch_events = [
            (start, end)
            for start, end, _ in launch_runtime
            if start >= host_boundary_start and end <= host_boundary_end
        ]

        device_events = sorted(
            (start, end, name)
            for start, end, name in gpu_activity
            if start >= frame_start and end <= frame_end
        )
        preprocess_kernels = [
            (start, end)
            for start, end, name in device_events
            if "fused_preprocess_kernel" in name
        ]
        cub_kernels = [
            (start, end)
            for start, end, name in device_events
            if "DeviceSelect" in name
        ]
        if len(preprocess_kernels) != 1 or not cub_kernels:
            raise RuntimeError(f"missing preprocess/CUB GPU activity in frame {index}")
        device_start = preprocess_kernels[0][0]
        device_end = max(end for _, end in cub_kernels)
        graph_activity = clipped(
            ((start, end) for start, end, _ in device_events),
            device_start,
            device_end,
        )
        device_span_ns = device_end - device_start
        busy_ns = merged_duration(graph_activity)
        gap_ns = device_span_ns - busy_ns
        launch_ns = sum(end - start for start, end in launch_events)
        frame_rows.append(
            {
                "frame_index": index,
                "frame_total_ms": (frame_end - frame_start) / 1e6,
                "enqueue_host_range_ms": (enqueue[0][1] - enqueue[0][0]) / 1e6,
                "launch_api_sum_ms": launch_ns / 1e6,
                "launch_api_calls": len(launch_events),
                "graph_device_span_ms": device_span_ns / 1e6,
                "graph_device_busy_ms": busy_ns / 1e6,
                "graph_device_gap_ms": gap_ns / 1e6,
                "graph_device_gap_percent": gap_ns / device_span_ns * 100.0,
                "count_sync_host_ms": sum(
                    end - start for start, end in count_sync
                ) / 1e6,
                "payload_sync_host_ms": sum(
                    end - start for start, end in payload_sync
                ) / 1e6,
            }
        )

    metrics = {
        key: summarize(float(row[key]) for row in frame_rows)
        for key in (
            "frame_total_ms",
            "enqueue_host_range_ms",
            "launch_api_sum_ms",
            "graph_device_span_ms",
            "graph_device_busy_ms",
            "graph_device_gap_ms",
            "graph_device_gap_percent",
            "count_sync_host_ms",
            "payload_sync_host_ms",
        )
    }
    e2e_mean = float(app_summary["timings_ms"]["end_to_end"]["mean"])
    capture_mean = float(app_summary["timings_ms"]["capture"]["mean"])
    post_capture_mean = e2e_mean - capture_mean
    launch_median = float(metrics["launch_api_sum_ms"]["p50_ms"])
    gap_median = float(metrics["graph_device_gap_ms"]["p50_ms"])
    gap_percent_median = float(metrics["graph_device_gap_percent"]["p50_ms"])
    predicted_upper_bound_percent = gap_median / e2e_mean * 100.0

    gates = {
        "host_launch_ms": launch_median >= 0.50,
        "host_launch_share": launch_median / post_capture_mean >= 0.05,
        "gpu_gap": gap_median >= 0.30 or gap_percent_median >= 5.0,
        "predicted_e2e_upper_bound": predicted_upper_bound_percent >= 3.0,
    }
    result = {
        "result": "PASS",
        "decision": "IMPLEMENT_GRAPH" if all(gates.values()) else "SKIPPED_BY_EVIDENCE",
        "expected_frames": args.expected_frames,
        "app_e2e_mean_ms": e2e_mean,
        "app_capture_mean_ms": capture_mean,
        "post_capture_processing_mean_ms": post_capture_mean,
        "launch_api_share_of_post_capture_percent":
            launch_median / post_capture_mean * 100.0,
        "predicted_e2e_upper_bound_percent": predicted_upper_bound_percent,
        "gates": gates,
        "metrics": metrics,
        "boundaries": {
            "host_api_duration": "CUDA launch API self time inside the permitted submission boundary",
            "gpu_activity": "union of kernels/memcpy from fused preprocess start through final CUB kernel end",
            "critical_path_upper_bound": "median GPU gap divided by profiled app E2E mean; diagnostic upper bound only",
        },
        "notes": [
            "enqueue_host_range_ms is reported separately and is never treated as GPU activity.",
            "Nsight instrumentation inflates host API duration; adoption still requires paired no-profiler verification.",
            "The graph_device_gap estimate includes the existing preprocess synchronization boundary.",
            "H2D, count/payload D2H, CPU candidate scan, and CPU NMS remain outside the proposed Graph.",
        ],
    }
    write_rows(output_dir / "frame_decision_metrics.csv", frame_rows)
    (output_dir / "decision_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
