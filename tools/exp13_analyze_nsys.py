#!/usr/bin/env python3
"""Summarize Exp13 NVTX, CUDA memcpy, and CUDA kernel timeline evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable


USER_RANGES = (
    "frame_total",
    "capture",
    "h2d",
    "preprocess_kernel",
    "preprocess_sync",
    "tensorrt_enqueue",
    "d2h",
    "inference_sync",
    "decode",
    "nms",
    "output",
)


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_ns(durations: Iterable[int]) -> dict[str, float | int]:
    values = sorted(value / 1_000_000.0 for value in durations)
    if not values:
        return {"instances": 0}
    return {
        "instances": len(values),
        "total_ms": sum(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def merge_duration(intervals: Iterable[tuple[int, int]]) -> int:
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


def intersection_duration(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    right_values = list(right)
    intersections = []
    for left_start, left_end in left:
        for right_start, right_end in right_values:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end > start:
                intersections.append((start, end))
    return merge_duration(intersections)


def clipped(
    rows: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> list[tuple[int, int]]:
    return [
        (max(start, start_ns), min(end, end_ns))
        for start, end in rows
        if end > start_ns and start < end_ns
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--source-type", choices=("file", "camera"), required=True)
    args = parser.parse_args()
    if args.expected_frames <= 0 or args.warmup < 0:
        raise ValueError("invalid expected frame or warmup count")

    database = args.sqlite.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, name in nvtx_rows:
        by_name[name].append((start, end))
    frame_ranges = sorted(by_name["frame_total"])
    if len(frame_ranges) != args.expected_frames:
        raise RuntimeError(
            f"frame_total count {len(frame_ranges)} != {args.expected_frames}"
        )
    frame_start_ns = frame_ranges[0][0]
    frame_end_ns = frame_ranges[-1][1]
    frame_span_ns = frame_end_ns - frame_start_ns

    expected_total = {
        "frame_total": args.expected_frames,
        "capture": args.expected_frames,
        "decode": args.expected_frames,
        "nms": args.expected_frames,
        "output": args.expected_frames,
        "h2d": args.expected_frames + args.warmup,
        "preprocess_kernel": args.expected_frames + args.warmup,
        "preprocess_sync": args.expected_frames + args.warmup,
        "tensorrt_enqueue": args.expected_frames + args.warmup,
        "d2h": args.expected_frames + args.warmup,
        "inference_sync": args.expected_frames + args.warmup,
    }
    for name, expected in expected_total.items():
        if len(by_name[name]) != expected:
            raise RuntimeError(f"NVTX {name} count {len(by_name[name])} != {expected}")

    nvtx_summary = []
    for name in USER_RANGES:
        formal = [
            (start, end)
            for start, end in by_name[name]
            if start >= frame_start_ns and end <= frame_end_ns
        ]
        row: dict[str, object] = {
            "range": name,
            "overall_instances": len(by_name[name]),
        }
        row.update(summarize_ns(end - start for start, end in formal))
        if name != "frame_total" and frame_span_ns > 0:
            row["share_of_frame_span_percent"] = (
                float(row.get("total_ms", 0.0)) * 1_000_000.0 / frame_span_ns * 100.0
            )
        nvtx_summary.append(row)

    copy_labels = {
        row[0]: row[2] or row[1]
        for row in connection.execute("SELECT id, name, label FROM ENUM_CUDA_MEMCPY_OPER")
    }
    memcpy_rows = list(
        connection.execute(
            """
            SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE end > ? AND start < ?
            """,
            (frame_start_ns, frame_end_ns),
        )
    )
    memcpy_groups: dict[tuple[object, ...], list[tuple[int, int, int]]] = defaultdict(list)
    for start, end, byte_count, copy_kind, src_kind, dst_kind, stream_id in memcpy_rows:
        key = (
            copy_kind,
            copy_labels.get(copy_kind, f"copy_kind_{copy_kind}"),
            byte_count,
            src_kind,
            dst_kind,
            stream_id,
        )
        memcpy_groups[key].append((start, end, byte_count))
    memcpy_summary = []
    for key, events in sorted(memcpy_groups.items(), key=lambda item: str(item[0])):
        copy_kind, label, byte_count, src_kind, dst_kind, stream_id = key
        row = {
            "copy_kind": copy_kind,
            "label": label,
            "bytes_per_copy": byte_count,
            "src_kind": src_kind,
            "dst_kind": dst_kind,
            "stream_id": stream_id,
            "total_bytes": sum(event[2] for event in events),
            "bytes_per_frame": sum(event[2] for event in events) / args.expected_frames,
        }
        row.update(summarize_ns(end - start for start, end, _ in events))
        memcpy_summary.append(row)

    kernel_rows = list(
        connection.execute(
            """
            SELECT k.start, k.end, COALESCE(short_names.value, demangled.value), k.streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            LEFT JOIN StringIds AS short_names ON k.shortName = short_names.id
            LEFT JOIN StringIds AS demangled ON k.demangledName = demangled.id
            WHERE k.end > ? AND k.start < ?
            """,
            (frame_start_ns, frame_end_ns),
        )
    )
    kernel_groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end, name, stream_id in kernel_rows:
        kernel_groups[(name or "unknown", stream_id)].append((start, end))
    kernel_summary = []
    for (name, stream_id), events in kernel_groups.items():
        row = {"kernel": name, "stream_id": stream_id}
        row.update(summarize_ns(end - start for start, end in events))
        kernel_summary.append(row)
    kernel_summary.sort(key=lambda row: float(row.get("total_ms", 0.0)), reverse=True)

    kernel_intervals = clipped(
        ((start, end) for start, end, _, _ in kernel_rows),
        frame_start_ns,
        frame_end_ns,
    )
    memcpy_intervals = clipped(
        ((start, end) for start, end, *_ in memcpy_rows),
        frame_start_ns,
        frame_end_ns,
    )
    gpu_intervals = kernel_intervals + memcpy_intervals
    gpu_busy_ns = merge_duration(gpu_intervals)
    overlap_ns = intersection_duration(kernel_intervals, memcpy_intervals)

    summary = {
        "result": "PASS",
        "source_type": args.source_type,
        "sqlite": str(database),
        "expected_frames": args.expected_frames,
        "warmup": args.warmup,
        "frame_span_ms": frame_span_ns / 1_000_000.0,
        "wall_throughput_fps": args.expected_frames / (frame_span_ns / 1_000_000_000.0),
        "nvtx": {row["range"]: row for row in nvtx_summary},
        "cuda": {
            "kernel_instances": len(kernel_rows),
            "memcpy_instances": len(memcpy_rows),
            "gpu_busy_ms": gpu_busy_ns / 1_000_000.0,
            "gpu_busy_ratio_percent": gpu_busy_ns / frame_span_ns * 100.0,
            "gpu_idle_ratio_percent": (frame_span_ns - gpu_busy_ns) / frame_span_ns * 100.0,
            "kernel_memcpy_overlap_ms": overlap_ns / 1_000_000.0,
            "kernel_memcpy_overlap_ratio_percent": (
                overlap_ns / frame_span_ns * 100.0
            ),
        },
        "notes": [
            "GPU busy is the union of CUDA kernel and memcpy activity clipped to the formal frame window.",
            "Nsight profiler overhead means this wall throughput is diagnostic, not the final benchmark value.",
            "Nsight Systems alone cannot prove kernel-level compute-bound versus memory-bound behavior.",
        ],
    }
    write_csv(output_dir / "nvtx_summary.csv", nvtx_summary)
    write_csv(output_dir / "memcpy_summary.csv", memcpy_summary)
    write_csv(output_dir / "cuda_kernel_summary.csv", kernel_summary)
    (output_dir / "timeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
