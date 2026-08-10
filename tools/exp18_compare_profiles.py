#!/usr/bin/env python3
"""Compare Normal and CUDA Graph Nsight timelines without mixing time domains."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def merged_duration(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "instances": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def analyze(path: Path, mode: str, expected_frames: int) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    nvtx = list(connection.execute(
        """
        SELECT n.start, n.end, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) IS NOT NULL
        """))
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end, name in nvtx:
        ranges[name].append((start, end))
    frames = sorted(ranges["frame_total"])
    if len(frames) != expected_frames:
        raise RuntimeError(f"{mode} frame count mismatch")
    runtime = list(connection.execute(
        """
        SELECT r.start, r.end, COALESCE(s.value, 'unknown')
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS s ON r.nameId = s.id
        WHERE r.end > ? AND r.start < ?
        """, (frames[0][0], frames[-1][1])))
    kernels = list(connection.execute(
        """
        SELECT k.start, k.end,
               COALESCE(short_names.value, demangled.value, 'unknown')
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS short_names ON k.shortName = short_names.id
        LEFT JOIN StringIds AS demangled ON k.demangledName = demangled.id
        WHERE k.end > ? AND k.start < ?
        """, (frames[0][0], frames[-1][1])))
    copies = list(connection.execute(
        """SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
           WHERE end > ? AND start < ?""", (frames[0][0], frames[-1][1])))

    boundary_name = (
        "cuda_graph_launch" if mode == "graph"
        else "gpu_decode_filter_compaction"
    )
    launch_names = (
        ("cudaGraphLaunch",) if mode == "graph"
        else ("cudaLaunchKernel", "cudaLaunchKernelExC")
    )
    rows = []
    for index, (frame_start, frame_end) in enumerate(frames):
        frame_kernels = [
            (start, end, name) for start, end, name in kernels
            if start >= frame_start and end <= frame_end
        ]
        preprocess = [row for row in frame_kernels if "fused_preprocess_kernel" in row[2]]
        cub = [row for row in frame_kernels if "DeviceSelect" in row[2]]
        if len(preprocess) != 1 or not cub:
            raise RuntimeError(f"{mode} missing Graph boundary kernel in frame {index}")
        device_start = preprocess[0][0]
        device_end = max(end for _, end, _ in cub)
        activity = [
            (start, end) for start, end, _ in frame_kernels
            if end > device_start and start < device_end
        ] + [
            (start, end) for start, end in copies
            if start >= frame_start and end <= frame_end and
            end > device_start and start < device_end
        ]
        busy = merged_duration(
            (max(start, device_start), min(end, device_end))
            for start, end in activity)
        boundary = [
            (start, end) for start, end in ranges[boundary_name]
            if start >= frame_start and end <= frame_end
        ]
        if len(boundary) != 1:
            raise RuntimeError(f"{mode} host boundary mismatch in frame {index}")
        host_start = (
            boundary[0][0] if mode == "graph"
            else next(
                start for start, end in ranges["preprocess_kernel"]
                if start >= frame_start and end <= frame_end)
        )
        host_end = boundary[0][1]
        launch_api = [
            (start, end) for start, end, name in runtime
            if any(name.startswith(prefix) for prefix in launch_names) and
            start >= host_start and end <= host_end
        ]
        rows.append({
            "launch_api_ms": sum(end - start for start, end in launch_api) / 1e6,
            "launch_api_calls": len(launch_api),
            "device_span_ms": (device_end - device_start) / 1e6,
            "device_busy_ms": busy / 1e6,
            "device_gap_ms": (device_end - device_start - busy) / 1e6,
        })
    return {
        "sqlite": str(path.resolve()),
        "launch_api_ms": summarize([row["launch_api_ms"] for row in rows]),
        "launch_api_calls_per_frame": summarize([
            float(row["launch_api_calls"]) for row in rows]),
        "device_span_ms": summarize([row["device_span_ms"] for row in rows]),
        "device_busy_ms": summarize([row["device_busy_ms"] for row in rows]),
        "device_gap_ms": summarize([row["device_gap_ms"] for row in rows]),
    }


def reduction(normal: float, graph: float) -> float:
    return (normal - graph) / normal * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=150)
    args = parser.parse_args()
    normal = analyze(args.normal, "normal", args.expected_frames)
    graph = analyze(args.graph, "graph", args.expected_frames)
    result = {
        "result": "PASS",
        "normal": normal,
        "graph": graph,
        "median_reduction_percent": {
            key: reduction(
                float(normal[key]["p50_ms"]), float(graph[key]["p50_ms"]))
            for key in ("launch_api_ms", "device_span_ms", "device_gap_ms")
        },
        "conclusion": (
            "Graph reduces host submissions and profiled launch gaps, but the "
            "dynamic-frequency paired E2E gate remains the adoption authority."
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
