#!/usr/bin/env python3
"""Deterministic image+class+IoU Hungarian matching for Exp16 revalidation.

This comparator intentionally does not use CSV row position as detection identity.
It preserves the historical Exp16 comparator and writes new Gate-local evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    source_row: int
    image: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    candidate_index: int = -1


def iou(left: Detection, right: Detection) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def hungarian_min_cost(cost: list[list[float]]) -> list[int]:
    """Return row->column assignment for a square cost matrix (O(n^3))."""
    size = len(cost)
    if size == 0:
        return []
    if any(len(row) != size for row in cost):
        raise ValueError("Hungarian input must be square")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def match_group(
    left: list[Detection], right: list[Detection], threshold: float
) -> tuple[list[tuple[Detection, Detection, float]], list[Detection], list[Detection]]:
    left = sorted(left, key=lambda item: item.source_row)
    right = sorted(right, key=lambda item: item.source_row)
    size = max(len(left), len(right))
    if size == 0:
        return [], [], []
    overlaps = [[iou(a, b) for b in right] for a in left]
    cardinality_weight = 1000.0
    cost = [[0.0 for _ in range(size)] for _ in range(size)]
    for left_index in range(len(left)):
        for right_index in range(len(right)):
            overlap = overlaps[left_index][right_index]
            if overlap >= threshold:
                cost[left_index][right_index] = -(cardinality_weight + overlap)
    assignment = hungarian_min_cost(cost)
    matches: list[tuple[Detection, Detection, float]] = []
    used_right: set[int] = set()
    unmatched_left: list[Detection] = []
    for left_index, column in enumerate(assignment[: len(left)]):
        if column < len(right) and overlaps[left_index][column] >= threshold:
            matches.append((left[left_index], right[column], overlaps[left_index][column]))
            used_right.add(column)
        else:
            unmatched_left.append(left[left_index])
    unmatched_right = [item for index, item in enumerate(right) if index not in used_right]
    return matches, unmatched_left, unmatched_right


def load_detections(path: Path) -> list[Detection]:
    result: list[Detection] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"class_id", "confidence", "x1", "y1", "x2", "y2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required detection columns")
        image_key = "image" if "image" in reader.fieldnames else "frame_index"
        if image_key not in reader.fieldnames:
            raise ValueError(f"{path} requires image or frame_index")
        for row_number, row in enumerate(reader, start=2):
            result.append(
                Detection(
                    source_row=row_number,
                    image=row[image_key],
                    class_id=int(row["class_id"]),
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    candidate_index=int(row.get("candidate_index", -1)),
                )
            )
    return result


def grouped(items: Iterable[Detection]) -> dict[tuple[str, int], list[Detection]]:
    result: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for item in items:
        result[(item.image, item.class_id)].append(item)
    return result


def write_detection(writer: csv.writer, side: str, item: Detection) -> None:
    writer.writerow(
        [side, item.source_row, item.image, item.class_id, item.candidate_index, item.confidence,
         item.x1, item.y1, item.x2, item.y2]
    )


def run(left_path: Path, right_path: Path, output_dir: Path, threshold: float) -> dict:
    left_groups = grouped(load_detections(left_path))
    right_groups = grouped(load_detections(right_path))
    matches = []
    unmatched_left = []
    unmatched_right = []
    for key in sorted(set(left_groups) | set(right_groups)):
        group_matches, group_left, group_right = match_group(
            left_groups.get(key, []), right_groups.get(key, []), threshold
        )
        matches.extend(group_matches)
        unmatched_left.extend(group_left)
        unmatched_right.extend(group_right)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "matched.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "class_id", "left_source_row", "right_source_row", "iou",
             "left_candidate_index", "right_candidate_index", "candidate_index_equal",
             "left_confidence", "right_confidence", "confidence_delta",
             "left_x1", "left_y1", "left_x2", "left_y2",
             "right_x1", "right_y1", "right_x2", "right_y2"]
        )
        for left, right, overlap in matches:
            writer.writerow(
                [left.image, left.class_id, left.source_row, right.source_row, overlap,
                 left.candidate_index, right.candidate_index,
                 int(left.candidate_index == right.candidate_index),
                 left.confidence, right.confidence, right.confidence - left.confidence,
                 left.x1, left.y1, left.x2, left.y2,
                 right.x1, right.y1, right.x2, right.y2]
            )
    for name, side, items in (
        ("unmatched_left.csv", "left", unmatched_left),
        ("unmatched_right.csv", "right", unmatched_right),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["side", "source_row", "image", "class_id", "candidate_index", "confidence",
                 "x1", "y1", "x2", "y2"]
            )
            for item in items:
                write_detection(writer, side, item)
    total = len(matches) + len(unmatched_left) + len(unmatched_right)
    overlaps = sorted(item[2] for item in matches)
    confidence_abs = sorted(abs(right.confidence - left.confidence) for left, right, _ in matches)
    box_abs = [
        abs(right_value - left_value)
        for left, right, _ in matches
        for left_value, right_value in zip(
            (left.x1, left.y1, left.x2, left.y2),
            (right.x1, right.y1, right.x2, right.y2),
        )
    ]

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return math.nan
        position = (len(values) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    summary = {
        "result": "PASS",
        "matching": "image+class+maximum-cardinality maximum-IoU Hungarian",
        "iou_threshold": threshold,
        "left": str(left_path),
        "right": str(right_path),
        "matched": len(matches),
        "unmatched_left": len(unmatched_left),
        "unmatched_right": len(unmatched_right),
        "unmatched_rate": (
            (len(unmatched_left) + len(unmatched_right)) / total if total else 0.0
        ),
        "matched_iou": {
            "mean": sum(overlaps) / len(overlaps) if overlaps else math.nan,
            "p05": percentile(overlaps, 0.05),
            "p50": percentile(overlaps, 0.50),
            "minimum": overlaps[0] if overlaps else math.nan,
        },
        "confidence_abs_delta": {
            "mean": sum(confidence_abs) / len(confidence_abs) if confidence_abs else math.nan,
            "p95": percentile(confidence_abs, 0.95),
            "maximum": confidence_abs[-1] if confidence_abs else math.nan,
        },
        "bbox_coordinate_abs_delta": {
            "p95": percentile(sorted(box_abs), 0.95),
            "maximum": max(box_abs, default=math.nan),
        },
        "candidate_index_agreement": sum(
            left.candidate_index == right.candidate_index for left, right, _ in matches
        ) / len(matches) if matches else math.nan,
        "unmatched_near_confidence_floor_0.01": sum(
            abs(item.confidence - 0.25) <= 0.01
            for item in unmatched_left + unmatched_right
        ),
        "threshold_crossing": "requires raw candidate forensic; not inferred from floor-filtered lists",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def self_test() -> None:
    def detection(row: int, image: str, cls: int, box: tuple[float, float, float, float]) -> Detection:
        return Detection(row, image, cls, 0.5, *box, candidate_index=row)

    left = [
        detection(2, "a", 0, (0, 0, 10, 10)),
        detection(3, "a", 0, (20, 20, 30, 30)),
    ]
    right = [left[1], left[0]]
    matches, missing_left, missing_right = match_group(left, right, 0.5)
    assert [(a.source_row, b.source_row) for a, b, _ in matches] == [(2, 2), (3, 3)]
    assert not missing_left and not missing_right
    below = detection(4, "a", 0, (9, 9, 19, 19))
    matches, missing_left, missing_right = match_group([left[0]], [below], 0.5)
    assert not matches and len(missing_left) == 1 and len(missing_right) == 1
    assert grouped([left[0], detection(5, "a", 1, (0, 0, 10, 10))]).keys() == {
        ("a", 0), ("a", 1)
    }
    print("PASS: deterministic Hungarian matcher self-test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.left is None or args.right is None or args.output_dir is None):
        parser.error("--left, --right, and --output-dir are required unless --self-test")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(run(args.left, args.right, args.output_dir, args.iou_threshold), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
