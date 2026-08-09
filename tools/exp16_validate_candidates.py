#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--plugin-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(args.summary)

    raw = np.fromfile(args.raw, dtype=np.float32)
    if raw.size != 7 * 8400:
        raise RuntimeError(f"raw element count mismatch: {raw.size}")
    raw = raw.reshape(7, 8400)

    reference = []
    for index in range(8400):
        class_id = 0
        confidence = float(raw[4, index])
        for category in range(1, 3):
            value = float(raw[4 + category, index])
            if value > confidence:
                confidence = value
                class_id = category
        cx, cy, width, height = (float(raw[channel, index]) for channel in range(4))
        if not np.isfinite(confidence) or confidence < args.confidence:
            continue
        if not all(np.isfinite(value) for value in (cx, cy, width, height)):
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        x1 = np.float32(np.clip(cx - 0.5 * width, 0.0, 640.0))
        y1 = np.float32(np.clip(cy - 0.5 * height, 0.0, 640.0))
        x2 = np.float32(np.clip(cx + 0.5 * width, 0.0, 640.0))
        y2 = np.float32(np.clip(cy + 0.5 * height, 0.0, 640.0))
        if x2 <= x1 or y2 <= y1:
            continue
        reference.append((index, class_id, np.float32(confidence), x1, y1, x2, y2))

    count = int(np.fromfile(args.plugin_output / "count.bin", dtype=np.int32)[0])
    boxes = np.fromfile(args.plugin_output / "boxes_scores.bin", dtype=np.float32).reshape(count, 5)
    classes = np.fromfile(args.plugin_output / "classes.bin", dtype=np.int32)
    indices = np.fromfile(args.plugin_output / "indices.bin", dtype=np.int32)
    if len(reference) != count:
        raise RuntimeError(f"count mismatch: reference={len(reference)} plugin={count}")

    reference_indices = np.asarray([item[0] for item in reference], dtype=np.int32)
    reference_classes = np.asarray([item[1] for item in reference], dtype=np.int32)
    reference_boxes = np.asarray(
        [[item[3], item[4], item[5], item[6], item[2]] for item in reference],
        dtype=np.float32,
    ).reshape(-1, 5)
    index_equal = bool(np.array_equal(reference_indices, indices))
    class_equal = bool(np.array_equal(reference_classes, classes))
    box_error = np.abs(reference_boxes[:, :4] - boxes[:, :4])
    confidence_error = np.abs(reference_boxes[:, 4] - boxes[:, 4])
    box_max = float(box_error.max(initial=0.0))
    confidence_max = float(confidence_error.max(initial=0.0))
    finite = bool(np.isfinite(boxes).all())
    passed = index_equal and class_equal and finite and box_max <= 1e-3 and confidence_max <= 1e-6
    result = {
        "status": "PASS" if passed else "FAIL",
        "count": count,
        "indices_exact": index_equal,
        "classes_exact": class_equal,
        "all_plugin_values_finite": finite,
        "box_max_abs_error": box_max,
        "confidence_max_abs_error": confidence_max,
        "box_tolerance": 1e-3,
        "confidence_tolerance": 1e-6,
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
