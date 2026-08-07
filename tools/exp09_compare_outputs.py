#!/usr/bin/env python3
"""Compare C++ TensorRT output with the frozen Python TensorRT reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exp07_trt_consistency import (
    detection_error,
    nms_predictions,
    tensor_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--max-abs", type=float, default=0.001)
    parser.add_argument("--mean-abs", type=float, default=0.00001)
    parser.add_argument("--relative-l2", type=float, default=0.00001)
    parser.add_argument("--box-max-abs", type=float, default=0.1)
    parser.add_argument("--confidence-max-abs", type=float, default=0.001)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    expected_elements = 1 * 7 * 8400
    reference = np.fromfile(reference_path, dtype=np.float32)
    candidate = np.fromfile(candidate_path, dtype=np.float32)
    if reference.size != expected_elements or candidate.size != expected_elements:
        raise RuntimeError(
            f"unexpected element count: {reference.size}, {candidate.size}"
        )
    reference = reference.reshape(1, 7, 8400)
    candidate = candidate.reshape(1, 7, 8400)
    raw = tensor_error(reference, candidate)
    reference_detections = nms_predictions(reference, 0.25, 0.70, 3)
    candidate_detections = nms_predictions(candidate, 0.25, 0.70, 3)
    detections = detection_error(reference_detections, candidate_detections)
    gates = {
        "shape_and_finite": bool(raw["shape_equal"] and raw["finite"]),
        "max_abs": bool(raw["max_abs_error"] <= args.max_abs),
        "mean_abs": bool(raw["mean_abs_error"] <= args.mean_abs),
        "relative_l2": bool(raw["relative_l2_error"] <= args.relative_l2),
        "detection_count": bool(detections["count_equal"]),
        "classes": bool(detections["classes_equal"]),
        "boxes": bool(detections["max_box_abs_error"] <= args.box_max_abs),
        "confidence": bool(
            detections["max_confidence_abs_error"]
            <= args.confidence_max_abs
        ),
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp09 C++ vs Python TensorRT output consistency",
        "result": "PASS" if passed else "FAIL",
        "configuration": {
            "shape": [1, 7, 8400],
            "confidence": 0.25,
            "nms_iou": 0.70,
        },
        "thresholds": {
            "max_abs": args.max_abs,
            "mean_abs": args.mean_abs,
            "relative_l2": args.relative_l2,
            "box_max_abs": args.box_max_abs,
            "confidence_max_abs": args.confidence_max_abs,
        },
        "sha256": {
            "reference": file_sha256(reference_path),
            "candidate": file_sha256(candidate_path),
        },
        "raw_tensor": raw,
        "detections": detections,
        "gates": gates,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"reference_sha256={summary['sha256']['reference']}",
                f"candidate_sha256={summary['sha256']['candidate']}",
                f"raw_max_abs_error={raw['max_abs_error']}",
                f"raw_mean_abs_error={raw['mean_abs_error']}",
                f"raw_relative_l2_error={raw['relative_l2_error']}",
                f"detection_count={detections['candidate_count']}",
                f"classes_equal={detections['classes_equal']}",
                f"box_max_abs_error={detections['max_box_abs_error']}",
                f"confidence_max_abs_error={detections['max_confidence_abs_error']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result={summary['result']} max_abs={raw['max_abs_error']} "
        f"relative_l2={raw['relative_l2_error']} "
        f"detections={detections['candidate_count']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
