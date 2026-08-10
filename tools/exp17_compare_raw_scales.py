#!/usr/bin/env python3
"""Compare FP16, implicit INT8, and explicit-QDQ raw outputs by feature scale."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from exp07_trt_consistency import preprocess_image
from exp16_revalidate_metrics import EngineSession


SCALES = {"p3": (0, 6400), "p4": (6400, 8000), "p5": (8000, 8400)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-engine", required=True, type=Path)
    parser.add_argument("--implicit-engine", required=True, type=Path)
    parser.add_argument("--qdq-engine", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root.iterdir() if path.suffix.lower() in extensions)


class ErrorState:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_error_sq = 0.0
        self.sum_reference_sq = 0.0
        self.max_abs = 0.0
        self.samples: list[np.ndarray] = []

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64)).reshape(-1)
        ref = reference.astype(np.float64).reshape(-1)
        self.count += error.size
        self.sum_abs += float(error.sum())
        self.sum_error_sq += float(np.dot(error, error))
        self.sum_reference_sq += float(np.dot(ref, ref))
        self.max_abs = max(self.max_abs, float(error.max(initial=0.0)))
        stride = max(1, error.size // 4096)
        self.samples.append(error[::stride][:4096].copy())

    def summary(self) -> dict[str, float | int]:
        sample = np.concatenate(self.samples)
        return {
            "count": self.count,
            "mean_abs": self.sum_abs / self.count,
            "p50_abs": float(np.percentile(sample, 50)),
            "p95_abs": float(np.percentile(sample, 95)),
            "p99_abs": float(np.percentile(sample, 99)),
            "max_abs": self.max_abs,
            "relative_l2": math.sqrt(self.sum_error_sq / self.sum_reference_sq)
            if self.sum_reference_sq else 0.0,
        }


def main() -> int:
    args = parse_args()
    for path in (args.fp16_engine, args.implicit_engine, args.qdq_engine):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    images = image_paths(args.images)
    if args.limit <= 0 or args.limit > len(images):
        raise ValueError(f"invalid limit {args.limit}; available={len(images)}")
    images = images[: args.limit]
    sessions = {
        "fp16": EngineSession(args.fp16_engine, None),
        "implicit_int8": EngineSession(args.implicit_engine, None),
        "explicit_qdq": EngineSession(args.qdq_engine, None),
    }
    states: dict[str, dict[str, dict[str, Any]]] = {}
    for backend in ("implicit_int8", "explicit_qdq"):
        states[backend] = {}
        for scale in SCALES:
            states[backend][scale] = {
                "score": ErrorState(),
                "bbox": ErrorState(),
                "class_changes_all": 0,
                "class_changes_reference_above": 0,
                "class_changes_union_above": 0,
                "cross_up": 0,
                "cross_down": 0,
                "reference_above": 0,
                "candidate_above": 0,
                "candidate_count": 0,
            }
    started = time.perf_counter()
    for image_index, image in enumerate(images):
        input_array, _ = preprocess_image(image, 640)
        raw_by_backend = {
            name: next(iter(session.infer(input_array).values())).reshape(7, 8400)
            for name, session in sessions.items()
        }
        reference = raw_by_backend["fp16"]
        reference_scores = reference[4:, :].max(axis=0)
        reference_classes = reference[4:, :].argmax(axis=0)
        for backend in ("implicit_int8", "explicit_qdq"):
            candidate = raw_by_backend[backend]
            candidate_scores = candidate[4:, :].max(axis=0)
            candidate_classes = candidate[4:, :].argmax(axis=0)
            for scale, (start, end) in SCALES.items():
                state = states[backend][scale]
                ref_score = reference_scores[start:end]
                cand_score = candidate_scores[start:end]
                state["score"].update(ref_score, cand_score)
                state["bbox"].update(reference[:4, start:end], candidate[:4, start:end])
                ref_above = ref_score >= args.confidence
                cand_above = cand_score >= args.confidence
                class_changed = reference_classes[start:end] != candidate_classes[start:end]
                state["class_changes_all"] += int(np.count_nonzero(class_changed))
                state["class_changes_reference_above"] += int(
                    np.count_nonzero(class_changed & ref_above)
                )
                state["class_changes_union_above"] += int(
                    np.count_nonzero(class_changed & (ref_above | cand_above))
                )
                state["cross_up"] += int(np.count_nonzero(~ref_above & cand_above))
                state["cross_down"] += int(np.count_nonzero(ref_above & ~cand_above))
                state["reference_above"] += int(np.count_nonzero(ref_above))
                state["candidate_above"] += int(np.count_nonzero(cand_above))
                state["candidate_count"] += end - start
        if (image_index + 1) % 10 == 0 or image_index + 1 == len(images):
            print(f"progress={image_index + 1}/{len(images)}", flush=True)

    comparisons: dict[str, Any] = {}
    for backend, scale_states in states.items():
        comparisons[backend] = {}
        for scale, state in scale_states.items():
            comparisons[backend][scale] = {
                "score_error": state["score"].summary(),
                "bbox_error": state["bbox"].summary(),
                "class_changes_all": state["class_changes_all"],
                "class_changes_reference_above": state["class_changes_reference_above"],
                "class_changes_union_above": state["class_changes_union_above"],
                "threshold_cross_up": state["cross_up"],
                "threshold_cross_down": state["cross_down"],
                "threshold_cross_total": state["cross_up"] + state["cross_down"],
                "reference_above_threshold": state["reference_above"],
                "candidate_above_threshold": state["candidate_above"],
                "candidate_count": state["candidate_count"],
            }
    summary = {
        "experiment": "Exp17 P3/P4/P5 raw score and bbox error audit",
        "result": "PASS",
        "configuration": {
            "images": len(images),
            "confidence": args.confidence,
            "scales": {name: [start, end] for name, (start, end) in SCALES.items()},
            "elapsed_seconds": time.perf_counter() - started,
            "matching": "same image and candidate index; raw tensor comparison before NMS",
        },
        "comparisons_vs_frozen_fp16": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["result=PASS", f"images={len(images)}"]
    for backend, scale_results in comparisons.items():
        for scale, result in scale_results.items():
            lines.append(
                f"{backend}_{scale} score_rel_l2={result['score_error']['relative_l2']:.12g} "
                f"bbox_rel_l2={result['bbox_error']['relative_l2']:.12g} "
                f"crossings={result['threshold_cross_total']} "
                f"class_changes_ref_above={result['class_changes_reference_above']} "
                f"class_changes_union_above={result['class_changes_union_above']}"
            )
    lines.append("")
    (args.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
