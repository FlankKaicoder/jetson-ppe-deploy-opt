#!/usr/bin/env python3
"""Apply frozen Exp08 accuracy gates to FP16 and INT8 test results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--exp07-reference", required=True)
    parser.add_argument("--fp16-metrics", required=True)
    parser.add_argument("--int8-metrics", required=True)
    parser.add_argument("--fp16-scale", required=True)
    parser.add_argument("--int8-scale", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--map50-95-max-drop", type=float, default=0.010)
    parser.add_argument("--map50-max-drop", type=float, default=0.015)
    parser.add_argument("--tiny-small-max-drop", type=float, default=0.050)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tiny_small_recall(document: dict[str, Any]) -> dict[str, Any]:
    rows = {
        row["size_group"]: row for row in document["per_size"]
    }
    tiny = rows["tiny"]
    small = rows["small"]
    gt = int(tiny["gt"]) + int(small["gt"])
    tp = int(tiny["tp"]) + int(small["tp"])
    return {"gt": gt, "tp": tp, "recall": tp / gt if gt else 0.0}


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    exp07_reference_path = Path(args.exp07_reference).resolve()
    fp16_metrics_path = Path(args.fp16_metrics).resolve()
    int8_metrics_path = Path(args.int8_metrics).resolve()
    fp16_scale_path = Path(args.fp16_scale).resolve()
    int8_scale_path = Path(args.int8_scale).resolve()
    fp16_engine = Path(args.fp16_engine).resolve()
    int8_engine = Path(args.int8_engine).resolve()
    paths = (
        exp07_reference_path,
        fp16_metrics_path,
        int8_metrics_path,
        fp16_scale_path,
        int8_scale_path,
        fp16_engine,
        int8_engine,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    exp07_reference = load_json(exp07_reference_path)
    fp16_metrics_doc = load_json(fp16_metrics_path)
    int8_metrics_doc = load_json(int8_metrics_path)
    fp16_metrics = fp16_metrics_doc["metrics"]
    int8_metrics = int8_metrics_doc["metrics"]
    fp16_scale = tiny_small_recall(load_json(fp16_scale_path))
    int8_scale = tiny_small_recall(load_json(int8_scale_path))
    deltas = {
        key: float(int8_metrics[key]) - float(fp16_metrics[key])
        for key in ("precision", "recall", "map50", "map50_95")
    }
    deltas["tiny_small_recall"] = int8_scale["recall"] - fp16_scale["recall"]
    gates = {
        "map50_95": deltas["map50_95"] >= -args.map50_95_max_drop,
        "map50": deltas["map50"] >= -args.map50_max_drop,
        "tiny_small_recall": (
            deltas["tiny_small_recall"] >= -args.tiny_small_max_drop
        ),
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp08 FP16 vs INT8 full-test and scale consistency",
        "result": "PASS" if passed else "FAIL",
        "configuration": {
            "split": "test",
            "images": 219,
            "instances": 840,
            "imgsz": 640,
            "batch": 1,
            "rect": False,
            "scale_confidence": 0.25,
            "scale_nms_iou": 0.70,
            "scale_match_iou": 0.50,
        },
        "thresholds": {
            "map50_95_max_drop": args.map50_95_max_drop,
            "map50_max_drop": args.map50_max_drop,
            "tiny_small_recall_max_drop": args.tiny_small_max_drop,
        },
        "same_run_metrics": {
            "fp16": fp16_metrics,
            "int8": int8_metrics,
        },
        "exp07_reference_metrics": {
            name: exp07_reference["evaluations"][name]["metrics"]
            for name in ("pytorch", "tensorrt_fp32", "tensorrt_fp16")
        },
        "scale_metrics": {
            "fp16_tiny_small": fp16_scale,
            "int8_tiny_small": int8_scale,
        },
        "int8_minus_fp16": deltas,
        "gates": gates,
        "engines": {
            "fp16": {
                "path": str(fp16_engine),
                "bytes": fp16_engine.stat().st_size,
                "sha256": sha256_file(fp16_engine),
            },
            "int8": {
                "path": str(int8_engine),
                "bytes": int8_engine.stat().st_size,
                "sha256": sha256_file(int8_engine),
            },
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"result={summary['result']}",
        f"fp16_precision={fp16_metrics['precision']:.12g}",
        f"fp16_recall={fp16_metrics['recall']:.12g}",
        f"fp16_map50={fp16_metrics['map50']:.12g}",
        f"fp16_map50_95={fp16_metrics['map50_95']:.12g}",
        f"int8_precision={int8_metrics['precision']:.12g}",
        f"int8_recall={int8_metrics['recall']:.12g}",
        f"int8_map50={int8_metrics['map50']:.12g}",
        f"int8_map50_95={int8_metrics['map50_95']:.12g}",
        f"map50_delta={deltas['map50']:.12g}",
        f"map50_95_delta={deltas['map50_95']:.12g}",
        f"fp16_tiny_small_recall={fp16_scale['recall']:.12g}",
        f"int8_tiny_small_recall={int8_scale['recall']:.12g}",
        f"tiny_small_recall_delta={deltas['tiny_small_recall']:.12g}",
        f"map50_gate={'PASS' if gates['map50'] else 'FAIL'}",
        f"map50_95_gate={'PASS' if gates['map50_95'] else 'FAIL'}",
        f"tiny_small_gate={'PASS' if gates['tiny_small_recall'] else 'FAIL'}",
        "",
    ]
    (report_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(" ".join(lines[:1] + lines[-4:-1]), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
