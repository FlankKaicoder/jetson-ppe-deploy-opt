#!/usr/bin/env python3
"""Apply the frozen Exp08 accuracy gates to the Explicit Q/DQ baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fp16-metrics", required=True)
    parser.add_argument("--qdq-metrics", required=True)
    parser.add_argument("--fp16-scale", required=True)
    parser.add_argument("--qdq-scale", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--qdq-engine", required=True)
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
    rows = {row["size_group"]: row for row in document["per_size"]}
    tiny = rows["tiny"]
    small = rows["small"]
    gt = int(tiny["gt"]) + int(small["gt"])
    tp = int(tiny["tp"]) + int(small["tp"])
    return {"gt": gt, "tp": tp, "recall": tp / gt if gt else 0.0}


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    paths = {
        "fp16_metrics": Path(args.fp16_metrics).resolve(),
        "qdq_metrics": Path(args.qdq_metrics).resolve(),
        "fp16_scale": Path(args.fp16_scale).resolve(),
        "qdq_scale": Path(args.qdq_scale).resolve(),
        "fp16_engine": Path(args.fp16_engine).resolve(),
        "qdq_engine": Path(args.qdq_engine).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    fp16_metrics = load_json(paths["fp16_metrics"])["metrics"]
    qdq_metrics = load_json(paths["qdq_metrics"])["metrics"]
    fp16_scale = tiny_small_recall(load_json(paths["fp16_scale"]))
    qdq_scale = tiny_small_recall(load_json(paths["qdq_scale"]))
    deltas = {
        key: float(qdq_metrics[key]) - float(fp16_metrics[key])
        for key in ("precision", "recall", "map50", "map50_95")
    }
    deltas["tiny_small_recall"] = qdq_scale["recall"] - fp16_scale["recall"]
    gates = {
        "map50_95": deltas["map50_95"] >= -args.map50_95_max_drop,
        "map50": deltas["map50"] >= -args.map50_max_drop,
        "tiny_small_recall": deltas["tiny_small_recall"] >= -args.tiny_small_max_drop,
    }
    passed = all(gates.values())
    summary = {
        "experiment": "Exp17 Explicit Q/DQ full-test baseline",
        "result": "PASS" if passed else "REJECTED",
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
        "same_run_metrics": {"fp16": fp16_metrics, "explicit_qdq": qdq_metrics},
        "scale_metrics": {
            "fp16_tiny_small": fp16_scale,
            "explicit_qdq_tiny_small": qdq_scale,
        },
        "explicit_qdq_minus_fp16": deltas,
        "gates": gates,
        "engines": {
            name: {
                "path": str(paths[name]),
                "bytes": paths[name].stat().st_size,
                "sha256": sha256_file(paths[name]),
            }
            for name in ("fp16_engine", "qdq_engine")
        },
        "decision_boundary": (
            "This baseline is diagnostic even if rejected; sensitivity analysis continues "
            "without rewriting Exp08 or changing thresholds."
        ),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"result={summary['result']}",
        f"fp16_map50={fp16_metrics['map50']:.12g}",
        f"qdq_map50={qdq_metrics['map50']:.12g}",
        f"map50_delta={deltas['map50']:.12g}",
        f"fp16_map50_95={fp16_metrics['map50_95']:.12g}",
        f"qdq_map50_95={qdq_metrics['map50_95']:.12g}",
        f"map50_95_delta={deltas['map50_95']:.12g}",
        f"fp16_tiny_small_recall={fp16_scale['recall']:.12g}",
        f"qdq_tiny_small_recall={qdq_scale['recall']:.12g}",
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
