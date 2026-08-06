#!/usr/bin/env python3
"""Evaluate one Ultralytics backend and serialize validation metrics."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
from ultralytics import YOLO


METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    if not model_path.is_file() or not data_yaml.is_file():
        raise FileNotFoundError(f"model={model_path} data={data_yaml}")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path), task="detect")
    start = time.perf_counter()
    result = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        rect=False,
        plots=False,
        save_json=False,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    results_dict = normalize(result.results_dict)
    metrics = {
        name: float(results_dict[key]) for name, key in METRIC_KEYS.items()
    }
    document = {
        "model_path": str(model_path),
        "device": args.device,
        "batch": args.batch,
        "workers": args.workers,
        "rect": False,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "all_results": results_dict,
        "speed_ms_per_image": normalize(result.speed),
    }
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
