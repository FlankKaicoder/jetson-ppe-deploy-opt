from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch.nn as nn
from ultralytics import YOLO

from models.blocks.attention_block import (
    count_p3_attention_wrappers,
    describe_p3_attention,
)


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    eval_root = Path(args.eval_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(weights))
    model = yolo.model
    wrapper_count = count_p3_attention_wrappers(model)
    attention = describe_p3_attention(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    metrics = yolo.val(
        data=str(data),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        plots=True,
        project=str(eval_root),
        name=args.run_name,
        exist_ok=False,
        verbose=True,
    )
    results_dict = metrics.results_dict
    overall = {key: float(results_dict[key]) for key in METRIC_KEYS}
    if not all(math.isfinite(value) for value in overall.values()):
        raise RuntimeError(f"non-finite overall metrics: {overall}")

    box = metrics.box
    indices = int_list(box.ap_class_index)
    precision = float_list(box.p)
    recall = float_list(box.r)
    ap50 = float_list(box.ap50)
    ap5095 = float_list(box.ap)
    per_class = []
    for position, class_id in enumerate(indices):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "precision": precision[position],
                "recall": recall[position],
                "mAP50": ap50[position],
                "mAP50-95": ap5095[position],
            }
        )

    speed = {
        str(key): float(value)
        for key, value in metrics.speed.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    checks = {
        "wrapper_count": wrapper_count == 1,
        "class_count": len(per_class) == 3,
        "finite_metrics": True,
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "result": result,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "data": str(data),
        "split": "test",
        "imgsz": 640,
        "batch": 16,
        "parameters": parameters,
        "wrapper_count": wrapper_count,
        "attention": attention,
        "checks": checks,
        "overall_metrics": overall,
        "per_class_metrics": per_class,
        "speed_ms_per_image": speed,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    lines = [
        "Exp05.1 P3 Residual CBAM-Lite Independent Test",
        f"result={result}",
        f"weights={weights}",
        f"parameters={parameters}",
        f"wrapper_count={wrapper_count}",
        f"mAP50={overall['metrics/mAP50(B)']:.8f}",
        f"mAP50-95={overall['metrics/mAP50-95(B)']:.8f}",
        f"precision={overall['metrics/precision(B)']:.8f}",
        f"recall={overall['metrics/recall(B)']:.8f}",
        f"per_class={json.dumps(per_class, ensure_ascii=False, sort_keys=True)}",
        f"speed_ms_per_image={json.dumps(speed, sort_keys=True)}",
    ]
    (report_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
