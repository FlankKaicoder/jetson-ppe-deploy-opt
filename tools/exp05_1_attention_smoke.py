from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(model: nn.Module) -> nn.Module:
        return getattr(model, "module", model)

from models.blocks.attention_block import (
    count_p3_attention_wrappers,
    describe_p3_attention,
)
from models.exp05_trainer import AttentionDetectionTrainer


def load_last_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("results.csv contains no rows")
    result: dict[str, Any] = {}
    for key, value in rows[-1].items():
        key = key.strip()
        value = value.strip() if value is not None else ""
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def finite_metrics(row: dict[str, Any]) -> bool:
    required = (
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    return all(
        isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
        for key in required
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    pretrained = Path(args.pretrained).resolve()
    data = Path(args.data).resolve()
    output_root = Path(args.output_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    train_arguments = {
        "model": str(pretrained),
        "data": str(data),
        "epochs": 2,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 8,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.0015,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "amp": True,
        "cache": False,
        "patience": 2,
        "val": True,
        "save": True,
        "plots": False,
        "project": str(output_root),
        "name": args.run_name,
        "exist_ok": False,
        "verbose": True,
    }
    (report_dir / "train_arguments.json").write_text(
        json.dumps(train_arguments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    start = time.perf_counter()
    trainer = AttentionDetectionTrainer(overrides=train_arguments)
    trainer.train()
    elapsed_seconds = time.perf_counter() - start

    train_dir = Path(trainer.save_dir).resolve()
    best_pt = train_dir / "weights" / "best.pt"
    last_pt = train_dir / "weights" / "last.pt"
    results_csv = train_dir / "results.csv"
    required_artifacts = {
        "best_pt": best_pt.is_file(),
        "last_pt": last_pt.is_file(),
        "results_csv": results_csv.is_file(),
    }
    if not all(required_artifacts.values()):
        raise RuntimeError(f"missing smoke artifacts: {required_artifacts}")

    row = load_last_row(results_csv)
    trainer_model = de_parallel(trainer.model)
    trainer_attention = describe_p3_attention(trainer_model)
    ema_model = de_parallel(trainer.ema.ema)
    ema_attention = describe_p3_attention(ema_model)

    checkpoint_model = YOLO(str(best_pt)).model
    checkpoint_attention = describe_p3_attention(checkpoint_model)

    scale_values = [
        abs(float(item["residual_scale"]))
        for item in trainer_attention + ema_attention + checkpoint_attention
    ]
    checks = {
        "artifacts": all(required_artifacts.values()),
        "finite_metrics": finite_metrics(row),
        "trainer_wrapper_count": count_p3_attention_wrappers(trainer_model) == 1,
        "ema_wrapper_count": count_p3_attention_wrappers(ema_model) == 1,
        "checkpoint_wrapper_count": count_p3_attention_wrappers(checkpoint_model) == 1,
        "residual_scale_updated": bool(scale_values and max(scale_values) > 1e-8),
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "overall": overall,
        "run_name": args.run_name,
        "elapsed_seconds": elapsed_seconds,
        "train_dir": str(train_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "results_csv": str(results_csv),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "last_metrics": row,
        "trainer_attention": trainer_attention,
        "ema_attention": ema_attention,
        "checkpoint_attention": checkpoint_attention,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "Exp05.1 P3 Residual CBAM-Lite Smoke Training",
        f"overall={overall}",
        f"run_name={args.run_name}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        f"train_dir={train_dir}",
        f"best_pt={best_pt}",
        f"checks={json.dumps(checks, sort_keys=True)}",
        f"last_mAP50={row.get('metrics/mAP50(B)')}",
        f"last_mAP50_95={row.get('metrics/mAP50-95(B)')}",
        f"trainer_attention={json.dumps(trainer_attention, sort_keys=True)}",
        f"ema_attention={json.dumps(ema_attention, sort_keys=True)}",
        f"checkpoint_attention={json.dumps(checkpoint_attention, sort_keys=True)}",
    ]
    (report_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(summary_lines))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
