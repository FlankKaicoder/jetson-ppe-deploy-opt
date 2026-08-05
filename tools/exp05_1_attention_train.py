from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            key = key.strip()
            value = value.strip() if value is not None else ""
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    if not rows:
        raise RuntimeError("results.csv contains no rows")
    return rows


def best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    if not valid:
        raise RuntimeError(f"no finite values for {key}")
    return max(valid, key=lambda row: float(row[key]))


def selected_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    )
    return {key: row.get(key) for key in keys}


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
        "epochs": 100,
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
        "patience": 100,
        "val": True,
        "save": True,
        "save_period": 10,
        "plots": True,
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
    args_yaml = train_dir / "args.yaml"
    required_artifacts = {
        "best_pt": best_pt.is_file(),
        "last_pt": last_pt.is_file(),
        "results_csv": results_csv.is_file(),
        "args_yaml": args_yaml.is_file(),
    }
    if not all(required_artifacts.values()):
        raise RuntimeError(f"missing formal training artifacts: {required_artifacts}")

    shutil.copy2(results_csv, report_dir / "results.csv")
    shutil.copy2(args_yaml, report_dir / "args.yaml")
    rows = load_rows(results_csv)
    final_metrics = selected_metrics(rows[-1])
    best_map50 = selected_metrics(best_row(rows, "metrics/mAP50(B)"))
    best_map5095 = selected_metrics(best_row(rows, "metrics/mAP50-95(B)"))

    trainer_model = de_parallel(trainer.model)
    ema_model = de_parallel(trainer.ema.ema)
    checkpoint_model = YOLO(str(best_pt)).model
    trainer_attention = describe_p3_attention(trainer_model)
    ema_attention = describe_p3_attention(ema_model)
    checkpoint_attention = describe_p3_attention(checkpoint_model)

    checks = {
        "artifacts": all(required_artifacts.values()),
        "epoch_count": len(rows) == 100,
        "trainer_wrapper_count": count_p3_attention_wrappers(trainer_model) == 1,
        "ema_wrapper_count": count_p3_attention_wrappers(ema_model) == 1,
        "checkpoint_wrapper_count": count_p3_attention_wrappers(checkpoint_model) == 1,
        "residual_scale_updated": bool(
            checkpoint_attention
            and abs(float(checkpoint_attention[0]["residual_scale"])) > 1e-8
        ),
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "overall": overall,
        "run_name": args.run_name,
        "elapsed_seconds": elapsed_seconds,
        "train_dir": str(train_dir),
        "best_pt": str(best_pt),
        "best_pt_sha256": sha256_file(best_pt),
        "last_pt": str(last_pt),
        "required_artifacts": required_artifacts,
        "checks": checks,
        "epoch_count": len(rows),
        "final_metrics": final_metrics,
        "best_map50": best_map50,
        "best_map50_95": best_map5095,
        "trainer_attention": trainer_attention,
        "ema_attention": ema_attention,
        "checkpoint_attention": checkpoint_attention,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Exp05.1 P3 Residual CBAM-Lite 100-Epoch Training",
        f"overall={overall}",
        f"run_name={args.run_name}",
        f"elapsed_seconds={elapsed_seconds:.3f}",
        f"train_dir={train_dir}",
        f"best_pt={best_pt}",
        f"best_pt_sha256={summary['best_pt_sha256']}",
        f"checks={json.dumps(checks, sort_keys=True)}",
        f"final_metrics={json.dumps(final_metrics, sort_keys=True)}",
        f"best_map50={json.dumps(best_map50, sort_keys=True)}",
        f"best_map50_95={json.dumps(best_map5095, sort_keys=True)}",
        f"checkpoint_attention={json.dumps(checkpoint_attention, sort_keys=True)}",
    ]
    (report_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
