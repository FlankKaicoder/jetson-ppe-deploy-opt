#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def stage(name: str) -> None:
    print(
        f"\n========== stage={name} "
        f"time={time.strftime('%F %T')} ==========",
        flush=True,
    )


def scalar_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, float] = {}

    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue

    return result


def find_best_rows(results_csv: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_count": 0,
        "best_map50": None,
        "best_map50_95": None,
    }

    if not results_csv.is_file():
        return result

    with results_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    result["row_count"] = len(rows)

    def maximum(column: str) -> dict[str, float] | None:
        candidates = []

        for row in rows:
            try:
                candidates.append((
                    float(row[column]),
                    row,
                ))
            except (KeyError, TypeError, ValueError):
                continue

        if not candidates:
            return None

        _, row = max(
            candidates,
            key=lambda item: item[0],
        )

        return {
            "epoch": int(float(row["epoch"])),
            "precision": float(
                row["metrics/precision(B)"]
            ),
            "recall": float(
                row["metrics/recall(B)"]
            ),
            "map50": float(
                row["metrics/mAP50(B)"]
            ),
            "map50_95": float(
                row["metrics/mAP50-95(B)"]
            ),
        }

    result["best_map50"] = maximum(
        "metrics/mAP50(B)"
    )

    result["best_map50_95"] = maximum(
        "metrics/mAP50-95(B)"
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-name", required=True)

    args = parser.parse_args()

    config = Path(args.config).resolve()
    pretrained = Path(args.pretrained).resolve()
    data_yaml = Path(args.data).resolve()
    output_root = Path(args.output_root).resolve()
    report_dir = Path(args.report_dir).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage("preflight")

    required_files = (
        config,
        pretrained,
        data_yaml,
    )

    for path in required_files:
        print(f"required_file={path}", flush=True)

        if not path.is_file():
            raise FileNotFoundError(
                f"required file not found: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"required file is empty: {path}"
            )

    print(f"python={sys.executable}", flush=True)
    print(f"config_sha256={file_sha256(config)}", flush=True)
    print(
        f"pretrained_sha256={file_sha256(pretrained)}",
        flush=True,
    )
    print(
        f"data_yaml_sha256={file_sha256(data_yaml)}",
        flush=True,
    )

    stage("environment")

    import torch
    import ultralytics
    from ultralytics import YOLO

    print(f"torch={torch.__version__}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print(
        f"ultralytics={ultralytics.__version__}",
        flush=True,
    )
    print(
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print(
        f"cuda_device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    stage("model_construction")

    model = YOLO(str(config), task="detect")

    detect = model.model.model[-1]

    initial_stride = [
        float(value)
        for value in model.model.stride.cpu().tolist()
    ]

    print(f"initial_nc={detect.nc}", flush=True)
    print(f"initial_nl={detect.nl}", flush=True)
    print(f"initial_stride={initial_stride}", flush=True)

    if detect.nl != 4:
        raise RuntimeError(
            f"expected detect_nl=4, got {detect.nl}"
        )

    if initial_stride != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(
            f"unexpected stride: {initial_stride}"
        )

    parameters_before_training = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    stage("pretrained_weight_transfer")

    model.load(str(pretrained))

    print(
        "pretrained_weight_transfer_completed=YES",
        flush=True,
    )

    stage("formal_training")

    start = time.perf_counter()

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        amp=True,
        cache=False,
        patience=100,
        project=str(output_root),
        name=args.run_name,
        exist_ok=False,
        pretrained=True,
        val=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - start

    stage("artifact_validation")

    trainer = getattr(model, "trainer", None)

    if trainer is None:
        raise RuntimeError(
            "Ultralytics trainer was not created"
        )

    save_dir = Path(trainer.save_dir).resolve()
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"
    args_yaml = save_dir / "args.yaml"

    trained_detect = model.model.model[-1]

    trained_stride = [
        float(value)
        for value in model.model.stride.cpu().tolist()
    ]

    metrics = scalar_metrics(
        getattr(trainer, "metrics", {})
    )

    training_extrema = find_best_rows(results_csv)

    passed = (
        best_weight.is_file()
        and best_weight.stat().st_size > 0
        and last_weight.is_file()
        and last_weight.stat().st_size > 0
        and results_csv.is_file()
        and training_extrema["row_count"] == 100
        and trained_detect.nc == 3
        and trained_detect.nl == 4
        and trained_stride
        == [4.0, 8.0, 16.0, 32.0]
    )

    summary = {
        "experiment": (
            "Exp03.2 custom YOLO11n-P2 "
            "formal 100-epoch training"
        ),
        "result": "PASS" if passed else "FAIL",
        "config": str(config),
        "config_sha256": file_sha256(config),
        "pretrained": str(pretrained),
        "pretrained_sha256": file_sha256(pretrained),
        "data_yaml": str(data_yaml),
        "save_dir": str(save_dir),
        "best_weight": (
            str(best_weight)
            if best_weight.is_file()
            else ""
        ),
        "best_weight_sha256": (
            file_sha256(best_weight)
            if best_weight.is_file()
            else ""
        ),
        "last_weight": (
            str(last_weight)
            if last_weight.is_file()
            else ""
        ),
        "last_weight_sha256": (
            file_sha256(last_weight)
            if last_weight.is_file()
            else ""
        ),
        "results_csv": str(results_csv),
        "args_yaml": str(args_yaml),
        "parameters_before_training": (
            parameters_before_training
        ),
        "trained_nc": trained_detect.nc,
        "detect_nl": trained_detect.nl,
        "stride": trained_stride,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "seed": 42,
        "optimizer": "AdamW",
        "lr0": 0.0015,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "final_metrics": metrics,
        "training_extrema": training_extrema,
    }

    (report_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if results_csv.is_file():
        (report_dir / "results.csv").write_bytes(
            results_csv.read_bytes()
        )

    if args_yaml.is_file():
        (report_dir / "args.yaml").write_bytes(
            args_yaml.read_bytes()
        )

    lines = [
        "============================================================",
        " Exp03.2 YOLO11n-P2 Formal Training Summary",
        "============================================================",
        f"result={summary['result']}",
        f"config={config}",
        f"config_sha256={summary['config_sha256']}",
        f"pretrained={pretrained}",
        (
            "pretrained_sha256="
            f"{summary['pretrained_sha256']}"
        ),
        f"data_yaml={data_yaml}",
        f"save_dir={save_dir}",
        (
            "best_weight="
            + (
                str(best_weight)
                if best_weight.is_file()
                else "NOT_GENERATED"
            )
        ),
        (
            "best_weight_sha256="
            f"{summary['best_weight_sha256']}"
        ),
        (
            "last_weight="
            + (
                str(last_weight)
                if last_weight.is_file()
                else "NOT_GENERATED"
            )
        ),
        (
            "last_weight_sha256="
            f"{summary['last_weight_sha256']}"
        ),
        f"results_csv={results_csv}",
        f"parameters={parameters_before_training}",
        f"trained_nc={trained_detect.nc}",
        f"detect_nl={trained_detect.nl}",
        f"stride={trained_stride}",
        "epochs=100",
        "imgsz=640",
        "batch=16",
        "workers=8",
        "seed=42",
        "optimizer=AdamW",
        "lr0=0.0015",
        "lrf=0.01",
        "momentum=0.9",
        "weight_decay=0.0005",
        f"elapsed_seconds={elapsed:.3f}",
    ]

    best_map50 = training_extrema.get(
        "best_map50"
    )
    best_map50_95 = training_extrema.get(
        "best_map50_95"
    )

    if best_map50 is not None:
        lines.extend([
            "",
            "========== maximum validation mAP50 ==========",
            f"epoch={best_map50['epoch']}",
            f"precision={best_map50['precision']:.8f}",
            f"recall={best_map50['recall']:.8f}",
            f"map50={best_map50['map50']:.8f}",
            f"map50_95={best_map50['map50_95']:.8f}",
        ])

    if best_map50_95 is not None:
        lines.extend([
            "",
            "========== maximum validation mAP50-95 ==========",
            f"epoch={best_map50_95['epoch']}",
            f"precision={best_map50_95['precision']:.8f}",
            f"recall={best_map50_95['recall']:.8f}",
            f"map50={best_map50_95['map50']:.8f}",
            f"map50_95={best_map50_95['map50_95']:.8f}",
        ])

    lines.extend([
        "",
        "========== final trainer metrics ==========",
    ])

    for key, value in sorted(metrics.items()):
        lines.append(f"{key}={value:.8f}")

    lines.append("")
    lines.append(
        "exp03_2_yolo11n_p2_formal_training="
        + ("PASS" if passed else "FAIL")
    )

    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        (report_dir / "summary.txt").read_text(
            encoding="utf-8"
        ),
        end="",
    )

    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        stage("fatal_error")
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
