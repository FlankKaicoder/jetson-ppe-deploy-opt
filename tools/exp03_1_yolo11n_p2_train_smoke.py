#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


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


def stage(name: str) -> None:
    print(
        f"\n========== stage={name} "
        f"time={time.strftime('%F %T')} ==========",
        flush=True,
    )


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

    print(f"python={sys.executable}", flush=True)
    print(f"config={config}", flush=True)
    print(f"pretrained={pretrained}", flush=True)
    print(f"data_yaml={data_yaml}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"report_dir={report_dir}", flush=True)
    print(f"run_name={args.run_name}", flush=True)

    for path in (config, pretrained, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(
                f"required file not found: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"required file is empty: {path}"
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

    initial_detect = model.model.model[-1]
    initial_stride = [
        float(value)
        for value in model.model.stride.cpu().tolist()
    ]

    print(
        f"initial_detect_nc={initial_detect.nc}",
        flush=True,
    )
    print(
        f"initial_detect_nl={initial_detect.nl}",
        flush=True,
    )
    print(
        f"initial_stride={initial_stride}",
        flush=True,
    )

    if initial_detect.nl != 4:
        raise RuntimeError(
            f"expected 4 detection levels, "
            f"got {initial_detect.nl}"
        )

    if initial_stride != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(
            f"unexpected stride: {initial_stride}"
        )

    stage("pretrained_weight_transfer")

    model.load(str(pretrained))

    print(
        "pretrained_weight_transfer_completed=YES",
        flush=True,
    )

    stage("training")

    train_start = time.perf_counter()

    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
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
        patience=1,
        project=str(output_root),
        name=args.run_name,
        exist_ok=False,
        val=True,
        save=True,
        plots=False,
        verbose=True,
    )

    elapsed = time.perf_counter() - train_start

    stage("post_training_validation")

    trainer = getattr(model, "trainer", None)

    if trainer is None:
        raise RuntimeError(
            "Ultralytics trainer was not created"
        )

    save_dir = Path(trainer.save_dir).resolve()
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    trained_detect = model.model.model[-1]

    trained_stride = [
        float(value)
        for value in model.model.stride.cpu().tolist()
    ]

    metrics = scalar_metrics(
        getattr(trainer, "metrics", {})
    )

    passed = (
        best_weight.is_file()
        and best_weight.stat().st_size > 0
        and last_weight.is_file()
        and last_weight.stat().st_size > 0
        and results_csv.is_file()
        and trained_detect.nc == 3
        and trained_detect.nl == 4
        and trained_stride
        == [4.0, 8.0, 16.0, 32.0]
    )

    summary = {
        "experiment": (
            "Exp03.1 custom YOLO11n-P2 "
            "one-epoch training smoke"
        ),
        "result": "PASS" if passed else "FAIL",
        "config": str(config),
        "pretrained": str(pretrained),
        "data_yaml": str(data_yaml),
        "save_dir": str(save_dir),
        "best_weight": (
            str(best_weight)
            if best_weight.is_file()
            else ""
        ),
        "last_weight": (
            str(last_weight)
            if last_weight.is_file()
            else ""
        ),
        "results_csv": (
            str(results_csv)
            if results_csv.is_file()
            else ""
        ),
        "initial_detect_nc": initial_detect.nc,
        "trained_detect_nc": trained_detect.nc,
        "detect_nl": trained_detect.nl,
        "stride": trained_stride,
        "epochs": 1,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "seed": 42,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "metrics": metrics,
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

    lines = [
        "============================================================",
        " Exp03.1 YOLO11n-P2 Training Smoke Summary",
        "============================================================",
        f"result={summary['result']}",
        f"config={config}",
        f"pretrained={pretrained}",
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
            "last_weight="
            + (
                str(last_weight)
                if last_weight.is_file()
                else "NOT_GENERATED"
            )
        ),
        (
            "results_csv="
            + (
                str(results_csv)
                if results_csv.is_file()
                else "NOT_GENERATED"
            )
        ),
        f"initial_detect_nc={initial_detect.nc}",
        f"trained_detect_nc={trained_detect.nc}",
        f"detect_nl={trained_detect.nl}",
        f"stride={trained_stride}",
        "epochs=1",
        "imgsz=640",
        "batch=8",
        "workers=0",
        "seed=42",
        f"elapsed_seconds={elapsed:.3f}",
    ]

    for key, value in sorted(metrics.items()):
        lines.append(f"metric_{key}={value:.8f}")

    lines.append(
        "exp03_1_yolo11n_p2_training_smoke="
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
