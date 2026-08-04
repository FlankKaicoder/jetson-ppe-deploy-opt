#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path


def print_stage(name: str) -> None:
    print(
        f"\n========== stage={name} "
        f"time={time.strftime('%F %T')} ==========",
        flush=True,
    )


def scalar_metrics(metrics: object) -> dict[str, float]:
    result: dict[str, float] = {}

    if not isinstance(metrics, dict):
        return result

    for key, value in metrics.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return result


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-name", required=True)

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    output_root = Path(args.output_root).resolve()
    report_dir = Path(args.report_dir).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print_stage("path_check")

    print(f"python_executable={sys.executable}", flush=True)
    print(f"data_yaml={data_yaml}", flush=True)
    print(f"model_path={model_path}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"report_dir={report_dir}", flush=True)
    print(f"run_name={args.run_name}", flush=True)

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"dataset yaml not found: {data_yaml}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"model file not found: {model_path}"
        )

    if model_path.stat().st_size == 0:
        raise RuntimeError(
            f"model file is empty: {model_path}"
        )

    print(
        f"model_size_bytes={model_path.stat().st_size}",
        flush=True,
    )

    # 避免训练过程中尝试自动下载远程资源。
    os.environ.setdefault("YOLO_OFFLINE", "true")

    print_stage("import_torch")

    import torch

    print(f"torch_version={torch.__version__}", flush=True)
    print(f"torch_cuda_version={torch.version.cuda}", flush=True)
    print(
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(
        f"cuda_device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        "cuda_capability="
        f"{torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    print_stage("import_ultralytics")

    import ultralytics
    from ultralytics import YOLO

    print(
        f"ultralytics_version={ultralytics.__version__}",
        flush=True,
    )

    print_stage("model_load_begin")

    model = YOLO(
        str(model_path),
        task="detect",
    )

    print_stage("model_load_end")

    train_start = time.perf_counter()

    print_stage("train_begin")

    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        cache=False,
        seed=42,
        deterministic=True,
        project=str(output_root),
        name=args.run_name,
        exist_ok=False,
        amp=True,
        plots=False,
        save=True,
        val=True,
        verbose=True,
    )

    train_elapsed = time.perf_counter() - train_start

    print_stage("train_end")

    trainer = getattr(model, "trainer", None)

    if trainer is None:
        raise RuntimeError(
            "Ultralytics trainer object was not created"
        )

    save_dir = Path(trainer.save_dir).resolve()

    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    metrics = scalar_metrics(
        getattr(trainer, "metrics", {})
    )

    passed = (
        save_dir.is_dir()
        and last_weight.is_file()
        and results_csv.is_file()
    )

    summary = {
        "experiment": (
            "Exp02.5 YOLO11n one-epoch "
            "training smoke"
        ),
        "result": "PASS" if passed else "FAIL",
        "data_yaml": str(data_yaml),
        "model_path": str(model_path),
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
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "epochs": 1,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "seed": 42,
        "train_elapsed_seconds": train_elapsed,
        "metrics": metrics,
    }

    summary_json = report_dir / "summary.json"
    summary_txt = report_dir / "summary.txt"

    summary_json.write_text(
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
        " Exp02.5 YOLO11n Training Smoke Summary",
        "============================================================",
        f"result={summary['result']}",
        f"data_yaml={data_yaml}",
        f"model_path={model_path}",
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
        f"torch_version={torch.__version__}",
        f"torch_cuda_version={torch.version.cuda}",
        f"ultralytics_version={ultralytics.__version__}",
        f"cuda_device={torch.cuda.get_device_name(0)}",
        "epochs=1",
        "imgsz=640",
        "batch=8",
        "workers=0",
        "seed=42",
        f"train_elapsed_seconds={train_elapsed:.3f}",
    ]

    for key, value in sorted(metrics.items()):
        lines.append(f"metric_{key}={value:.8f}")

    lines.append(
        "exp02_5_training_smoke="
        + ("PASS" if passed else "FAIL")
    )

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(summary_txt.read_text(encoding="utf-8"), end="")

    return 0 if passed else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_stage("fatal_error")
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
