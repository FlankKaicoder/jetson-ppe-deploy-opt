#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]

    if hasattr(value, "tolist"):
        return normalize(value.tolist())

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, Path):
        return str(value)

    return value


def read_training_extrema(
    results_csv: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "results_csv": str(results_csv),
        "row_count": 0,
        "max_map50": None,
        "max_map50_95": None,
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

    def find_max(column: str) -> dict[str, Any] | None:
        valid_rows: list[tuple[float, dict[str, str]]] = []

        for row in rows:
            raw_value = row.get(column, "").strip()

            try:
                value = float(raw_value)
            except ValueError:
                continue

            valid_rows.append((value, row))

        if not valid_rows:
            return None

        value, row = max(
            valid_rows,
            key=lambda item: item[0],
        )

        return {
            "epoch": int(float(row["epoch"])),
            "value": value,
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

    result["max_map50"] = find_max(
        "metrics/mAP50(B)"
    )
    result["max_map50_95"] = find_max(
        "metrics/mAP50-95(B)"
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-name", required=True)

    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    train_dir = Path(args.train_dir).resolve()
    eval_root = Path(args.eval_root).resolve()
    report_dir = Path(args.report_dir).resolve()

    eval_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("========== Exp02.7 preflight ==========", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"weights={weights}", flush=True)
    print(f"data_yaml={data_yaml}", flush=True)
    print(f"train_dir={train_dir}", flush=True)
    print(f"eval_root={eval_root}", flush=True)
    print(f"report_dir={report_dir}", flush=True)
    print(f"run_name={args.run_name}", flush=True)

    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(
            f"valid weight not found: {weights}"
        )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"dataset yaml not found: {data_yaml}"
        )

    training_extrema = read_training_extrema(
        train_dir / "results.csv"
    )

    print()
    print(
        "========== environment import ==========",
        flush=True,
    )

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
        raise RuntimeError("CUDA is not available")

    print(
        f"cuda_device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    print()
    print("========== model load ==========", flush=True)

    model = YOLO(str(weights), task="detect")

    print()
    print(
        "========== independent test evaluation ==========",
        flush=True,
    )

    start = time.perf_counter()

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        plots=True,
        save_json=False,
        project=str(eval_root),
        name=args.run_name,
        exist_ok=False,
        verbose=True,
    )

    elapsed = time.perf_counter() - start

    validator = getattr(model, "validator", None)
    eval_dir_value = getattr(
        validator,
        "save_dir",
        eval_root / args.run_name,
    )
    eval_dir = Path(eval_dir_value).resolve()

    overall_metrics = normalize(
        getattr(metrics, "results_dict", {})
    )

    speed = normalize(
        getattr(metrics, "speed", {})
    )

    try:
        class_summary = normalize(metrics.summary())
    except Exception as exc:
        class_summary = []
        print(
            "WARNING: metrics.summary() unavailable: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    summary = {
        "experiment": (
            "Exp02.7 YOLO11n independent "
            "test-set evaluation"
        ),
        "result": "PASS",
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "report_dir": str(report_dir),
        "split": "test",
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "training_extrema": training_extrema,
        "test_metrics": overall_metrics,
        "test_speed_ms_per_image": speed,
        "test_class_summary": class_summary,
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
        " Exp02.7 YOLO11n Independent Test Summary",
        "============================================================",
        "result=PASS",
        f"weights={weights}",
        f"data_yaml={data_yaml}",
        f"train_dir={train_dir}",
        f"eval_dir={eval_dir}",
        "split=test",
        "imgsz=640",
        "batch=16",
        "workers=8",
        f"torch_version={torch.__version__}",
        f"torch_cuda_version={torch.version.cuda}",
        f"ultralytics_version={ultralytics.__version__}",
        f"cuda_device={torch.cuda.get_device_name(0)}",
        f"elapsed_seconds={elapsed:.3f}",
    ]

    max_map50 = training_extrema.get("max_map50")
    max_map50_95 = training_extrema.get("max_map50_95")

    if max_map50 is not None:
        lines.extend([
            "",
            "========== training maximum val mAP50 ==========",
            f"epoch={max_map50['epoch']}",
            f"precision={max_map50['precision']:.8f}",
            f"recall={max_map50['recall']:.8f}",
            f"map50={max_map50['map50']:.8f}",
            f"map50_95={max_map50['map50_95']:.8f}",
        ])

    if max_map50_95 is not None:
        lines.extend([
            "",
            "========== training maximum val mAP50-95 ==========",
            f"epoch={max_map50_95['epoch']}",
            f"precision={max_map50_95['precision']:.8f}",
            f"recall={max_map50_95['recall']:.8f}",
            f"map50={max_map50_95['map50']:.8f}",
            f"map50_95={max_map50_95['map50_95']:.8f}",
        ])

    lines.extend([
        "",
        "========== test overall metrics ==========",
    ])

    for key, value in sorted(overall_metrics.items()):
        if isinstance(value, (int, float)):
            lines.append(f"{key}={value:.8f}")
        else:
            lines.append(f"{key}={value}")

    lines.extend([
        "",
        "========== test speed ms/image ==========",
    ])

    for key, value in sorted(speed.items()):
        if isinstance(value, (int, float)):
            lines.append(f"{key}={value:.8f}")
        else:
            lines.append(f"{key}={value}")

    lines.extend([
        "",
        "========== test class summary ==========",
        json.dumps(
            class_summary,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "exp02_7_baseline_test=PASS",
    ])

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(summary_txt.read_text(encoding="utf-8"), end="")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nFATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
