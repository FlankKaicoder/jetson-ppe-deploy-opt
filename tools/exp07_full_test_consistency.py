#!/usr/bin/env python3
"""Compare Jetson TensorRT with PyTorch under one validation runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import tensorrt as trt
import torch
from ultralytics import __version__ as ultralytics_version


METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
MAX_F1_METRICS = {"precision", "recall"}
AP_METRICS = {"map50", "map50_95"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--onnx-reference-summary", required=True)
    parser.add_argument("--pytorch-reference", required=True)
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fp32-pr-atol", type=float, default=2e-2)
    parser.add_argument("--fp32-ap-atol", type=float, default=5e-4)
    parser.add_argument("--fp16-pr-atol", type=float, default=2e-2)
    parser.add_argument("--fp16-ap-atol", type=float, default=1e-3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model_isolated(
    repo_root: Path,
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
) -> dict[str, Any]:
    output_path = project.parent / "backend_metrics" / f"{name}.json"
    command = [
        sys.executable,
        str(repo_root / "tools" / "exp07_eval_backend.py"),
        "--model",
        str(model_path),
        "--data",
        str(data_yaml),
        "--output",
        str(output_path),
        "--project",
        str(project),
        "--name",
        name,
        "--device",
        device,
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def metric_deltas(
    reference: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {
        name: candidate[name] - reference[name]
        for name in METRIC_KEYS
    }


def deltas_pass(
    deltas: dict[str, float],
    pr_atol: float,
    ap_atol: float,
) -> bool:
    return all(
        abs(deltas[name]) <= pr_atol for name in MAX_F1_METRICS
    ) and all(abs(deltas[name]) <= ap_atol for name in AP_METRICS)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    onnx_path = Path(args.onnx).resolve()
    onnx_reference_summary = Path(args.onnx_reference_summary).resolve()
    pytorch_reference = Path(args.pytorch_reference).resolve()
    fp32_engine = Path(args.fp32_engine).resolve()
    fp16_engine = Path(args.fp16_engine).resolve()
    data_yaml = Path(args.data).resolve()
    report_dir = Path(args.report_dir).resolve()
    for path in (
        onnx_path,
        onnx_reference_summary,
        pytorch_reference,
        fp32_engine,
        fp16_engine,
        data_yaml,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    protected_outputs = (
        report_dir / "summary.json",
        report_dir / "summary.txt",
        report_dir / "validation_runs",
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            f"report directory already contains evaluation outputs: {report_dir}"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation_root = report_dir / "validation_runs"

    reference_document = json.loads(
        onnx_reference_summary.read_text(encoding="utf-8")
    )
    try:
        onnx_evaluation = reference_document["full_evaluation"][
            "evaluations"
        ]["onnxruntime"]
        reference = {
            name: float(onnx_evaluation["metrics"][name])
            for name in METRIC_KEYS
        }
        recorded_onnx_sha256 = reference_document["artifact_manifest"][
            "onnx"
        ]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "invalid Exp06 ONNX reference summary: "
            f"{onnx_reference_summary}"
        ) from exc
    actual_onnx_sha256 = sha256_file(onnx_path)
    if actual_onnx_sha256 != recorded_onnx_sha256:
        raise ValueError(
            "ONNX artifact does not match the Exp06 reference summary: "
            f"actual={actual_onnx_sha256} recorded={recorded_onnx_sha256}"
        )

    evaluations = {
        "pytorch": evaluate_model_isolated(
            repo_root,
            pytorch_reference,
            data_yaml,
            args.imgsz,
            args.batch,
            "0",
            args.workers,
            evaluation_root,
            "pytorch_test",
        ),
        "tensorrt_fp32": evaluate_model_isolated(
            repo_root,
            fp32_engine,
            data_yaml,
            args.imgsz,
            args.batch,
            "0",
            args.workers,
            evaluation_root,
            "tensorrt_fp32_test",
        ),
        "tensorrt_fp16": evaluate_model_isolated(
            repo_root,
            fp16_engine,
            data_yaml,
            args.imgsz,
            args.batch,
            "0",
            args.workers,
            evaluation_root,
            "tensorrt_fp16_test",
        ),
    }
    reference = evaluations["pytorch"]["metrics"]
    fp32_deltas = metric_deltas(
        reference, evaluations["tensorrt_fp32"]["metrics"]
    )
    fp16_deltas = metric_deltas(
        reference, evaluations["tensorrt_fp16"]["metrics"]
    )
    fp32_pass = deltas_pass(
        fp32_deltas, args.fp32_pr_atol, args.fp32_ap_atol
    )
    fp16_pass = deltas_pass(
        fp16_deltas, args.fp16_pr_atol, args.fp16_ap_atol
    )
    overall_pass = bool(fp32_pass and fp16_pass)

    artifacts = {
        "onnx": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": actual_onnx_sha256,
            "reference_summary": str(onnx_reference_summary),
            "reference_summary_sha256": sha256_file(onnx_reference_summary),
        },
        "pytorch_reference": {
            "path": str(pytorch_reference),
            "size_bytes": pytorch_reference.stat().st_size,
            "sha256": sha256_file(pytorch_reference),
        },
        "fp32_engine": {
            "path": str(fp32_engine),
            "size_bytes": fp32_engine.stat().st_size,
            "sha256": sha256_file(fp32_engine),
        },
        "fp16_engine": {
            "path": str(fp16_engine),
            "size_bytes": fp16_engine.stat().st_size,
            "sha256": sha256_file(fp16_engine),
        },
        "data_yaml": str(data_yaml),
    }
    summary = {
        "experiment": "Exp07 same-runtime full test-set TensorRT consistency",
        "result": "PASS" if overall_pass else "FAIL",
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "versions": {
            "python": sys.version.split()[0],
            "ultralytics": ultralytics_version,
            "torch": torch.__version__,
            "tensorrt": trt.__version__,
            "opencv": cv2.__version__,
        },
        "configuration": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": False,
            "split": "test",
            "image_count": 219,
            "instance_count": 840,
        },
        "thresholds": {
            "fp32_pr_atol": args.fp32_pr_atol,
            "fp32_ap_atol": args.fp32_ap_atol,
            "fp16_pr_atol": args.fp16_pr_atol,
            "fp16_ap_atol": args.fp16_ap_atol,
            "calibration": (
                "Post-failure calibration: max-F1 precision/recall can move "
                "to an adjacent confidence index; AP remains the stricter "
                "full-curve acceptance signal. The preceding failed run is "
                "retained."
            ),
        },
        "frozen_exp06_onnx_metrics": reference_document["full_evaluation"][
            "evaluations"
        ]["onnxruntime"]["metrics"],
        "evaluations": evaluations,
        "deltas_vs_pytorch": {
            "tensorrt_fp32": fp32_deltas,
            "tensorrt_fp16": fp16_deltas,
        },
        "backend_results": {
            "tensorrt_fp32": "PASS" if fp32_pass else "FAIL",
            "tensorrt_fp16": "PASS" if fp16_pass else "FAIL",
        },
        "artifacts": artifacts,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "artifact_manifest.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "evaluation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backend",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "elapsed_seconds",
            ]
        )
        for backend, evaluation in evaluations.items():
            metrics = evaluation["metrics"]
            writer.writerow(
                [
                    backend,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["map50"],
                    metrics["map50_95"],
                    evaluation["elapsed_seconds"],
                ]
            )

    lines = [
        "============================================================",
        " Exp07 Same-Runtime Full Test-Set Consistency Summary",
        "============================================================",
        f"result={'PASS' if overall_pass else 'FAIL'}",
        f"git_branch={summary['git_branch']}",
        f"git_commit={summary['git_commit']}",
        "test_image_count=219",
        "test_instance_count=840",
        f"fp32_result={'PASS' if fp32_pass else 'FAIL'}",
        f"fp16_result={'PASS' if fp16_pass else 'FAIL'}",
    ]
    for backend, evaluation in evaluations.items():
        for metric, value in evaluation["metrics"].items():
            lines.append(f"{backend}_{metric}={value:.12g}")
    for backend, deltas in (
        ("tensorrt_fp32", fp32_deltas),
        ("tensorrt_fp16", fp16_deltas),
    ):
        for metric, value in deltas.items():
            lines.append(f"{backend}_minus_pytorch_{metric}={value:.12g}")
    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print((report_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
