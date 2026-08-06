#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
MAX_F1_METRIC_NAMES = {"precision", "recall"}
AP_METRIC_NAMES = {"map50", "map50_95"}


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def resolve_dataset_images(
    yaml_path: Path,
    split: str,
) -> tuple[list[Path], dict[int, str]]:
    config = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    split_value = config.get(split)
    if split_value is None:
        raise KeyError(
            f"dataset YAML does not contain split: {split}"
        )

    entries = (
        split_value
        if isinstance(split_value, list)
        else [split_value]
    )
    images: list[Path] = []

    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()

        if path.is_dir():
            images.extend(
                file
                for file in path.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                images.append(image_path.resolve())
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(
                f"dataset split path not found: {path}"
            )

    names_value = config.get("names", {})
    if isinstance(names_value, list):
        names = {
            index: str(name)
            for index, name in enumerate(names_value)
        }
    else:
        names = {
            int(index): str(name)
            for index, name in names_value.items()
        }

    resolved = sorted(set(images))
    if not resolved:
        raise RuntimeError(
            f"no images found for dataset split: {split}"
        )
    return resolved, names


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [
        index
        for index, value in enumerate(parts)
        if value == "images"
    ]
    if not indices:
        return image_path.with_suffix(".txt")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_count(image_path: Path) -> int:
    label_path = image_to_label_path(image_path)
    if not label_path.is_file():
        return 0
    return sum(
        1
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )


def select_probe_image(images: list[Path]) -> tuple[Path, int]:
    ranked = sorted(
        ((label_count(path), str(path), path) for path in images),
        key=lambda item: (-item[0], item[1]),
    )
    count, _, path = ranked[0]
    return path, count


def letterbox_image(
    image: np.ndarray,
    size: int,
    padding_value: int = 114,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))

    if (resized_width, resized_height) != (width, height):
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized = image.copy()

    width_padding = size - resized_width
    height_padding = size - resized_height
    half_width = width_padding / 2
    half_height = height_padding / 2

    left = int(round(half_width - 0.1))
    right = int(round(half_width + 0.1))
    top = int(round(half_height - 0.1))
    bottom = int(round(half_height + 0.1))

    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(padding_value,) * 3,
    )
    if letterboxed.shape[:2] != (size, size):
        raise RuntimeError(
            "letterbox shape mismatch: "
            f"{letterboxed.shape[:2]} != {(size, size)}"
        )

    metadata = {
        "original_height": height,
        "original_width": width,
        "resize_ratio": ratio,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "padding_left": left,
        "padding_right": right,
        "padding_top": top,
        "padding_bottom": bottom,
    }
    return letterboxed, metadata


def preprocess_image(
    image_path: Path,
    imgsz: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    letterboxed, metadata = letterbox_image(image, imgsz)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    tensor = chw.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    metadata.update(
        {
            "image_path": str(image_path),
            "input_shape": list(tensor.shape),
            "input_dtype": str(tensor.dtype),
            "input_min": float(tensor.min()),
            "input_max": float(tensor.max()),
        }
    )
    return tensor, metadata


def extract_pytorch_prediction(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise TypeError(
        "unable to locate decoded prediction tensor in "
        f"PyTorch output type {type(output).__name__}"
    )


def tensor_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": False,
            "max_abs_error": math.inf,
            "mean_abs_error": math.inf,
            "relative_l2_error": math.inf,
        }

    difference = reference.astype(np.float64) - candidate.astype(
        np.float64
    )
    absolute = np.abs(difference)
    denominator = max(
        float(np.linalg.norm(reference.astype(np.float64))),
        1e-12,
    )
    return {
        "shape_equal": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "finite": bool(
            np.isfinite(reference).all()
            and np.isfinite(candidate).all()
        ),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()),
        "relative_l2_error": float(
            np.linalg.norm(difference) / denominator
        ),
    }


def nms_predictions(
    raw_prediction: np.ndarray,
    confidence: float,
    iou: float,
    class_count: int,
) -> np.ndarray:
    prediction = torch.from_numpy(raw_prediction.copy()).float()
    results = non_max_suppression(
        prediction,
        conf_thres=confidence,
        iou_thres=iou,
        nc=class_count,
        max_det=300,
    )
    if len(results) != 1:
        raise RuntimeError(
            f"expected one NMS result, received {len(results)}"
        )
    return results[0].detach().cpu().numpy()


def sort_detections(detections: np.ndarray) -> np.ndarray:
    if detections.size == 0:
        return detections.reshape(0, 6)
    order = np.lexsort(
        (
            detections[:, 0],
            -detections[:, 4],
            detections[:, 5],
        )
    )
    return detections[order]


def detection_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    reference = sort_detections(reference)
    candidate = sort_detections(candidate)
    count_equal = reference.shape[0] == candidate.shape[0]
    if not count_equal:
        return {
            "count_equal": False,
            "reference_count": int(reference.shape[0]),
            "candidate_count": int(candidate.shape[0]),
            "classes_equal": False,
            "max_box_abs_error": math.inf,
            "max_confidence_abs_error": math.inf,
        }

    if reference.shape[0] == 0:
        return {
            "count_equal": True,
            "reference_count": 0,
            "candidate_count": 0,
            "classes_equal": True,
            "max_box_abs_error": 0.0,
            "max_confidence_abs_error": 0.0,
        }

    return {
        "count_equal": True,
        "reference_count": int(reference.shape[0]),
        "candidate_count": int(candidate.shape[0]),
        "classes_equal": bool(
            np.array_equal(reference[:, 5], candidate[:, 5])
        ),
        "max_box_abs_error": float(
            np.abs(reference[:, :4] - candidate[:, :4]).max()
        ),
        "max_confidence_abs_error": float(
            np.abs(reference[:, 4] - candidate[:, 4]).max()
        ),
    }


def metric_subset(results_dict: dict[str, Any]) -> dict[str, float]:
    subset: dict[str, float] = {}
    for name, key in METRIC_KEYS.items():
        if key not in results_dict:
            raise KeyError(f"validation metric missing: {key}")
        subset[name] = float(results_dict[key])
    return subset


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
) -> dict[str, Any]:
    model = YOLO(str(model_path), task="detect")
    start = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        rect=False,
        plots=False,
        save_json=False,
        project=str(project),
        name=name,
        exist_ok=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    results_dict = normalize(metrics.results_dict)
    return {
        "model_path": str(model_path),
        "device": device,
        "batch": batch,
        "workers": workers,
        "rect": False,
        "elapsed_seconds": elapsed,
        "metrics": metric_subset(results_dict),
        "all_results": results_dict,
        "speed_ms_per_image": normalize(metrics.speed),
    }


def write_metric_csv(
    path: Path,
    evaluations: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-abs-atol", type=float, default=1e-3)
    parser.add_argument("--mean-abs-atol", type=float, default=2e-5)
    parser.add_argument("--relative-l2-atol", type=float, default=1e-5)
    parser.add_argument("--box-atol", type=float, default=1e-3)
    parser.add_argument("--confidence-atol", type=float, default=1e-5)
    parser.add_argument("--max-f1-metric-atol", type=float, default=5e-4)
    parser.add_argument("--ap-metric-atol", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--full-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    report_dir = Path(args.report_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"valid weights not found: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data YAML not found: {data_yaml}")

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if any(artifact_dir.iterdir()):
        raise FileExistsError(
            f"artifact directory is not empty: {artifact_dir}"
        )

    weight_sha256 = sha256_file(weights)
    if weight_sha256 != args.expected_weight_sha256:
        raise RuntimeError(
            "frozen weight SHA256 mismatch: "
            f"{weight_sha256} != {args.expected_weight_sha256}"
        )

    images, names = resolve_dataset_images(data_yaml, "test")
    probe_image, probe_label_count = select_probe_image(images)
    input_array, preprocess_metadata = preprocess_image(
        probe_image,
        args.imgsz,
    )

    source_copy = artifact_dir / "source_best.pt"
    shutil.copy2(weights, source_copy)
    copied_sha256 = sha256_file(source_copy)
    if copied_sha256 != weight_sha256:
        raise RuntimeError("copied source weight SHA256 mismatch")

    print("========== Exp06 export configuration ==========")
    print(f"run_kind={args.run_kind}")
    print(f"repo_root={repo_root}")
    print(f"weights={weights}")
    print(f"weight_sha256={weight_sha256}")
    print(f"data_yaml={data_yaml}")
    print(f"test_image_count={len(images)}")
    print(f"probe_image={probe_image}")
    print(f"probe_label_count={probe_label_count}")
    print(f"artifact_dir={artifact_dir}")
    print(f"imgsz={args.imgsz}")
    print(f"batch={args.batch}")
    print(f"opset={args.opset}")

    export_model = YOLO(str(source_copy), task="detect")
    export_start = time.perf_counter()
    exported_value = export_model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=False,
        simplify=False,
        nms=False,
        optimize=False,
        device="cpu",
        verbose=False,
    )
    export_seconds = time.perf_counter() - export_start
    exported_path = Path(str(exported_value)).resolve()
    if not exported_path.is_file() or exported_path.stat().st_size == 0:
        raise RuntimeError(
            f"Ultralytics did not produce a valid ONNX file: {exported_path}"
        )

    onnx_path = artifact_dir / (
        f"yolo11n_baseline_exp06_b{args.batch}_"
        f"{args.imgsz}_opset{args.opset}.onnx"
    )
    if onnx_path.exists():
        raise FileExistsError(f"target ONNX already exists: {onnx_path}")
    if exported_path != onnx_path:
        exported_path.rename(onnx_path)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_sha256 = sha256_file(onnx_path)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = session.get_inputs()
    ort_outputs = session.get_outputs()
    if len(ort_inputs) != 1:
        raise RuntimeError(
            f"expected one ONNX input, received {len(ort_inputs)}"
        )
    if len(ort_outputs) != 1:
        raise RuntimeError(
            f"expected one ONNX output, received {len(ort_outputs)}"
        )

    input_name = ort_inputs[0].name
    ort_start = time.perf_counter()
    ort_values = session.run(None, {input_name: input_array})
    ort_forward_seconds = time.perf_counter() - ort_start
    ort_prediction = np.asarray(ort_values[0], dtype=np.float32)

    pytorch_yolo = YOLO(str(weights), task="detect")
    pytorch_model = pytorch_yolo.model.float().eval().cpu()
    input_tensor = torch.from_numpy(input_array)
    with torch.no_grad():
        pytorch_start = time.perf_counter()
        pytorch_output = pytorch_model(input_tensor)
        pytorch_forward_seconds = time.perf_counter() - pytorch_start
    pytorch_prediction_tensor = extract_pytorch_prediction(
        pytorch_output
    )
    pytorch_prediction = (
        pytorch_prediction_tensor.detach().cpu().numpy().astype(np.float32)
    )

    raw_error = tensor_error(
        pytorch_prediction,
        ort_prediction,
    )
    raw_pass = bool(
        raw_error["shape_equal"]
        and raw_error["finite"]
        and raw_error["max_abs_error"] <= args.max_abs_atol
        and raw_error["mean_abs_error"] <= args.mean_abs_atol
        and raw_error["relative_l2_error"] <= args.relative_l2_atol
    )

    class_count = len(names)
    pytorch_detections = nms_predictions(
        pytorch_prediction,
        args.confidence,
        args.nms_iou,
        class_count,
    )
    ort_detections = nms_predictions(
        ort_prediction,
        args.confidence,
        args.nms_iou,
        class_count,
    )
    detection_metrics = detection_error(
        pytorch_detections,
        ort_detections,
    )
    detection_pass = bool(
        detection_metrics["count_equal"]
        and detection_metrics["classes_equal"]
        and detection_metrics["max_box_abs_error"] <= args.box_atol
        and detection_metrics["max_confidence_abs_error"]
        <= args.confidence_atol
    )

    evaluations: dict[str, dict[str, Any]] = {}
    evaluation_deltas: dict[str, float] = {}
    evaluation_pass = True

    if args.full_eval:
        evaluation_root = artifact_dir / "validation_runs"
        evaluations["pytorch"] = evaluate_model(
            weights,
            data_yaml,
            args.imgsz,
            1,
            "0",
            args.workers,
            evaluation_root,
            "pytorch_test",
        )
        evaluations["onnxruntime"] = evaluate_model(
            onnx_path,
            data_yaml,
            args.imgsz,
            1,
            "cpu",
            args.workers,
            evaluation_root,
            "onnxruntime_test",
        )
        for metric_name in METRIC_KEYS:
            delta = (
                evaluations["onnxruntime"]["metrics"][metric_name]
                - evaluations["pytorch"]["metrics"][metric_name]
            )
            evaluation_deltas[metric_name] = float(delta)
        evaluation_pass = all(
            abs(evaluation_deltas[name]) <= args.max_f1_metric_atol
            for name in MAX_F1_METRIC_NAMES
        ) and all(
            abs(evaluation_deltas[name]) <= args.ap_metric_atol
            for name in AP_METRIC_NAMES
        )
        write_metric_csv(
            report_dir / "evaluation_metrics.csv",
            evaluations,
        )

    overall_pass = bool(
        raw_pass
        and detection_pass
        and evaluation_pass
    )

    artifact_manifest = {
        "source_weight": {
            "original_path": str(weights),
            "artifact_copy": str(source_copy),
            "size_bytes": source_copy.stat().st_size,
            "sha256": copied_sha256,
        },
        "onnx": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": onnx_sha256,
            "opset": args.opset,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "dynamic": False,
            "simplify": False,
            "half": False,
            "nms": False,
        },
    }

    summary = {
        "experiment": "Exp06 PyTorch to ONNX export and consistency",
        "run_kind": args.run_kind,
        "result": "PASS" if overall_pass else "FAIL",
        "repo_root": str(repo_root),
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "python_executable": sys.executable,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "data_yaml": str(data_yaml),
        "test_image_count": len(images),
        "class_names": names,
        "probe_image": str(probe_image),
        "probe_label_count": probe_label_count,
        "preprocess": preprocess_metadata,
        "onnx_checker": "PASS",
        "onnx_input": {
            "name": ort_inputs[0].name,
            "shape": normalize(ort_inputs[0].shape),
            "type": ort_inputs[0].type,
        },
        "onnx_outputs": [
            {
                "name": output.name,
                "shape": normalize(output.shape),
                "type": output.type,
            }
            for output in ort_outputs
        ],
        "raw_tensor": {
            **raw_error,
            "thresholds": {
                "max_abs_atol": args.max_abs_atol,
                "mean_abs_atol": args.mean_abs_atol,
                "relative_l2_atol": args.relative_l2_atol,
            },
            "result": "PASS" if raw_pass else "FAIL",
        },
        "detections": {
            **detection_metrics,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "box_atol": args.box_atol,
            "confidence_atol": args.confidence_atol,
            "result": "PASS" if detection_pass else "FAIL",
        },
        "timing_seconds": {
            "export": export_seconds,
            "pytorch_probe_forward": pytorch_forward_seconds,
            "onnxruntime_probe_forward": ort_forward_seconds,
        },
        "artifact_manifest": artifact_manifest,
        "full_evaluation": {
            "enabled": args.full_eval,
            "thresholds": {
                "precision_recall_at_max_f1_atol": args.max_f1_metric_atol,
                "average_precision_atol": args.ap_metric_atol,
            },
            "precision_recall_note": (
                "Ultralytics reports precision and recall at the confidence index "
                "selected by the smoothed maximum mean F1 curve; tiny confidence "
                "perturbations can move that index. AP metrics remain separately "
                "guarded by the stricter threshold."
            ),
            "evaluations": evaluations,
            "onnx_minus_pytorch": evaluation_deltas,
            "result": "PASS" if evaluation_pass else "FAIL",
        },
    }

    (report_dir / "artifact_manifest.json").write_text(
        json.dumps(
            artifact_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "tensor_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reference",
                "candidate",
                "shape_equal",
                "finite",
                "max_abs_error",
                "mean_abs_error",
                "relative_l2_error",
                "result",
            ]
        )
        writer.writerow(
            [
                "pytorch_fp32",
                "onnxruntime_fp32",
                raw_error["shape_equal"],
                raw_error["finite"],
                raw_error["max_abs_error"],
                raw_error["mean_abs_error"],
                raw_error["relative_l2_error"],
                "PASS" if raw_pass else "FAIL",
            ]
        )

    lines = [
        "============================================================",
        " Exp06 PyTorch to ONNX Summary",
        "============================================================",
        f"result={'PASS' if overall_pass else 'FAIL'}",
        f"run_kind={args.run_kind}",
        f"git_branch={summary['git_branch']}",
        f"git_commit={summary['git_commit']}",
        f"source_weight={weights}",
        f"source_weight_sha256={weight_sha256}",
        f"onnx_path={onnx_path}",
        f"onnx_size_bytes={onnx_path.stat().st_size}",
        f"onnx_sha256={onnx_sha256}",
        f"onnx_input_name={ort_inputs[0].name}",
        f"onnx_input_shape={normalize(ort_inputs[0].shape)}",
        f"onnx_output_name={ort_outputs[0].name}",
        f"onnx_output_shape={normalize(ort_outputs[0].shape)}",
        f"probe_image={probe_image}",
        f"probe_label_count={probe_label_count}",
        f"raw_max_abs_error={raw_error['max_abs_error']:.12g}",
        f"raw_mean_abs_error={raw_error['mean_abs_error']:.12g}",
        f"raw_relative_l2_error={raw_error['relative_l2_error']:.12g}",
        f"raw_tensor_result={'PASS' if raw_pass else 'FAIL'}",
        f"pytorch_detection_count={detection_metrics['reference_count']}",
        f"onnxruntime_detection_count={detection_metrics['candidate_count']}",
        f"detection_max_box_abs_error={detection_metrics['max_box_abs_error']:.12g}",
        f"detection_max_confidence_abs_error={detection_metrics['max_confidence_abs_error']:.12g}",
        f"detection_result={'PASS' if detection_pass else 'FAIL'}",
        f"full_evaluation_enabled={args.full_eval}",
        f"full_evaluation_result={'PASS' if evaluation_pass else 'FAIL'}",
    ]
    if args.full_eval:
        for backend, evaluation in evaluations.items():
            for metric_name, value in evaluation["metrics"].items():
                lines.append(
                    f"{backend}_{metric_name}={value:.12g}"
                )
        for metric_name, delta in evaluation_deltas.items():
            lines.append(
                f"onnx_minus_pytorch_{metric_name}={delta:.12g}"
            )
    (report_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print((report_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        print(
            f"FATAL: {type(exception).__name__}: {exception}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
