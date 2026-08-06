#!/usr/bin/env python3
"""Compare one TensorRT engine against ONNX Runtime on a fixed image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-abs-atol", type=float, required=True)
    parser.add_argument("--mean-abs-atol", type=float, required=True)
    parser.add_argument("--relative-l2-atol", type=float, required=True)
    parser.add_argument("--box-atol", type=float, required=True)
    parser.add_argument("--confidence-atol", type=float, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


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
            f"letterbox shape mismatch: {letterboxed.shape[:2]}"
        )
    return letterboxed, {
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
    tensor = np.expand_dims(chw.astype(np.float32) / 255.0, axis=0)
    metadata.update(
        {
            "input_shape": list(tensor.shape),
            "input_dtype": str(tensor.dtype),
            "input_min": float(tensor.min()),
            "input_max": float(tensor.max()),
        }
    )
    return tensor, metadata


def torch_dtype(data_type: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if data_type not in mapping:
        raise TypeError(f"unsupported TensorRT dtype: {data_type}")
    return mapping[data_type]


def run_tensorrt(
    engine_path: Path,
    input_array: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable on Jetson")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    io_metadata: list[dict[str, Any]] = []
    input_names: list[str] = []
    output_names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        engine_shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        io_metadata.append(
            {
                "name": name,
                "mode": str(mode),
                "engine_shape": list(engine_shape),
                "dtype": str(dtype),
            }
        )
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    if len(input_names) != 1:
        raise RuntimeError(f"expected one engine input, got {input_names}")
    if not output_names:
        raise RuntimeError("engine has no output tensors")

    input_name = input_names[0]
    expected_shape = tuple(engine.get_tensor_shape(input_name))
    if -1 in expected_shape:
        if not context.set_input_shape(input_name, tuple(input_array.shape)):
            raise RuntimeError("failed to set dynamic TensorRT input shape")
    elif expected_shape != tuple(input_array.shape):
        raise RuntimeError(
            f"engine input shape {expected_shape} != {input_array.shape}"
        )

    input_tensor = torch.from_numpy(input_array).to(
        device="cuda",
        dtype=torch_dtype(engine.get_tensor_dtype(input_name)),
    )
    tensors: dict[str, torch.Tensor] = {input_name: input_tensor}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        if not shape or any(dimension <= 0 for dimension in shape):
            raise RuntimeError(f"invalid runtime output shape for {name}: {shape}")
        tensors[name] = torch.empty(
            shape,
            device="cuda",
            dtype=torch_dtype(engine.get_tensor_dtype(name)),
        )

    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, int(tensor.data_ptr())):
            raise RuntimeError(f"failed to bind TensorRT tensor: {name}")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
    stream.synchronize()

    start = time.perf_counter()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT timed execute_async_v3 returned false")
    stream.synchronize()
    elapsed = time.perf_counter() - start

    outputs = {
        name: tensors[name].detach().float().cpu().numpy()
        for name in output_names
    }
    return outputs, {"tensors": io_metadata}, elapsed


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
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
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
            np.isfinite(reference).all() and np.isfinite(candidate).all()
        ),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()),
        "relative_l2_error": float(np.linalg.norm(difference) / denominator),
    }


def nms_predictions(
    raw_prediction: np.ndarray,
    confidence: float,
    iou: float,
    class_count: int,
) -> np.ndarray:
    results = non_max_suppression(
        torch.from_numpy(raw_prediction.copy()).float(),
        conf_thres=confidence,
        iou_thres=iou,
        nc=class_count,
        max_det=300,
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one NMS result, got {len(results)}")
    return results[0].detach().cpu().numpy()


def sort_detections(detections: np.ndarray) -> np.ndarray:
    if detections.size == 0:
        return detections.reshape(0, 6)
    order = np.lexsort(
        (detections[:, 0], -detections[:, 4], detections[:, 5])
    )
    return detections[order]


def detection_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    reference = sort_detections(reference)
    candidate = sort_detections(candidate)
    if reference.shape[0] != candidate.shape[0]:
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


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    image_path = Path(args.image).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in (onnx_path, engine_path, image_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_array, preprocess = preprocess_image(image_path, args.imgsz)
    providers = ort.get_available_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(f"ORT CPU provider unavailable: {providers}")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = session.get_inputs()
    ort_outputs = session.get_outputs()
    if len(ort_inputs) != 1 or not ort_outputs:
        raise RuntimeError("unexpected ONNX Runtime I/O count")

    ort_start = time.perf_counter()
    ort_values = session.run(None, {ort_inputs[0].name: input_array})
    ort_seconds = time.perf_counter() - ort_start
    ort_by_name = {
        output.name: np.asarray(value, dtype=np.float32)
        for output, value in zip(ort_outputs, ort_values)
    }
    trt_by_name, engine_io, trt_seconds = run_tensorrt(
        engine_path,
        input_array,
    )
    common_outputs = sorted(set(ort_by_name) & set(trt_by_name))
    if len(common_outputs) != 1:
        raise RuntimeError(
            f"expected one common output, got {common_outputs}; "
            f"ORT={list(ort_by_name)}, TRT={list(trt_by_name)}"
        )
    output_name = common_outputs[0]
    ort_prediction = ort_by_name[output_name]
    trt_prediction = trt_by_name[output_name]
    raw = tensor_error(ort_prediction, trt_prediction)
    raw_pass = bool(
        raw["shape_equal"]
        and raw["finite"]
        and raw["max_abs_error"] <= args.max_abs_atol
        and raw["mean_abs_error"] <= args.mean_abs_atol
        and raw["relative_l2_error"] <= args.relative_l2_atol
    )

    class_count = int(ort_prediction.shape[1] - 4)
    if class_count <= 0:
        raise RuntimeError(f"invalid decoded class count: {class_count}")
    ort_detections = nms_predictions(
        ort_prediction,
        args.confidence,
        args.nms_iou,
        class_count,
    )
    trt_detections = nms_predictions(
        trt_prediction,
        args.confidence,
        args.nms_iou,
        class_count,
    )
    detections = detection_error(ort_detections, trt_detections)
    detection_pass = bool(
        detections["count_equal"]
        and detections["classes_equal"]
        and detections["max_box_abs_error"] <= args.box_atol
        and detections["max_confidence_abs_error"]
        <= args.confidence_atol
    )
    overall_pass = bool(raw_pass and detection_pass)

    thresholds = {
        "max_abs_atol": args.max_abs_atol,
        "mean_abs_atol": args.mean_abs_atol,
        "relative_l2_atol": args.relative_l2_atol,
        "box_atol": args.box_atol,
        "confidence_atol": args.confidence_atol,
    }
    artifact_manifest = {
        "onnx": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": sha256_file(onnx_path),
        },
        "engine": {
            "path": str(engine_path),
            "size_bytes": engine_path.stat().st_size,
            "sha256": sha256_file(engine_path),
            "precision": args.precision,
        },
        "probe_image": {
            "path": str(image_path),
            "size_bytes": image_path.stat().st_size,
            "sha256": sha256_file(image_path),
        },
    }
    summary = {
        "experiment": "Exp07 TensorRT consistency smoke",
        "result": "PASS" if overall_pass else "FAIL",
        "precision": args.precision,
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "providers": providers,
        "preprocess": preprocess,
        "onnx_io": {
            "input": {
                "name": ort_inputs[0].name,
                "shape": normalize(ort_inputs[0].shape),
                "type": ort_inputs[0].type,
            },
            "outputs": [
                {
                    "name": output.name,
                    "shape": normalize(output.shape),
                    "type": output.type,
                }
                for output in ort_outputs
            ],
        },
        "engine_io": engine_io,
        "raw_tensor": {
            **raw,
            "result": "PASS" if raw_pass else "FAIL",
        },
        "detections": {
            **detections,
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "result": "PASS" if detection_pass else "FAIL",
        },
        "thresholds": thresholds,
        "timing_seconds": {
            "onnxruntime_single_forward": ort_seconds,
            "tensorrt_single_forward": trt_seconds,
        },
        "artifact_manifest": artifact_manifest,
    }
    summary = normalize(summary)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "tensor_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
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
                "onnxruntime_fp32",
                f"tensorrt_{args.precision}",
                raw["shape_equal"],
                raw["finite"],
                raw["max_abs_error"],
                raw["mean_abs_error"],
                raw["relative_l2_error"],
                "PASS" if raw_pass else "FAIL",
            ]
        )

    lines = [
        "============================================================",
        " Exp07 TensorRT Consistency Summary",
        "============================================================",
        f"result={'PASS' if overall_pass else 'FAIL'}",
        f"precision={args.precision}",
        f"git_branch={summary['git_branch']}",
        f"git_commit={summary['git_commit']}",
        f"onnx_sha256={artifact_manifest['onnx']['sha256']}",
        f"engine_path={engine_path}",
        f"engine_size_bytes={engine_path.stat().st_size}",
        f"engine_sha256={artifact_manifest['engine']['sha256']}",
        f"probe_image_sha256={artifact_manifest['probe_image']['sha256']}",
        f"output_name={output_name}",
        f"output_shape={list(ort_prediction.shape)}",
        f"raw_max_abs_error={raw['max_abs_error']:.12g}",
        f"raw_mean_abs_error={raw['mean_abs_error']:.12g}",
        f"raw_relative_l2_error={raw['relative_l2_error']:.12g}",
        f"raw_tensor_result={'PASS' if raw_pass else 'FAIL'}",
        f"onnxruntime_detection_count={detections['reference_count']}",
        f"tensorrt_detection_count={detections['candidate_count']}",
        f"detection_classes_equal={detections['classes_equal']}",
        f"detection_max_box_abs_error={detections['max_box_abs_error']:.12g}",
        f"detection_max_confidence_abs_error={detections['max_confidence_abs_error']:.12g}",
        f"detection_result={'PASS' if detection_pass else 'FAIL'}",
        f"onnxruntime_single_forward_seconds={ort_seconds:.12g}",
        f"tensorrt_single_forward_seconds={trt_seconds:.12g}",
    ]
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
