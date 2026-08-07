#!/usr/bin/env python3
"""Execute one INT8 engine input and verify shape and finite outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from exp07_trt_consistency import (
    nms_predictions,
    preprocess_image,
    run_tensorrt,
    sha256_file,
    tensor_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    image_path = Path(args.image).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in (onnx_path, engine_path, image_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_array, preprocess = preprocess_image(image_path, args.imgsz)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_output = np.asarray(
        session.run(None, {session.get_inputs()[0].name: input_array})[0],
        dtype=np.float32,
    )
    trt_outputs, engine_io, elapsed = run_tensorrt(engine_path, input_array)
    if len(trt_outputs) != 1:
        raise RuntimeError(f"expected one TensorRT output: {list(trt_outputs)}")
    output_name, trt_output = next(iter(trt_outputs.items()))
    raw = tensor_error(ort_output, trt_output)
    shape_expected = list(ort_output.shape) == [1, 7, 8400]
    finite = bool(np.isfinite(trt_output).all())
    class_count = int(ort_output.shape[1] - 4)
    ort_detections = nms_predictions(
        ort_output, args.confidence, args.nms_iou, class_count
    )
    trt_detections = nms_predictions(
        trt_output, args.confidence, args.nms_iou, class_count
    )
    passed = bool(raw["shape_equal"] and shape_expected and finite)
    summary = {
        "experiment": "Exp08 INT8 execution smoke",
        "result": "PASS" if passed else "FAIL",
        "onnx_sha256": sha256_file(onnx_path),
        "engine_sha256": sha256_file(engine_path),
        "probe_image_sha256": sha256_file(image_path),
        "preprocess": preprocess,
        "output_name": output_name,
        "output_shape": list(trt_output.shape),
        "finite": finite,
        "engine_io": engine_io,
        "raw_tensor_diagnostic": raw,
        "nms_diagnostic": {
            "confidence": args.confidence,
            "iou": args.nms_iou,
            "onnx_detection_count": int(ort_detections.shape[0]),
            "int8_detection_count": int(trt_detections.shape[0]),
        },
        "tensorrt_single_forward_seconds": elapsed,
        "note": "Raw/NMS deltas are diagnostic only; full-test gates are frozen separately.",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"engine_sha256={summary['engine_sha256']}",
                f"output_shape={summary['output_shape']}",
                f"finite={summary['finite']}",
                f"raw_max_abs_error={raw['max_abs_error']}",
                f"raw_mean_abs_error={raw['mean_abs_error']}",
                f"raw_relative_l2_error={raw['relative_l2_error']}",
                f"onnx_detection_count={ort_detections.shape[0]}",
                f"int8_detection_count={trt_detections.shape[0]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
