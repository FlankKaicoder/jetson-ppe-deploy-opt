#!/usr/bin/env python3
"""Create and audit a small Explicit Q/DQ PTQ ONNX smoke candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2  # Import system NumPy consumers before adding the isolated ONNX path.
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp08-builder", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--onnx-package-root",
        default="/home/nvidia/.local/jetson-ppe-exp16-py",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--calibration-chunk-size", type=int, default=8)
    parser.add_argument(
        "--calibration-method",
        choices=("entropy", "minmax", "percentile"),
        default="entropy",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exp08_preprocess(builder_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("exp08_builder_frozen", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Exp08 builder: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_image


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    manifest_path = Path(args.manifest).resolve()
    builder_path = Path(args.exp08_builder).resolve()
    output_onnx = Path(args.output_onnx).resolve()
    report_dir = Path(args.report_dir).resolve()
    package_root = Path(args.onnx_package_root).resolve()
    for path in (onnx_path, manifest_path, builder_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    if output_onnx.exists():
        raise FileExistsError(f"refusing to overwrite: {output_onnx}")
    report_dir.mkdir(parents=True, exist_ok=True)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "qdq_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite: {summary_path}")

    # Keep Jetson's ABI-compatible NumPy 1.x loaded, then expose the isolated
    # ONNX/protobuf packages without importing their NumPy 2.x copy.
    sys.path.insert(0, str(package_root))
    import onnx  # pylint: disable=import-outside-toplevel
    from onnx import numpy_helper  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    from onnxruntime.quantization import (  # pylint: disable=import-outside-toplevel
        CalibrationDataReader,
        CalibrationMethod,
        QuantType,
    )
    from onnxruntime.quantization.calibrate import (  # pylint: disable=import-outside-toplevel
        TensorsData,
        create_calibrator,
    )
    from onnxruntime.quantization.qdq_quantizer import (  # pylint: disable=import-outside-toplevel
        QDQQuantizer,
    )
    from onnxruntime.quantization.quant_utils import (  # pylint: disable=import-outside-toplevel
        load_model_with_shape_infer,
    )

    method_map = {
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train":
        raise RuntimeError("calibration manifest is not train-only")
    entries = manifest.get("images", [])
    if args.limit <= 0 or args.limit > len(entries):
        raise ValueError(f"invalid limit={args.limit}; available={len(entries)}")
    if args.calibration_chunk_size <= 0:
        raise ValueError("calibration chunk size must be positive")
    entries = entries[: args.limit]
    calibration_root = manifest_path.parent.parent
    image_paths = [
        (calibration_root / entry["archive_path"]).resolve() for entry in entries
    ]
    for entry, image_path in zip(entries, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != entry["sha256"]:
            raise RuntimeError(f"calibration image hash mismatch: {image_path}")

    preprocess_image = load_exp08_preprocess(builder_path)

    class Reader(CalibrationDataReader):
        def __init__(self, start_index: int, end_index: int) -> None:
            self.start_index = start_index
            self.end_index = end_index
            self.index = start_index

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index >= self.end_index:
                return None
            tensor = preprocess_image(image_paths[self.index], args.imgsz)
            self.index += 1
            return {"images": np.expand_dims(tensor, axis=0)}

        def rewind(self) -> None:
            self.index = self.start_index

    op_types_to_quantize = ["Conv", "MatMul", "Softmax"]
    quantizer_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "CalibTensorRangeSymmetric": True,
        "DedicatedQDQPair": True,
        "QuantizeBias": False,
        "OpTypesToExcludeOutputQuantization": ["Conv", "MatMul"],
    }

    start = time.perf_counter()
    source_model = load_model_with_shape_infer(onnx_path)
    with tempfile.TemporaryDirectory(prefix="exp17.quant.") as temporary_dir:
        calibrator = create_calibrator(
            onnx_path,
            op_types_to_calibrate=op_types_to_quantize,
            augmented_model_path=str(Path(temporary_dir) / "augmented_model.onnx"),
            calibrate_method=method_map[args.calibration_method],
            extra_options={"symmetric": True},
        )
        calibration_chunks = 0
        for start_index in range(0, len(image_paths), args.calibration_chunk_size):
            end_index = min(
                start_index + args.calibration_chunk_size,
                len(image_paths),
            )
            calibrator.collect_data(Reader(start_index, end_index))
            calibration_chunks += 1
            print(
                f"calibration_chunk={calibration_chunks} "
                f"images={end_index}/{len(image_paths)}",
                flush=True,
            )
        tensors_range = calibrator.compute_data()
        if not isinstance(tensors_range, TensorsData):
            raise TypeError(f"unexpected calibration range type: {type(tensors_range)}")
        del calibrator
    quantizer = QDQQuantizer(
        source_model,
        True,
        False,
        QuantType.QInt8,
        QuantType.QInt8,
        tensors_range,
        [],
        [],
        op_types_to_quantize,
        quantizer_options,
    )
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(str(output_onnx), False)
    quantization_seconds = time.perf_counter() - start
    if not output_onnx.is_file() or output_onnx.stat().st_size == 0:
        raise RuntimeError("quantizer did not create an ONNX file")

    model = onnx.load(str(output_onnx), load_external_data=False)
    onnx.checker.check_model(model)
    op_counts = Counter(node.op_type for node in model.graph.node)
    q_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dq_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    scale_arrays = []
    zero_point_arrays = []
    unresolved_scales = 0
    unresolved_zero_points = 0
    for node in q_nodes + dq_nodes:
        if len(node.input) >= 2 and node.input[1] in initializers:
            scale_arrays.append(np.asarray(initializers[node.input[1]], dtype=np.float64))
        else:
            unresolved_scales += 1
        if len(node.input) >= 3 and node.input[2] in initializers:
            zero_point_arrays.append(np.asarray(initializers[node.input[2]]))
        else:
            unresolved_zero_points += 1
    all_scales_positive_finite = bool(
        scale_arrays
        and all(np.isfinite(item).all() and (item > 0).all() for item in scale_arrays)
    )
    all_resolved_zero_points_zero = bool(
        zero_point_arrays and all((item == 0).all() for item in zero_point_arrays)
    )
    input_shape = [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    passed = bool(
        q_nodes
        and dq_nodes
        and input_shape == [1, 3, args.imgsz, args.imgsz]
        and output_shape == [1, 7, 8400]
        and all_scales_positive_finite
        and all_resolved_zero_points_zero
    )
    summary = {
        "experiment": "Exp17 Explicit Q/DQ PTQ generation",
        "stage": "formal" if len(image_paths) == 256 else "smoke",
        "result": "PASS" if passed else "FAIL",
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
        },
        "configuration": {
            "calibration_method": args.calibration_method,
            "calibration_images": len(image_paths),
            "calibration_chunk_size": args.calibration_chunk_size,
            "calibration_chunks": calibration_chunks,
            "quant_format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel_weights": True,
            "activation_symmetric": True,
            "weight_symmetric": True,
            "quantize_bias": False,
            "op_types_to_quantize": op_types_to_quantize,
            "op_types_to_exclude_output_quantization": ["Conv", "MatMul"],
        },
        "inputs": {
            "onnx_sha256": sha256_file(onnx_path),
            "manifest_sha256": sha256_file(manifest_path),
            "exp08_builder_sha256": sha256_file(builder_path),
            "calibration_image_sha256": [entry["sha256"] for entry in entries],
        },
        "output": {
            "path": str(output_onnx),
            "sha256": sha256_file(output_onnx),
            "bytes": output_onnx.stat().st_size,
            "node_count": len(model.graph.node),
            "quantize_linear_count": len(q_nodes),
            "dequantize_linear_count": len(dq_nodes),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "top_op_counts": dict(op_counts.most_common(20)),
        },
        "scale_audit": {
            "resolved_scale_inputs": len(scale_arrays),
            "unresolved_scale_inputs": unresolved_scales,
            "all_resolved_scales_positive_finite": all_scales_positive_finite,
            "resolved_zero_point_inputs": len(zero_point_arrays),
            "unresolved_zero_point_inputs": unresolved_zero_points,
            "all_resolved_zero_points_zero": all_resolved_zero_points_zero,
        },
        "quantization_seconds": quantization_seconds,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "qdq_summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"calibration_images={len(image_paths)}",
                f"quantize_linear_count={len(q_nodes)}",
                f"dequantize_linear_count={len(dq_nodes)}",
                f"all_scales_positive_finite={all_scales_positive_finite}",
                f"all_zero_points_zero={all_resolved_zero_points_zero}",
                f"output_onnx_sha256={summary['output']['sha256']}",
                f"quantization_seconds={quantization_seconds:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result={summary['result']} Q={len(q_nodes)} DQ={len(dq_nodes)} "
        f"seconds={quantization_seconds:.3f} output={output_onnx}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
