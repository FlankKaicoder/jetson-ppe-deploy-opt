#!/usr/bin/env python3
"""Audit whether the frozen Exp08 path uses implicit calibration or explicit Q/DQ."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


IMPLICIT_MARKERS = (
    "trt.IInt8EntropyCalibrator2",
    "trt.BuilderFlag.INT8",
    "config.int8_calibrator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--exp08-builder", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--fp16-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expect-mode",
        choices=("implicit", "explicit"),
        default="implicit",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value_info: Any) -> list[int | str | None]:
    dims: list[int | str | None] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def tensor_description(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "element_type": int(tensor_type.elem_type),
        "shape": tensor_shape(value_info),
    }


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    paths = {
        "onnx": Path(args.onnx).resolve(),
        "exp08_builder": Path(args.exp08_builder).resolve(),
        "calibration_cache": Path(args.calibration_cache).resolve(),
        "calibration_manifest": Path(args.calibration_manifest).resolve(),
        "fp16_engine": Path(args.fp16_engine).resolve(),
        "int8_engine": Path(args.int8_engine).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite: {summary_path}")

    source_lines = paths["exp08_builder"].read_text(encoding="utf-8").splitlines()
    source_evidence: dict[str, list[int]] = {}
    for marker in IMPLICIT_MARKERS:
        source_evidence[marker] = [
            index
            for index, line in enumerate(source_lines, start=1)
            if marker in line
        ]

    model = onnx.load(str(paths["onnx"]), load_external_data=False)
    onnx.checker.check_model(model)
    op_counts = Counter(node.op_type for node in model.graph.node)
    q_count = int(op_counts.get("QuantizeLinear", 0))
    dq_count = int(op_counts.get("DequantizeLinear", 0))
    all_implicit_markers_present = all(source_evidence.values())
    if q_count or dq_count:
        detected_mode = "explicit"
    elif all_implicit_markers_present:
        detected_mode = "implicit"
    else:
        detected_mode = "unclassified"

    manifest = json.loads(paths["calibration_manifest"].read_text(encoding="utf-8"))
    calibration_images = manifest.get("images", [])
    result = "PASS" if detected_mode == args.expect_mode else "FAIL"
    summary = {
        "experiment": "Exp17 R0 quantization mode audit",
        "result": result,
        "expected_mode": args.expect_mode,
        "detected_mode": detected_mode,
        "interpretation": (
            "Exp08 uses implicit TensorRT INT8 calibration/cache; Explicit Q/DQ "
            "PTQ baseline is required before sensitivity and mixed precision."
            if detected_mode == "implicit"
            else "See detected_mode and evidence before choosing the Exp17 path."
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "onnx": onnx.__version__,
        },
        "toolchain_availability": {
            "modelopt": module_available("modelopt"),
            "modelopt_onnx": module_available("modelopt.onnx"),
            "onnxruntime": module_available("onnxruntime"),
            "onnx_graphsurgeon": module_available("onnx_graphsurgeon"),
            "tensorrt": module_available("tensorrt"),
            "torch": module_available("torch"),
        },
        "source_evidence": {
            "path": str(paths["exp08_builder"]),
            "sha256": sha256_file(paths["exp08_builder"]),
            "markers": source_evidence,
            "all_implicit_markers_present": all_implicit_markers_present,
        },
        "onnx": {
            "path": str(paths["onnx"]),
            "sha256": sha256_file(paths["onnx"]),
            "node_count": len(model.graph.node),
            "opset_imports": [
                {"domain": item.domain, "version": int(item.version)}
                for item in model.opset_import
            ],
            "quantize_linear_count": q_count,
            "dequantize_linear_count": dq_count,
            "inputs": [tensor_description(item) for item in model.graph.input],
            "outputs": [tensor_description(item) for item in model.graph.output],
            "top_op_counts": dict(op_counts.most_common(20)),
        },
        "calibration": {
            "manifest_path": str(paths["calibration_manifest"]),
            "manifest_sha256": sha256_file(paths["calibration_manifest"]),
            "source_split": manifest.get("source_split"),
            "image_count": len(calibration_images),
            "cache_path": str(paths["calibration_cache"]),
            "cache_sha256": sha256_file(paths["calibration_cache"]),
            "cache_bytes": paths["calibration_cache"].stat().st_size,
        },
        "engines": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
                "bytes": paths[name].stat().st_size,
            }
            for name in ("fp16_engine", "int8_engine")
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={result}",
                f"expected_mode={args.expect_mode}",
                f"detected_mode={detected_mode}",
                f"onnx_nodes={len(model.graph.node)}",
                f"quantize_linear_count={q_count}",
                f"dequantize_linear_count={dq_count}",
                f"calibration_images={len(calibration_images)}",
                f"modelopt_available={summary['toolchain_availability']['modelopt']}",
                f"onnxruntime_available={summary['toolchain_availability']['onnxruntime']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result={result} detected_mode={detected_mode} "
        f"Q={q_count} DQ={dq_count} calibration_images={len(calibration_images)}",
        flush=True,
    )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
