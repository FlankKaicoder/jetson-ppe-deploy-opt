#!/usr/bin/env python3
"""Measure Detect-head activation range, clipping, and simulated INT8 error."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2  # Load the system NumPy ABI consumer before the isolated ONNX packages.
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-onnx", required=True)
    parser.add_argument("--qdq-onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp08-builder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--sample-per-image", type=int, default=2048)
    parser.add_argument(
        "--onnx-package-root", default="/home/nvidia/.local/jetson-ppe-exp16-py"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preprocess(path: Path):
    spec = importlib.util.spec_from_file_location("exp08_builder_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_image


def group(text: str) -> tuple[str, str]:
    lower = text.lower()
    feature_scale = "unassigned"
    for index, label in enumerate(("p3", "p4", "p5")):
        if f"model.23/cv2.{index}" in lower or f"model.23/cv3.{index}" in lower:
            feature_scale = label
            break
    if "model.23/dfl" in lower or "/dfl/" in lower:
        branch = "dfl"
    elif "model.23/cv2" in lower:
        branch = "regression"
    elif "model.23/cv3" in lower:
        branch = "classification"
    elif "model.23" in lower:
        branch = "detect_other"
    else:
        branch = "backbone_neck"
    return feature_scale, branch


def main() -> int:
    args = parse_args()
    fp32_path = Path(args.fp32_onnx).resolve()
    qdq_path = Path(args.qdq_onnx).resolve()
    manifest_path = Path(args.manifest).resolve()
    builder_path = Path(args.exp08_builder).resolve()
    output_dir = Path(args.output_dir).resolve()
    package_root = Path(args.onnx_package_root).resolve()
    for path in (fp32_path, qdq_path, manifest_path, builder_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.limit <= 0 or args.sample_per_image <= 0:
        raise ValueError("limit and sample-per-image must be positive")
    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(package_root))
    import onnx  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    from onnx import TensorProto, helper, numpy_helper  # pylint: disable=import-outside-toplevel

    qdq = onnx.load(str(qdq_path))
    initializers = {item.name: numpy_helper.to_array(item) for item in qdq.graph.initializer}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in qdq.graph.node:
        for tensor in node.input:
            consumers[tensor].append(node)
    selected: list[dict[str, Any]] = []
    for node in (item for item in qdq.graph.node if item.op_type == "QuantizeLinear"):
        text = "|".join([node.name, node.input[0], node.output[0]])
        feature_scale, branch = group(text)
        if branch == "backbone_neck":
            continue
        scale = initializers[node.input[1]].astype(np.float64).reshape(-1)
        if scale.size != 1:
            continue  # activation audit; per-channel entries are weights
        selected.append(
            {
                "tensor": node.input[0],
                "scale": float(scale[0]),
                "feature_scale": feature_scale,
                "branch": branch,
                "q_node": node.name,
            }
        )
    if len(selected) != 20 or len({item["tensor"] for item in selected}) != 20:
        raise RuntimeError(f"expected 20 unique head activations, got {len(selected)}")

    fp32 = onnx.load(str(fp32_path))
    available = {value for node in fp32.graph.node for value in node.output}
    missing = [item["tensor"] for item in selected if item["tensor"] not in available]
    if missing:
        raise RuntimeError(f"selected tensors missing from FP32 graph: {missing}")
    existing_outputs = {item.name for item in fp32.graph.output}
    for item in selected:
        if item["tensor"] not in existing_outputs:
            fp32.graph.output.append(
                helper.make_tensor_value_info(item["tensor"], TensorProto.FLOAT, None)
            )
    with tempfile.TemporaryDirectory(prefix="exp17_activation_") as temporary:
        augmented = Path(temporary) / "augmented.onnx"
        onnx.save(fp32, str(augmented))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        available_providers = ort.get_available_providers()
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available_providers else "CPUExecutionProvider"
        session = ort.InferenceSession(str(augmented), sess_options=options, providers=[provider])

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("images", [])
        if manifest.get("source_split") != "train" or args.limit > len(entries):
            raise RuntimeError("invalid train calibration manifest or limit")
        entries = entries[: args.limit]
        root = manifest_path.parent.parent
        images = [(root / entry["archive_path"]).resolve() for entry in entries]
        for entry, image in zip(entries, images):
            if not image.is_file() or sha256_file(image) != entry["sha256"]:
                raise RuntimeError(f"calibration image validation failed: {image}")
        preprocess = load_preprocess(builder_path)
        names = [item["tensor"] for item in selected]
        stats: dict[str, dict[str, Any]] = {
            item["tensor"]: {
                "count": 0,
                "min": math.inf,
                "max": -math.inf,
                "clip_low": 0,
                "clip_high": 0,
                "signal_sq": 0.0,
                "error_sq": 0.0,
                "dot": 0.0,
                "dequant_sq": 0.0,
                "max_abs_error": 0.0,
                "samples": [],
            }
            for item in selected
        }
        started = time.perf_counter()
        for image_index, image in enumerate(images):
            input_array = np.expand_dims(preprocess(image, 640), axis=0)
            outputs = session.run(names, {session.get_inputs()[0].name: input_array})
            for item, output in zip(selected, outputs):
                values = np.asarray(output, dtype=np.float32).reshape(-1)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite activation: {item['tensor']}")
                scale = item["scale"]
                lower, upper = -128.0 * scale, 127.0 * scale
                quantized = np.clip(np.rint(values / scale), -128, 127)
                dequantized = quantized * scale
                error = values - dequantized
                state = stats[item["tensor"]]
                state["count"] += int(values.size)
                state["min"] = min(state["min"], float(values.min()))
                state["max"] = max(state["max"], float(values.max()))
                state["clip_low"] += int(np.count_nonzero(values < lower))
                state["clip_high"] += int(np.count_nonzero(values > upper))
                state["signal_sq"] += float(np.dot(values, values))
                state["error_sq"] += float(np.dot(error, error))
                state["dot"] += float(np.dot(values, dequantized))
                state["dequant_sq"] += float(np.dot(dequantized, dequantized))
                state["max_abs_error"] = max(state["max_abs_error"], float(np.abs(error).max()))
                stride = max(1, values.size // args.sample_per_image)
                state["samples"].append(values[::stride][: args.sample_per_image].copy())
            if (image_index + 1) % 8 == 0 or image_index + 1 == len(images):
                print(f"progress={image_index + 1}/{len(images)}", flush=True)
        elapsed = time.perf_counter() - started

    records: list[dict[str, Any]] = []
    for item in selected:
        state = stats[item["tensor"]]
        samples = np.concatenate(state.pop("samples"))
        count = state["count"]
        clip_count = state["clip_low"] + state["clip_high"]
        denominator = math.sqrt(state["signal_sq"] * state["dequant_sq"])
        records.append(
            {
                **item,
                "count": count,
                "observed_min": state["min"],
                "observed_max": state["max"],
                "sample_p001": float(np.percentile(samples, 0.1)),
                "sample_p01": float(np.percentile(samples, 1)),
                "sample_p50": float(np.percentile(samples, 50)),
                "sample_p99": float(np.percentile(samples, 99)),
                "sample_p999": float(np.percentile(samples, 99.9)),
                "representable_low": -128.0 * item["scale"],
                "representable_high": 127.0 * item["scale"],
                "clip_low": state["clip_low"],
                "clip_high": state["clip_high"],
                "clip_ratio": clip_count / count,
                "relative_l2": math.sqrt(state["error_sq"] / state["signal_sq"])
                if state["signal_sq"] else 0.0,
                "cosine": state["dot"] / denominator if denominator else 1.0,
                "max_abs_error": state["max_abs_error"],
            }
        )
    with (output_dir / "activation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    ranked_clipping = sorted(records, key=lambda item: item["clip_ratio"], reverse=True)
    ranked_error = sorted(records, key=lambda item: item["relative_l2"], reverse=True)
    summary = {
        "experiment": "Exp17 Detect-head activation/dynamic-range/clipping audit",
        "result": "PASS",
        "configuration": {
            "images": args.limit,
            "source_split": "train",
            "provider": provider,
            "sample_per_image_per_tensor": args.sample_per_image,
            "elapsed_seconds": elapsed,
            "tensors": len(records),
        },
        "inputs": {
            "fp32_onnx_sha256": sha256_file(fp32_path),
            "qdq_onnx_sha256": sha256_file(qdq_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "top_clipping": ranked_clipping[:10],
        "top_relative_l2": ranked_error[:10],
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                "result=PASS",
                f"images={args.limit}",
                f"provider={provider}",
                f"elapsed_seconds={elapsed:.6f}",
                f"max_clip_ratio={ranked_clipping[0]['clip_ratio']:.12g}",
                f"max_clip_tensor={ranked_clipping[0]['tensor']}",
                f"max_relative_l2={ranked_error[0]['relative_l2']:.12g}",
                f"max_relative_l2_tensor={ranked_error[0]['tensor']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
