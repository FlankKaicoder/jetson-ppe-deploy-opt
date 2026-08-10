#!/usr/bin/env python3
"""Audit explicit Q/DQ scale placement and coarse YOLO Detect-head groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output-dir", required=True)
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


def functional_group(text: str) -> tuple[str, str]:
    lower = text.lower()
    scale = "unassigned"
    for index, label in enumerate(("p3", "p4", "p5")):
        if f"model.23/cv2.{index}" in lower or f"model.23/cv3.{index}" in lower:
            scale = label
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
    return scale, branch


def scalar_or_stats(array: np.ndarray) -> dict[str, Any]:
    values = array.astype(np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "all_finite": bool(np.isfinite(values).all()),
        "all_positive": bool((values > 0).all()),
    }


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    output_dir = Path(args.output_dir).resolve()
    package_root = Path(args.onnx_package_root).resolve()
    if not onnx_path.is_file() or not package_root.is_dir():
        raise FileNotFoundError(onnx_path if not onnx_path.is_file() else package_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(package_root))
    import onnx  # pylint: disable=import-outside-toplevel
    from onnx import numpy_helper  # pylint: disable=import-outside-toplevel

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    records: list[dict[str, Any]] = []
    for index, node in enumerate(n for n in model.graph.node if n.op_type == "QuantizeLinear"):
        source, scale_name, zero_name = node.input[:3]
        scale = initializers[scale_name]
        zero = initializers[zero_name]
        downstream = consumers.get(node.output[0], [])
        text = "|".join(
            [node.name, source, node.output[0]]
            + [item.name for item in downstream]
            + [value for item in downstream for value in item.output]
        )
        feature_scale, branch = functional_group(text)
        scale_stats = scalar_or_stats(scale)
        zero_values = zero.astype(np.int64).reshape(-1)
        records.append(
            {
                "index": index,
                "node": node.name,
                "source_tensor": source,
                "quantized_tensor": node.output[0],
                "scale_name": scale_name,
                "zero_point_name": zero_name,
                "granularity": "per_tensor" if scale.size == 1 else "per_channel",
                "scale_count": scale_stats["count"],
                "scale_min": scale_stats["min"],
                "scale_max": scale_stats["max"],
                "scale_mean": scale_stats["mean"],
                "signed_int8_abs_range_min": scale_stats["min"] * 127.0,
                "signed_int8_abs_range_max": scale_stats["max"] * 127.0,
                "zero_point_min": int(zero_values.min()),
                "zero_point_max": int(zero_values.max()),
                "feature_scale": feature_scale,
                "branch": branch,
                "downstream": ";".join(f"{item.op_type}:{item.name}" for item in downstream),
                "valid": bool(
                    scale_stats["all_finite"]
                    and scale_stats["all_positive"]
                    and np.all(zero_values == 0)
                ),
            }
        )

    fieldnames = list(records[0])
    with (output_dir / "qdq_scales.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    group_counts = Counter((item["feature_scale"], item["branch"]) for item in records)
    summary = {
        "experiment": "Exp17 static Q/DQ scale and placement audit",
        "result": "PASS" if records and all(item["valid"] for item in records) else "FAIL",
        "onnx": {"path": str(onnx_path), "bytes": onnx_path.stat().st_size,
                 "sha256": sha256_file(onnx_path)},
        "graph": {
            "nodes": len(model.graph.node),
            "quantize_linear": len(records),
            "dequantize_linear": sum(n.op_type == "DequantizeLinear" for n in model.graph.node),
        },
        "placement": {
            "granularity": dict(Counter(item["granularity"] for item in records)),
            "feature_scale_branch": {
                f"{scale}/{branch}": count
                for (scale, branch), count in sorted(group_counts.items())
            },
            "head_records": [
                item for item in records if item["branch"] != "backbone_neck"
            ],
        },
        "gates": {
            "all_scales_positive_finite": all(item["valid"] for item in records),
            "all_zero_points_zero": all(
                item["zero_point_min"] == item["zero_point_max"] == 0 for item in records
            ),
            "head_group_discovered": any(
                item["branch"] != "backbone_neck" for item in records
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"result={summary['result']}",
                f"quantize_linear={len(records)}",
                f"dequantize_linear={summary['graph']['dequantize_linear']}",
                f"head_records={len(summary['placement']['head_records'])}",
                *[
                    f"group_{name.replace('/', '_')}={count}"
                    for name, count in summary["placement"]["feature_scale_branch"].items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["placement"]["feature_scale_branch"], indent=2))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
