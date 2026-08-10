#!/usr/bin/env python3
"""Create coarse mixed-precision candidates by bypassing selected activation Q/DQ pairs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdq-onnx", required=True)
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


def classify(text: str) -> tuple[str, str]:
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


def make_candidate(model: Any, predicate: Callable[[str, str], bool]) -> tuple[Any, list[dict[str, str]]]:
    candidate = copy.deepcopy(model)
    nodes = list(candidate.graph.node)
    consumers: dict[str, list[Any]] = {}
    for node in nodes:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    remove_ids: set[int] = set()
    bypass: dict[str, str] = {}
    removed: list[dict[str, str]] = []
    for node in nodes:
        if node.op_type != "QuantizeLinear":
            continue
        source = node.input[0]
        scale, branch = classify("|".join([node.name, source, node.output[0]]))
        if not predicate(scale, branch):
            continue
        downstream = consumers.get(node.output[0], [])
        if not downstream or any(item.op_type != "DequantizeLinear" for item in downstream):
            raise RuntimeError(f"Q output is not exclusively consumed by DQ: {node.name}")
        remove_ids.add(id(node))
        for dq in downstream:
            remove_ids.add(id(dq))
            bypass[dq.output[0]] = source
        removed.append({"q_node": node.name, "source": source, "scale": scale, "branch": branch})
    if not removed:
        raise RuntimeError("candidate predicate selected no activation Q nodes")
    for node in nodes:
        if id(node) in remove_ids:
            continue
        for index, tensor in enumerate(node.input):
            if tensor in bypass:
                node.input[index] = bypass[tensor]
    kept = [node for node in nodes if id(node) not in remove_ids]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    return candidate, removed


def main() -> int:
    args = parse_args()
    source_path = Path(args.qdq_onnx).resolve()
    output_dir = Path(args.output_dir).resolve()
    package_root = Path(args.onnx_package_root).resolve()
    if not source_path.is_file() or not package_root.is_dir():
        raise FileNotFoundError(source_path if not source_path.is_file() else package_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(package_root))
    import onnx  # pylint: disable=import-outside-toplevel

    model = onnx.load(str(source_path))
    onnx.checker.check_model(model)
    specifications: dict[str, Callable[[str, str], bool]] = {
        "p3_classification": lambda scale, branch: scale == "p3" and branch == "classification",
        "classification": lambda _scale, branch: branch == "classification",
        "dfl": lambda _scale, branch: branch == "dfl",
        "detect_head": lambda _scale, branch: branch in {"classification", "regression", "dfl", "detect_other"},
    }
    records: dict[str, Any] = {}
    for name, predicate in specifications.items():
        candidate, removed = make_candidate(model, predicate)
        onnx.checker.check_model(candidate)
        path = output_dir / f"yolo11n_exp17_mixed_{name}.onnx"
        onnx.save(candidate, str(path))
        reloaded = onnx.load(str(path))
        onnx.checker.check_model(reloaded)
        records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "removed_activation_qdq_pairs": len(removed),
            "removed": removed,
            "quantize_linear": sum(n.op_type == "QuantizeLinear" for n in reloaded.graph.node),
            "dequantize_linear": sum(n.op_type == "DequantizeLinear" for n in reloaded.graph.node),
            "output_contract": [
                {"name": item.name, "element_type": item.type.tensor_type.elem_type}
                for item in reloaded.graph.output
            ],
        }
    summary = {
        "experiment": "Exp17 coarse mixed-precision Q/DQ graph candidates",
        "result": "PASS",
        "method": "bypass selected activation Q->DQ pairs; weights and all other graph nodes unchanged",
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "candidates": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            ["result=PASS"]
            + [
                f"{name} removed_pairs={record['removed_activation_qdq_pairs']} "
                f"Q={record['quantize_linear']} DQ={record['dequantize_linear']} "
                f"sha256={record['sha256']}"
                for name, record in records.items()
            ]
            + [""]
        ),
        encoding="utf-8",
    )
    print(json.dumps({name: record["removed_activation_qdq_pairs"] for name, record in records.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
