#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


PLUGIN_NAME = "PpeYoloDecodeCompact"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "com.flankkaicoder.ppe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--retain-raw-output", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("output and manifest must not already exist")

    model = onnx.load(args.input)
    graph = gs.import_onnx(model)
    if len(graph.outputs) != 1:
        raise RuntimeError("expected exactly one original graph output")
    raw = graph.outputs[0]
    if raw.name != "output0" or list(raw.shape) != [1, 7, 8400] or raw.dtype != np.float32:
        raise RuntimeError(f"unexpected raw output contract: {raw.name} {raw.shape} {raw.dtype}")

    boxes_scores = gs.Variable("boxes_scores", dtype=np.float32, shape=[1, 8400, 5])
    classes = gs.Variable("classes", dtype=np.int32, shape=[1, 8400])
    indices = gs.Variable("indices", dtype=np.int32, shape=[1, 8400])
    count = gs.Variable("count", dtype=np.int32, shape=[1])
    node = gs.Node(
        op=PLUGIN_NAME,
        name="exp16_postprocess",
        domain=PLUGIN_NAMESPACE,
        attrs={
            "confidence_threshold": np.float32(args.confidence),
            "plugin_version": PLUGIN_VERSION,
            "plugin_namespace": PLUGIN_NAMESPACE,
        },
        inputs=[raw],
        outputs=[boxes_scores, classes, indices, count],
    )
    graph.nodes.append(node)
    graph.outputs = (
        [raw, boxes_scores, classes, indices, count]
        if args.retain_raw_output
        else [boxes_scores, classes, indices, count]
    )
    graph.cleanup().toposort()
    modified = gs.export_onnx(graph)
    if not any(item.domain == PLUGIN_NAMESPACE for item in modified.opset_import):
        modified.opset_import.append(onnx.helper.make_opsetid(PLUGIN_NAMESPACE, 1))
    onnx.checker.check_model(modified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(modified, args.output)

    manifest = {
        "status": "PASS",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "plugin_name": PLUGIN_NAME,
        "plugin_version": PLUGIN_VERSION,
        "plugin_namespace": PLUGIN_NAMESPACE,
        "confidence_threshold": args.confidence,
        "diagnostic_retain_raw_output": args.retain_raw_output,
        "input_contract": {"name": "output0", "shape": [1, 7, 8400], "dtype": "float32"},
        "outputs": ([
            {"name": "output0", "shape": [1, 7, 8400], "dtype": "float32"},
        ] if args.retain_raw_output else []) + [
            {"name": "boxes_scores", "shape": [1, 8400, 5], "dtype": "float32"},
            {"name": "classes", "shape": [1, 8400], "dtype": "int32"},
            {"name": "indices", "shape": [1, 8400], "dtype": "int32"},
            {"name": "count", "shape": [1], "dtype": "int32"},
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
