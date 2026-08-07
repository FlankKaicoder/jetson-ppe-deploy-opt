#!/usr/bin/env python3
"""Prepare one frozen input tensor and Python TensorRT reference output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exp07_trt_consistency import preprocess_image, run_tensorrt, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> int:
    args = parse_args()
    engine = Path(args.engine).resolve()
    image = Path(args.image).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in (engine, image):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_array, preprocess = preprocess_image(image, 640)
    outputs, engine_io, elapsed = run_tensorrt(engine, input_array)
    if len(outputs) != 1:
        raise RuntimeError(f"expected one output, got {list(outputs)}")
    output_name, output = next(iter(outputs.items()))
    input_array = np.ascontiguousarray(input_array, dtype=np.float32)
    output = np.ascontiguousarray(output, dtype=np.float32)
    if list(input_array.shape) != [1, 3, 640, 640]:
        raise RuntimeError(f"unexpected input shape: {input_array.shape}")
    if list(output.shape) != [1, 7, 8400] or not np.isfinite(output).all():
        raise RuntimeError(f"invalid reference output: {output.shape}")

    input_path = report_dir / "input_fp32_nchw.bin"
    output_path = report_dir / "python_trt_output_fp32.bin"
    input_array.tofile(input_path)
    output.tofile(output_path)
    summary = {
        "result": "PASS",
        "engine_sha256": sha256_file(engine),
        "image_sha256": sha256_file(image),
        "input": {
            "path": str(input_path),
            "shape": list(input_array.shape),
            "dtype": str(input_array.dtype),
            "bytes": input_path.stat().st_size,
            "sha256": array_sha256(input_array),
        },
        "output": {
            "path": str(output_path),
            "name": output_name,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "bytes": output_path.stat().st_size,
            "sha256": array_sha256(output),
            "finite": bool(np.isfinite(output).all()),
        },
        "preprocess": preprocess,
        "engine_io": engine_io,
        "python_trt_forward_seconds": elapsed,
    }
    (report_dir / "reference.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "reference.txt").write_text(
        "\n".join(
            [
                "result=PASS",
                f"engine_sha256={summary['engine_sha256']}",
                f"image_sha256={summary['image_sha256']}",
                f"input_sha256={summary['input']['sha256']}",
                f"output_sha256={summary['output']['sha256']}",
                f"input_bytes={summary['input']['bytes']}",
                f"output_bytes={summary['output']['bytes']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"result=PASS input_sha256={summary['input']['sha256']} "
        f"output_sha256={summary['output']['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
