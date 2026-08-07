#!/usr/bin/env python3
"""Build a TensorRT INT8 engine from a train-only calibration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorrt as trt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument("--allow-cache-read", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_image(image_path: Path, imgsz: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read calibration image: {image_path}")
    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    width_padding = imgsz - resized_width
    height_padding = imgsz - resized_height
    left = int(round(width_padding / 2 - 0.1))
    right = int(round(width_padding / 2 + 0.1))
    top = int(round(height_padding / 2 - 0.1))
    bottom = int(round(height_padding / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if image.shape[:2] != (imgsz, imgsz):
        raise RuntimeError(f"unexpected letterbox shape: {image.shape}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return image.astype(np.float32) / 255.0


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: list[Path],
        batch_size: int,
        imgsz: int,
        cache_path: Path,
        allow_cache_read: bool,
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if len(image_paths) % batch_size != 0:
            raise ValueError(
                f"image count {len(image_paths)} is not divisible by batch {batch_size}"
            )
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.cache_path = cache_path
        self.allow_cache_read = allow_cache_read
        self.index = 0
        self.batch_count = 0
        self.device_input = torch.empty(
            (batch_size, 3, imgsz, imgsz),
            dtype=torch.float32,
            device="cuda",
        )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.index >= len(self.image_paths):
            return None
        batch_paths = self.image_paths[self.index:self.index + self.batch_size]
        if len(batch_paths) != self.batch_size:
            raise RuntimeError("incomplete final calibration batch")
        host_batch = np.stack(
            [preprocess_image(path, self.imgsz) for path in batch_paths],
            axis=0,
        )
        self.device_input.copy_(torch.from_numpy(host_batch))
        torch.cuda.synchronize()
        self.index += self.batch_size
        self.batch_count += 1
        print(
            f"calibration_batch={self.batch_count} "
            f"images={self.index}/{len(self.image_paths)}",
            flush=True,
        )
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        if self.allow_cache_read and self.cache_path.is_file():
            cache = self.cache_path.read_bytes()
            print(f"calibration_cache_read_bytes={len(cache)}", flush=True)
            return cache
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        print(f"calibration_cache_written_bytes={len(cache)}", flush=True)


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    manifest_path = Path(args.manifest).resolve()
    engine_path = Path(args.engine).resolve()
    cache_path = Path(args.cache).resolve()
    report_dir = Path(args.report_dir).resolve()
    if not onnx_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"onnx={onnx_path} manifest={manifest_path}")
    if engine_path.exists() or cache_path.exists():
        raise FileExistsError(f"engine/cache already exists: {engine_path} {cache_path}")
    report_dir.mkdir(parents=True, exist_ok=True)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0 or args.imgsz <= 0:
        raise ValueError("batch-size and imgsz must be positive")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train":
        raise RuntimeError("calibration manifest is not train-only")
    entries = manifest.get("images", [])
    if not entries:
        raise RuntimeError("calibration manifest contains no images")
    if args.limit:
        entries = entries[:args.limit]
    calibration_root = manifest_path.parent.parent
    image_paths = [
        (calibration_root / entry["archive_path"]).resolve()
        for entry in entries
    ]
    for entry, image_path in zip(entries, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != entry["sha256"]:
            raise RuntimeError(f"calibration image SHA256 mismatch: {image_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_int8:
        raise RuntimeError("TensorRT reports no fast INT8 support")
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"unexpected network I/O: {network.num_inputs}/{network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    if tuple(input_tensor.shape) != (1, 3, args.imgsz, args.imgsz):
        raise RuntimeError(f"unexpected ONNX input shape: {input_tensor.shape}")

    calibrator = EntropyCalibrator(
        image_paths=image_paths,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        cache_path=cache_path,
        allow_cache_read=args.allow_cache_read,
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        args.workspace_mib * 1024 * 1024,
    )
    config.builder_optimization_level = args.builder_optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.int8_calibrator = calibrator

    start = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if serialized_engine is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    engine_bytes = bytes(serialized_engine)
    engine_path.write_bytes(engine_bytes)
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        raise RuntimeError("TensorRT did not write a calibration cache")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("failed to deserialize newly built INT8 engine")
    io_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_tensors.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "shape": list(engine.get_tensor_shape(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
            }
        )
    inspector = engine.create_engine_inspector()
    inspector_text = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    (report_dir / "engine_inspector.json").write_text(
        inspector_text + ("\n" if not inspector_text.endswith("\n") else ""),
        encoding="utf-8",
    )

    summary = normalize(
        {
            "experiment": "Exp08 TensorRT INT8 build",
            "result": "PASS",
            "versions": {
                "python": sys.version.split()[0],
                "tensorrt": trt.__version__,
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
            },
            "onnx": {
                "path": onnx_path,
                "sha256": sha256_file(onnx_path),
                "bytes": onnx_path.stat().st_size,
            },
            "manifest": {
                "path": manifest_path,
                "sha256": sha256_file(manifest_path),
                "source_split": manifest["source_split"],
                "available_images": len(manifest["images"]),
                "used_images": len(image_paths),
                "batch_size": args.batch_size,
                "calibration_batches": calibrator.batch_count,
            },
            "builder": {
                "int8": True,
                "fp16_fallback": True,
                "tf32": False,
                "workspace_mib": args.workspace_mib,
                "optimization_level": args.builder_optimization_level,
                "platform_has_fast_int8": builder.platform_has_fast_int8,
                "allow_cache_read": args.allow_cache_read,
                "build_seconds": build_seconds,
            },
            "engine": {
                "path": engine_path,
                "sha256": sha256_file(engine_path),
                "bytes": engine_path.stat().st_size,
                "io_tensors": io_tensors,
            },
            "calibration_cache": {
                "path": cache_path,
                "sha256": sha256_file(cache_path),
                "bytes": cache_path.stat().st_size,
            },
        }
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "summary.txt").write_text(
        "\n".join(
            [
                "result=PASS",
                f"onnx_sha256={summary['onnx']['sha256']}",
                f"manifest_sha256={summary['manifest']['sha256']}",
                f"calibration_images={len(image_paths)}",
                f"calibration_batches={calibrator.batch_count}",
                f"build_seconds={build_seconds:.6f}",
                f"engine_bytes={engine_path.stat().st_size}",
                f"engine_sha256={summary['engine']['sha256']}",
                f"cache_bytes={cache_path.stat().st_size}",
                f"cache_sha256={summary['calibration_cache']['sha256']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        "result=PASS "
        f"engine_sha256={summary['engine']['sha256']} "
        f"engine_bytes={summary['engine']['bytes']} "
        f"build_seconds={build_seconds:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
