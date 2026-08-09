#!/usr/bin/env python3
"""Collect Exp16 detections and compute Gate-local conf-floor metrics."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

from exp07_trt_consistency import preprocess_image, torch_dtype


CONFIDENCE_FLOOR = 0.25
NMS_IOU = 0.70


@dataclass
class Box:
    image: str
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    candidate_index: int = -1
    size_group: str = ""


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def nms(items: list[Box]) -> list[Box]:
    ordered = sorted(items, key=lambda item: (-item.confidence, item.candidate_index))
    kept: list[Box] = []
    for item in ordered:
        if any(
            item.class_id == accepted.class_id
            and box_iou(item.xyxy, accepted.xyxy) > NMS_IOU
            for accepted in kept
        ):
            continue
        kept.append(item)
    return kept


def inverse_box(box: np.ndarray, metadata: dict) -> tuple[float, float, float, float]:
    ratio = float(metadata["resize_ratio"])
    left = float(metadata["padding_left"])
    top = float(metadata["padding_top"])
    width = float(metadata["original_width"])
    height = float(metadata["original_height"])
    return (
        float(np.clip((box[0] - left) / ratio, 0.0, width)),
        float(np.clip((box[1] - top) / ratio, 0.0, height)),
        float(np.clip((box[2] - left) / ratio, 0.0, width)),
        float(np.clip((box[3] - top) / ratio, 0.0, height)),
    )


class EngineSession:
    def __init__(self, engine_path: Path, plugin_path: Path | None) -> None:
        self.plugin_handle = None
        if plugin_path is not None:
            self.plugin_handle = ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
            init = self.plugin_handle.ppeInitPlugin
            init.restype = ctypes.c_bool
            if not init():
                raise RuntimeError("ppeInitPlugin returned false")
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed")
        self.input_name = ""
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if not self.input_name:
            raise RuntimeError("engine input not found")
        self.input = torch.empty(
            tuple(self.engine.get_tensor_shape(self.input_name)),
            device="cuda", dtype=torch_dtype(self.engine.get_tensor_dtype(self.input_name)),
        )
        self.outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self.outputs[name] = torch.empty(
                shape, device="cuda", dtype=torch_dtype(self.engine.get_tensor_dtype(name))
            )
        self.context.set_tensor_address(self.input_name, int(self.input.data_ptr()))
        for name, tensor in self.outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        self.stream = torch.cuda.Stream()

    def infer(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        source = torch.from_numpy(input_array)
        with torch.cuda.stream(self.stream):
            self.input.copy_(source, non_blocking=False)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("execute_async_v3 returned false")
        self.stream.synchronize()
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in self.outputs.items()
        }


def decode_raw(raw: np.ndarray, metadata: dict, image: str) -> list[Box]:
    raw = raw.reshape(7, 8400)
    classes = np.argmax(raw[4:, :], axis=0)
    scores = raw[4:, :].max(axis=0)
    result: list[Box] = []
    for index in np.flatnonzero(np.isfinite(scores) & (scores >= CONFIDENCE_FLOOR)):
        cx, cy, width, height = (float(raw[channel, index]) for channel in range(4))
        if not np.isfinite([cx, cy, width, height]).all() or width <= 0 or height <= 0:
            continue
        network = np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
        )
        source = inverse_box(network, metadata)
        if source[2] > source[0] and source[3] > source[1]:
            result.append(Box(image, int(classes[index]), float(scores[index]), source, int(index)))
    return nms(result)


def decode_plugin(outputs: dict[str, np.ndarray], metadata: dict, image: str) -> list[Box]:
    required = {"boxes_scores", "classes", "indices", "count"}
    if set(outputs) != required:
        raise RuntimeError(f"unexpected Plugin outputs: {sorted(outputs)}")
    count = int(outputs["count"].reshape(-1)[0])
    if not 0 <= count <= 8400:
        raise RuntimeError(f"invalid Plugin count: {count}")
    boxes = outputs["boxes_scores"].reshape(8400, 5)
    classes = outputs["classes"].reshape(8400)
    indices = outputs["indices"].reshape(8400)
    result = []
    for position in range(count):
        source = inverse_box(boxes[position, :4], metadata)
        if source[2] <= source[0] or source[3] <= source[1]:
            continue
        result.append(
            Box(image, int(classes[position]), float(boxes[position, 4]), source,
                int(indices[position]))
        )
    return nms(result)


def image_paths(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root.iterdir() if path.suffix.lower() in extensions)


def collect(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    images = image_paths(args.images)
    if args.max_images > 0:
        images = images[: args.max_images]
    elif len(images) != 219:
        raise RuntimeError(f"expected 219 test images, got {len(images)}")
    session = EngineSession(args.engine, args.plugin)
    plugin_mode = args.plugin is not None
    predictions_path = args.output / "predictions.csv"
    total = 0
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "detection_index", "candidate_index", "class_id", "confidence",
             "x1", "y1", "x2", "y2"]
        )
        for image_index, path in enumerate(images):
            input_array, metadata = preprocess_image(path, 640)
            outputs = session.infer(input_array)
            detections = (
                decode_plugin(outputs, metadata, path.name)
                if plugin_mode else decode_raw(next(iter(outputs.values())), metadata, path.name)
            )
            for detection_index, item in enumerate(detections):
                writer.writerow(
                    [item.image, detection_index, item.candidate_index, item.class_id,
                     f"{item.confidence:.9f}", *[f"{value:.9f}" for value in item.xyxy]]
                )
            total += len(detections)
            if (image_index + 1) % 25 == 0:
                print(f"progress={image_index + 1}/219", flush=True)
    summary = {
        "result": "PASS", "mode": "plugin" if plugin_mode else "raw",
        "engine": str(args.engine), "plugin": str(args.plugin) if args.plugin else None,
        "images": str(args.images), "image_count": len(images),
        "detection_count": total, "confidence_floor": CONFIDENCE_FLOOR,
        "nms_iou": NMS_IOU, "predictions": str(predictions_path),
    }
    (args.output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def load_predictions(path: Path) -> list[Box]:
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.append(
                Box(row["image"], int(row["class_id"]), float(row["confidence"]),
                    tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2")),
                    int(row["candidate_index"]))
            )
    return result


def load_ground_truth(images_root: Path) -> list[Box]:
    result = []
    for image_path in image_paths(images_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        height, width = image.shape[:2]
        label = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not label.is_file():
            label = image_path.parents[2] / "labels" / image_path.parent.name / (image_path.stem + ".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            cls, cx, cy, bw, bh = (float(value) for value in line.split()[:5])
            x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
            x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
            area_ratio = bw * bh
            size = "tiny" if area_ratio < 0.0025 else "small" if area_ratio < 0.01 else \
                "medium" if area_ratio < 0.04 else "large"
            result.append(Box(image_path.name, int(cls), 1.0, (x1, y1, x2, y2), size_group=size))
    return result


def assignments(predictions: list[Box], targets: list[Box], threshold: float):
    target_groups: dict[tuple[str, int], list[Box]] = defaultdict(list)
    for target in targets:
        target_groups[(target.image, target.class_id)].append(target)
    used: dict[tuple[str, int], set[int]] = defaultdict(set)
    records = []
    ordered = sorted(
        predictions,
        key=lambda item: (-item.confidence, item.image, item.class_id, item.candidate_index),
    )
    for prediction in ordered:
        key = (prediction.image, prediction.class_id)
        candidates = target_groups.get(key, [])
        choices = [
            (box_iou(prediction.xyxy, target.xyxy), index)
            for index, target in enumerate(candidates) if index not in used[key]
        ]
        overlap, target_index = max(choices, default=(0.0, -1))
        matched = overlap >= threshold
        if matched:
            used[key].add(target_index)
        records.append((prediction, matched, overlap,
                        candidates[target_index] if matched else None))
    return records, used


def average_precision(records, gt_count: int) -> float:
    if gt_count == 0:
        return float("nan")
    tp = np.cumsum([int(item[1]) for item in records])
    fp = np.cumsum([int(not item[1]) for item in records])
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([
        precision[recall >= level].max(initial=0.0) for level in np.linspace(0, 1, 101)
    ]))


def evaluate(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    predictions = load_predictions(args.predictions)
    targets = load_ground_truth(args.images)
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    per_threshold = {}
    for threshold in thresholds:
        class_aps = []
        for class_id in sorted({item.class_id for item in targets}):
            class_predictions = [item for item in predictions if item.class_id == class_id]
            class_targets = [item for item in targets if item.class_id == class_id]
            records, _ = assignments(class_predictions, class_targets, threshold)
            class_aps.append(average_precision(records, len(class_targets)))
        per_threshold[f"{threshold:.2f}"] = float(np.nanmean(class_aps))
    fixed_records, used = assignments(predictions, targets, 0.50)
    tp = sum(int(item[1]) for item in fixed_records)
    fp = len(fixed_records) - tp
    fn = len(targets) - tp
    matched_target_ids = {
        (key, index) for key, indices in used.items() for index in indices
    }
    size_counts = {}
    grouped_targets: dict[tuple[str, int], list[Box]] = defaultdict(list)
    for target in targets:
        grouped_targets[(target.image, target.class_id)].append(target)
    for size in ("tiny", "small", "medium", "large"):
        members = [
            (key, index) for key, items in grouped_targets.items()
            for index, target in enumerate(items) if target.size_group == size
        ]
        matched = sum(item in matched_target_ids for item in members)
        size_counts[size] = {"gt": len(members), "tp": matched,
                             "recall": matched / len(members) if members else 0.0}
    tiny_small_gt = size_counts["tiny"]["gt"] + size_counts["small"]["gt"]
    tiny_small_tp = size_counts["tiny"]["tp"] + size_counts["small"]["tp"]
    summary = {
        "result": "PASS",
        "scope": "Gate-local metrics from detections hard-filtered at confidence 0.25",
        "metric_names": {
            "map50": "mAP50@conf_floor_0.25",
            "map50_95": "mAP50-95@conf_floor_0.25",
        },
        "image_count": len(image_paths(args.images)), "ground_truth_count": len(targets),
        "prediction_count": len(predictions), "confidence_floor": CONFIDENCE_FLOOR,
        "fixed_iou": 0.50, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / len(targets) if targets else 0.0,
        "map50_at_conf_floor_0.25": per_threshold["0.50"],
        "map50_95_at_conf_floor_0.25": float(np.mean(list(per_threshold.values()))),
        "ap_by_iou": per_threshold, "size": size_counts,
        "tiny_small_recall": tiny_small_tp / tiny_small_gt if tiny_small_gt else 0.0,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def gate(args: argparse.Namespace) -> int:
    variants = {
        name: json.loads((args.root / name / "metrics" / "metrics.json").read_text())
        for name in ("F0", "B1", "B2", "P")
    }
    high_paths = {
        "precision": lambda value: value["precision"],
        "recall": lambda value: value["recall"],
        "mAP50@conf_floor_0.25": lambda value: value["map50_at_conf_floor_0.25"],
        "mAP50-95@conf_floor_0.25": lambda value: value["map50_95_at_conf_floor_0.25"],
        "TP": lambda value: value["tp"],
        "tiny_recall": lambda value: value["size"]["tiny"]["recall"],
        "small_recall": lambda value: value["size"]["small"]["recall"],
        "tiny_small_recall": lambda value: value["tiny_small_recall"],
    }
    low_paths = {
        "FP": lambda value: value["fp"],
        "FN": lambda value: value["fn"],
    }
    checks = []
    for name, getter in high_paths.items():
        baseline = [getter(variants[item]) for item in ("F0", "B1", "B2")]
        plugin = getter(variants["P"])
        checks.append({"metric": name, "direction": "higher", "baseline_min": min(baseline),
                       "baseline_max": max(baseline), "plugin": plugin,
                       "pass": plugin + 1e-12 >= min(baseline)})
    for name, getter in low_paths.items():
        baseline = [getter(variants[item]) for item in ("F0", "B1", "B2")]
        plugin = getter(variants["P"])
        checks.append({"metric": name, "direction": "lower", "baseline_min": min(baseline),
                       "baseline_max": max(baseline), "plugin": plugin,
                       "pass": plugin <= max(baseline)})

    match_names = ("F0_B1", "F0_B2", "B1_B2", "F0_P", "B1_P", "B2_P")
    matches = {
        name: json.loads((args.root / "matches" / name / "summary.json").read_text())
        for name in match_names
    }
    baseline_names = match_names[:3]
    plugin_names = match_names[3:]
    pair_checks = (
        ("unmatched_rate", "lower", lambda value: value["unmatched_rate"]),
        ("confidence_abs_delta_max", "lower",
         lambda value: value["confidence_abs_delta"]["maximum"]),
        ("matched_iou_min", "higher", lambda value: value["matched_iou"]["minimum"]),
        ("matched_iou_p05", "higher", lambda value: value["matched_iou"]["p05"]),
        ("candidate_index_agreement", "higher",
         lambda value: value["candidate_index_agreement"]),
    )
    for metric, direction, getter in pair_checks:
        baseline_values = [getter(matches[name]) for name in baseline_names]
        plugin_values = [getter(matches[name]) for name in plugin_names]
        if direction == "lower":
            limit = max(baseline_values)
            passed = max(plugin_values) <= limit + 1e-12
        else:
            limit = min(baseline_values)
            passed = min(plugin_values) + 1e-12 >= limit
        checks.append({"metric": metric, "direction": direction,
                       "baseline_pair_values": dict(zip(baseline_names, baseline_values)),
                       "plugin_pair_values": dict(zip(plugin_names, plugin_values)),
                       "limit": limit, "pass": passed})
    passed = all(item["pass"] for item in checks)
    summary = {
        "result": "PASS" if passed else "REJECTED",
        "decision_scope": "Exp16 Deployment Semantic Revalidation Gate R3",
        "baseline_envelope": ["F0", "B1", "B2"],
        "plugin_candidate": "P", "checks": checks,
        "performance_authorized": passed,
    }
    output = args.root / "r3_gate_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--engine", type=Path, required=True)
    collect_parser.add_argument("--plugin", type=Path)
    collect_parser.add_argument("--images", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--max-images", type=int, default=0)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--images", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("engine", "plugin", "images", "predictions"):
        path = getattr(args, name, None)
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    if args.command == "collect":
        return collect(args)
    if args.command == "evaluate":
        return evaluate(args)
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
