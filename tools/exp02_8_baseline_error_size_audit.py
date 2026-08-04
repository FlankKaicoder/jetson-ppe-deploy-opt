#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_dataset(
    yaml_path: Path,
) -> tuple[list[Path], dict[int, str]]:
    config = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )

    root = Path(config.get("path", yaml_path.parent))

    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    test_value = config.get("test")

    if test_value is None:
        raise KeyError("dataset YAML does not contain test")

    test_entries = (
        test_value
        if isinstance(test_value, list)
        else [test_value]
    )

    images: list[Path] = []

    for entry in test_entries:
        path = Path(entry)

        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if path.is_dir():
            images.extend(
                file
                for file in path.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()

                if not line:
                    continue

                image_path = Path(line)

                if not image_path.is_absolute():
                    image_path = path.parent / image_path

                images.append(image_path.resolve())
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(
                f"test path not found: {path}"
            )

    images = sorted(set(images))

    names_value = config.get("names", {})

    if isinstance(names_value, list):
        names = {
            index: str(name)
            for index, name in enumerate(names_value)
        }
    else:
        names = {
            int(index): str(name)
            for index, name in names_value.items()
        }

    return images, names


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, value in enumerate(parts)
        if value == "images"
    ]

    if not image_indices:
        raise ValueError(
            f"'images' directory not found in path: {image_path}"
        )

    parts[image_indices[-1]] = "labels"

    return Path(*parts).with_suffix(".txt")


def box_iou(
    first: list[float],
    second: list[float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection = intersection_width * intersection_height

    first_area = max(
        0.0,
        first[2] - first[0],
    ) * max(
        0.0,
        first[3] - first[1],
    )

    second_area = max(
        0.0,
        second[2] - second[0],
    ) * max(
        0.0,
        second[3] - second[1],
    )

    union = first_area + second_area - intersection

    return safe_div(intersection, union)


def area_group(area_ratio: float) -> str:
    if area_ratio < 0.0025:
        return "tiny"

    if area_ratio < 0.01:
        return "small"

    if area_ratio < 0.04:
        return "medium"

    return "large"


def load_ground_truth(
    image_path: Path,
) -> tuple[list[dict[str, Any]], Any]:
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"failed to read image: {image_path}"
        )

    height, width = image.shape[:2]
    label_path = image_to_label_path(image_path)

    targets: list[dict[str, Any]] = []

    if not label_path.is_file():
        return targets, image

    for line_number, line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        fields = line.split()

        if len(fields) < 5:
            raise ValueError(
                f"invalid label line: "
                f"{label_path}:{line_number}"
            )

        class_id = int(float(fields[0]))
        center_x = float(fields[1]) * width
        center_y = float(fields[2]) * height
        box_width = float(fields[3]) * width
        box_height = float(fields[4]) * height

        x1 = center_x - box_width / 2.0
        y1 = center_y - box_height / 2.0
        x2 = center_x + box_width / 2.0
        y2 = center_y + box_height / 2.0

        ratio = (
            max(0.0, box_width)
            * max(0.0, box_height)
            / float(width * height)
        )

        targets.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2],
            "area_ratio": ratio,
            "size_group": area_group(ratio),
            "matched": False,
        })

    return targets, image


def match_predictions(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    match_iou: float,
) -> None:
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: predictions[index]["confidence"],
        reverse=True,
    )

    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]

        best_target_index = None
        best_iou = 0.0

        for target_index, target in enumerate(targets):
            if target["matched"]:
                continue

            if target["class_id"] != prediction["class_id"]:
                continue

            current_iou = box_iou(
                target["box"],
                prediction["box"],
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_target_index = target_index

        if (
            best_target_index is not None
            and best_iou >= match_iou
        ):
            targets[best_target_index]["matched"] = True
            predictions[prediction_index]["status"] = "TP"
            predictions[prediction_index]["match_iou"] = best_iou
        else:
            predictions[prediction_index]["status"] = "FP"
            predictions[prediction_index]["match_iou"] = best_iou


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_audit_image(
    image: Any,
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    names: dict[int, str],
    destination: Path,
) -> None:
    canvas = image.copy()

    for target in targets:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in target["box"]
        ]

        color = (
            (0, 180, 0)
            if target["matched"]
            else (0, 0, 255)
        )

        status = "GT-TP" if target["matched"] else "GT-FN"

        label = (
            f"{status} "
            f"{names.get(target['class_id'], target['class_id'])} "
            f"{target['size_group']}"
        )

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    for prediction in predictions:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in prediction["box"]
        ]

        if prediction["status"] == "FP":
            color = (0, 165, 255)
            status = "PRED-FP"
        else:
            color = (255, 180, 0)
            status = "PRED-TP"

        label = (
            f"{status} "
            f"{names.get(prediction['class_id'], prediction['class_id'])} "
            f"{prediction['confidence']:.2f}"
        )

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        cv2.putText(
            canvas,
            label,
            (x1, min(canvas.shape[0] - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(str(destination), canvas)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-visuals", type=int, default=30)

    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    visual_dir = output_dir / "worst_visuals"

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise FileNotFoundError(weights)

    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    images, names = load_dataset(data_yaml)

    print("========== Exp02.8 configuration ==========")
    print(f"python={sys.executable}")
    print(f"weights={weights}")
    print(f"data_yaml={data_yaml}")
    print(f"image_count={len(images)}")
    print(f"imgsz={args.imgsz}")
    print(f"batch={args.batch}")
    print(f"conf={args.conf}")
    print(f"nms_iou={args.nms_iou}")
    print(f"match_iou={args.match_iou}")
    print(f"output_dir={output_dir}")

    if not images:
        raise RuntimeError("no test images found")

    model = YOLO(str(weights), task="detect")

    class_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    )

    size_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    image_records: list[dict[str, Any]] = []
    detailed_records: list[dict[str, Any]] = []

    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.nms_iou,
        device=0,
        stream=True,
        verbose=False,
        save=False,
    )

    for index, (image_path, result) in enumerate(
        zip(images, results),
        start=1,
    ):
        image_path = image_path.resolve()
        targets, image = load_ground_truth(image_path)

        predictions: list[dict[str, Any]] = []

        if result.boxes is not None:
            xyxy_values = (
                result.boxes.xyxy.detach().cpu().tolist()
            )
            confidence_values = (
                result.boxes.conf.detach().cpu().tolist()
            )
            class_values = (
                result.boxes.cls.detach().cpu().tolist()
            )

            for box, confidence, class_value in zip(
                xyxy_values,
                confidence_values,
                class_values,
            ):
                predictions.append({
                    "class_id": int(class_value),
                    "confidence": float(confidence),
                    "box": [float(value) for value in box],
                    "status": "FP",
                    "match_iou": 0.0,
                })

        match_predictions(
            targets,
            predictions,
            args.match_iou,
        )

        tp = sum(
            1
            for target in targets
            if target["matched"]
        )

        fn = len(targets) - tp

        fp = sum(
            1
            for prediction in predictions
            if prediction["status"] == "FP"
        )

        tiny_fn = sum(
            1
            for target in targets
            if not target["matched"]
            and target["size_group"] == "tiny"
        )

        small_fn = sum(
            1
            for target in targets
            if not target["matched"]
            and target["size_group"] == "small"
        )

        for target in targets:
            class_id = target["class_id"]
            class_stats[class_id]["gt"] += 1
            size_stats[target["size_group"]]["gt"] += 1

            if target["matched"]:
                class_stats[class_id]["tp"] += 1
                size_stats[target["size_group"]]["tp"] += 1
            else:
                class_stats[class_id]["fn"] += 1
                size_stats[target["size_group"]]["fn"] += 1

        for prediction in predictions:
            if prediction["status"] == "FP":
                class_stats[
                    prediction["class_id"]
                ]["fp"] += 1

        error_score = (
            fn * 5
            + fp * 2
            + tiny_fn * 3
            + small_fn
        )

        record = {
            "image": str(image_path),
            "gt": len(targets),
            "predictions": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tiny_fn": tiny_fn,
            "small_fn": small_fn,
            "error_score": error_score,
        }

        image_records.append(record)

        detailed_records.append({
            "record": record,
            "image": image,
            "targets": targets,
            "predictions": predictions,
        })

        if index % 25 == 0 or index == len(images):
            print(
                f"processed={index}/{len(images)}",
                flush=True,
            )

    class_rows: list[dict[str, Any]] = []

    for class_id in sorted(class_stats):
        values = class_stats[class_id]

        precision = safe_div(
            values["tp"],
            values["tp"] + values["fp"],
        )

        recall = safe_div(
            values["tp"],
            values["gt"],
        )

        class_rows.append({
            "class_id": class_id,
            "class_name": names.get(
                class_id,
                str(class_id),
            ),
            "gt": values["gt"],
            "tp": values["tp"],
            "fp": values["fp"],
            "fn": values["fn"],
            "precision_at_conf_0_25": precision,
            "recall_at_conf_0_25": recall,
        })

    size_order = [
        "tiny",
        "small",
        "medium",
        "large",
    ]

    size_rows: list[dict[str, Any]] = []

    for group in size_order:
        values = size_stats[group]

        size_rows.append({
            "size_group": group,
            "gt": values["gt"],
            "tp": values["tp"],
            "fn": values["fn"],
            "recall_at_conf_0_25": safe_div(
                values["tp"],
                values["gt"],
            ),
        })

    worst_records = sorted(
        detailed_records,
        key=lambda item: (
            item["record"]["error_score"],
            item["record"]["fn"],
            item["record"]["fp"],
        ),
        reverse=True,
    )

    worst_csv_rows: list[dict[str, Any]] = []

    for rank, item in enumerate(
        worst_records[: args.max_visuals],
        start=1,
    ):
        image_path = Path(item["record"]["image"])

        visual_name = (
            f"{rank:03d}_"
            f"score{item['record']['error_score']}_"
            f"fn{item['record']['fn']}_"
            f"fp{item['record']['fp']}_"
            f"{image_path.stem}.jpg"
        )

        visual_path = visual_dir / visual_name

        draw_audit_image(
            item["image"],
            item["targets"],
            item["predictions"],
            names,
            visual_path,
        )

        worst_csv_rows.append({
            "rank": rank,
            **item["record"],
            "visual": str(visual_path),
        })

    write_csv(
        output_dir / "per_image.csv",
        image_records,
        [
            "image",
            "gt",
            "predictions",
            "tp",
            "fp",
            "fn",
            "tiny_fn",
            "small_fn",
            "error_score",
        ],
    )

    write_csv(
        output_dir / "per_class.csv",
        class_rows,
        [
            "class_id",
            "class_name",
            "gt",
            "tp",
            "fp",
            "fn",
            "precision_at_conf_0_25",
            "recall_at_conf_0_25",
        ],
    )

    write_csv(
        output_dir / "per_size.csv",
        size_rows,
        [
            "size_group",
            "gt",
            "tp",
            "fn",
            "recall_at_conf_0_25",
        ],
    )

    write_csv(
        output_dir / "worst_samples.csv",
        worst_csv_rows,
        [
            "rank",
            "image",
            "gt",
            "predictions",
            "tp",
            "fp",
            "fn",
            "tiny_fn",
            "small_fn",
            "error_score",
            "visual",
        ],
    )

    total_gt = sum(row["gt"] for row in class_rows)
    total_tp = sum(row["tp"] for row in class_rows)
    total_fp = sum(row["fp"] for row in class_rows)
    total_fn = sum(row["fn"] for row in class_rows)

    overall_precision = safe_div(
        total_tp,
        total_tp + total_fp,
    )

    overall_recall = safe_div(
        total_tp,
        total_gt,
    )

    summary = {
        "experiment": (
            "Exp02.8 baseline error and "
            "object-size audit"
        ),
        "result": "PASS",
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "image_count": len(images),
        "confidence_threshold": args.conf,
        "nms_iou": args.nms_iou,
        "matching_iou": args.match_iou,
        "size_definition": {
            "tiny": "area_ratio < 0.0025",
            "small": "0.0025 <= area_ratio < 0.01",
            "medium": "0.01 <= area_ratio < 0.04",
            "large": "area_ratio >= 0.04",
        },
        "overall": {
            "gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision_at_conf_0_25": overall_precision,
            "recall_at_conf_0_25": overall_recall,
        },
        "per_class": class_rows,
        "per_size": size_rows,
        "worst_visual_count": len(worst_csv_rows),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "============================================================",
        " Exp02.8 Baseline Error and Size Audit",
        "============================================================",
        "result=PASS",
        f"weights={weights}",
        f"data_yaml={data_yaml}",
        f"image_count={len(images)}",
        f"confidence_threshold={args.conf}",
        f"matching_iou={args.match_iou}",
        "",
        "========== overall ==========",
        f"gt={total_gt}",
        f"tp={total_tp}",
        f"fp={total_fp}",
        f"fn={total_fn}",
        f"precision_at_conf_0_25={overall_precision:.8f}",
        f"recall_at_conf_0_25={overall_recall:.8f}",
        "",
        "========== per class ==========",
    ]

    for row in class_rows:
        lines.append(
            f"{row['class_name']}: "
            f"gt={row['gt']} "
            f"tp={row['tp']} "
            f"fp={row['fp']} "
            f"fn={row['fn']} "
            f"P={row['precision_at_conf_0_25']:.8f} "
            f"R={row['recall_at_conf_0_25']:.8f}"
        )

    lines.extend([
        "",
        "========== per size ==========",
    ])

    for row in size_rows:
        lines.append(
            f"{row['size_group']}: "
            f"gt={row['gt']} "
            f"tp={row['tp']} "
            f"fn={row['fn']} "
            f"R={row['recall_at_conf_0_25']:.8f}"
        )

    lines.extend([
        "",
        "NOTE: These precision/recall values use a fixed "
        "confidence threshold of 0.25 and class-aware "
        "IoU matching at 0.50. They are an error audit, "
        "not a replacement for Ultralytics mAP metrics.",
        "",
        f"worst_visual_dir={visual_dir}",
        "exp02_8_baseline_error_size_audit=PASS",
    ])

    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        (output_dir / "summary.txt").read_text(
            encoding="utf-8"
        ),
        end="",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
