from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from models.blocks.reparam_block import (
    ConvBN,
    RepConvBlock,
)


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


SIZE_ORDER = [
    "tiny",
    "small",
    "medium",
    "large",
]


EXPECTED_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
}


# Frozen Exp02.8 reference. The new implementation must reproduce
# these exact counts before the Exp04 comparison is accepted.
EXPECTED_BASELINE = {
    "overall": {
        "gt": 840,
        "tp": 731,
        "fp": 169,
        "fn": 109,
    },
    "per_class": {
        "person": {
            "gt": 337,
            "tp": 288,
            "fp": 86,
            "fn": 49,
        },
        "helmet": {
            "gt": 259,
            "tp": 232,
            "fp": 26,
            "fn": 27,
        },
        "safety_vest": {
            "gt": 244,
            "tp": 211,
            "fp": 57,
            "fn": 33,
        },
    },
    "per_size": {
        "tiny": {
            "gt": 29,
            "tp": 24,
            "fn": 5,
        },
        "small": {
            "gt": 114,
            "tp": 89,
            "fn": 25,
        },
        "medium": {
            "gt": 222,
            "tp": 194,
            "fn": 28,
        },
        "large": {
            "gt": 475,
            "tp": 424,
            "fn": 51,
        },
    },
}


def resolve_path(
    value: str,
    dataset_root: Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = dataset_root / path

    return path.resolve()


def collect_images_from_entry(
    entry: str,
    dataset_root: Path,
) -> list[Path]:
    path = resolve_path(
        entry,
        dataset_root,
    )

    if path.is_dir():
        return sorted(
            item.resolve()
            for item in path.rglob("*")
            if (
                item.is_file()
                and item.suffix.lower()
                in IMAGE_SUFFIXES
            )
        )

    if (
        path.is_file()
        and path.suffix.lower()
        in IMAGE_SUFFIXES
    ):
        return [path]

    if (
        path.is_file()
        and path.suffix.lower() == ".txt"
    ):
        images: list[Path] = []

        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            candidate = Path(line)

            if not candidate.is_absolute():
                candidate = (
                    dataset_root
                    / candidate
                )

            images.append(
                candidate.resolve()
            )

        return sorted(images)

    raise FileNotFoundError(
        "unsupported or missing dataset entry: "
        f"{path}"
    )


def load_test_images(
    data_yaml: Path,
) -> tuple[
    list[Path],
    dict[int, str],
]:
    data = yaml.safe_load(
        data_yaml.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise TypeError(
            "dataset YAML is not a dictionary"
        )

    configured_root = data.get(
        "path",
        data_yaml.parent,
    )

    dataset_root = Path(
        configured_root
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            data_yaml.parent
            / dataset_root
        )

    dataset_root = dataset_root.resolve()

    test_entry = data.get("test")

    if test_entry is None:
        raise KeyError(
            "dataset YAML does not contain test"
        )

    entries = (
        test_entry
        if isinstance(test_entry, list)
        else [test_entry]
    )

    images: list[Path] = []

    for entry in entries:
        images.extend(
            collect_images_from_entry(
                str(entry),
                dataset_root,
            )
        )

    images = sorted(set(images))

    if not images:
        raise RuntimeError(
            "no test images were found"
        )

    names_raw = data.get("names")

    if isinstance(names_raw, list):
        names = {
            index: str(name)
            for index, name
            in enumerate(names_raw)
        }

    elif isinstance(names_raw, dict):
        names = {
            int(index): str(name)
            for index, name
            in names_raw.items()
        }

    else:
        raise TypeError(
            "dataset names must be list or dict"
        )

    return images, names


def label_path_for_image(
    image_path: Path,
) -> Path:
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, part
        in enumerate(parts)
        if part == "images"
    ]

    if not image_indices:
        raise RuntimeError(
            "cannot derive label path because "
            f"'images' is absent: {image_path}"
        )

    parts[image_indices[-1]] = "labels"

    return Path(*parts).with_suffix(
        ".txt"
    )


def size_group(
    area_ratio: float,
) -> str:
    if area_ratio < 0.0025:
        return "tiny"

    if area_ratio < 0.01:
        return "small"

    if area_ratio < 0.04:
        return "medium"

    return "large"


def load_ground_truth(
    label_path: Path,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not label_path.is_file():
        return []

    records: list[
        dict[str, Any]
    ] = []

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
            raise RuntimeError(
                "invalid label row: "
                f"{label_path}:{line_number}"
            )

        class_id = int(
            float(fields[0])
        )

        center_x = float(fields[1])
        center_y = float(fields[2])
        box_width = float(fields[3])
        box_height = float(fields[4])

        x1 = (
            center_x
            - box_width / 2.0
        ) * width

        y1 = (
            center_y
            - box_height / 2.0
        ) * height

        x2 = (
            center_x
            + box_width / 2.0
        ) * width

        y2 = (
            center_y
            + box_height / 2.0
        ) * height

        area_ratio = (
            box_width
            * box_height
        )

        records.append(
            {
                "class_id": class_id,
                "xyxy": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "area_ratio": (
                    area_ratio
                ),
                "size": size_group(
                    area_ratio
                ),
            }
        )

    return records


def box_iou(
    first: list[float],
    second: list[float],
) -> float:
    intersection_x1 = max(
        first[0],
        second[0],
    )

    intersection_y1 = max(
        first[1],
        second[1],
    )

    intersection_x2 = min(
        first[2],
        second[2],
    )

    intersection_y2 = min(
        first[3],
        second[3],
    )

    intersection_width = max(
        0.0,
        intersection_x2
        - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2
        - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

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

    union_area = (
        first_area
        + second_area
        - intersection_area
    )

    if union_area <= 0.0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def empty_class_counters() -> dict[
    str,
    dict[str, int],
]:
    return {
        name: {
            "gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for name
        in EXPECTED_NAMES.values()
    }


def empty_size_counters() -> dict[
    str,
    dict[str, int],
]:
    return {
        name: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
        for name in SIZE_ORDER
    }


def finalize_class_counters(
    counters: dict[
        str,
        dict[str, int],
    ],
) -> dict[
    str,
    dict[str, float | int],
]:
    result: dict[
        str,
        dict[str, float | int],
    ] = {}

    for name, values in counters.items():
        denominator_precision = (
            values["tp"]
            + values["fp"]
        )

        denominator_recall = (
            values["tp"]
            + values["fn"]
        )

        precision = (
            values["tp"]
            / denominator_precision
            if denominator_precision
            else 0.0
        )

        recall = (
            values["tp"]
            / denominator_recall
            if denominator_recall
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if (
                precision
                + recall
            ) > 0.0
            else 0.0
        )

        result[name] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return result


def finalize_size_counters(
    counters: dict[
        str,
        dict[str, int],
    ],
) -> dict[
    str,
    dict[str, float | int],
]:
    result: dict[
        str,
        dict[str, float | int],
    ] = {}

    for name, values in counters.items():
        denominator = (
            values["tp"]
            + values["fn"]
        )

        recall = (
            values["tp"]
            / denominator
            if denominator
            else 0.0
        )

        result[name] = {
            **values,
            "recall": recall,
        }

    return result


def audit_model(
    model_name: str,
    weights: Path,
    images: list[Path],
    names: dict[int, str],
    confidence: float,
    nms_iou: float,
    matching_iou: float,
    batch: int,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print(f" Auditing model={model_name}")
    print("=" * 72)
    print(f"weights={weights}")
    print(f"image_count={len(images)}")

    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}"
        )

    yolo = YOLO(
        str(weights)
    )

    loaded_names = {
        int(key): str(value)
        for key, value
        in yolo.names.items()
    }

    if loaded_names != names:
        raise RuntimeError(
            "model and dataset class names differ: "
            f"{loaded_names} != {names}"
        )

    results = yolo.predict(
        source=[
            str(path)
            for path in images
        ],
        imgsz=640,
        conf=confidence,
        iou=nms_iou,
        batch=batch,
        device=0,
        half=False,
        augment=False,
        max_det=300,
        save=False,
        verbose=False,
        stream=False,
    )

    if len(results) != len(images):
        raise RuntimeError(
            "prediction result count mismatch: "
            f"{len(results)} != "
            f"{len(images)}"
        )

    class_counters = (
        empty_class_counters()
    )

    size_counters = (
        empty_size_counters()
    )

    per_image: list[
        dict[str, Any]
    ] = []

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_index, (
        image_path,
        result,
    ) in enumerate(
        zip(images, results),
        start=1,
    ):
        height, width = (
            int(result.orig_shape[0]),
            int(result.orig_shape[1]),
        )

        label_path = (
            label_path_for_image(
                image_path
            )
        )

        ground_truth = (
            load_ground_truth(
                label_path,
                width,
                height,
            )
        )

        predictions: list[
            dict[str, Any]
        ] = []

        boxes = result.boxes

        if boxes is not None:
            xyxy_rows = (
                boxes.xyxy
                .detach()
                .cpu()
                .tolist()
            )

            confidence_rows = (
                boxes.conf
                .detach()
                .cpu()
                .tolist()
            )

            class_rows = (
                boxes.cls
                .detach()
                .cpu()
                .tolist()
            )

            for box, score, class_id in zip(
                xyxy_rows,
                confidence_rows,
                class_rows,
            ):
                predictions.append(
                    {
                        "class_id": int(
                            class_id
                        ),
                        "confidence": float(
                            score
                        ),
                        "xyxy": [
                            float(value)
                            for value in box
                        ],
                    }
                )

        predictions.sort(
            key=lambda item: (
                item["confidence"]
            ),
            reverse=True,
        )

        matched_gt: set[int] = set()
        matched_prediction: set[int] = set()

        for prediction_index, prediction in enumerate(
            predictions
        ):
            best_gt_index = None
            best_iou = -1.0

            for gt_index, gt in enumerate(
                ground_truth
            ):
                if gt_index in matched_gt:
                    continue

                if (
                    gt["class_id"]
                    != prediction[
                        "class_id"
                    ]
                ):
                    continue

                overlap = box_iou(
                    prediction["xyxy"],
                    gt["xyxy"],
                )

                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_index = (
                        gt_index
                    )

            if (
                best_gt_index is not None
                and best_iou
                >= matching_iou
            ):
                matched_gt.add(
                    best_gt_index
                )

                matched_prediction.add(
                    prediction_index
                )

        image_tp = len(matched_gt)

        image_fp = (
            len(predictions)
            - len(
                matched_prediction
            )
        )

        image_fn = (
            len(ground_truth)
            - len(matched_gt)
        )

        total_gt += len(
            ground_truth
        )

        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn

        for gt_index, gt in enumerate(
            ground_truth
        ):
            class_name = names[
                gt["class_id"]
            ]

            size_name = gt["size"]

            class_counters[
                class_name
            ]["gt"] += 1

            size_counters[
                size_name
            ]["gt"] += 1

            if gt_index in matched_gt:
                class_counters[
                    class_name
                ]["tp"] += 1

                size_counters[
                    size_name
                ]["tp"] += 1
            else:
                class_counters[
                    class_name
                ]["fn"] += 1

                size_counters[
                    size_name
                ]["fn"] += 1

        for prediction_index, prediction in enumerate(
            predictions
        ):
            if (
                prediction_index
                in matched_prediction
            ):
                continue

            class_name = names[
                prediction["class_id"]
            ]

            class_counters[
                class_name
            ]["fp"] += 1

        per_image.append(
            {
                "image_index": (
                    image_index
                ),
                "image": str(
                    image_path
                ),
                "label": str(
                    label_path
                ),
                "gt": len(
                    ground_truth
                ),
                "predictions": len(
                    predictions
                ),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "error_score": (
                    image_fp
                    + image_fn
                ),
            }
        )

        if (
            image_index % 25 == 0
            or image_index
            == len(images)
        ):
            print(
                "processed="
                f"{image_index}/"
                f"{len(images)}"
            )

    precision_denominator = (
        total_tp + total_fp
    )

    recall_denominator = (
        total_tp + total_fn
    )

    precision = (
        total_tp
        / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        total_tp
        / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (
            precision
            + recall
        ) > 0.0
        else 0.0
    )

    per_class = (
        finalize_class_counters(
            class_counters
        )
    )

    per_size = (
        finalize_size_counters(
            size_counters
        )
    )

    tiny_small_gt = (
        per_size["tiny"]["gt"]
        + per_size["small"]["gt"]
    )

    tiny_small_tp = (
        per_size["tiny"]["tp"]
        + per_size["small"]["tp"]
    )

    tiny_small_fn = (
        per_size["tiny"]["fn"]
        + per_size["small"]["fn"]
    )

    medium_large_gt = (
        per_size["medium"]["gt"]
        + per_size["large"]["gt"]
    )

    medium_large_tp = (
        per_size["medium"]["tp"]
        + per_size["large"]["tp"]
    )

    medium_large_fn = (
        per_size["medium"]["fn"]
        + per_size["large"]["fn"]
    )

    return {
        "model": model_name,
        "weights": str(weights),
        "overall": {
            "gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_class": per_class,
        "per_size": per_size,
        "combined_size": {
            "tiny_small": {
                "gt": tiny_small_gt,
                "tp": tiny_small_tp,
                "fn": tiny_small_fn,
                "recall": (
                    tiny_small_tp
                    / tiny_small_gt
                    if tiny_small_gt
                    else 0.0
                ),
            },
            "medium_large": {
                "gt": medium_large_gt,
                "tp": medium_large_tp,
                "fn": medium_large_fn,
                "recall": (
                    medium_large_tp
                    / medium_large_gt
                    if medium_large_gt
                    else 0.0
                ),
            },
        },
        "per_image": per_image,
    }


def count_fields_match(
    actual: dict[str, Any],
    expected: dict[str, int],
) -> bool:
    return all(
        int(actual[key])
        == int(value)
        for key, value
        in expected.items()
    )


def baseline_reproduced(
    baseline: dict[str, Any],
) -> tuple[
    bool,
    dict[str, bool],
]:
    checks: dict[str, bool] = {
        "overall": count_fields_match(
            baseline["overall"],
            EXPECTED_BASELINE[
                "overall"
            ],
        )
    }

    for class_name, expected in (
        EXPECTED_BASELINE[
            "per_class"
        ].items()
    ):
        checks[
            f"class_{class_name}"
        ] = count_fields_match(
            baseline["per_class"][
                class_name
            ],
            expected,
        )

    for size_name, expected in (
        EXPECTED_BASELINE[
            "per_size"
        ].items()
    ):
        checks[
            f"size_{size_name}"
        ] = count_fields_match(
            baseline["per_size"][
                size_name
            ],
            expected,
        )

    return (
        all(checks.values()),
        checks,
    )


def numeric_difference(
    baseline: float | int,
    candidate: float | int,
) -> float:
    return (
        float(candidate)
        - float(baseline)
    )


def build_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    overall_keys = [
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]

    overall = {
        key: {
            "baseline": (
                baseline["overall"][key]
            ),
            "candidate": (
                candidate["overall"][key]
            ),
            "difference": (
                numeric_difference(
                    baseline["overall"][
                        key
                    ],
                    candidate["overall"][
                        key
                    ],
                )
            ),
        }
        for key in overall_keys
    }

    per_class: dict[
        str,
        dict[str, Any],
    ] = {}

    for class_name in (
        EXPECTED_NAMES.values()
    ):
        per_class[
            class_name
        ] = {}

        for key in [
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]:
            per_class[
                class_name
            ][key] = {
                "baseline": (
                    baseline[
                        "per_class"
                    ][class_name][key]
                ),
                "candidate": (
                    candidate[
                        "per_class"
                    ][class_name][key]
                ),
                "difference": (
                    numeric_difference(
                        baseline[
                            "per_class"
                        ][class_name][key],
                        candidate[
                            "per_class"
                        ][class_name][key],
                    )
                ),
            }

    per_size: dict[
        str,
        dict[str, Any],
    ] = {}

    for size_name in SIZE_ORDER:
        per_size[size_name] = {}

        for key in [
            "tp",
            "fn",
            "recall",
        ]:
            per_size[
                size_name
            ][key] = {
                "baseline": (
                    baseline[
                        "per_size"
                    ][size_name][key]
                ),
                "candidate": (
                    candidate[
                        "per_size"
                    ][size_name][key]
                ),
                "difference": (
                    numeric_difference(
                        baseline[
                            "per_size"
                        ][size_name][key],
                        candidate[
                            "per_size"
                        ][size_name][key],
                    )
                ),
            }

    combined_size: dict[
        str,
        dict[str, Any],
    ] = {}

    for group in [
        "tiny_small",
        "medium_large",
    ]:
        combined_size[group] = {}

        for key in [
            "tp",
            "fn",
            "recall",
        ]:
            combined_size[
                group
            ][key] = {
                "baseline": (
                    baseline[
                        "combined_size"
                    ][group][key]
                ),
                "candidate": (
                    candidate[
                        "combined_size"
                    ][group][key]
                ),
                "difference": (
                    numeric_difference(
                        baseline[
                            "combined_size"
                        ][group][key],
                        candidate[
                            "combined_size"
                        ][group][key],
                    )
                ),
            }

    return {
        "overall": overall,
        "per_class": per_class,
        "per_size": per_size,
        "combined_size": (
            combined_size
        ),
    }


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-weights",
        required=True,
    )

    parser.add_argument(
        "--rep-deploy-weights",
        required=True,
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    baseline_weights = Path(
        args.baseline_weights
    ).resolve()

    rep_deploy_weights = Path(
        args.rep_deploy_weights
    ).resolve()

    data_yaml = Path(
        args.data
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        torch.serialization.add_safe_globals(
            [
                RepConvBlock,
                ConvBN,
            ]
        )
    except Exception:
        pass

    images, names = load_test_images(
        data_yaml
    )

    if names != EXPECTED_NAMES:
        raise RuntimeError(
            "unexpected dataset class names: "
            f"{names}"
        )

    settings = {
        "imgsz": 640,
        "confidence_threshold": 0.25,
        "nms_iou": 0.70,
        "matching_iou": 0.50,
        "batch": args.batch,
        "image_count": len(images),
    }

    baseline = audit_model(
        model_name="baseline",
        weights=baseline_weights,
        images=images,
        names=names,
        confidence=settings[
            "confidence_threshold"
        ],
        nms_iou=settings[
            "nms_iou"
        ],
        matching_iou=settings[
            "matching_iou"
        ],
        batch=args.batch,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rep_deploy = audit_model(
        model_name="rep_deploy",
        weights=rep_deploy_weights,
        images=images,
        names=names,
        confidence=settings[
            "confidence_threshold"
        ],
        nms_iou=settings[
            "nms_iou"
        ],
        matching_iou=settings[
            "matching_iou"
        ],
        batch=args.batch,
    )

    (
        baseline_match,
        baseline_match_checks,
    ) = baseline_reproduced(
        baseline
    )

    comparison = build_comparison(
        baseline,
        rep_deploy,
    )

    checks = {
        "image_count_219": (
            len(images) == 219
        ),
        "class_names": (
            names == EXPECTED_NAMES
        ),
        "baseline_reference_reproduced": (
            baseline_match
        ),
        "baseline_gt_840": (
            baseline[
                "overall"
            ]["gt"] == 840
        ),
        "rep_gt_840": (
            rep_deploy[
                "overall"
            ]["gt"] == 840
        ),
        "same_gt_count": (
            baseline[
                "overall"
            ]["gt"]
            == rep_deploy[
                "overall"
            ]["gt"]
        ),
        "finite_overall_metrics": all(
            math.isfinite(
                float(
                    result["overall"][
                        key
                    ]
                )
            )
            for result in [
                baseline,
                rep_deploy,
            ]
            for key in [
                "precision",
                "recall",
                "f1",
            ]
        ),
    }

    result = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    payload = {
        "experiment": (
            "Exp04.5b Reparameterization "
            "Error and Size Audit"
        ),
        "result": result,
        "settings": settings,
        "data_yaml": str(data_yaml),
        "baseline": {
            key: value
            for key, value
            in baseline.items()
            if key != "per_image"
        },
        "rep_deploy": {
            key: value
            for key, value
            in rep_deploy.items()
            if key != "per_image"
        },
        "comparison": comparison,
        "expected_baseline": (
            EXPECTED_BASELINE
        ),
        "baseline_reference_checks": (
            baseline_match_checks
        ),
        "checks": checks,
        "route_decision": (
            "PENDING_MANUAL_REVIEW"
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_rows(
        output_dir
        / "baseline_per_image.csv",
        baseline["per_image"],
    )

    write_rows(
        output_dir
        / "rep_deploy_per_image.csv",
        rep_deploy["per_image"],
    )

    overall_rows = []

    for model_name, model_result in [
        ("baseline", baseline),
        ("rep_deploy", rep_deploy),
    ]:
        overall_rows.append(
            {
                "model": model_name,
                **model_result[
                    "overall"
                ],
            }
        )

    write_rows(
        output_dir
        / "overall_metrics.csv",
        overall_rows,
    )

    class_rows = []

    for model_name, model_result in [
        ("baseline", baseline),
        ("rep_deploy", rep_deploy),
    ]:
        for class_name, values in (
            model_result[
                "per_class"
            ].items()
        ):
            class_rows.append(
                {
                    "model": model_name,
                    "class_name": (
                        class_name
                    ),
                    **values,
                }
            )

    write_rows(
        output_dir
        / "per_class_metrics.csv",
        class_rows,
    )

    size_rows = []

    for model_name, model_result in [
        ("baseline", baseline),
        ("rep_deploy", rep_deploy),
    ]:
        for size_name, values in (
            model_result[
                "per_size"
            ].items()
        ):
            size_rows.append(
                {
                    "model": model_name,
                    "size": size_name,
                    **values,
                }
            )

        for size_name, values in (
            model_result[
                "combined_size"
            ].items()
        ):
            size_rows.append(
                {
                    "model": model_name,
                    "size": size_name,
                    **values,
                }
            )

    write_rows(
        output_dir
        / "per_size_metrics.csv",
        size_rows,
    )

    lines = [
        "=" * 72,
        (
            " Exp04.5b Reparameterization "
            "Error and Size Audit"
        ),
        "=" * 72,
        f"result={result}",
        f"data_yaml={data_yaml}",
        (
            "image_count="
            f"{len(images)}"
        ),
        (
            "confidence_threshold="
            f"{settings['confidence_threshold']}"
        ),
        (
            "nms_iou="
            f"{settings['nms_iou']}"
        ),
        (
            "matching_iou="
            f"{settings['matching_iou']}"
        ),
        f"batch={args.batch}",
    ]

    for model_name, model_result in [
        ("baseline", baseline),
        ("rep_deploy", rep_deploy),
    ]:
        overall = model_result[
            "overall"
        ]

        lines.extend(
            [
                "",
                (
                    "========== "
                    f"{model_name} "
                    "=========="
                ),
                (
                    "weights="
                    f"{model_result['weights']}"
                ),
                f"gt={overall['gt']}",
                f"tp={overall['tp']}",
                f"fp={overall['fp']}",
                f"fn={overall['fn']}",
                (
                    "precision="
                    f"{overall['precision']:.12g}"
                ),
                (
                    "recall="
                    f"{overall['recall']:.12g}"
                ),
                (
                    "f1="
                    f"{overall['f1']:.12g}"
                ),
            ]
        )

        for class_name, values in (
            model_result[
                "per_class"
            ].items()
        ):
            lines.append(
                "class="
                f"{class_name},"
                f"gt={values['gt']},"
                f"tp={values['tp']},"
                f"fp={values['fp']},"
                f"fn={values['fn']},"
                f"precision={values['precision']:.12g},"
                f"recall={values['recall']:.12g},"
                f"f1={values['f1']:.12g}"
            )

        for size_name, values in (
            model_result[
                "per_size"
            ].items()
        ):
            lines.append(
                "size="
                f"{size_name},"
                f"gt={values['gt']},"
                f"tp={values['tp']},"
                f"fn={values['fn']},"
                f"recall={values['recall']:.12g}"
            )

        for size_name, values in (
            model_result[
                "combined_size"
            ].items()
        ):
            lines.append(
                "size="
                f"{size_name},"
                f"gt={values['gt']},"
                f"tp={values['tp']},"
                f"fn={values['fn']},"
                f"recall={values['recall']:.12g}"
            )

    lines.extend(
        [
            "",
            "========== differences ==========",
        ]
    )

    for key, item in (
        comparison[
            "overall"
        ].items()
    ):
        lines.append(
            f"overall_{key}_difference="
            f"{item['difference']:.12g}"
        )

    for size_name, values in (
        comparison[
            "per_size"
        ].items()
    ):
        lines.append(
            f"{size_name}_recall_difference="
            f"{values['recall']['difference']:.12g}"
        )

    for size_name, values in (
        comparison[
            "combined_size"
        ].items()
    ):
        lines.append(
            f"{size_name}_recall_difference="
            f"{values['recall']['difference']:.12g}"
        )

    lines.extend(
        [
            "",
            (
                "========== baseline "
                "reproduction =========="
            ),
        ]
    )

    for name, passed in (
        baseline_match_checks.items()
    ):
        lines.append(
            f"{name}="
            f"{'PASS' if passed else 'FAIL'}"
        )

    lines.extend(
        [
            "",
            "========== checks ==========",
        ]
    )

    for name, passed in checks.items():
        lines.append(
            f"{name}="
            f"{'PASS' if passed else 'FAIL'}"
        )

    lines.extend(
        [
            "",
            (
                "route_decision="
                "PENDING_MANUAL_REVIEW"
            ),
            f"overall={result}",
        ]
    )

    summary_text = (
        "\n".join(lines)
        + "\n"
    )

    (
        output_dir
        / "summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        summary_text,
        end="",
    )

    return (
        0
        if result == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
