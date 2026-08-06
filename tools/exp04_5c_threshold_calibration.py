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
from tools.exp04_5b_reparam_error_size_audit import (
    EXPECTED_NAMES,
    SIZE_ORDER,
    box_iou,
    collect_images_from_entry,
    label_path_for_image,
    load_ground_truth,
)


EXPECTED_FIXED_TEST = {
    "baseline": {
        "gt": 840,
        "tp": 731,
        "fp": 169,
        "fn": 109,
    },
    "rep_deploy": {
        "gt": 840,
        "tp": 721,
        "fp": 142,
        "fn": 119,
    },
}


def load_split_images(
    data_yaml: Path,
    split: str,
) -> tuple[list[Path], dict[int, str]]:
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

    split_entry = data.get(split)

    if split_entry is None:
        raise KeyError(
            f"dataset YAML does not contain split={split}"
        )

    entries = (
        split_entry
        if isinstance(split_entry, list)
        else [split_entry]
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
            f"no images found for split={split}"
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


def collect_prediction_cache(
    model_name: str,
    weights: Path,
    images: list[Path],
    names: dict[int, str],
    batch: int,
) -> dict[str, Any]:
    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}"
        )

    print()
    print("=" * 74)
    print(
        f" Collecting low-confidence predictions: {model_name}"
    )
    print("=" * 74)
    print(f"weights={weights}")
    print(f"image_count={len(images)}")

    yolo = YOLO(str(weights))

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
        conf=0.001,
        iou=0.70,
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
            f"{len(results)} != {len(images)}"
        )

    records: list[dict[str, Any]] = []

    total_prediction_count = 0
    maximum_prediction_count = 0

    for index, (
        image_path,
        result,
    ) in enumerate(
        zip(images, results),
        start=1,
    ):
        height = int(
            result.orig_shape[0]
        )

        width = int(
            result.orig_shape[1]
        )

        label_path = label_path_for_image(
            image_path
        )

        ground_truth = load_ground_truth(
            label_path,
            width,
            height,
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

            for (
                xyxy,
                confidence,
                class_id,
            ) in zip(
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
                            confidence
                        ),
                        "xyxy": [
                            float(value)
                            for value in xyxy
                        ],
                    }
                )

        predictions.sort(
            key=lambda item: (
                item["confidence"]
            ),
            reverse=True,
        )

        total_prediction_count += len(
            predictions
        )

        maximum_prediction_count = max(
            maximum_prediction_count,
            len(predictions),
        )

        records.append(
            {
                "image": str(image_path),
                "ground_truth": (
                    ground_truth
                ),
                "predictions": (
                    predictions
                ),
            }
        )

        if (
            index % 50 == 0
            or index == len(images)
        ):
            print(
                f"cached={index}/{len(images)}"
            )

    return {
        "model": model_name,
        "weights": str(weights),
        "image_count": len(images),
        "total_prediction_count": (
            total_prediction_count
        ),
        "maximum_prediction_count": (
            maximum_prediction_count
        ),
        "records": records,
    }


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


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator) / float(
        denominator
    )


def evaluate_threshold(
    cache: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    class_counters = (
        empty_class_counters()
    )

    size_counters = (
        empty_size_counters()
    )

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0

    for record in cache["records"]:
        ground_truth = record[
            "ground_truth"
        ]

        predictions = [
            prediction
            for prediction
            in record["predictions"]
            if (
                prediction["confidence"]
                >= threshold
            )
        ]

        matched_gt: set[int] = set()
        matched_prediction: set[int] = (
            set()
        )

        for (
            prediction_index,
            prediction,
        ) in enumerate(predictions):
            best_gt_index = None
            best_overlap = -1.0

            for gt_index, gt in enumerate(
                ground_truth
            ):
                if gt_index in matched_gt:
                    continue

                if (
                    gt["class_id"]
                    != prediction["class_id"]
                ):
                    continue

                overlap = box_iou(
                    prediction["xyxy"],
                    gt["xyxy"],
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt_index = (
                        gt_index
                    )

            if (
                best_gt_index is not None
                and best_overlap >= 0.50
            ):
                matched_gt.add(
                    best_gt_index
                )

                matched_prediction.add(
                    prediction_index
                )

        total_gt += len(ground_truth)
        total_predictions += len(
            predictions
        )

        total_tp += len(matched_gt)

        total_fp += (
            len(predictions)
            - len(matched_prediction)
        )

        total_fn += (
            len(ground_truth)
            - len(matched_gt)
        )

        for gt_index, gt in enumerate(
            ground_truth
        ):
            class_name = EXPECTED_NAMES[
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

        for (
            prediction_index,
            prediction,
        ) in enumerate(predictions):
            if (
                prediction_index
                in matched_prediction
            ):
                continue

            class_name = EXPECTED_NAMES[
                prediction["class_id"]
            ]

            class_counters[
                class_name
            ]["fp"] += 1

    precision = safe_ratio(
        total_tp,
        total_tp + total_fp,
    )

    recall = safe_ratio(
        total_tp,
        total_tp + total_fn,
    )

    f1 = safe_ratio(
        2.0 * precision * recall,
        precision + recall,
    )

    per_class: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        class_name,
        values,
    ) in class_counters.items():
        class_precision = safe_ratio(
            values["tp"],
            values["tp"]
            + values["fp"],
        )

        class_recall = safe_ratio(
            values["tp"],
            values["tp"]
            + values["fn"],
        )

        per_class[class_name] = {
            **values,
            "precision": (
                class_precision
            ),
            "recall": class_recall,
            "f1": safe_ratio(
                2.0
                * class_precision
                * class_recall,
                class_precision
                + class_recall,
            ),
        }

    per_size: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        size_name,
        values,
    ) in size_counters.items():
        per_size[size_name] = {
            **values,
            "recall": safe_ratio(
                values["tp"],
                values["tp"]
                + values["fn"],
            ),
        }

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
        "threshold": threshold,
        "overall": {
            "gt": total_gt,
            "predictions": (
                total_predictions
            ),
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
                "recall": safe_ratio(
                    tiny_small_tp,
                    tiny_small_gt,
                ),
            },
            "medium_large": {
                "gt": medium_large_gt,
                "tp": medium_large_tp,
                "fn": medium_large_fn,
                "recall": safe_ratio(
                    medium_large_tp,
                    medium_large_gt,
                ),
            },
        },
    }


def make_thresholds() -> list[float]:
    thresholds = [0.001]

    thresholds.extend(
        round(index / 100.0, 2)
        for index in range(1, 51)
    )

    return sorted(set(thresholds))


def sweep_cache(
    cache: dict[str, Any],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    return [
        evaluate_threshold(
            cache,
            threshold,
        )
        for threshold in thresholds
    ]


def flatten_sweep_row(
    model: str,
    split: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "threshold": (
            result["threshold"]
        ),
        "gt": (
            result["overall"]["gt"]
        ),
        "predictions": (
            result["overall"][
                "predictions"
            ]
        ),
        "tp": (
            result["overall"]["tp"]
        ),
        "fp": (
            result["overall"]["fp"]
        ),
        "fn": (
            result["overall"]["fn"]
        ),
        "precision": (
            result["overall"][
                "precision"
            ]
        ),
        "recall": (
            result["overall"]["recall"]
        ),
        "f1": (
            result["overall"]["f1"]
        ),
        "tiny_tp": (
            result["per_size"][
                "tiny"
            ]["tp"]
        ),
        "tiny_fn": (
            result["per_size"][
                "tiny"
            ]["fn"]
        ),
        "tiny_recall": (
            result["per_size"][
                "tiny"
            ]["recall"]
        ),
        "small_tp": (
            result["per_size"][
                "small"
            ]["tp"]
        ),
        "small_fn": (
            result["per_size"][
                "small"
            ]["fn"]
        ),
        "small_recall": (
            result["per_size"][
                "small"
            ]["recall"]
        ),
        "tiny_small_tp": (
            result["combined_size"][
                "tiny_small"
            ]["tp"]
        ),
        "tiny_small_fn": (
            result["combined_size"][
                "tiny_small"
            ]["fn"]
        ),
        "tiny_small_recall": (
            result["combined_size"][
                "tiny_small"
            ]["recall"]
        ),
        "medium_large_recall": (
            result["combined_size"][
                "medium_large"
            ]["recall"]
        ),
    }


def find_threshold_result(
    sweep: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    matches = [
        result
        for result in sweep
        if abs(
            result["threshold"]
            - threshold
        ) <= 1e-12
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "threshold result not found or ambiguous: "
            f"{threshold}"
        )

    return matches[0]


def select_best_f1(
    sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(
        sweep,
        key=lambda result: (
            result["overall"]["f1"],
            result["overall"][
                "precision"
            ],
            result["threshold"],
        ),
    )


def select_precision_match(
    sweep: list[dict[str, Any]],
    target_precision: float,
) -> dict[str, Any]:
    return min(
        sweep,
        key=lambda result: (
            abs(
                result["overall"][
                    "precision"
                ]
                - target_precision
            ),
            -result["overall"]["f1"],
            -result["combined_size"][
                "tiny_small"
            ]["recall"],
            result["threshold"],
        ),
    )


def summarize_point(
    label: str,
    model: str,
    split: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "split": split,
        **flatten_sweep_row(
            model,
            split,
            result,
        ),
    }


def compare_points(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    return {
        "tp_difference": (
            candidate["overall"]["tp"]
            - baseline["overall"]["tp"]
        ),
        "fp_difference": (
            candidate["overall"]["fp"]
            - baseline["overall"]["fp"]
        ),
        "fn_difference": (
            candidate["overall"]["fn"]
            - baseline["overall"]["fn"]
        ),
        "precision_difference": (
            candidate["overall"][
                "precision"
            ]
            - baseline["overall"][
                "precision"
            ]
        ),
        "recall_difference": (
            candidate["overall"][
                "recall"
            ]
            - baseline["overall"][
                "recall"
            ]
        ),
        "f1_difference": (
            candidate["overall"]["f1"]
            - baseline["overall"]["f1"]
        ),
        "tiny_recall_difference": (
            candidate["per_size"][
                "tiny"
            ]["recall"]
            - baseline["per_size"][
                "tiny"
            ]["recall"]
        ),
        "small_recall_difference": (
            candidate["per_size"][
                "small"
            ]["recall"]
            - baseline["per_size"][
                "small"
            ]["recall"]
        ),
        "tiny_small_recall_difference": (
            candidate["combined_size"][
                "tiny_small"
            ]["recall"]
            - baseline["combined_size"][
                "tiny_small"
            ]["recall"]
        ),
    }


def fixed_count_match(
    result: dict[str, Any],
    expected: dict[str, int],
) -> bool:
    return all(
        int(result["overall"][key])
        == int(value)
        for key, value
        in expected.items()
    )


def write_csv(
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


def format_result_lines(
    prefix: str,
    result: dict[str, Any],
) -> list[str]:
    overall = result["overall"]

    tiny = result["per_size"]["tiny"]
    small = result["per_size"]["small"]

    tiny_small = result[
        "combined_size"
    ]["tiny_small"]

    return [
        f"{prefix}_threshold={result['threshold']}",
        f"{prefix}_gt={overall['gt']}",
        f"{prefix}_tp={overall['tp']}",
        f"{prefix}_fp={overall['fp']}",
        f"{prefix}_fn={overall['fn']}",
        (
            f"{prefix}_precision="
            f"{overall['precision']:.12g}"
        ),
        (
            f"{prefix}_recall="
            f"{overall['recall']:.12g}"
        ),
        (
            f"{prefix}_f1="
            f"{overall['f1']:.12g}"
        ),
        (
            f"{prefix}_tiny_tp="
            f"{tiny['tp']}"
        ),
        (
            f"{prefix}_tiny_fn="
            f"{tiny['fn']}"
        ),
        (
            f"{prefix}_tiny_recall="
            f"{tiny['recall']:.12g}"
        ),
        (
            f"{prefix}_small_recall="
            f"{small['recall']:.12g}"
        ),
        (
            f"{prefix}_tiny_small_tp="
            f"{tiny_small['tp']}"
        ),
        (
            f"{prefix}_tiny_small_fn="
            f"{tiny_small['fn']}"
        ),
        (
            f"{prefix}_tiny_small_recall="
            f"{tiny_small['recall']:.12g}"
        ),
    ]


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

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    val_images, val_names = (
        load_split_images(
            data_yaml,
            "val",
        )
    )

    test_images, test_names = (
        load_split_images(
            data_yaml,
            "test",
        )
    )

    if (
        val_names != EXPECTED_NAMES
        or test_names != EXPECTED_NAMES
    ):
        raise RuntimeError(
            "unexpected dataset class names"
        )

    model_paths = {
        "baseline": baseline_weights,
        "rep_deploy": (
            rep_deploy_weights
        ),
    }

    caches: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for model_name, weights in (
        model_paths.items()
    ):
        caches[model_name] = {}

        caches[model_name]["val"] = (
            collect_prediction_cache(
                model_name=(
                    f"{model_name}_val"
                ),
                weights=weights,
                images=val_images,
                names=val_names,
                batch=args.batch,
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        caches[model_name]["test"] = (
            collect_prediction_cache(
                model_name=(
                    f"{model_name}_test"
                ),
                weights=weights,
                images=test_images,
                names=test_names,
                batch=args.batch,
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    thresholds = make_thresholds()

    sweeps: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = {}

    all_sweep_rows: list[
        dict[str, Any]
    ] = []

    for model_name in model_paths:
        sweeps[model_name] = {}

        for split in ["val", "test"]:
            sweep = sweep_cache(
                caches[model_name][split],
                thresholds,
            )

            sweeps[
                model_name
            ][split] = sweep

            all_sweep_rows.extend(
                flatten_sweep_row(
                    model_name,
                    split,
                    result,
                )
                for result in sweep
            )

    baseline_val_fixed = (
        find_threshold_result(
            sweeps["baseline"]["val"],
            0.25,
        )
    )

    rep_val_fixed = (
        find_threshold_result(
            sweeps["rep_deploy"]["val"],
            0.25,
        )
    )

    baseline_test_fixed = (
        find_threshold_result(
            sweeps["baseline"]["test"],
            0.25,
        )
    )

    rep_test_fixed = (
        find_threshold_result(
            sweeps["rep_deploy"]["test"],
            0.25,
        )
    )

    baseline_val_best_f1 = (
        select_best_f1(
            sweeps["baseline"]["val"]
        )
    )

    rep_val_best_f1 = (
        select_best_f1(
            sweeps[
                "rep_deploy"
            ]["val"]
        )
    )

    baseline_test_at_best_f1 = (
        evaluate_threshold(
            caches["baseline"]["test"],
            baseline_val_best_f1[
                "threshold"
            ],
        )
    )

    rep_test_at_best_f1 = (
        evaluate_threshold(
            caches["rep_deploy"]["test"],
            rep_val_best_f1[
                "threshold"
            ],
        )
    )

    rep_val_precision_matched = (
        select_precision_match(
            sweeps[
                "rep_deploy"
            ]["val"],
            baseline_val_fixed[
                "overall"
            ]["precision"],
        )
    )

    rep_test_precision_matched = (
        evaluate_threshold(
            caches["rep_deploy"]["test"],
            rep_val_precision_matched[
                "threshold"
            ],
        )
    )

    fixed_comparison = compare_points(
        baseline_test_fixed,
        rep_test_fixed,
    )

    calibrated_f1_comparison = (
        compare_points(
            baseline_test_at_best_f1,
            rep_test_at_best_f1,
        )
    )

    precision_matched_comparison = (
        compare_points(
            baseline_test_fixed,
            rep_test_precision_matched,
        )
    )

    selected_rows = [
        summarize_point(
            "baseline_val_fixed_025",
            "baseline",
            "val",
            baseline_val_fixed,
        ),
        summarize_point(
            "rep_val_fixed_025",
            "rep_deploy",
            "val",
            rep_val_fixed,
        ),
        summarize_point(
            "baseline_test_fixed_025",
            "baseline",
            "test",
            baseline_test_fixed,
        ),
        summarize_point(
            "rep_test_fixed_025",
            "rep_deploy",
            "test",
            rep_test_fixed,
        ),
        summarize_point(
            "baseline_val_best_f1",
            "baseline",
            "val",
            baseline_val_best_f1,
        ),
        summarize_point(
            "rep_val_best_f1",
            "rep_deploy",
            "val",
            rep_val_best_f1,
        ),
        summarize_point(
            "baseline_test_at_val_best_f1",
            "baseline",
            "test",
            baseline_test_at_best_f1,
        ),
        summarize_point(
            "rep_test_at_val_best_f1",
            "rep_deploy",
            "test",
            rep_test_at_best_f1,
        ),
        summarize_point(
            "rep_val_precision_matched",
            "rep_deploy",
            "val",
            rep_val_precision_matched,
        ),
        summarize_point(
            "rep_test_precision_matched",
            "rep_deploy",
            "test",
            rep_test_precision_matched,
        ),
    ]

    checks = {
        "val_image_count_217": (
            len(val_images) == 217
        ),
        "test_image_count_219": (
            len(test_images) == 219
        ),
        "class_names": (
            val_names == EXPECTED_NAMES
            and test_names
            == EXPECTED_NAMES
        ),
        "baseline_fixed_test_reproduced": (
            fixed_count_match(
                baseline_test_fixed,
                EXPECTED_FIXED_TEST[
                    "baseline"
                ],
            )
        ),
        "rep_fixed_test_reproduced": (
            fixed_count_match(
                rep_test_fixed,
                EXPECTED_FIXED_TEST[
                    "rep_deploy"
                ],
            )
        ),
        "baseline_best_threshold_valid": (
            0.001
            <= baseline_val_best_f1[
                "threshold"
            ]
            <= 0.50
        ),
        "rep_best_threshold_valid": (
            0.001
            <= rep_val_best_f1[
                "threshold"
            ]
            <= 0.50
        ),
        "rep_precision_match_valid": (
            0.001
            <= rep_val_precision_matched[
                "threshold"
            ]
            <= 0.50
        ),
        "all_selected_metrics_finite": all(
            math.isfinite(
                float(row[key])
            )
            for row in selected_rows
            for key in [
                "precision",
                "recall",
                "f1",
                "tiny_recall",
                "small_recall",
                "tiny_small_recall",
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
            "Exp04.5c Validation-Calibrated "
            "Confidence Threshold Analysis"
        ),
        "result": result,
        "data_yaml": str(data_yaml),
        "thresholds": thresholds,
        "baseline_val_fixed": (
            baseline_val_fixed
        ),
        "rep_val_fixed": (
            rep_val_fixed
        ),
        "baseline_test_fixed": (
            baseline_test_fixed
        ),
        "rep_test_fixed": (
            rep_test_fixed
        ),
        "baseline_val_best_f1": (
            baseline_val_best_f1
        ),
        "rep_val_best_f1": (
            rep_val_best_f1
        ),
        "baseline_test_at_best_f1": (
            baseline_test_at_best_f1
        ),
        "rep_test_at_best_f1": (
            rep_test_at_best_f1
        ),
        "rep_val_precision_matched": (
            rep_val_precision_matched
        ),
        "rep_test_precision_matched": (
            rep_test_precision_matched
        ),
        "fixed_comparison": (
            fixed_comparison
        ),
        "calibrated_f1_comparison": (
            calibrated_f1_comparison
        ),
        "precision_matched_comparison": (
            precision_matched_comparison
        ),
        "checks": checks,
        "route_decision": (
            "PENDING_CALIBRATION_REVIEW"
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

    write_csv(
        output_dir
        / "threshold_sweep.csv",
        all_sweep_rows,
    )

    write_csv(
        output_dir
        / "selected_operating_points.csv",
        selected_rows,
    )

    lines = [
        "=" * 78,
        (
            " Exp04.5c Validation-Calibrated "
            "Confidence Threshold Analysis"
        ),
        "=" * 78,
        f"result={result}",
        f"data_yaml={data_yaml}",
        (
            "val_image_count="
            f"{len(val_images)}"
        ),
        (
            "test_image_count="
            f"{len(test_images)}"
        ),
        "",
        "========== fixed conf=0.25 test ==========",
    ]

    lines.extend(
        format_result_lines(
            "baseline_fixed",
            baseline_test_fixed,
        )
    )

    lines.append("")

    lines.extend(
        format_result_lines(
            "rep_fixed",
            rep_test_fixed,
        )
    )

    lines.extend(
        [
            "",
            (
                "fixed_f1_difference="
                f"{fixed_comparison['f1_difference']:.12g}"
            ),
            (
                "fixed_tiny_recall_difference="
                f"{fixed_comparison['tiny_recall_difference']:.12g}"
            ),
            (
                "fixed_tiny_small_recall_difference="
                f"{fixed_comparison['tiny_small_recall_difference']:.12g}"
            ),
            "",
            "========== validation best-F1 thresholds ==========",
            (
                "baseline_val_best_f1_threshold="
                f"{baseline_val_best_f1['threshold']}"
            ),
            (
                "baseline_val_best_f1="
                f"{baseline_val_best_f1['overall']['f1']:.12g}"
            ),
            (
                "rep_val_best_f1_threshold="
                f"{rep_val_best_f1['threshold']}"
            ),
            (
                "rep_val_best_f1="
                f"{rep_val_best_f1['overall']['f1']:.12g}"
            ),
            "",
            "========== test using val best-F1 thresholds ==========",
        ]
    )

    lines.extend(
        format_result_lines(
            "baseline_calibrated",
            baseline_test_at_best_f1,
        )
    )

    lines.append("")

    lines.extend(
        format_result_lines(
            "rep_calibrated",
            rep_test_at_best_f1,
        )
    )

    lines.extend(
        [
            "",
            (
                "calibrated_tp_difference="
                f"{calibrated_f1_comparison['tp_difference']}"
            ),
            (
                "calibrated_fp_difference="
                f"{calibrated_f1_comparison['fp_difference']}"
            ),
            (
                "calibrated_fn_difference="
                f"{calibrated_f1_comparison['fn_difference']}"
            ),
            (
                "calibrated_f1_difference="
                f"{calibrated_f1_comparison['f1_difference']:.12g}"
            ),
            (
                "calibrated_tiny_recall_difference="
                f"{calibrated_f1_comparison['tiny_recall_difference']:.12g}"
            ),
            (
                "calibrated_small_recall_difference="
                f"{calibrated_f1_comparison['small_recall_difference']:.12g}"
            ),
            (
                "calibrated_tiny_small_recall_difference="
                f"{calibrated_f1_comparison['tiny_small_recall_difference']:.12g}"
            ),
            "",
            "========== Rep precision-matched calibration ==========",
            (
                "baseline_val_fixed_precision="
                f"{baseline_val_fixed['overall']['precision']:.12g}"
            ),
            (
                "rep_val_precision_matched_threshold="
                f"{rep_val_precision_matched['threshold']}"
            ),
            (
                "rep_val_precision_matched_precision="
                f"{rep_val_precision_matched['overall']['precision']:.12g}"
            ),
        ]
    )

    lines.extend(
        format_result_lines(
            "rep_precision_matched_test",
            rep_test_precision_matched,
        )
    )

    lines.extend(
        [
            "",
            (
                "precision_matched_tp_difference="
                f"{precision_matched_comparison['tp_difference']}"
            ),
            (
                "precision_matched_fp_difference="
                f"{precision_matched_comparison['fp_difference']}"
            ),
            (
                "precision_matched_fn_difference="
                f"{precision_matched_comparison['fn_difference']}"
            ),
            (
                "precision_matched_f1_difference="
                f"{precision_matched_comparison['f1_difference']:.12g}"
            ),
            (
                "precision_matched_tiny_recall_difference="
                f"{precision_matched_comparison['tiny_recall_difference']:.12g}"
            ),
            (
                "precision_matched_tiny_small_recall_difference="
                f"{precision_matched_comparison['tiny_small_recall_difference']:.12g}"
            ),
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
                "PENDING_CALIBRATION_REVIEW"
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
