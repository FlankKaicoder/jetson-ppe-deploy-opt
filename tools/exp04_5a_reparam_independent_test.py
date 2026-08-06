from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from models.blocks.reparam_block import (
    ConvBN,
    RepConvBlock,
)
from models.reparam_yolo import (
    all_reparam_blocks_deployed,
    count_reparam_blocks,
)


OVERALL_METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def extract_overall_metrics(
    metrics: Any,
) -> dict[str, float]:
    results_dict = getattr(
        metrics,
        "results_dict",
        None,
    )

    if not isinstance(results_dict, dict):
        raise RuntimeError(
            "validation results_dict is unavailable"
        )

    result: dict[str, float] = {}

    for key in OVERALL_METRIC_KEYS:
        value = results_dict.get(key)

        if value is None:
            raise RuntimeError(
                f"missing validation metric: {key}"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise RuntimeError(
                f"non-finite validation metric: "
                f"{key}={numeric}"
            )

        result[key] = numeric

    return result


def extract_speed(
    metrics: Any,
) -> dict[str, float]:
    speed = getattr(
        metrics,
        "speed",
        {},
    )

    if not isinstance(speed, dict):
        return {}

    result: dict[str, float] = {}

    for key, value in speed.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            result[str(key)] = numeric

    return result


def to_float_list(value: Any) -> list[float]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        value = list(value)

    return [
        float(item)
        for item in value
    ]


def to_int_list(value: Any) -> list[int]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        value = list(value)

    return [
        int(item)
        for item in value
    ]


def extract_per_class_metrics(
    metrics: Any,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    box = getattr(metrics, "box", None)

    if box is None:
        raise RuntimeError(
            "metrics.box is unavailable"
        )

    class_indices = to_int_list(
        getattr(
            box,
            "ap_class_index",
            None,
        )
    )

    precision = to_float_list(
        getattr(box, "p", None)
    )

    recall = to_float_list(
        getattr(box, "r", None)
    )

    ap50 = to_float_list(
        getattr(box, "ap50", None)
    )

    ap5095 = to_float_list(
        getattr(box, "ap", None)
    )

    lengths = {
        len(class_indices),
        len(precision),
        len(recall),
        len(ap50),
        len(ap5095),
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "per-class metric length mismatch: "
            f"indices={len(class_indices)}, "
            f"precision={len(precision)}, "
            f"recall={len(recall)}, "
            f"ap50={len(ap50)}, "
            f"ap5095={len(ap5095)}"
        )

    records: list[dict[str, Any]] = []

    for position, class_index in enumerate(
        class_indices
    ):
        records.append(
            {
                "class_id": class_index,
                "class_name": names.get(
                    class_index,
                    str(class_index),
                ),
                "precision": precision[position],
                "recall": recall[position],
                "mAP50": ap50[position],
                "mAP50-95": ap5095[position],
            }
        )

    return records


def inspect_model_structure(
    yolo: YOLO,
) -> dict[str, Any]:
    model = yolo.model

    reparam_count = count_reparam_blocks(
        model
    )

    block_states: list[dict[str, Any]] = []

    if hasattr(model, "model"):
        for index, module in enumerate(
            model.model
        ):
            if not isinstance(
                module,
                RepConvBlock,
            ):
                continue

            block_states.append(
                {
                    "index": index,
                    "deploy": bool(
                        module.deploy
                    ),
                    "in_channels": (
                        module.in_channels
                    ),
                    "out_channels": (
                        module.out_channels
                    ),
                    "stride": (
                        module.stride
                    ),
                }
            )

    return {
        "parameter_count": parameter_count(
            model
        ),
        "reparam_count": reparam_count,
        "all_reparam_deployed": (
            all_reparam_blocks_deployed(
                model
            )
            if reparam_count > 0
            else False
        ),
        "reparam_blocks": block_states,
    }


def evaluate_model(
    model_name: str,
    weights: Path,
    data_yaml: Path,
    validation_root: Path,
) -> dict[str, Any]:
    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}"
        )

    print()
    print("=" * 76)
    print(f" Evaluating model={model_name}")
    print("=" * 76)
    print(f"weights={weights}")

    yolo = YOLO(
        str(weights)
    )

    structure = inspect_model_structure(
        yolo
    )

    names = {
        int(key): str(value)
        for key, value
        in yolo.names.items()
    }

    metrics = yolo.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        plots=False,
        save_json=False,
        project=str(validation_root),
        name=model_name,
        exist_ok=False,
        verbose=True,
    )

    overall = extract_overall_metrics(
        metrics
    )

    per_class = extract_per_class_metrics(
        metrics,
        names,
    )

    speed = extract_speed(metrics)

    return {
        "model_name": model_name,
        "weights": str(weights),
        "weights_sha256": sha256_file(
            weights
        ),
        "names": names,
        "structure": structure,
        "overall": overall,
        "per_class": per_class,
        "speed": speed,
    }


def metric_differences(
    reference: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, dict[str, float]]:
    result: dict[
        str,
        dict[str, float],
    ] = {}

    for key in OVERALL_METRIC_KEYS:
        signed = (
            candidate[key]
            - reference[key]
        )

        result[key] = {
            "reference": reference[key],
            "candidate": candidate[key],
            "signed_difference": signed,
            "absolute_difference": abs(
                signed
            ),
        }

    return result


def maximum_absolute_difference(
    differences: dict[
        str,
        dict[str, float],
    ],
) -> float:
    return max(
        item["absolute_difference"]
        for item in differences.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-weights",
        required=True,
    )

    parser.add_argument(
        "--rep-training-weights",
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

    args = parser.parse_args()

    baseline_weights = Path(
        args.baseline_weights
    ).resolve()

    rep_training_weights = Path(
        args.rep_training_weights
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

    validation_root = (
        output_dir
        / "validation_runs"
    )

    validation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data YAML not found: {data_yaml}"
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

    models = [
        (
            "baseline",
            baseline_weights,
        ),
        (
            "rep_training",
            rep_training_weights,
        ),
        (
            "rep_deploy",
            rep_deploy_weights,
        ),
    ]

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name, weights in models:
        results[model_name] = (
            evaluate_model(
                model_name=model_name,
                weights=weights,
                data_yaml=data_yaml,
                validation_root=(
                    validation_root
                ),
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_vs_rep_training = (
        metric_differences(
            results["baseline"][
                "overall"
            ],
            results["rep_training"][
                "overall"
            ],
        )
    )

    baseline_vs_rep_deploy = (
        metric_differences(
            results["baseline"][
                "overall"
            ],
            results["rep_deploy"][
                "overall"
            ],
        )
    )

    training_vs_deploy = (
        metric_differences(
            results["rep_training"][
                "overall"
            ],
            results["rep_deploy"][
                "overall"
            ],
        )
    )

    train_deploy_max_difference = (
        maximum_absolute_difference(
            training_vs_deploy
        )
    )

    baseline_structure = (
        results["baseline"][
            "structure"
        ]
    )

    training_structure = (
        results["rep_training"][
            "structure"
        ]
    )

    deploy_structure = (
        results["rep_deploy"][
            "structure"
        ]
    )

    expected_names = {
        0: "person",
        1: "helmet",
        2: "safety_vest",
    }

    checks = {
        "baseline_has_no_reparam": (
            baseline_structure[
                "reparam_count"
            ] == 0
        ),
        "rep_training_count": (
            training_structure[
                "reparam_count"
            ] == 2
        ),
        "rep_training_form": (
            training_structure[
                "reparam_count"
            ] == 2
            and not training_structure[
                "all_reparam_deployed"
            ]
            and all(
                not item["deploy"]
                for item in training_structure[
                    "reparam_blocks"
                ]
            )
        ),
        "rep_deploy_count": (
            deploy_structure[
                "reparam_count"
            ] == 2
        ),
        "rep_deploy_form": (
            deploy_structure[
                "all_reparam_deployed"
            ]
        ),
        "class_names_consistent": all(
            result["names"]
            == expected_names
            for result in results.values()
        ),
        "all_metrics_finite": all(
            math.isfinite(value)
            for result in results.values()
            for value in result[
                "overall"
            ].values()
        ),
        "training_deploy_metric_equivalence": (
            train_deploy_max_difference
            <= 1e-5
        ),
    }

    experiment_result = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    payload = {
        "experiment": (
            "Exp04.5a Reparameterization "
            "Independent Test Evaluation"
        ),
        "result": experiment_result,
        "data_yaml": str(data_yaml),
        "models": results,
        "baseline_vs_rep_training": (
            baseline_vs_rep_training
        ),
        "baseline_vs_rep_deploy": (
            baseline_vs_rep_deploy
        ),
        "training_vs_deploy": (
            training_vs_deploy
        ),
        "training_deploy_max_metric_difference": (
            train_deploy_max_difference
        ),
        "checks": checks,
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

    overall_csv = (
        output_dir
        / "overall_metrics.csv"
    )

    with overall_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "model",
            "weights",
            "parameter_count",
            "reparam_count",
            "all_reparam_deployed",
            *OVERALL_METRIC_KEYS,
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for model_name, result in (
            results.items()
        ):
            writer.writerow(
                {
                    "model": model_name,
                    "weights": (
                        result["weights"]
                    ),
                    "parameter_count": (
                        result["structure"][
                            "parameter_count"
                        ]
                    ),
                    "reparam_count": (
                        result["structure"][
                            "reparam_count"
                        ]
                    ),
                    "all_reparam_deployed": (
                        result["structure"][
                            "all_reparam_deployed"
                        ]
                    ),
                    **result["overall"],
                    "preprocess_ms": (
                        result["speed"].get(
                            "preprocess"
                        )
                    ),
                    "inference_ms": (
                        result["speed"].get(
                            "inference"
                        )
                    ),
                    "postprocess_ms": (
                        result["speed"].get(
                            "postprocess"
                        )
                    ),
                }
            )

    per_class_csv = (
        output_dir
        / "per_class_metrics.csv"
    )

    with per_class_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "model",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for model_name, result in (
            results.items()
        ):
            for row in result[
                "per_class"
            ]:
                writer.writerow(
                    {
                        "model": model_name,
                        **row,
                    }
                )

    lines = [
        "=" * 76,
        (
            " Exp04.5a Reparameterization "
            "Independent Test Evaluation"
        ),
        "=" * 76,
        f"result={experiment_result}",
        f"data_yaml={data_yaml}",
    ]

    for model_name in [
        "baseline",
        "rep_training",
        "rep_deploy",
    ]:
        result = results[model_name]

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
                    f"{result['weights']}"
                ),
                (
                    "weights_sha256="
                    f"{result['weights_sha256']}"
                ),
                (
                    "parameter_count="
                    f"{result['structure']['parameter_count']}"
                ),
                (
                    "reparam_count="
                    f"{result['structure']['reparam_count']}"
                ),
                (
                    "all_reparam_deployed="
                    f"{result['structure']['all_reparam_deployed']}"
                ),
            ]
        )

        for key, value in result[
            "overall"
        ].items():
            lines.append(
                f"{key}={value:.12g}"
            )

        lines.append(
            "speed="
            f"{json.dumps(result['speed'])}"
        )

        for row in result["per_class"]:
            lines.append(
                "class="
                f"{row['class_name']},"
                f"precision={row['precision']:.12g},"
                f"recall={row['recall']:.12g},"
                f"mAP50={row['mAP50']:.12g},"
                f"mAP50-95={row['mAP50-95']:.12g}"
            )

    lines.extend(
        [
            "",
            (
                "========== baseline vs "
                "rep deploy =========="
            ),
        ]
    )

    for key, item in (
        baseline_vs_rep_deploy.items()
    ):
        lines.append(
            f"{key}_difference="
            f"{item['signed_difference']:.12g}"
        )

    lines.extend(
        [
            "",
            (
                "========== rep training vs "
                "rep deploy =========="
            ),
        ]
    )

    for key, item in (
        training_vs_deploy.items()
    ):
        lines.append(
            f"{key}_difference="
            f"{item['signed_difference']:.12g}"
        )

    lines.append(
        "training_deploy_max_metric_difference="
        f"{train_deploy_max_difference:.12g}"
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
            f"overall={experiment_result}",
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

    print(summary_text, end="")

    return (
        0
        if experiment_result == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
