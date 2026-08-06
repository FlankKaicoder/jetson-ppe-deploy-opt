from __future__ import annotations

import argparse
import copy
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
    switch_reparam_blocks_to_deploy,
)
from tools.exp04_3_yolo11n_rep_train import (
    compare_outputs,
    inspect_reparam_blocks,
    parameter_count,
    sha256_file,
)


def extract_validation_metrics(
    metrics: Any,
) -> dict[str, float]:
    raw = getattr(
        metrics,
        "results_dict",
        None,
    )

    if not isinstance(raw, dict):
        raise RuntimeError(
            "validation results_dict is unavailable"
        )

    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    result: dict[str, float] = {}

    for key in keys:
        value = raw.get(key)

        if value is None:
            raise RuntimeError(
                f"validation metric missing: {key}"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise RuntimeError(
                f"validation metric is not finite: "
                f"{key}={numeric}"
            )

        result[key] = numeric

    return result


def extract_speed(
    metrics: Any,
) -> dict[str, float]:
    raw = getattr(
        metrics,
        "speed",
        {},
    )

    if not isinstance(raw, dict):
        return {}

    result: dict[str, float] = {}

    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            result[str(key)] = numeric

    return result


def save_fp32_deploy_checkpoint(
    source_checkpoint: Path,
    deploy_model: nn.Module,
    deploy_path: Path,
    conversion_metadata: dict[str, Any],
) -> None:
    checkpoint = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "source checkpoint is not a dictionary"
        )

    saved_model = copy.deepcopy(
        deploy_model
    ).float().cpu().eval()

    for parameter in saved_model.parameters():
        parameter.requires_grad = False

    checkpoint["model"] = saved_model
    checkpoint["ema"] = None
    checkpoint["optimizer"] = None
    checkpoint["updates"] = None
    checkpoint["best_fitness"] = None
    checkpoint["epoch"] = -1

    checkpoint[
        "reparameterization"
    ] = conversion_metadata

    deploy_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        deploy_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-weights",
        required=True,
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--artifact-dir",
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        required=True,
    )

    args = parser.parse_args()

    training_weights = Path(
        args.training_weights
    ).resolve()

    data_yaml = Path(
        args.data
    ).resolve()

    artifact_dir = Path(
        args.artifact_dir
    ).resolve()

    report_dir = Path(
        args.report_dir
    ).resolve()

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not training_weights.is_file():
        raise FileNotFoundError(
            "training checkpoint not found: "
            f"{training_weights}"
        )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"dataset YAML not found: {data_yaml}"
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

    torch.manual_seed(123)
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        torch.set_float32_matmul_precision(
            "highest"
        )
    except Exception:
        pass

    training_yolo = YOLO(
        str(training_weights)
    )

    training_model = (
        training_yolo.model
        .float()
        .cpu()
        .eval()
    )

    training_rep_count = (
        count_reparam_blocks(
            training_model
        )
    )

    training_block_details = (
        inspect_reparam_blocks(
            training_model
        )
    )

    training_form = (
        training_rep_count == 2
        and all(
            not item["deploy"]
            for item in training_block_details
        )
    )

    training_parameter_count = (
        parameter_count(
            training_model
        )
    )

    input_tensor = torch.randn(
        1,
        3,
        640,
        640,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        training_output = training_model(
            input_tensor
        )

    deploy_model = copy.deepcopy(
        training_model
    )

    converted_block_count = (
        switch_reparam_blocks_to_deploy(
            deploy_model
        )
    )

    deploy_rep_count = (
        count_reparam_blocks(
            deploy_model
        )
    )

    deploy_form = (
        all_reparam_blocks_deployed(
            deploy_model
        )
    )

    deploy_parameter_count = (
        parameter_count(
            deploy_model
        )
    )

    with torch.inference_mode():
        deploy_output = deploy_model(
            input_tensor
        )

    training_vs_deploy = (
        compare_outputs(
            training_output,
            deploy_output,
        )
    )

    deploy_path = (
        artifact_dir
        / "weights"
        / "best_deploy_fp32.pt"
    )

    conversion_metadata = {
        "source_checkpoint": str(
            training_weights
        ),
        "source_sha256": sha256_file(
            training_weights
        ),
        "converted_block_count": (
            converted_block_count
        ),
        "training_parameter_count": (
            training_parameter_count
        ),
        "deploy_parameter_count": (
            deploy_parameter_count
        ),
        "conversion_error": (
            training_vs_deploy
        ),
    }

    save_fp32_deploy_checkpoint(
        source_checkpoint=training_weights,
        deploy_model=deploy_model,
        deploy_path=deploy_path,
        conversion_metadata=(
            conversion_metadata
        ),
    )

    if not deploy_path.is_file():
        raise RuntimeError(
            "deploy checkpoint was not generated"
        )

    reloaded_yolo = YOLO(
        str(deploy_path)
    )

    reloaded_model = (
        reloaded_yolo.model
        .float()
        .cpu()
        .eval()
    )

    reloaded_rep_count = (
        count_reparam_blocks(
            reloaded_model
        )
    )

    reloaded_deploy_form = (
        all_reparam_blocks_deployed(
            reloaded_model
        )
    )

    reloaded_parameter_count = (
        parameter_count(
            reloaded_model
        )
    )

    with torch.inference_mode():
        reloaded_output = (
            reloaded_model(
                input_tensor
            )
        )

    deploy_vs_reloaded = (
        compare_outputs(
            deploy_output,
            reloaded_output,
        )
    )

    # Full validation is intentionally run from two separately loaded
    # checkpoints so that validation-time fusion or caching from one
    # model cannot affect the other.
    validation_root = (
        artifact_dir
        / "validation"
    )

    training_val_yolo = YOLO(
        str(training_weights)
    )

    training_metrics_object = (
        training_val_yolo.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=16,
            workers=8,
            device=0,
            plots=False,
            save_json=False,
            project=str(
                validation_root
            ),
            name="training_form",
            exist_ok=False,
            verbose=True,
        )
    )

    deploy_val_yolo = YOLO(
        str(deploy_path)
    )

    deploy_metrics_object = (
        deploy_val_yolo.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=16,
            workers=8,
            device=0,
            plots=False,
            save_json=False,
            project=str(
                validation_root
            ),
            name="deploy_form",
            exist_ok=False,
            verbose=True,
        )
    )

    training_metrics = (
        extract_validation_metrics(
            training_metrics_object
        )
    )

    deploy_metrics = (
        extract_validation_metrics(
            deploy_metrics_object
        )
    )

    training_speed = extract_speed(
        training_metrics_object
    )

    deploy_speed = extract_speed(
        deploy_metrics_object
    )

    metric_differences: dict[
        str,
        dict[str, float],
    ] = {}

    maximum_metric_abs_difference = 0.0

    for key in training_metrics:
        signed_difference = (
            deploy_metrics[key]
            - training_metrics[key]
        )

        absolute_difference = abs(
            signed_difference
        )

        maximum_metric_abs_difference = max(
            maximum_metric_abs_difference,
            absolute_difference,
        )

        metric_differences[key] = {
            "training": (
                training_metrics[key]
            ),
            "deploy": (
                deploy_metrics[key]
            ),
            "signed_difference": (
                signed_difference
            ),
            "absolute_difference": (
                absolute_difference
            ),
        }

    thresholds = {
        "forward_max_abs_error": 5e-4,
        "forward_mean_abs_error": 1e-6,
        "forward_relative_l2_error": 1e-6,
        "save_reload_max_abs_error": 1e-7,
        "validation_metric_abs_difference": (
            1e-5
        ),
    }

    checks = {
        "training_rep_count": (
            training_rep_count == 2
        ),
        "training_form": training_form,
        "converted_block_count": (
            converted_block_count == 2
        ),
        "deploy_rep_count": (
            deploy_rep_count == 2
        ),
        "deploy_form": deploy_form,
        "parameter_count_reduced": (
            deploy_parameter_count
            < training_parameter_count
        ),
        "forward_max_abs_error": (
            training_vs_deploy[
                "max_abs_error"
            ]
            <= thresholds[
                "forward_max_abs_error"
            ]
        ),
        "forward_mean_abs_error": (
            training_vs_deploy[
                "mean_abs_error"
            ]
            <= thresholds[
                "forward_mean_abs_error"
            ]
        ),
        "forward_relative_l2_error": (
            training_vs_deploy[
                "relative_l2_error"
            ]
            <= thresholds[
                "forward_relative_l2_error"
            ]
        ),
        "deploy_checkpoint_generated": (
            deploy_path.is_file()
        ),
        "deploy_checkpoint_reload": (
            reloaded_rep_count == 2
        ),
        "reloaded_deploy_form": (
            reloaded_deploy_form
        ),
        "reloaded_parameter_count": (
            reloaded_parameter_count
            == deploy_parameter_count
        ),
        "save_reload_max_abs_error": (
            deploy_vs_reloaded[
                "max_abs_error"
            ]
            <= thresholds[
                "save_reload_max_abs_error"
            ]
        ),
        "validation_metric_equivalence": (
            maximum_metric_abs_difference
            <= thresholds[
                "validation_metric_abs_difference"
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
            "Exp04.4 Rep Training-to-Deploy "
            "Conversion and Validation"
        ),
        "result": result,
        "training_weights": str(
            training_weights
        ),
        "training_weights_sha256": (
            sha256_file(
                training_weights
            )
        ),
        "deploy_weights": str(
            deploy_path
        ),
        "deploy_weights_sha256": (
            sha256_file(
                deploy_path
            )
        ),
        "data_yaml": str(data_yaml),
        "training_rep_count": (
            training_rep_count
        ),
        "training_form": training_form,
        "training_block_details": (
            training_block_details
        ),
        "training_parameter_count": (
            training_parameter_count
        ),
        "converted_block_count": (
            converted_block_count
        ),
        "deploy_rep_count": (
            deploy_rep_count
        ),
        "deploy_form": deploy_form,
        "deploy_parameter_count": (
            deploy_parameter_count
        ),
        "reloaded_rep_count": (
            reloaded_rep_count
        ),
        "reloaded_deploy_form": (
            reloaded_deploy_form
        ),
        "reloaded_parameter_count": (
            reloaded_parameter_count
        ),
        "training_vs_deploy": (
            training_vs_deploy
        ),
        "deploy_vs_reloaded": (
            deploy_vs_reloaded
        ),
        "training_validation_metrics": (
            training_metrics
        ),
        "deploy_validation_metrics": (
            deploy_metrics
        ),
        "metric_differences": (
            metric_differences
        ),
        "maximum_metric_abs_difference": (
            maximum_metric_abs_difference
        ),
        "training_validation_speed": (
            training_speed
        ),
        "deploy_validation_speed": (
            deploy_speed
        ),
        "thresholds": thresholds,
        "checks": checks,
    }

    (
        report_dir
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

    lines = [
        "=" * 74,
        (
            " Exp04.4 Rep Training-to-Deploy "
            "Conversion and Validation"
        ),
        "=" * 74,
        f"result={result}",
        (
            "training_weights="
            f"{training_weights}"
        ),
        (
            "training_weights_sha256="
            f"{payload['training_weights_sha256']}"
        ),
        (
            "deploy_weights="
            f"{deploy_path}"
        ),
        (
            "deploy_weights_sha256="
            f"{payload['deploy_weights_sha256']}"
        ),
        f"data_yaml={data_yaml}",
        "",
        "========== structure ==========",
        (
            "training_rep_count="
            f"{training_rep_count}"
        ),
        (
            "training_form="
            f"{training_form}"
        ),
        (
            "training_parameter_count="
            f"{training_parameter_count}"
        ),
        (
            "converted_block_count="
            f"{converted_block_count}"
        ),
        (
            "deploy_rep_count="
            f"{deploy_rep_count}"
        ),
        f"deploy_form={deploy_form}",
        (
            "deploy_parameter_count="
            f"{deploy_parameter_count}"
        ),
        (
            "reloaded_rep_count="
            f"{reloaded_rep_count}"
        ),
        (
            "reloaded_deploy_form="
            f"{reloaded_deploy_form}"
        ),
        (
            "reloaded_parameter_count="
            f"{reloaded_parameter_count}"
        ),
        "",
        "========== forward equivalence ==========",
        (
            "training_vs_deploy_max_abs_error="
            f"{training_vs_deploy['max_abs_error']:.12g}"
        ),
        (
            "training_vs_deploy_mean_abs_error="
            f"{training_vs_deploy['mean_abs_error']:.12g}"
        ),
        (
            "training_vs_deploy_relative_l2_error="
            f"{training_vs_deploy['relative_l2_error']:.12g}"
        ),
        (
            "deploy_vs_reloaded_max_abs_error="
            f"{deploy_vs_reloaded['max_abs_error']:.12g}"
        ),
        "",
        "========== validation metrics ==========",
    ]

    for key in training_metrics:
        item = metric_differences[key]

        lines.extend(
            [
                f"{key}_training={item['training']:.12g}",
                f"{key}_deploy={item['deploy']:.12g}",
                (
                    f"{key}_signed_difference="
                    f"{item['signed_difference']:.12g}"
                ),
            ]
        )

    lines.extend(
        [
            (
                "maximum_metric_abs_difference="
                f"{maximum_metric_abs_difference:.12g}"
            ),
            "",
            "========== validation speed ==========",
            (
                "training_speed="
                f"{json.dumps(training_speed)}"
            ),
            (
                "deploy_speed="
                f"{json.dumps(deploy_speed)}"
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
            f"overall={result}",
        ]
    )

    summary_text = (
        "\n".join(lines)
        + "\n"
    )

    (
        report_dir
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
