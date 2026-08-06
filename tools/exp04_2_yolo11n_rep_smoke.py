from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from ultralytics.utils.torch_utils import (
        de_parallel,
    )
except ImportError:
    def de_parallel(
        model: nn.Module,
    ) -> nn.Module:
        return getattr(
            model,
            "module",
            model,
        )

from models.blocks.reparam_block import (
    ConvBN,
    RepConvBlock,
)
from models.reparam_trainer import (
    ReparamDetectionTrainer,
)
from models.reparam_yolo import (
    all_reparam_blocks_deployed,
    count_reparam_blocks,
    find_neck_downsample_indices,
    switch_reparam_blocks_to_deploy,
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def parameter_count(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def flatten_tensors(
    value: Any,
    prefix: str = "output",
) -> list[tuple[str, torch.Tensor]]:
    tensors: list[
        tuple[str, torch.Tensor]
    ] = []

    if isinstance(value, torch.Tensor):
        tensors.append(
            (prefix, value)
        )

    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            tensors.extend(
                flatten_tensors(
                    item,
                    f"{prefix}.tuple[{index}]",
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            tensors.extend(
                flatten_tensors(
                    item,
                    f"{prefix}.list[{index}]",
                )
            )

    elif isinstance(value, dict):
        for key in sorted(value):
            tensors.extend(
                flatten_tensors(
                    value[key],
                    f"{prefix}.dict[{key}]",
                )
            )

    return tensors


def compare_outputs(
    reference: Any,
    candidate: Any,
) -> dict[str, Any]:
    reference_tensors = flatten_tensors(
        reference
    )

    candidate_tensors = flatten_tensors(
        candidate
    )

    if len(reference_tensors) != len(
        candidate_tensors
    ):
        raise RuntimeError(
            "output tensor count mismatch"
        )

    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    difference_square_sum = 0.0
    reference_square_sum = 0.0

    per_tensor: list[
        dict[str, Any]
    ] = []

    for (
        reference_path,
        reference_tensor,
    ), (
        candidate_path,
        candidate_tensor,
    ) in zip(
        reference_tensors,
        candidate_tensors,
    ):
        if reference_path != candidate_path:
            raise RuntimeError(
                "output path mismatch: "
                f"{reference_path} != "
                f"{candidate_path}"
            )

        if (
            reference_tensor.shape
            != candidate_tensor.shape
        ):
            raise RuntimeError(
                "output shape mismatch at "
                f"{reference_path}"
            )

        difference = (
            reference_tensor
            - candidate_tensor
        )

        absolute = difference.abs()

        tensor_maximum = (
            float(absolute.max())
            if absolute.numel()
            else 0.0
        )

        tensor_mean = (
            float(absolute.mean())
            if absolute.numel()
            else 0.0
        )

        maximum = max(
            maximum,
            tensor_maximum,
        )

        absolute_sum += float(
            absolute.sum()
        )

        element_count += (
            absolute.numel()
        )

        difference_float = (
            difference.float()
        )

        reference_float = (
            reference_tensor.float()
        )

        difference_square_sum += float(
            torch.sum(
                difference_float
                * difference_float
            )
        )

        reference_square_sum += float(
            torch.sum(
                reference_float
                * reference_float
            )
        )

        per_tensor.append(
            {
                "path": reference_path,
                "shape": list(
                    reference_tensor.shape
                ),
                "max_abs_error": (
                    tensor_maximum
                ),
                "mean_abs_error": (
                    tensor_mean
                ),
            }
        )

    mean_absolute = (
        absolute_sum
        / max(element_count, 1)
    )

    relative_l2 = (
        difference_square_sum ** 0.5
        / max(
            reference_square_sum ** 0.5,
            1e-12,
        )
    )

    return {
        "tensor_count": len(
            reference_tensors
        ),
        "element_count": (
            element_count
        ),
        "max_abs_error": maximum,
        "mean_abs_error": (
            mean_absolute
        ),
        "relative_l2_error": (
            relative_l2
        ),
        "per_tensor": per_tensor,
    }


def load_last_metrics_row(
    results_csv: Path,
) -> dict[str, Any]:
    with results_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            "results.csv contains no rows"
        )

    converted: dict[
        str,
        Any,
    ] = {}

    for key, value in rows[-1].items():
        normalized_key = key.strip()

        if value is None:
            converted[
                normalized_key
            ] = None
            continue

        normalized_value = (
            value.strip()
        )

        try:
            converted[
                normalized_key
            ] = float(
                normalized_value
            )
        except ValueError:
            converted[
                normalized_key
            ] = normalized_value

    return converted


def inspect_reparam_blocks(
    model: nn.Module,
) -> list[dict[str, Any]]:
    details: list[
        dict[str, Any]
    ] = []

    if not hasattr(model, "model"):
        return details

    for index, module in enumerate(
        model.model
    ):
        if not isinstance(
            module,
            RepConvBlock,
        ):
            continue

        item: dict[
            str,
            Any,
        ] = {
            "index": index,
            "deploy": module.deploy,
            "in_channels": (
                module.in_channels
            ),
            "out_channels": (
                module.out_channels
            ),
            "stride": module.stride,
            "parameter_count": (
                parameter_count(module)
            ),
        }

        if not module.deploy:
            gamma = (
                module.branch_1x1[1]
                .weight.detach()
                .float()
            )

            item[
                "branch_1x1_bn_gamma_abs_max"
            ] = float(
                gamma.abs().max()
            )

            item[
                "branch_1x1_bn_gamma_abs_mean"
            ] = float(
                gamma.abs().mean()
            )

            item[
                "branch_1x1_bn_gamma_nonzero_count"
            ] = int(
                torch.count_nonzero(
                    gamma
                )
            )

        details.append(item)

    return details


def metric_value(
    metrics: dict[str, Any],
    key: str,
) -> float | None:
    value = metrics.get(key)

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    return None


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained",
        required=True,
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        required=True,
    )

    args = parser.parse_args()

    pretrained = Path(
        args.pretrained
    ).resolve()

    data_yaml = Path(
        args.data
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    report_dir = Path(
        args.report_dir
    ).resolve()

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not pretrained.is_file():
        raise FileNotFoundError(
            f"pretrained model not found: "
            f"{pretrained}"
        )

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"dataset YAML not found: "
            f"{data_yaml}"
        )

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Register custom classes for current PyTorch safe-loading behavior.
    try:
        torch.serialization.add_safe_globals(
            [
                RepConvBlock,
                ConvBN,
            ]
        )
    except Exception:
        pass

    base_yolo = YOLO(
        str(pretrained)
    )

    base_model = (
        base_yolo.model
    )

    base_candidate_indices = (
        find_neck_downsample_indices(
            base_model
        )
    )

    base_parameter_count = (
        parameter_count(
            base_model
        )
    )

    train_arguments = {
        "model": str(pretrained),
        "data": str(data_yaml),
        "epochs": 1,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": 0,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.0015,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "amp": True,
        "cache": False,
        "patience": 100,
        "val": True,
        "save": True,
        "plots": False,
        "project": str(output_root),
        "name": args.run_name,
        "exist_ok": False,
        "verbose": True,
    }

    (
        report_dir
        / "train_arguments.json"
    ).write_text(
        json.dumps(
            train_arguments,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    trainer = ReparamDetectionTrainer(
        overrides=train_arguments
    )

    trainer.train()

    train_dir = Path(
        trainer.save_dir
    ).resolve()

    best_pt = (
        train_dir
        / "weights"
        / "best.pt"
    )

    last_pt = (
        train_dir
        / "weights"
        / "last.pt"
    )

    results_csv = (
        train_dir
        / "results.csv"
    )

    required_artifacts = {
        "best_pt": best_pt.is_file(),
        "last_pt": last_pt.is_file(),
        "results_csv": (
            results_csv.is_file()
        ),
    }

    trainer_model = de_parallel(
        trainer.model
    )

    trainer_rep_count = (
        count_reparam_blocks(
            trainer_model
        )
    )

    ema_rep_count = None

    if (
        getattr(
            trainer,
            "ema",
            None,
        )
        is not None
        and getattr(
            trainer.ema,
            "ema",
            None,
        )
        is not None
    ):
        ema_rep_count = (
            count_reparam_blocks(
                de_parallel(
                    trainer.ema.ema
                )
            )
        )

    if not all(
        required_artifacts.values()
    ):
        raise RuntimeError(
            "required training artifacts "
            "were not generated"
        )

    last_metrics = (
        load_last_metrics_row(
            results_csv
        )
    )

    # Reload the real serialized best checkpoint. This verifies that the
    # custom Python module can be reconstructed from disk.
    reloaded_yolo = YOLO(
        str(best_pt)
    )

    reloaded_model = (
        reloaded_yolo.model
        .float()
        .cpu()
        .eval()
    )

    checkpoint_rep_count = (
        count_reparam_blocks(
            reloaded_model
        )
    )

    checkpoint_block_details = (
        inspect_reparam_blocks(
            reloaded_model
        )
    )

    checkpoint_training_form = (
        checkpoint_rep_count == 2
        and all(
            not item["deploy"]
            for item
            in checkpoint_block_details
        )
    )

    learned_1x1_branches = (
        checkpoint_rep_count == 2
        and all(
            item.get(
                "branch_1x1_bn_gamma_nonzero_count",
                0,
            ) > 0
            for item
            in checkpoint_block_details
        )
    )

    # Strict CPU FP32 post-training conversion replay.
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = (
        False
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.allow_tf32 = (
        False
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        False
    )

    try:
        torch.set_float32_matmul_precision(
            "highest"
        )
    except Exception:
        pass

    torch.manual_seed(123)

    input_tensor = torch.randn(
        1,
        3,
        320,
        320,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        training_output = (
            reloaded_model(
                input_tensor
            )
        )

    deploy_model = copy.deepcopy(
        reloaded_model
    )

    converted_block_count = (
        switch_reparam_blocks_to_deploy(
            deploy_model
        )
    )

    deploy_parameter_count = (
        parameter_count(
            deploy_model
        )
    )

    with torch.inference_mode():
        deploy_output = (
            deploy_model(
                input_tensor
            )
        )

    conversion_error = (
        compare_outputs(
            training_output,
            deploy_output,
        )
    )

    all_blocks_deployed = (
        all_reparam_blocks_deployed(
            deploy_model
        )
    )

    thresholds = {
        "deploy_max_abs_error": (
            5e-4
        ),
        "deploy_mean_abs_error": (
            1e-6
        ),
        "deploy_relative_l2_error": (
            1e-6
        ),
    }

    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    metric_values = {
        key: metric_value(
            last_metrics,
            key,
        )
        for key in metric_keys
    }

    finite_metrics = all(
        value is not None
        and math.isfinite(value)
        for value
        in metric_values.values()
    )

    checks = {
        "base_candidate_indices": (
            base_candidate_indices
            == [17, 20]
        ),
        "trainer_rep_count": (
            trainer_rep_count == 2
        ),
        "ema_rep_count": (
            ema_rep_count == 2
        ),
        "best_pt_generated": (
            required_artifacts[
                "best_pt"
            ]
        ),
        "last_pt_generated": (
            required_artifacts[
                "last_pt"
            ]
        ),
        "results_csv_generated": (
            required_artifacts[
                "results_csv"
            ]
        ),
        "checkpoint_reload": (
            checkpoint_rep_count == 2
        ),
        "checkpoint_training_form": (
            checkpoint_training_form
        ),
        "new_1x1_branches_learned": (
            learned_1x1_branches
        ),
        "converted_block_count": (
            converted_block_count == 2
        ),
        "all_blocks_deployed": (
            all_blocks_deployed
        ),
        "deploy_max_abs_error": (
            conversion_error[
                "max_abs_error"
            ]
            <= thresholds[
                "deploy_max_abs_error"
            ]
        ),
        "deploy_mean_abs_error": (
            conversion_error[
                "mean_abs_error"
            ]
            <= thresholds[
                "deploy_mean_abs_error"
            ]
        ),
        "deploy_relative_l2_error": (
            conversion_error[
                "relative_l2_error"
            ]
            <= thresholds[
                "deploy_relative_l2_error"
            ]
        ),
        "training_metrics_finite": (
            finite_metrics
        ),
    }

    result = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    payload = {
        "experiment": (
            "Exp04.2 YOLO11n-Rep "
            "One-Epoch Training Smoke Test"
        ),
        "result": result,
        "pretrained": str(
            pretrained
        ),
        "pretrained_sha256": (
            sha256_file(
                pretrained
            )
        ),
        "data_yaml": str(
            data_yaml
        ),
        "train_dir": str(
            train_dir
        ),
        "report_dir": str(
            report_dir
        ),
        "base_candidate_indices": (
            base_candidate_indices
        ),
        "base_parameter_count": (
            base_parameter_count
        ),
        "trainer_reparam_source": (
            getattr(
                trainer,
                "reparam_source",
                None,
            )
        ),
        "trainer_reparam_manifest": (
            getattr(
                trainer,
                "reparam_manifest",
                None,
            )
        ),
        "trainer_rep_count": (
            trainer_rep_count
        ),
        "ema_rep_count": (
            ema_rep_count
        ),
        "best_pt": str(
            best_pt
        ),
        "best_pt_sha256": (
            sha256_file(
                best_pt
            )
        ),
        "last_pt": str(
            last_pt
        ),
        "results_csv": str(
            results_csv
        ),
        "required_artifacts": (
            required_artifacts
        ),
        "last_metrics": (
            last_metrics
        ),
        "selected_metrics": (
            metric_values
        ),
        "checkpoint_rep_count": (
            checkpoint_rep_count
        ),
        "checkpoint_block_details": (
            checkpoint_block_details
        ),
        "checkpoint_training_form": (
            checkpoint_training_form
        ),
        "learned_1x1_branches": (
            learned_1x1_branches
        ),
        "converted_block_count": (
            converted_block_count
        ),
        "all_blocks_deployed": (
            all_blocks_deployed
        ),
        "deploy_parameter_count": (
            deploy_parameter_count
        ),
        "conversion_error": (
            conversion_error
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
        "=" * 68,
        (
            " Exp04.2 YOLO11n-Rep "
            "One-Epoch Training Smoke Test"
        ),
        "=" * 68,
        f"result={result}",
        f"pretrained={pretrained}",
        (
            "pretrained_sha256="
            f"{payload['pretrained_sha256']}"
        ),
        f"data_yaml={data_yaml}",
        f"train_dir={train_dir}",
        f"report_dir={report_dir}",
        "",
        (
            "base_candidate_indices="
            f"{base_candidate_indices}"
        ),
        (
            "base_parameter_count="
            f"{base_parameter_count}"
        ),
        (
            "trainer_reparam_source="
            f"{payload['trainer_reparam_source']}"
        ),
        (
            "trainer_rep_count="
            f"{trainer_rep_count}"
        ),
        (
            "ema_rep_count="
            f"{ema_rep_count}"
        ),
        "",
        (
            "checkpoint_rep_count="
            f"{checkpoint_rep_count}"
        ),
        (
            "checkpoint_training_form="
            f"{checkpoint_training_form}"
        ),
        (
            "learned_1x1_branches="
            f"{learned_1x1_branches}"
        ),
    ]

    for item in checkpoint_block_details:
        lines.extend(
            [
                "",
                (
                    "reparam_layer_index="
                    f"{item['index']}"
                ),
                (
                    "reparam_layer_deploy="
                    f"{item['deploy']}"
                ),
                (
                    "branch_1x1_gamma_abs_max="
                    f"{item.get('branch_1x1_bn_gamma_abs_max', 'NA')}"
                ),
                (
                    "branch_1x1_gamma_nonzero_count="
                    f"{item.get('branch_1x1_bn_gamma_nonzero_count', 'NA')}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "converted_block_count="
                f"{converted_block_count}"
            ),
            (
                "all_blocks_deployed="
                f"{all_blocks_deployed}"
            ),
            (
                "deploy_parameter_count="
                f"{deploy_parameter_count}"
            ),
            (
                "deploy_max_abs_error="
                f"{conversion_error['max_abs_error']:.12g}"
            ),
            (
                "deploy_mean_abs_error="
                f"{conversion_error['mean_abs_error']:.12g}"
            ),
            (
                "deploy_relative_l2_error="
                f"{conversion_error['relative_l2_error']:.12g}"
            ),
            "",
            "========== one-epoch metrics ==========",
        ]
    )

    for key, value in metric_values.items():
        lines.append(
            f"{key}={value}"
        )

    lines.extend(
        [
            "",
            "========== artifacts ==========",
            f"best_pt={best_pt}",
            (
                "best_pt_sha256="
                f"{payload['best_pt_sha256']}"
            ),
            f"last_pt={last_pt}",
            f"results_csv={results_csv}",
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
