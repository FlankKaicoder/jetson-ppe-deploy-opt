from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from models.reparam_yolo import (
    all_reparam_blocks_deployed,
    count_reparam_blocks,
    find_neck_downsample_indices,
    replace_neck_downsample_convs,
    switch_reparam_blocks_to_deploy,
)


def parameter_count(
    module: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def shape_tree(
    value: Any,
) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)

    if isinstance(value, tuple):
        return [
            shape_tree(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            shape_tree(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): shape_tree(item)
            for key, item in value.items()
        }

    if value is None:
        return None

    return type(value).__name__


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
        return tensors

    if isinstance(value, tuple):
        for index, item in enumerate(value):
            tensors.extend(
                flatten_tensors(
                    item,
                    f"{prefix}.tuple[{index}]",
                )
            )

        return tensors

    if isinstance(value, list):
        for index, item in enumerate(value):
            tensors.extend(
                flatten_tensors(
                    item,
                    f"{prefix}.list[{index}]",
                )
            )

        return tensors

    if isinstance(value, dict):
        for key in sorted(value):
            tensors.extend(
                flatten_tensors(
                    value[key],
                    f"{prefix}.dict[{key}]",
                )
            )

        return tensors

    return tensors


def compare_outputs(
    reference: Any,
    candidate: Any,
) -> dict[str, Any]:
    reference_tensors = (
        flatten_tensors(reference)
    )

    candidate_tensors = (
        flatten_tensors(candidate)
    )

    if len(reference_tensors) != len(
        candidate_tensors
    ):
        raise RuntimeError(
            "output tensor count mismatch: "
            f"{len(reference_tensors)} != "
            f"{len(candidate_tensors)}"
        )

    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    difference_square_sum = 0.0
    reference_square_sum = 0.0

    tensor_records: list[
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
                f"{reference_path}: "
                f"{list(reference_tensor.shape)} != "
                f"{list(candidate_tensor.shape)}"
            )

        difference = (
            reference_tensor
            - candidate_tensor
        )

        absolute = difference.abs()

        tensor_maximum = float(
            absolute.max()
        ) if absolute.numel() else 0.0

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

        difference_square_sum += float(
            torch.sum(
                difference.float()
                * difference.float()
            )
        )

        reference_float = (
            reference_tensor.float()
        )

        reference_square_sum += float(
            torch.sum(
                reference_float
                * reference_float
            )
        )

        tensor_records.append(
            {
                "path": reference_path,
                "shape": list(
                    reference_tensor.shape
                ),
                "max_abs_error": (
                    tensor_maximum
                ),
                "mean_abs_error": float(
                    absolute.mean()
                ) if absolute.numel() else 0.0,
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
        "element_count": element_count,
        "max_abs_error": maximum,
        "mean_abs_error": mean_absolute,
        "relative_l2_error": (
            relative_l2
        ),
        "per_tensor": tensor_records,
    }


def graph_records(
    detection_model: nn.Module,
) -> list[dict[str, Any]]:
    records: list[
        dict[str, Any]
    ] = []

    for index, module in enumerate(
        detection_model.model
    ):
        records.append(
            {
                "index": index,
                "from": json.dumps(
                    getattr(
                        module,
                        "f",
                        None,
                    )
                ),
                "class": (
                    module.__class__.__name__
                ),
                "type": str(
                    getattr(
                        module,
                        "type",
                        module.__class__.__name__,
                    )
                ),
                "parameter_count": (
                    parameter_count(module)
                ),
            }
        )

    return records


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if requested.startswith("cuda"):
        return torch.device(requested)

    return torch.device(
        f"cuda:{requested}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    args = parser.parse_args()

    weights = Path(
        args.weights
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}"
        )

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.benchmark = False

    device = resolve_device(
        args.device
    )

    yolo = YOLO(
        str(weights)
    )

    baseline_model = (
        yolo.model
        .float()
        .to(device)
        .eval()
    )

    rep_model = copy.deepcopy(
        baseline_model
    ).to(device).eval()

    original_graph = graph_records(
        baseline_model
    )

    graph_csv = (
        output_dir
        / "top_level_graph.csv"
    )

    with graph_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                original_graph[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            original_graph
        )

    target_indices = (
        find_neck_downsample_indices(
            baseline_model
        )
    )

    if len(target_indices) != 2:
        raise RuntimeError(
            "expected exactly two Neck downsample Conv candidates, "
            f"found {len(target_indices)}: {target_indices}"
        )

    target_shapes: dict[
        str,
        dict[str, Any],
    ] = {}

    hook_handles = []

    for index in target_indices:
        def capture_shape(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            layer_index: int = index,
        ) -> None:
            target_shapes[
                str(layer_index)
            ] = {
                "input": shape_tree(
                    inputs
                ),
                "output": shape_tree(
                    output
                ),
            }

        hook_handles.append(
            baseline_model.model[
                index
            ].register_forward_hook(
                capture_shape
            )
        )

    input_tensor = torch.randn(
        1,
        3,
        args.imgsz,
        args.imgsz,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        baseline_output = (
            baseline_model(
                input_tensor
            )
        )

    for handle in hook_handles:
        handle.remove()

    baseline_parameter_count = (
        parameter_count(
            baseline_model
        )
    )

    replacement_manifest = (
        replace_neck_downsample_convs(
            rep_model,
            indices=target_indices,
        )
    )

    rep_training_parameter_count = (
        parameter_count(
            rep_model
        )
    )

    rep_training_count = (
        count_reparam_blocks(
            rep_model
        )
    )

    with torch.inference_mode():
        rep_training_output = (
            rep_model(
                input_tensor
            )
        )

    baseline_vs_training = (
        compare_outputs(
            baseline_output,
            rep_training_output,
        )
    )

    training_state_path = (
        output_dir
        / "reparam_training_state.pt"
    )

    torch.save(
        rep_model.state_dict(),
        training_state_path,
    )

    converted_block_count = (
        switch_reparam_blocks_to_deploy(
            rep_model
        )
    )

    rep_deploy_parameter_count = (
        parameter_count(
            rep_model
        )
    )

    with torch.inference_mode():
        rep_deploy_output = (
            rep_model(
                input_tensor
            )
        )

    training_vs_deploy = (
        compare_outputs(
            rep_training_output,
            rep_deploy_output,
        )
    )

    baseline_vs_deploy = (
        compare_outputs(
            baseline_output,
            rep_deploy_output,
        )
    )

    deploy_state_path = (
        output_dir
        / "reparam_deploy_state.pt"
    )

    torch.save(
        rep_model.state_dict(),
        deploy_state_path,
    )

    deployed = (
        all_reparam_blocks_deployed(
            rep_model
        )
    )

    threshold_initial = 1e-5
    threshold_deploy = 1e-4
    threshold_relative_l2 = 1e-5

    overall = (
        "PASS"
        if (
            target_indices == [17, 20]
            and rep_training_count == 2
            and converted_block_count == 2
            and deployed
            and (
                baseline_vs_training[
                    "max_abs_error"
                ]
                <= threshold_initial
            )
            and (
                training_vs_deploy[
                    "max_abs_error"
                ]
                <= threshold_deploy
            )
            and (
                baseline_vs_deploy[
                    "max_abs_error"
                ]
                <= threshold_deploy
            )
            and (
                training_vs_deploy[
                    "relative_l2_error"
                ]
                <= threshold_relative_l2
            )
        )
        else "FAIL"
    )

    payload = {
        "experiment": (
            "Exp04.1 YOLO11n "
            "Reparameterization Integration Probe"
        ),
        "weights": str(weights),
        "device": str(device),
        "imgsz": args.imgsz,
        "torch_version": (
            torch.__version__
        ),
        "baseline_output_shape": (
            shape_tree(
                baseline_output
            )
        ),
        "target_indices": (
            target_indices
        ),
        "target_shapes": (
            target_shapes
        ),
        "replacement_manifest": (
            replacement_manifest
        ),
        "baseline_parameter_count": (
            baseline_parameter_count
        ),
        "rep_training_parameter_count": (
            rep_training_parameter_count
        ),
        "rep_deploy_parameter_count": (
            rep_deploy_parameter_count
        ),
        "rep_training_block_count": (
            rep_training_count
        ),
        "converted_block_count": (
            converted_block_count
        ),
        "all_blocks_deployed": (
            deployed
        ),
        "baseline_vs_training": (
            baseline_vs_training
        ),
        "training_vs_deploy": (
            training_vs_deploy
        ),
        "baseline_vs_deploy": (
            baseline_vs_deploy
        ),
        "threshold_initial": (
            threshold_initial
        ),
        "threshold_deploy": (
            threshold_deploy
        ),
        "threshold_relative_l2": (
            threshold_relative_l2
        ),
        "training_state": str(
            training_state_path
        ),
        "deploy_state": str(
            deploy_state_path
        ),
        "overall": overall,
    }

    summary_json = (
        output_dir
        / "summary.json"
    )

    summary_json.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_csv = (
        output_dir
        / "replacement_manifest.csv"
    )

    with manifest_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                replacement_manifest[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            replacement_manifest
        )

    lines = [
        "=" * 60,
        (
            " Exp04.1 YOLO11n "
            "Reparameterization Integration Probe"
        ),
        "=" * 60,
        f"weights={weights}",
        f"device={device}",
        f"imgsz={args.imgsz}",
        (
            "torch_version="
            f"{torch.__version__}"
        ),
        (
            "target_indices="
            f"{target_indices}"
        ),
        (
            "target_shapes="
            f"{json.dumps(target_shapes)}"
        ),
        (
            "baseline_output_shape="
            f"{json.dumps(shape_tree(baseline_output))}"
        ),
        "",
        (
            "baseline_parameter_count="
            f"{baseline_parameter_count}"
        ),
        (
            "rep_training_parameter_count="
            f"{rep_training_parameter_count}"
        ),
        (
            "rep_deploy_parameter_count="
            f"{rep_deploy_parameter_count}"
        ),
        (
            "rep_training_block_count="
            f"{rep_training_count}"
        ),
        (
            "converted_block_count="
            f"{converted_block_count}"
        ),
        (
            "all_blocks_deployed="
            f"{deployed}"
        ),
        "",
        (
            "baseline_vs_training_max_abs_error="
            f"{baseline_vs_training['max_abs_error']:.12g}"
        ),
        (
            "baseline_vs_training_mean_abs_error="
            f"{baseline_vs_training['mean_abs_error']:.12g}"
        ),
        (
            "baseline_vs_training_relative_l2_error="
            f"{baseline_vs_training['relative_l2_error']:.12g}"
        ),
        "",
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
        "",
        (
            "baseline_vs_deploy_max_abs_error="
            f"{baseline_vs_deploy['max_abs_error']:.12g}"
        ),
        (
            "baseline_vs_deploy_mean_abs_error="
            f"{baseline_vs_deploy['mean_abs_error']:.12g}"
        ),
        (
            "baseline_vs_deploy_relative_l2_error="
            f"{baseline_vs_deploy['relative_l2_error']:.12g}"
        ),
        "",
        (
            "threshold_initial="
            f"{threshold_initial}"
        ),
        (
            "threshold_deploy="
            f"{threshold_deploy}"
        ),
        (
            "threshold_relative_l2="
            f"{threshold_relative_l2}"
        ),
        f"overall={overall}",
    ]

    summary_text = (
        "\n".join(lines)
        + "\n"
    )

    summary_txt = (
        output_dir
        / "summary.txt"
    )

    summary_txt.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        summary_text,
        end="",
    )

    return (
        0
        if overall == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
