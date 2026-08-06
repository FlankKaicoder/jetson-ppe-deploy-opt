from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.blocks.reparam_block import (
    RepConvBlock,
)


def count_modules(
    module: nn.Module,
    module_type: type[nn.Module],
) -> int:
    return sum(
        1
        for item in module.modules()
        if isinstance(item, module_type)
    )


def parameter_count(
    module: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def warm_batch_norm(
    block: RepConvBlock,
    shape: tuple[int, int, int, int],
) -> None:
    """
    Update BN running mean and variance with random data.

    Using non-default BN statistics makes the fusion test more
    meaningful than comparing only freshly initialized layers.
    """

    block.train()

    with torch.no_grad():
        for _ in range(8):
            block(
                torch.randn(*shape)
            )

    block.eval()


def run_case(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    name = case["name"]
    shape = tuple(case["shape"])

    block = RepConvBlock(
        in_channels=case["in_channels"],
        out_channels=case["out_channels"],
        stride=case["stride"],
        deploy=False,
    )

    warm_batch_norm(
        block,
        shape,
    )

    input_tensor = torch.randn(
        *shape
    )

    conv_before = count_modules(
        block,
        nn.Conv2d,
    )

    bn_before = count_modules(
        block,
        nn.BatchNorm2d,
    )

    params_before = parameter_count(
        block
    )

    with torch.no_grad():
        reference_output = block(
            input_tensor
        )

    training_state_path = (
        output_dir
        / f"{name}_training_state.pt"
    )

    torch.save(
        block.state_dict(),
        training_state_path,
    )

    block.switch_to_deploy()

    # Verify that repeated conversion does not damage the module.
    block.switch_to_deploy()

    with torch.no_grad():
        deploy_output = block(
            input_tensor
        )

    absolute_difference = (
        reference_output
        - deploy_output
    ).abs()

    reference_norm = (
        torch.linalg.vector_norm(
            reference_output
        ).clamp_min(1e-12)
    )

    relative_l2_error = float(
        torch.linalg.vector_norm(
            reference_output
            - deploy_output
        )
        / reference_norm
    )

    deploy_state_path = (
        output_dir
        / f"{name}_deploy_state.pt"
    )

    torch.save(
        block.state_dict(),
        deploy_state_path,
    )

    result = {
        **case,
        "output_shape": list(
            deploy_output.shape
        ),
        "conv_before": conv_before,
        "conv_after": count_modules(
            block,
            nn.Conv2d,
        ),
        "bn_before": bn_before,
        "bn_after": count_modules(
            block,
            nn.BatchNorm2d,
        ),
        "params_before": params_before,
        "params_after": parameter_count(
            block
        ),
        "max_abs_error": float(
            absolute_difference.max()
        ),
        "mean_abs_error": float(
            absolute_difference.mean()
        ),
        "relative_l2_error": (
            relative_l2_error
        ),
        "training_state": str(
            training_state_path
        ),
        "deploy_state": str(
            deploy_state_path
        ),
    }

    result["result"] = (
        "PASS"
        if (
            result["max_abs_error"]
            <= 1e-5
            and result["conv_after"] == 1
            and result["bn_after"] == 0
        )
        else "FAIL"
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(42)
    torch.set_num_threads(1)

    cases = [
        {
            "name": (
                "same_channels_identity"
            ),
            "in_channels": 16,
            "out_channels": 16,
            "stride": 1,
            "shape": [
                2,
                16,
                32,
                32,
            ],
        },
        {
            "name": (
                "channel_projection"
            ),
            "in_channels": 16,
            "out_channels": 24,
            "stride": 1,
            "shape": [
                2,
                16,
                32,
                32,
            ],
        },
        {
            "name": (
                "stride2_projection"
            ),
            "in_channels": 16,
            "out_channels": 24,
            "stride": 2,
            "shape": [
                2,
                16,
                32,
                32,
            ],
        },
    ]

    results = [
        run_case(
            case,
            output_dir,
        )
        for case in cases
    ]

    overall = (
        "PASS"
        if all(
            item["result"] == "PASS"
            for item in results
        )
        else "FAIL"
    )

    payload = {
        "experiment": (
            "Exp04.0 "
            "Reparameterization Block Probe"
        ),
        "torch_version": (
            torch.__version__
        ),
        "threshold_max_abs_error": (
            1e-5
        ),
        "overall": overall,
        "cases": results,
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

    metrics_csv = (
        output_dir
        / "metrics.csv"
    )

    with metrics_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    lines = [
        "=" * 60,
        (
            " Exp04.0 Reparameterization "
            "Block Probe"
        ),
        "=" * 60,
        (
            "torch_version="
            f"{torch.__version__}"
        ),
        (
            "threshold_max_abs_error="
            "1e-5"
        ),
    ]

    for item in results:
        lines.extend(
            [
                "",
                (
                    "case="
                    f"{item['name']}"
                ),
                (
                    "output_shape="
                    f"{item['output_shape']}"
                ),
                (
                    "conv_before="
                    f"{item['conv_before']}"
                ),
                (
                    "conv_after="
                    f"{item['conv_after']}"
                ),
                (
                    "bn_before="
                    f"{item['bn_before']}"
                ),
                (
                    "bn_after="
                    f"{item['bn_after']}"
                ),
                (
                    "params_before="
                    f"{item['params_before']}"
                ),
                (
                    "params_after="
                    f"{item['params_after']}"
                ),
                (
                    "max_abs_error="
                    f"{item['max_abs_error']:.12g}"
                ),
                (
                    "mean_abs_error="
                    f"{item['mean_abs_error']:.12g}"
                ),
                (
                    "relative_l2_error="
                    f"{item['relative_l2_error']:.12g}"
                ),
                (
                    "result="
                    f"{item['result']}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            f"overall={overall}",
        ]
    )

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
