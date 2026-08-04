#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO


def describe_output(
    value: Any,
    prefix: str = "output",
) -> list[str]:
    lines: list[str] = []

    if isinstance(value, torch.Tensor):
        lines.append(
            f"{prefix}: tensor "
            f"shape={list(value.shape)} "
            f"dtype={value.dtype}"
        )
        return lines

    if isinstance(value, (list, tuple)):
        lines.append(
            f"{prefix}: {type(value).__name__} "
            f"length={len(value)}"
        )

        for index, item in enumerate(value):
            lines.extend(
                describe_output(
                    item,
                    f"{prefix}[{index}]",
                )
            )

        return lines

    lines.append(
        f"{prefix}: type={type(value).__name__}"
    )

    return lines


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        required=True,
    )

    args = parser.parse_args()

    config = Path(args.config).resolve()
    report_dir = Path(args.report_dir).resolve()

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not config.is_file():
        raise FileNotFoundError(
            f"custom model config not found: {config}"
        )

    print("============================================================")
    print(" Exp03.0 Custom YOLO11n-P2 Architecture Probe")
    print("============================================================")
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"ultralytics={ultralytics.__version__}")
    print(f"config={config}")

    print()
    print("========== model construction ==========")

    model = YOLO(
        str(config),
        task="detect",
    )

    network = model.model
    network.eval()

    model.info(
        detailed=False,
        verbose=True,
    )

    parameters = sum(
        parameter.numel()
        for parameter in network.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )

    stride = [
        float(value)
        for value in (
            network.stride
            .detach()
            .cpu()
            .tolist()
        )
    ]

    detect = network.model[-1]

    detect_class = type(detect).__name__
    detect_nc = getattr(detect, "nc", None)
    detect_nl = getattr(detect, "nl", None)
    detect_from = getattr(detect, "f", None)

    print()
    print("========== architecture properties ==========")
    print(f"parameters={parameters}")
    print(
        f"trainable_parameters="
        f"{trainable_parameters}"
    )
    print(f"stride={stride}")
    print(f"detect_class={detect_class}")
    print(f"detect_nc={detect_nc}")
    print(f"detect_nl={detect_nl}")
    print(f"detect_from={detect_from}")
    print(f"network_scale={network.yaml.get('scale')}")

    print()
    print("========== forward probe ==========")

    dummy = torch.zeros(
        1,
        3,
        640,
        640,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = network(dummy)

    output_lines = describe_output(output)

    for line in output_lines:
        print(line)

    expected_stride = [
        4.0,
        8.0,
        16.0,
        32.0,
    ]

    passed = (
        stride == expected_stride
        and detect_class == "Detect"
        and detect_nl == 4
        and list(detect_from) == [
            19,
            22,
            25,
            28,
        ]
    )

    summary = {
        "experiment": (
            "Exp03.0 custom YOLO11n-P2 "
            "architecture probe"
        ),
        "result": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "config": str(config),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "stride": stride,
        "detect_class": detect_class,
        "detect_nc": detect_nc,
        "detect_nl": detect_nl,
        "detect_from": list(detect_from),
        "network_scale": network.yaml.get("scale"),
        "forward_output": output_lines,
    }

    (report_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "============================================================",
        " Exp03.0 Custom YOLO11n-P2 Probe Summary",
        "============================================================",
        f"result={summary['result']}",
        f"config={config}",
        f"parameters={parameters}",
        f"trainable_parameters={trainable_parameters}",
        f"stride={stride}",
        f"detect_class={detect_class}",
        f"detect_nc={detect_nc}",
        f"detect_nl={detect_nl}",
        f"detect_from={list(detect_from)}",
        f"network_scale={network.yaml.get('scale')}",
        (
            "exp03_0_yolo11n_p2_probe="
            f"{summary['result']}"
        ),
    ]

    (report_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        (report_dir / "summary.txt").read_text(
            encoding="utf-8"
        ),
        end="",
    )

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
