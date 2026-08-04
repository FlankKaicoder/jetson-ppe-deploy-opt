#!/usr/bin/env python3

import hashlib
import json
import shutil
from pathlib import Path

import ultralytics
import yaml


REPO_DIR = Path(
    "/root/autodl-tmp/jetson-ppe-deploy-opt"
).resolve()

PACKAGE_DIR = Path(
    ultralytics.__file__
).resolve().parent

SOURCE_YAML = (
    PACKAGE_DIR
    / "cfg"
    / "models"
    / "11"
    / "yolo11.yaml"
)

OUTPUT_DIR = REPO_DIR / "configs" / "models"

FROZEN_SOURCE = (
    OUTPUT_DIR
    / "upstream_yolo11_ultralytics_8.4.95.yaml"
)

CUSTOM_YAML = OUTPUT_DIR / "yolo11n_p2.yaml"
MANIFEST = OUTPUT_DIR / "yolo11n_p2_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> int:
    print("========== source ==========")
    print(f"ultralytics={ultralytics.__version__}")
    print(f"package_dir={PACKAGE_DIR}")
    print(f"source_yaml={SOURCE_YAML}")

    if not SOURCE_YAML.is_file():
        raise FileNotFoundError(
            f"YOLO11 source YAML not found: {SOURCE_YAML}"
        )

    source_config = yaml.safe_load(
        SOURCE_YAML.read_text(encoding="utf-8")
    )

    backbone = source_config.get("backbone")
    original_head = source_config.get("head")
    scales = source_config.get("scales")

    if not isinstance(backbone, list):
        raise RuntimeError(
            "source YOLO11 backbone is not a list"
        )

    if len(backbone) != 11:
        raise RuntimeError(
            "unexpected YOLO11 backbone length: "
            f"{len(backbone)}"
        )

    if not isinstance(original_head, list):
        raise RuntimeError(
            "source YOLO11 head is not a list"
        )

    if not isinstance(scales, dict) or "n" not in scales:
        raise RuntimeError(
            "source YOLO11 YAML has no n scale"
        )

    print(f"backbone_layers={len(backbone)}")
    print(f"original_head_layers={len(original_head)}")
    print(f"original_last_head={original_head[-1]}")

    # 基于 YOLO11 原始 P3/P4/P5 Head 增加：
    #
    # P3 上采样 -> 与 Backbone P2 拼接 -> P2 输出
    #
    # 再通过自底向上的路径恢复 P3、P4、P5，
    # 最终 Detect 接收四个尺度：
    #
    # P2/4, P3/8, P4/16, P5/32
    custom_head = [
        # P5 -> P4
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        # P4 -> P3
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],

        # P3 -> P2
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [128, False]],

        # P2 -> P3
        [-1, 1, "Conv", [128, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],

        # P3 -> P4
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],

        # P4 -> P5
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],

        # 四尺度检测
        [[19, 22, 25, 28], 1, "Detect", ["nc"]],
    ]

    custom_config = {
        "nc": source_config.get("nc", 80),
        "scale": "n",
        "scales": scales,
        "backbone": backbone,
        "head": custom_head,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        SOURCE_YAML,
        FROZEN_SOURCE,
    )

    yaml_text = yaml.safe_dump(
        custom_config,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )

    header = (
        "# Custom YOLO11n-P2 configuration\n"
        "# Derived from the local Ultralytics "
        f"{ultralytics.__version__} yolo11.yaml\n"
        "# Detection outputs: P2/4, P3/8, "
        "P4/16, P5/32\n"
        "# This is a project experimental "
        "configuration, not an official "
        "Ultralytics YOLO11-P2 release.\n\n"
    )

    CUSTOM_YAML.write_text(
        header + yaml_text,
        encoding="utf-8",
    )

    manifest = {
        "result": "PASS",
        "ultralytics_version": ultralytics.__version__,
        "source_yaml": str(SOURCE_YAML),
        "source_sha256": sha256(SOURCE_YAML),
        "frozen_source": str(FROZEN_SOURCE),
        "frozen_source_sha256": sha256(FROZEN_SOURCE),
        "custom_yaml": str(CUSTOM_YAML),
        "custom_sha256": sha256(CUSTOM_YAML),
        "scale": "n",
        "detection_levels": [
            "P2/4",
            "P3/8",
            "P4/16",
            "P5/32",
        ],
        "detect_from": [19, 22, 25, 28],
        "custom_head_layers": len(custom_head),
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("========== generated ==========")
    print(f"frozen_source={FROZEN_SOURCE}")
    print(f"custom_yaml={CUSTOM_YAML}")
    print(f"manifest={MANIFEST}")
    print(f"source_sha256={manifest['source_sha256']}")
    print(f"custom_sha256={manifest['custom_sha256']}")
    print("exp03_0_create_yolo11n_p2_config=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
