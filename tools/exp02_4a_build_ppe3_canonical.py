#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SPLITS = ("train", "val", "test")

# 原始类别 → PPE3 类别
CLASS_MAPPING = {
    6: 0,  # Person -> person
    0: 1,  # helmet -> helmet
    2: 2,  # vest -> safety_vest
}

TARGET_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
}

SOURCE_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dataset-out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--report-out",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_hidden_relative(path: Path) -> bool:
    return any(
        part.startswith(".")
        for part in path.parts
    )


def read_boxes(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []

    boxes = []

    for line_number, raw_line in enumerate(
        label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        tokens = line.split()

        if len(tokens) != 5:
            raise RuntimeError(
                f"{label_path}:{line_number}: "
                f"expected 5 tokens, got {len(tokens)}"
            )

        class_float = float(tokens[0])

        if not class_float.is_integer():
            raise RuntimeError(
                f"{label_path}:{line_number}: "
                "class id is not an integer"
            )

        class_id = int(class_float)
        x, y, width, height = map(
            float,
            tokens[1:],
        )

        boxes.append(
            (
                class_id,
                x,
                y,
                width,
                height,
            )
        )

    return boxes


def center_inside(
    child: tuple[int, float, float, float, float],
    parent: tuple[int, float, float, float, float],
) -> bool:
    _, child_x, child_y, _, _ = child
    _, parent_x, parent_y, parent_w, parent_h = parent

    x1 = parent_x - parent_w / 2.0
    y1 = parent_y - parent_h / 2.0
    x2 = parent_x + parent_w / 2.0
    y2 = parent_y + parent_h / 2.0

    return (
        x1 <= child_x <= x2
        and y1 <= child_y <= y2
    )


def hardlink_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        os.link(source, destination)
        return "hardlink"

    except OSError:
        shutil.copy2(source, destination)
        return "copy"


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


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    dataset_out = args.dataset_out.resolve()
    report_out = args.report_out.resolve()

    if not source_root.is_dir():
        raise SystemExit(
            f"ERROR: source dataset not found: {source_root}"
        )

    if dataset_out.exists():
        raise SystemExit(
            f"ERROR: output already exists: {dataset_out}"
        )

    dataset_out.mkdir(
        parents=True,
        exist_ok=False,
    )

    report_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_out = dataset_out / "images" / "all"
    label_out = dataset_out / "labels" / "all"

    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    orphan_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    source_class_counts: Counter[int] = Counter()
    mapped_class_counts: Counter[int] = Counter()
    dropped_class_counts: Counter[int] = Counter()

    containment_total: Counter[int] = Counter()
    containment_inside_person: Counter[int] = Counter()

    image_hash_groups: dict[
        str,
        list[str],
    ] = defaultdict(list)

    valid_image_keys: dict[
        str,
        set[str],
    ] = {}

    hidden_images_ignored = 0
    transfer_counts: Counter[str] = Counter()
    zero_target_images = 0
    source_images = 0

    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split

        images = []

        for image_path in sorted(image_dir.rglob("*")):
            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            relative = image_path.relative_to(image_dir)

            if is_hidden_relative(relative):
                hidden_images_ignored += 1
                continue

            images.append(image_path)

        keys = {
            image.relative_to(image_dir)
            .with_suffix("")
            .as_posix()
            for image in images
        }

        valid_image_keys[split] = keys

        for image_path in images:
            source_images += 1

            relative = image_path.relative_to(image_dir)
            relative_key = relative.with_suffix("")

            label_path = (
                label_dir
                / relative_key
            ).with_suffix(".txt")

            raw_boxes = read_boxes(label_path)

            for box in raw_boxes:
                source_class_counts[box[0]] += 1

            person_boxes = [
                box
                for box in raw_boxes
                if box[0] == 6
            ]

            # 用于验证 helmet / vest / none / no_helmet
            # 与 Person 框的空间关系。
            for box in raw_boxes:
                class_id = box[0]

                if class_id not in (0, 2, 5, 7):
                    continue

                containment_total[class_id] += 1

                if any(
                    center_inside(box, person_box)
                    for person_box in person_boxes
                ):
                    containment_inside_person[
                        class_id
                    ] += 1

            mapped_boxes = []

            for box in raw_boxes:
                source_class_id = box[0]

                if source_class_id in CLASS_MAPPING:
                    mapped_class_id = CLASS_MAPPING[
                        source_class_id
                    ]

                    mapped_boxes.append(
                        (
                            mapped_class_id,
                            *box[1:],
                        )
                    )

                    mapped_class_counts[
                        mapped_class_id
                    ] += 1

                else:
                    dropped_class_counts[
                        source_class_id
                    ] += 1

            if not mapped_boxes:
                zero_target_images += 1

            safe_relative = "__".join(
                relative_key.parts
            )

            canonical_stem = (
                f"{split}__{safe_relative}"
            )

            canonical_image = (
                image_out
                / (
                    canonical_stem
                    + image_path.suffix.lower()
                )
            )

            canonical_label = (
                label_out
                / f"{canonical_stem}.txt"
            )

            if canonical_image.exists():
                raise RuntimeError(
                    f"Canonical filename collision: "
                    f"{canonical_image}"
                )

            transfer_action = hardlink_or_copy(
                image_path,
                canonical_image,
            )

            transfer_counts[transfer_action] += 1

            with canonical_label.open(
                "w",
                encoding="utf-8",
            ) as file:
                for box in mapped_boxes:
                    file.write(
                        f"{box[0]} "
                        f"{box[1]:.8f} "
                        f"{box[2]:.8f} "
                        f"{box[3]:.8f} "
                        f"{box[4]:.8f}\n"
                    )

            image_sha256 = sha256_file(
                image_path
            )

            image_hash_groups[
                image_sha256
            ].append(
                str(image_path)
            )

            manifest_rows.append(
                {
                    "source_split": split,
                    "source_image": str(image_path),
                    "source_label": (
                        str(label_path)
                        if label_path.is_file()
                        else ""
                    ),
                    "canonical_image": str(
                        canonical_image
                    ),
                    "canonical_label": str(
                        canonical_label
                    ),
                    "raw_box_count": len(raw_boxes),
                    "mapped_box_count": len(
                        mapped_boxes
                    ),
                    "dropped_box_count": (
                        len(raw_boxes)
                        - len(mapped_boxes)
                    ),
                    "image_sha256": image_sha256,
                    "transfer_action": transfer_action,
                }
            )

        label_paths = sorted(
            label_dir.rglob("*.txt")
        )

        for label_path in label_paths:
            relative = label_path.relative_to(
                label_dir
            )

            if is_hidden_relative(relative):
                continue

            key = relative.with_suffix("").as_posix()

            if key in keys:
                continue

            stem = relative.stem
            base_stem = (
                stem[:-3]
                if stem.endswith("(1)")
                else ""
            )

            base_label = (
                label_path.with_name(
                    base_stem + ".txt"
                )
                if base_stem
                else None
            )

            base_exists = (
                base_label.is_file()
                if base_label is not None
                else False
            )

            identical_to_base = False

            if base_exists and base_label is not None:
                identical_to_base = (
                    label_path.read_bytes()
                    == base_label.read_bytes()
                )

            orphan_rows.append(
                {
                    "split": split,
                    "orphan_label": str(label_path),
                    "base_label": (
                        str(base_label)
                        if base_label is not None
                        else ""
                    ),
                    "base_exists": base_exists,
                    "identical_to_base": (
                        identical_to_base
                    ),
                }
            )

    duplicate_group_count = 0

    for image_sha256, paths in sorted(
        image_hash_groups.items()
    ):
        if len(paths) < 2:
            continue

        duplicate_group_count += 1

        for path in paths:
            duplicate_rows.append(
                {
                    "group_id": (
                        f"duplicate_"
                        f"{duplicate_group_count:04d}"
                    ),
                    "sha256": image_sha256,
                    "group_size": len(paths),
                    "image_path": path,
                }
            )

    write_csv(
        report_out / "canonical_manifest.csv",
        manifest_rows,
        [
            "source_split",
            "source_image",
            "source_label",
            "canonical_image",
            "canonical_label",
            "raw_box_count",
            "mapped_box_count",
            "dropped_box_count",
            "image_sha256",
            "transfer_action",
        ],
    )

    write_csv(
        report_out / "orphan_label_analysis.csv",
        orphan_rows,
        [
            "split",
            "orphan_label",
            "base_label",
            "base_exists",
            "identical_to_base",
        ],
    )

    write_csv(
        report_out
        / "exact_duplicates_after_cleanup.csv",
        duplicate_rows,
        [
            "group_id",
            "sha256",
            "group_size",
            "image_path",
        ],
    )

    metadata = {
        "source_root": str(source_root),
        "dataset_out": str(dataset_out),
        "source_images": source_images,
        "hidden_images_ignored": (
            hidden_images_ignored
        ),
        "orphan_labels_ignored": len(
            orphan_rows
        ),
        "canonical_images": len(
            manifest_rows
        ),
        "canonical_labels": len(
            list(label_out.glob("*.txt"))
        ),
        "zero_target_images": (
            zero_target_images
        ),
        "source_class_counts": {
            SOURCE_NAMES.get(
                class_id,
                str(class_id),
            ): count
            for class_id, count
            in sorted(source_class_counts.items())
        },
        "mapped_class_counts": {
            TARGET_NAMES[class_id]: count
            for class_id, count
            in sorted(mapped_class_counts.items())
        },
        "dropped_class_counts": {
            SOURCE_NAMES.get(
                class_id,
                str(class_id),
            ): count
            for class_id, count
            in sorted(dropped_class_counts.items())
        },
        "containment_audit": {
            SOURCE_NAMES[class_id]: {
                "total": containment_total[
                    class_id
                ],
                "center_inside_person": (
                    containment_inside_person[
                        class_id
                    ]
                ),
                "ratio": (
                    containment_inside_person[
                        class_id
                    ]
                    / containment_total[class_id]
                    if containment_total[
                        class_id
                    ]
                    else None
                ),
            }
            for class_id in (0, 2, 5, 7)
        },
        "exact_duplicate_groups_after_cleanup": (
            duplicate_group_count
        ),
        "transfer_counts": dict(
            transfer_counts
        ),
    }

    (report_out / "summary.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset_yaml = {
        "path": str(dataset_out),
        "train": "images/all",
        "val": "images/all",
        "names": TARGET_NAMES,
        "note": (
            "Canonical unsplit PPE3 pool. "
            "Do not use for final training until "
            "group-aware split is generated."
        ),
    }

    (dataset_out / "canonical.yaml").write_text(
        yaml.safe_dump(
            dataset_yaml,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    (dataset_out / ".exp02_4a_complete").write_text(
        "PASS\n",
        encoding="utf-8",
    )

    print(
        "============================================================"
    )
    print(" Exp02.4a PPE3 Canonical Pool Summary")
    print(
        "============================================================"
    )
    print("build_execution=PASS")
    print(f"source_root={source_root}")
    print(f"dataset_out={dataset_out}")
    print(f"source_images={source_images}")
    print(
        f"hidden_images_ignored="
        f"{hidden_images_ignored}"
    )
    print(
        f"orphan_labels_ignored="
        f"{len(orphan_rows)}"
    )
    print(
        f"canonical_images={len(manifest_rows)}"
    )
    print(
        "canonical_labels="
        f"{len(list(label_out.glob('*.txt')))}"
    )
    print(
        f"zero_target_images="
        f"{zero_target_images}"
    )

    for class_id in range(3):
        print(
            f"class_{class_id}_"
            f"{TARGET_NAMES[class_id]}="
            f"{mapped_class_counts[class_id]}"
        )

    for source_class_id in (0, 2, 5, 7):
        total = containment_total[
            source_class_id
        ]

        inside = containment_inside_person[
            source_class_id
        ]

        ratio = (
            inside / total
            if total
            else 0.0
        )

        print(
            f"containment_"
            f"{SOURCE_NAMES[source_class_id]}="
            f"inside:{inside},"
            f"total:{total},"
            f"ratio:{ratio:.6f}"
        )

    print(
        "exact_duplicate_groups_after_cleanup="
        f"{duplicate_group_count}"
    )
    print(
        f"manifest="
        f"{report_out / 'canonical_manifest.csv'}"
    )
    print(
        f"summary_json="
        f"{report_out / 'summary.json'}"
    )
    print("exp02_4a_command_completed=YES")


if __name__ == "__main__":
    main()
